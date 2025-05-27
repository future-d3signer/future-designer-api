import wandb
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import CLIPModel, CLIPProcessor
from functools import partial

@dataclass
class EvaluationConfig:
    project_name: str = "furniture-poster-evaluation"
    batch_size: int = 100
    #prompt_template: str = "{type}, {style}, {color}, {material}, {shape}, {details}, {room_type}, {price_range}"
    prompt_template: str = "Professional studio photograph of a {type}, showcasing a {style} design. The piece features a {color} color palette and is crafted from {material}. It has a distinctive {shape} shape with {details} as a key design element. Perfect for a {room_type}, this {price_range} furniture piece is captured in optimal lighting conditions."
    log_dir: str = "logs"
    run_name: str = "Qwen2VL-7B"
    model_name: str = "openai/clip-vit-large-patch14"
    mode: str = "train"
    
class CLIPEvaluator:
    def __init__(self, dataset: Dict[str, Any], config: EvaluationConfig):
        self.dataset = dataset
        self.config = config
        self.setup_logging()
        
        self.clip_scores: List[float] = []
        self.total_score = 0
        self.successful_evals = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)

        self.clip_score_fn = partial(self.clip_score, model=self.model, processor=self.processor)
    
    def clip_score(self, image_tensor, text, model, processor, weight=2.5):
    # Process image and text
        inputs = processor(images=image_tensor, text=text, return_tensors="pt", padding=True)
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute image-text similarity
        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_embeds = model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            cosine_similarity = (image_embeds * text_embeds).sum(dim=-1)
            similarity = weight * torch.max(cosine_similarity, torch.zeros_like(cosine_similarity))
            
        return similarity

    def calculate_clip_score(self, images, prompts):
        if isinstance(images, torch.Tensor):
            image_numpy = images.cpu().numpy()
        else:
            image_numpy = np.array(images)
        
        # Handle single image case
        if len(image_numpy.shape) == 2:
            return 0
        
        # Ensure correct shape
        if len(image_numpy.shape) == 3:
            image_numpy = image_numpy.reshape(1, *image_numpy.shape)
        
        # Convert to uint8 and to tensor
        images_int = (image_numpy * 255).astype("uint8")
        image_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2).to(self.device)
        
        # Compute CLIP score
        clip_score = self.clip_score_fn(image_tensor, prompts).detach()
        return round(float(clip_score), 6)
    
    def setup_logging(self):
        """Configure logging to both file and console"""
        log_path = Path(self.config.log_dir)
        log_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
    
    def format_prompt(self, sample: Dict[str, str]) -> str:
        """Format prompt from sample data"""
        try:
            return self.config.prompt_template.format(**sample)
        except KeyError as e:
            logging.error(f"Missing key in sample data: {e}")
            return ""
    
    def log_batch_metrics(self, batch_idx: int):
        """Log metrics for the current batch"""
        recent_scores = self.clip_scores[-self.config.batch_size:]
        
        wandb.log({
            "clip_score_distribution": wandb.Histogram(np.array(recent_scores)),
            "batch_average": np.mean(recent_scores),
            "batch_std": np.std(recent_scores),
            "batch_step": batch_idx,
            "running_average": self.total_score / self.successful_evals
        })
        
        logging.info(f"Batch {batch_idx}: Average Score = {np.mean(recent_scores):.2f}")
    
    def log_final_metrics(self):
        """Log final evaluation metrics"""
        if not self.clip_scores:
            logging.warning("No successful evaluations to report")
            return
            
        scores_array = np.array(self.clip_scores)
        final_metrics = {
            "final_average_clip_score": np.mean(scores_array),
            "final_std": np.std(scores_array),
            "final_min": np.min(scores_array),
            "final_max": np.max(scores_array),
            "score_distribution_final": wandb.Histogram(scores_array),
            "successful_evaluations": self.successful_evals,
            "total_samples": len(self.dataset[self.config.mode]),
            "success_rate": (self.successful_evals / len(self.dataset[self.config.mode])) * 100
        }
        
        wandb.log(final_metrics)
        
        # Create final visualization
        wandb.log({"clip_scores": wandb.plot.line_series(
            xs=[[x for x in range(len(self.clip_scores))]],
            ys=[self.clip_scores],
            keys=["CLIP Score"],
            title="CLIP Scores Over Dataset",
            xname="Image Index")
        })
        
        logging.info(f"Evaluation completed. Average score: {final_metrics['final_average_clip_score']:.2f}")
    
    def evaluate(self):
        """Run evaluation on the dataset"""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config={
                    "dataset_size": len(self.dataset[self.config.mode]),
                    "batch_size": self.config.batch_size,
                    "prompt_template": self.config.prompt_template
                }
            )
            
            wandb.define_metric("individual_clip_score", summary="mean")
            wandb.define_metric("running_average", summary="last")
            
            for i in tqdm(range(len(self.dataset[self.config.mode])), desc="Evaluating"):
                try:
                    sample = self.dataset[self.config.mode][i]
                    prompt = self.format_prompt(sample)
    
                    if not prompt:
                        continue
                        
                    clip_score = self.calculate_clip_score(sample['image'], prompt) * 100
                    
                    self.total_score += clip_score
                    self.clip_scores.append(clip_score)
                    self.successful_evals += 1
                    
                    wandb.log({
                        "individual_clip_score": clip_score,
                        "step": i
                    })
                    
                    if i % self.config.batch_size == 0 and self.clip_scores:
                        self.log_batch_metrics(i)
                        
                except Exception as e:
                    logging.error(f"Error processing sample {i}: {str(e)}")
                    continue
                    
            self.log_final_metrics()
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise
        finally:
            wandb.finish()

def main():
    dataset = load_dataset("filnow/futniture-qwen2-vl-7b")

    config = EvaluationConfig()
    evaluator = CLIPEvaluator(dataset, config)
    evaluator.evaluate()

if __name__ == "__main__":
    main()