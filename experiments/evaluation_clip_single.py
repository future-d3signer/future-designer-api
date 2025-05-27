import wandb
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Set
import logging
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import CLIPModel, CLIPProcessor
from functools import partial

@dataclass
class EvaluationConfig:
    project_name: str = "furniture-labeling-single"
    batch_size: int = 100
    attributes: Set[str] = frozenset({
        'type', 'style', 'color', 'material', 
        'shape', 'details', 'room_type', 'price_range'
    })
    log_dir: str = "logs"
    run_name: str = "Qwen2VL-72B-0.8T-32-336"
    model_name: str = "openai/clip-vit-large-patch14-336"
    
class CLIPEvaluator:
    def __init__(self, dataset: Dict[str, Any], config: EvaluationConfig):
        self.dataset = dataset
        self.config = config
        self.setup_logging()
        
        # Initialize per-attribute tracking
        self.attribute_scores = {attr: [] for attr in config.attributes}
        self.total_scores = {attr: 0.0 for attr in config.attributes}
        self.successful_evals = {attr: 0 for attr in config.attributes}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model and processor to GPU
        self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)

        self.clip_score_fn = partial(self.clip_score, model=self.model, processor=self.processor)
    
    def clip_score(self, image_tensor, text, model, processor):
    # Process image and text
        inputs = processor(images=image_tensor, text=text, return_tensors="pt", padding=True)
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute image-text similarity
        with torch.no_grad():
            image_embeds = model.get_image_features(**{k: inputs[k] for k in ['pixel_values'] if k in inputs})
            text_embeds = model.get_text_features(**{k: inputs[k] for k in ['input_ids', 'attention_mask'] if k in inputs})
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            similarity = (image_embeds * text_embeds).sum(dim=-1)
        
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
    
    def format_attribute_prompt(self, sample: Dict[str, str], attribute: str) -> str:
        """Format prompt for a single attribute"""
        try:
            # Create focused prompts for each attribute
            prompts = {
                'type': f"This is a {sample['type']}",
                'style': f"This furniture piece has a {sample['style']} style",
                'color': f"The color of this furniture is {sample['color']}",
                'material': f"This furniture is made of {sample['material']}",
                'shape': f"The shape of this furniture is {sample['shape']}",
                'details': f"This furniture features {sample['details']}",
                'room_type': f"This furniture belongs in a {sample['room_type']}",
                'price_range': f"This is a {sample['price_range']} price range furniture piece"
            }
            return prompts[attribute]
        except KeyError as e:
            logging.error(f"Missing key in sample data: {e}")
        return ""
    
    def log_batch_metrics(self, batch_idx: int):
        """Log metrics for the current batch"""
        for attr in self.config.attributes:
            recent_scores = self.attribute_scores[attr][-self.config.batch_size:]
            if not recent_scores:
                continue
                
            wandb.log({
                f"{attr}_distribution": wandb.Histogram(np.array(recent_scores)),
                f"{attr}_batch_average": np.mean(recent_scores),
                f"{attr}_batch_std": np.std(recent_scores),
                f"batch_step": batch_idx
            })
        
        # Log overall average across all attributes
        all_recent_scores = []
        for attr in self.config.attributes:
            if self.attribute_scores[attr]:
                all_recent_scores.extend(self.attribute_scores[attr][-self.config.batch_size:])
        
        if all_recent_scores:
            wandb.log({
                "overall_batch_average": np.mean(all_recent_scores),
                "overall_batch_std": np.std(all_recent_scores)
            })
            
        logging.info(f"Batch {batch_idx}: Overall Average Score = {np.mean(all_recent_scores):.2f}")
    
    def log_final_metrics(self):
        """Log final evaluation metrics"""
        # Calculate final metrics for each attribute
        final_metrics = {}
        attribute_averages = []
        
        for attr in self.config.attributes:
            if not self.attribute_scores[attr]:
                logging.warning(f"No successful evaluations for attribute: {attr}")
                continue
                
            scores_array = np.array(self.attribute_scores[attr])
            avg_score = np.mean(scores_array)
            attribute_averages.append(avg_score)
            
            final_metrics.update({
                f"{attr}_final_average": avg_score,
                f"{attr}_final_std": np.std(scores_array),
                f"{attr}_final_min": np.min(scores_array),
                f"{attr}_final_max": np.max(scores_array),
                f"{attr}_distribution_final": wandb.Histogram(scores_array),
                f"{attr}_successful_evaluations": self.successful_evals[attr]
            })
            
            # Create per-attribute visualization
            wandb.log({f"{attr}_scores": wandb.plot.line_series(
                xs=[[x for x in range(len(self.attribute_scores[attr]))]],
                ys=[self.attribute_scores[attr]],
                keys=[f"{attr} Score"],
                title=f"CLIP Scores for {attr}",
                xname="Image Index")
            })
        
        # Calculate and log overall metrics
        if attribute_averages:
            final_metrics.update({
                "overall_final_average": np.mean(attribute_averages),
                "overall_final_std": np.std(attribute_averages),
                "total_samples": len(self.dataset['train']),
                "average_success_rate": np.mean([
                    (self.successful_evals[attr] / len(self.dataset['train'])) * 100 
                    for attr in self.config.attributes
                ])
            })
        
        wandb.log(final_metrics)
        logging.info(f"Evaluation completed. Overall average score: {final_metrics.get('overall_final_average', 0):.2f}")
    
    def evaluate(self):
        """Run evaluation on the dataset"""
        try:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config={
                    "dataset_size": len(self.dataset['train']),
                    "batch_size": self.config.batch_size,
                    "attributes": list(self.config.attributes)
                }
            )
            
            for attr in self.config.attributes:
                wandb.define_metric(f"{attr}_score", summary="mean")
            
            for i in tqdm(range(len(self.dataset['train'])), desc="Evaluating"):
                sample = self.dataset['train'][i]
                
                # Evaluate each attribute separately
                for attr in self.config.attributes:
                    try:
                        prompt = self.format_attribute_prompt(sample, attr)
                        if not prompt:
                            continue
                            
                        clip_score = self.calculate_clip_score(sample['image'], prompt) * 100
                        
                        self.total_scores[attr] += clip_score
                        self.attribute_scores[attr].append(clip_score)
                        self.successful_evals[attr] += 1
                        
                        wandb.log({
                            f"{attr}_score": clip_score,
                            "step": i
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing {attr} for sample {i}: {str(e)}")
                        continue
                
                if i % self.config.batch_size == 0:
                    self.log_batch_metrics(i)
                    
            self.log_final_metrics()
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise
        finally:
            wandb.finish()

def main():
    dataset = load_dataset("filnow/furniture-Qwen2VL-72B-0.8T")

    config = EvaluationConfig()
    evaluator = CLIPEvaluator(dataset, config)
    evaluator.evaluate()

if __name__ == "__main__":
    main()