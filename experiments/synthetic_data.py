import json
import torch
import random
from typing import List
from diffusers import StableDiffusion3Pipeline
from datasets import Dataset, Features, Value, Image
from PIL import Image as PILImage
import numpy as np


class EnhancedFurnitureDataset:
    def __init__(self):
        torch.set_float32_matmul_precision("high")

        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", 
                                                        torch_dtype=torch.float16,
                                                        use_safetensors=True,
                                                    ).to("cuda")  

    def generate_enhanced_prompt(self, item: dict) -> str:
        base_prompt = f"Professional product photography of a {item['style']} {item['color']} {item['type']}, "
        base_prompt += f"made of {item['material']}, {item['shape']} shape, featuring {item['details']}, "
        base_prompt += f"{item['room_type']} in mind, {item['price_range']} price range. "

        # Add technical requirements
        technical = "Pure white background, studio lighting, commercial product photography, 8k, ultra detailed, "
        technical += "centered composition, professional furniture catalog style"
        
        return base_prompt + technical
    
    def generate_furniture_image(self, pipe, prompt, negative_prompt):    
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=60,  
            guidance_scale=8.5,      
            width=512,              
            height=512,
            num_images_per_prompt=1
        ).images[0]

    def generate_training_sample(self, item: dict) -> List[dict]:
        samples = []
        
        prompt = self.generate_enhanced_prompt(item)
        
        image = self.generate_furniture_image(
            pipe=self.pipe,
            prompt=prompt,
            negative_prompt="defects, damage, watermarks, text, artifacts, blurry, noisy",
        )
        
        samples.append({
            "image": image,
            "attributes": item,
            "prompt": prompt
        })
            
        return samples

    def generate_base_item(self, attributes):
        return {
            "type": attributes["type"],
            "style": random.choice(attributes["style"]),
            "color": random.choice(attributes["color"]),
            "material": random.choice(attributes["material"]),
            "shape": random.choice(attributes["shape"]),
            "details": random.choice(attributes["details"]),
            "room_type": random.choice(attributes["room_type"]),
            "price_range": random.choice(attributes["price_range"])
        }
def save_to_huggingface(dataset, repo_name):
    # Prepare dataset dictionary
    dataset_dict = {
        "image": [],
        "type": [],
        "style": [],
        "color": [],
        "material": [],
        "shape": [],
        "details": [],
        "room_type": [],
        "price_range": [],
        "prompt": []
    }
    
    # Fill dataset dictionary
    for sample in dataset:
        attrs = sample["attributes"]
        for key in dataset_dict.keys():
            if key == "image":
                # Convert numpy array to PIL Image
                if isinstance(sample["image"], np.ndarray):
                    img = PILImage.fromarray(sample["image"])
                else:
                    img = sample["image"]
                dataset_dict[key].append(img)
            elif key == "prompt":
                dataset_dict[key].append(sample.get("prompt", ""))
            else:
                dataset_dict[key].append(attrs.get(key, ""))

    # Create features specification
    features = Features({
        "image": Image(),
        "type": Value("string"),
        "style": Value("string"),
        "color": Value("string"),
        "material": Value("string"),
        "shape": Value("string"),
        "details": Value("string"),
        "room_type": Value("string"),
        "price_range": Value("string"),
        "prompt": Value("string")
    })

    # Create HuggingFace dataset
    hf_dataset = Dataset.from_dict(dataset_dict, features=features)
    hf_dataset.push_to_hub(repo_name)
    
    return hf_dataset

def generate_enhanced_dataset(num_items_per_type=20):
    dataset = []
    generator = EnhancedFurnitureDataset()

    with open('furniture_attributes.json', 'r') as f:
        furniture_attributes = json.load(f)
    
    for furniture_type, attrs in furniture_attributes.items():
        attrs = attrs.copy()
        attrs["type"] = furniture_type
        
        for _ in range(num_items_per_type):
            item = generator.generate_base_item(attrs)
            
            samples = generator.generate_training_sample(item)
            dataset.extend(samples)
    
    return dataset

if __name__ == "__main__":
    enhanced_dataset = generate_enhanced_dataset(num_items_per_type=2500) 
    
    hf_dataset = save_to_huggingface(
        enhanced_dataset, 
        "filnow/furniture-synthetic-dataset"
    )