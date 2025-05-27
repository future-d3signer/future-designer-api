from huggingface_hub import login
login()

import wandb
wandb.login()

from datasets import load_dataset
from transformers import PaliGemmaProcessor
from transformers import PaliGemmaForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments
import os 
from transformers import Trainer
import json


def create_feature_dict(row):
    feature_dict = {
        'type': row['type'],
        'style': row['style'],
        'color': row['color'],
        'material': row['material'],
        'shape': row['shape'],
        'details': row['details'],
        'room_type': row['room_type'],
        'price_range': row['price_range']
    }
    # Convert dictionary to string
    return json.dumps(feature_dict)

def collate_fn(examples):
  question = "describe furniture on the image in JSON format."
  texts = ["<image>answer en " + question for example in examples]
  labels= [example['features_json'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")

  tokens = tokens.to(DTYPE).to(device)
  return tokens


ds = load_dataset("filnow/furniture-synthetic-dataset")
model_id ="google/paligemma2-3b-pt-448"

train_ds = ds["train"]
eval_dataset = ds["test"]

# First convert the existing code output to a feature column
json_features = [create_feature_dict(item) for item in train_ds]
json_features_eval = [create_feature_dict(item) for item in eval_dataset]

# Add new feature to dataset
train_ds = train_ds.add_column("features_json", json_features)
eval_dataset = eval_dataset.add_column("features_json", json_features_eval)

device = "cuda"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=8,                     # Increased rank for better capacity
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Core attention modules
        "gate_proj", "up_proj", "down_proj"       # MLP modules
    ],
    task_type="CAUSAL_LM",
)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")#, quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

DTYPE = model.dtype

processor = PaliGemmaProcessor.from_pretrained(model_id)

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

os.environ["WANDB_PROJECT"] = "furniture-paligemma"

args = TrainingArguments(
    # Increase epochs since your dataset isn't very large
    num_train_epochs=5,
    remove_unused_columns=False,
    
    # Increase batch size for better stability
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Reduced since we increased batch size
    
    # Learning rate schedule adjustments
    warmup_steps=100,  # Increase warmup steps
    learning_rate=1e-4,  # Slightly lower learning rate for better stability
    lr_scheduler_type="cosine",  # Add cosine scheduler for better convergence
    
    # Weight decay and optimizer settings
    weight_decay=0.01,  # Increase weight decay to prevent overfitting
    adam_beta2=0.999,
    
    # Evaluation and logging
    logging_steps=50,  # More frequent logging
    evaluation_strategy="steps",  # Add evaluation
    eval_steps=500,    # Evaluate every 500 steps
    
    # Saving settings
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=500,    # Save more frequently
    save_total_limit=3,  # Keep more checkpoints
    
    # Output and hardware settings
    output_dir="furniture-paligemma",
    bf16=True,
    report_to="wandb",
    run_name="furniture-paligemma-claude",
    dataloader_pin_memory=False,
    
    # Add gradient clipping to prevent instability
    max_grad_norm=1.0,
    
    # Add early stopping
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        args=args
        )

trainer.train()

wandb.finish()

trainer.push_to_hub()
