from huggingface_hub import login
login()

import wandb
wandb.login()

import json
from datasets import load_dataset
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import TrainingArguments
import os 
from transformers import Trainer


SYSTEM_MESSAGE = """You are a furniture expert. Analyze images and provide descriptions in this exact JSON structure:
{
    "type": "<must be one of: bed, chair, table, sofa>",
    "style": "<describe overall style>",
    "color": "<describe main color>",
    "material": "<describe primary material>"
    "details": "<describe one decorative feature>",
    "room_type": "<specify room type>"
}
Focus on maintaining this exact structure while providing relevant descriptions."""

def create_feature_dict(row):
    feature_dict = {
        'type': row['type'],
        'style': row['style'],
        'color': row['color'],
        'material': row['material'],
        'details': row['details'],
        'room_type': row['room_type']
    }
    return json.dumps(feature_dict)

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": "Describe this furniture piece in JSON format.",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample['features_json']}],
        },
    ]

ds = load_dataset("filnow/furniture-synthetic-dataset-30k")

train_dataset = ds['train']
eval_dataset = ds['test']

json_features = [create_feature_dict(item) for item in train_dataset]
json_features_eval = [create_feature_dict(item) for item in eval_dataset]

# Add new feature to dataset
train_dataset = train_dataset.add_column("features_json", json_features)
eval_dataset = eval_dataset.add_column("features_json", json_features_eval)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

processor.tokenizer.padding_side = "left"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation

peft_model = get_peft_model(model, peft_config)

# Print trainable parameters

peft_model.print_trainable_parameters()

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(format_data(example), tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(format_data(example))[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels 

    return batch  

os.environ["WANDB_PROJECT"] = "furniture-qwen2vl-30k"

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
    #evaluation_strategy="steps",  # Add evaluation
    eval_steps=500,    # Evaluate every 500 steps
    eval_strategy="steps",
    
    # Saving settings
    optim="adamw_torch_fused",  # Use fused AdamW for better performance
    save_strategy="steps",
    save_steps=500,    # Save more frequently
    save_total_limit=3,  # Keep more checkpoints
    
    # Output and hardware settings
    output_dir="furniture-qwen2vl-25k",
    bf16=True,
    report_to="wandb",
    run_name="furniture-qwen2.5-VLM-3B-30k",
    dataloader_pin_memory=False,
    
    max_grad_norm=1.0,

    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        args=args
        )

trainer.train()

wandb.finish()

trainer.push_to_hub()