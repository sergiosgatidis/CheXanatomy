"""
Simplified Paligemma Fine-tuning Script for CheXanatomy

This script provides a streamlined approach to fine-tuning Paligemma models
on chest X-ray anatomy data using our PaligemmaSampleGenerator.

Usage:
    python scripts/train_paligemma.py [--config path/to/config.yaml]

Requirements:
    - transformers
    - torch
    - peft (for LoRA)
    - wandb (optional, for logging)
    - pyyaml
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Paligemma model on CheXanatomy data')
parser.add_argument('--config', type=str, default='config.yaml', 
                   help='Path to config YAML file (default: config.yaml)')
args = parser.parse_args()

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
config = load_config(config_path)
from src.paligemma_generator import PaligemmaSampleGenerator
from PIL import Image
import torch
import random
import json
from torch.utils.data import Dataset
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# Optional wandb import
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    print("wandb not available. Install with: pip install wandb")

print("Starting Paligemma training script...")
print(f"Using config: {config_path}")
print(f"Using model: {config['model']['model_id']}")
print(f"Training data path: {config['data']['training_data_path']}")

# =============================================================================
# DATA LOADING
# =============================================================================

def find_image_info_files(data_path):
    """Find all image_info.json files in the data directory"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data path does not exist: {data_path}")
    
    # Look for image_info.json files
    json_files = list(data_path.glob("**/image_info.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No image_info.json files found in {data_path}")
    
    print(f"Found {len(json_files)} image_info.json files")
    return [str(f) for f in json_files]

# Find all training data files
file_paths = find_image_info_files(config['data']['training_data_path'])
random.shuffle(file_paths)  # Shuffle for better training

# Split into train/validation
split_index = int(len(file_paths) * (1 - config['data']['train_val_split']))
train_file_paths = file_paths[:split_index]
val_file_paths = file_paths[split_index:]

print(f"Training files: {len(train_file_paths)}")
print(f"Validation files: {len(val_file_paths)}")

# =============================================================================
# DATASET CLASS
# =============================================================================

class CheXanatomyDataset(Dataset):
    """
    Dataset class that dynamically generates training samples using PaligemmaSampleGenerator
    """
    
    def __init__(self, file_paths, enable_augmentation=True):
        self.file_paths = file_paths
        self.enable_augmentation = enable_augmentation
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            # Get the image info file path
            img_info_path = self.file_paths[idx]
            
            # Initialize generator with this file
            generator = PaligemmaSampleGenerator(
                img_info_path=img_info_path,
                vqvae_model_path=config['paths']['vqvae_model_path'],
                enable_augmentation=self.enable_augmentation
            )
                        
            # Get available structures from the image
            with open(img_info_path, 'r') as f:
                img_info = json.load(f)
            
            structures = list(img_info.get("structure_info", {}).keys())
            
            # Exclude specific structures that are not useful for training
            excluded_structures = ['torso_fat', 'subcutaneous_fat', 'intervertebral_discs']
            structures = [s for s in structures if s not in excluded_structures]
            
            if not structures:
                # If no structures, try next file
                return self.__getitem__((idx + 1) % len(self.file_paths))
            
            # Randomly select a task and structure
            task = random.choice(config['data']['available_tasks'])
            structure = random.choice(structures)
            
            # Generate the sample
            sample = generator.generate_sample(structure=structure, task_name=task)
            
            if sample is None:
                # If sample generation failed, try next file
                return self.__getitem__((idx + 1) % len(self.file_paths))
            
            # Resize image to training size
            image = sample.image.resize((config['model']['input_image_size'], config['model']['input_image_size']), resample=Image.BICUBIC)
            
            return {
                "prefix": sample.prefix,
                "suffix": sample.suffix, 
                "image": image,
                "task_type": sample.task_type,
                "structure": sample.structure
            }
            
        except Exception as e:
            print(f"Error processing file {self.file_paths[idx]}: {e}")
            # Try next file on error
            return self.__getitem__((idx + 1) % len(self.file_paths))

# =============================================================================
# MODEL SETUP
# =============================================================================

print("Loading model and processor...")

# Initialize wandb if available and configured
if wandb_available and config.get('wandb', {}).get('project'):
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'),
        config=config,
        tags=config['wandb'].get('tags', []),
        notes=config['wandb'].get('notes', '')
    )
    print("Wandb initialized")

# Load processor
processor = PaliGemmaProcessor.from_pretrained(config['model']['model_id'])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Setup quantization if using QLoRA
if config['optimization']['use_qlora']:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
else:
    bnb_config = None

# Load model
model = PaliGemmaForConditionalGeneration.from_pretrained(
    config['model']['model_id'],
    attn_implementation='eager',
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16
)

# Apply LoRA if enabled
if config['optimization']['use_lora'] or config['optimization']['use_qlora']:
    lora_config = LoraConfig(
        r=config['optimization']['lora_r'],
        target_modules=config['optimization']['lora_target_modules'],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configuration applied:")
    model.print_trainable_parameters()

# Freeze vision tower if specified
if config['optimization']['freeze_vision'] and not (config['optimization']['use_lora'] or config['optimization']['use_qlora']):
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
    print("Vision tower frozen")

# =============================================================================
# DATA COLLATION
# =============================================================================

def collate_fn(examples):
    """
    Collate function for batching training samples
    """
    texts = ["<image>" + example["prefix"] for example in examples]
    labels = [example['suffix'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    
    # Use processor to tokenize and process inputs
    tokens = processor(
        text=texts, 
        images=images, 
        suffix=labels,
        return_tensors="pt", 
        padding="longest"
    )
    
    return tokens

# =============================================================================
# TRAINING SETUP
# =============================================================================

# Create datasets
print("Creating datasets...")
train_ds = CheXanatomyDataset(train_file_paths, enable_augmentation=config['data']['enable_augmentation'])
val_ds = CheXanatomyDataset(val_file_paths, enable_augmentation=False)  # No augmentation for validation

# Create output directory
os.makedirs(config['paths']['model_output_dir'], exist_ok=True)

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=config['training']['num_train_epochs'],
    remove_unused_columns=False,
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    warmup_steps=config['training']['warmup_steps'],
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    adam_beta2=config['training']['adam_beta2'],
    logging_steps=config['logging']['logging_steps'],
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=config['logging']['save_steps'],
    do_eval=True,
    eval_strategy="steps",
    eval_steps=config['logging']['eval_steps'],
    push_to_hub=False,
    save_total_limit=config['logging']['save_total_limit'],
    bf16=True,
    dataloader_pin_memory=False,
    output_dir=config['paths']['model_output_dir'],
    run_name=config['logging']['run_name'],
    report_to=["wandb"] if wandb_available and config.get('wandb', {}).get('project') else [],
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    args=training_args
)

# =============================================================================
# TRAINING
# =============================================================================

if __name__ == "__main__":
    print("Starting training...")
    print(f"Total training steps: {len(train_ds) // (config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']) * config['training']['num_train_epochs']}")
    
    # Start training
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_output_dir'], config['logging']['run_name'])
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    print(f"Training completed! Model saved to: {final_model_path}")
    
    if wandb_available and config.get('wandb', {}).get('project'):
        wandb.finish()