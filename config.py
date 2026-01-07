"""
Configuration file for Paligemma fine-tuning with CheXanatomy

This file contains all the configuration parameters needed for training
the Paligemma model on chest X-ray anatomy data.
"""

import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model ID for PaliGemma from HuggingFace
MODEL_ID = "google/paligemma-3b-pt-224"

# Image size for training (resize all images to this size)
INPUT_IMAGE_SIZE = 224

# Output directory for saved models
MODEL_OUTPUT_DIR = "./models/paligemma_finetuned"

# Run name for logging and model saving
RUN_NAME = "chexanatomy_paligemma_v1"

# =============================================================================
# TRAINING DATA CONFIGURATION
# =============================================================================

# Path to training data directory (containing image_info.json files)
TRAINING_DATA_PATH = "./outputs"

# Train/validation split ratio (0.1 = 10% for validation)
TRAIN_VAL_SPLIT = 0.1

# Fraction of training data to use (1.0 = use all data)
TRAINING_DATA_FRACTION = 1.0

# Task types to include in training
AVAILABLE_TASKS = [
    'detection',
    'segmentation', 
    'bbox_token_identification',
    'mask_token_identification',
    'bbox_identification',
    'mask_identification',
    'orientation_identification'
]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Number of training epochs
NUM_TRAIN_EPOCHS = 3

# Batch sizes
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4

# Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)
GRADIENT_ACCUMULATION_STEPS = 4

# Learning rate
LEARNING_RATE = 5e-5

# Weight decay for regularization
WEIGHT_DECAY = 0.01

# Adam optimizer beta2 parameter
ADAM_BETA2 = 0.999

# =============================================================================
# LOGGING AND SAVING
# =============================================================================

# Steps between logging
LOGGING_STEPS = 10

# Steps between model saves
SAVE_STEPS = 500

# Steps between evaluations
EVAL_STEPS = 500

# Maximum number of saved checkpoints
SAVE_TOTAL_LIMIT = 3

# =============================================================================
# OPTIMIZATION STRATEGY
# =============================================================================

# Use LoRA (Low-Rank Adaptation) for efficient fine-tuning
USE_LORA = True

# Use QLoRA (Quantized LoRA) for even more memory efficiency
USE_QLORA = False

# Freeze vision tower (only train language components)
FREEZE_VISION = True

# Enable image augmentation during training
ENABLE_AUGMENTATION = True

# =============================================================================
# PATHS AND DEPENDENCIES
# =============================================================================

# Path to VQVAE model file for segmentation tasks
VQVAE_MODEL_PATH = "./models/vae-oid.npz"