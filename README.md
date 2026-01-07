# CheXanatomy: Chest X-ray VLM Training Data Generator

A minimal training data generation pipeline for chest X-ray analysis, based on anatomical structure detection and location token generation.

## Project Structure

```
cheXanatomy/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── src/                   # Core modules
│   ├── anatomy.py         # Bounding box & segmentation token generation
│   ├── vqvae.py          # VQVAE encoding for segmentation
│   └── decoder.py        # Token decoding utilities
├── scripts/              # Data processing scripts
│   └── process_data.py   # Main data processing script
├── models/               # Model files
│   └── vae-oid.npz      # VQVAE model weights
├── examples/             # Example usage
├── outputs/              # Generated data and results
└── old_code/            # Original reference code (not tracked)
```

## Core Training Tasks (Original Design)

Based on the proven task_definitions.py approach, the system generates 4 core task types:

1. **Detection**: `detect heart` → `<loc0400><loc0312><loc0703><loc0625> heart`
2. **Segmentation**: `segment lung` → `<seg045><seg023>...<seg089> lung`  
3. **Bbox Token Identification**: `caption <loc0400>...` → `heart`
4. **Mask Token Identification**: `caption <seg045>...` → `lung`

## Pipeline Integration

The generator is designed for **direct pipeline feeding** with optional file output:

```python
from chexanatomy import PaligemmaSampleGenerator

# Initialize with image data
generator = PaligemmaSampleGenerator(image_info)

# Generate training batch for pipeline
samples = generator.generate_training_batch(batch_size=32)

# Each sample has: prefix, suffix, image, task_type, structure
for sample in samples:
    train_pipeline(sample.prefix, sample.suffix, sample.image)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sergiosgatidis/cheXanatomy.git
cd cheXanatomy
```

2. Install the package in editable mode:
```bash
pip install -e .
```

This will install the `chexanatomy` package with all dependencies.

## Usage

### For Training Pipeline Integration

```bash
# Generate samples for direct pipeline feeding
python scripts/paligemma_training_generator.py --image_info outputs/image_info.json --num_samples 1000

# Process case directory directly
python scripts/paligemma_training_generator.py --case_dir /path/to/case --num_samples 100

# Demo mode (testing)
python scripts/paligemma_training_generator.py --image_info outputs/image_info.json --demo
```

### Basic Data Processing

```bash
# Process a single case to create image_info.json
python scripts/anatomy_training_data.py --single_case /path/to/case_directory

# Process multiple cases
python scripts/anatomy_training_data.py --input_dir /path/to/cases --output_dir outputs/data
```

### Expected Data Format

Your data should be organized as:
```
case_directory/
├── ct.png                 # Main chest X-ray image
├── lung_left.png         # Anatomical structure masks
├── heart.png
├── spine.png
└── ...
```

### Generated Training Data

The system generates training examples like:
- **Localization**: "Where is the lung?" → "The lung is located at: `<loc0234><loc0567><loc0789><loc0123>`"
- **Segmentation**: "Segment the heart" → "`<loc0234><loc0567><loc0789><loc0123><seg045><seg023>...`"
- **Detection**: "Is the spine visible?" → "Yes, the spine is visible."

## Requirements

- Python 3.9+
- TensorFlow 2.8+
- PIL (Pillow)
- NumPy

Install with: `pip install -r requirements.txt`