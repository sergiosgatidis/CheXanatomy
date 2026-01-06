# CheXtrain: Chest X-ray VLM Training Data Generator

A minimal training data generation pipeline for chest X-ray analysis, based on anatomical structure detection and location token generation.

## Project Structure

```
cheXtrain/
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

## Features

- Real VQVAE-based segmentation token generation
- Paligemma-compatible location token format (`<loc0000>`)
- Multiple training task types (localization, segmentation, detection)
- Modular, clean code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sergiosgatidis/cheXtrain.git
cd cheXtrain
```

2. Install the package in editable mode:
```bash
pip install -e .
```

This will install the `chextrain` package with all dependencies.

## Usage

### Process CheXsynth Data Format

```bash
# Navigate to scripts directory
cd scripts

# Process a single case
python anatomy_training_data.py --single_case /path/to/case_directory

# Process multiple cases
python anatomy_training_data.py --input_dir /path/to/cases --output_dir ../outputs/my_training_data
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