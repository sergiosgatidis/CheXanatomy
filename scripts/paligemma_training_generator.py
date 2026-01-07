#!/usr/bin/env python3
# ==============================================================================
# PALIGEMMA TRAINING GENERATOR - COMMAND LINE INTERFACE
# ==============================================================================
# 
# PURPOSE: CLI wrapper for the core PaligemmaSampleGenerator library.
#          Provides convenient command-line access for batch generation.
#
# USES: src/paligemma_generator.py (core library)
# COMPANIONS:
#   - src/paligemma_generator.py (import this for pipeline integration)
#   - examples/task_examples.py (see examples before using this CLI)
#
# CLI USAGE:
#   python scripts/paligemma_training_generator.py --image-info /path/to/info.json
#   python scripts/paligemma_training_generator.py --demo
#
# FOR PIPELINE INTEGRATION: Import PaligemmaSampleGenerator from src/ directly
# ==============================================================================

"""Paligemma Training Generator - Command Line Interface

Command-line script for generating Paligemma training samples from chest X-ray data.
This script provides a convenient CLI wrapper around the core PaligemmaSampleGenerator
library for batch generation and demonstration purposes.

PURPOSE:
--------
- CLI interface for the core PaligemmaSampleGenerator library
- Batch generation of training samples for pipeline feeding
- Demo functionality to showcase task generation capabilities
- File-based input processing (image_info.json, case directories)

USAGE MODES:
------------
1. From image_info.json:
   python scripts/paligemma_training_generator.py --image-info /path/to/image_info.json --samples 100

2. From case directory:
   python scripts/paligemma_training_generator.py --case-dir /path/to/case --samples 50

3. Demo mode:
   python scripts/paligemma_training_generator.py --demo

OUTPUT:
-------
- Returns list of TrainingSample objects for direct pipeline feeding
- No file output - designed for in-memory pipeline integration
- Supports filtering by task types

RELATIONSHIP TO OTHER MODULES:
------------------------------
- Imports and uses: src/paligemma_generator.py (core library)
- Complements: examples/task_examples.py (demonstration)
- Role: CLI wrapper for convenient command-line access to core functionality

NOTE: This script focuses on pipeline feeding rather than file-based workflows.
      For library integration, import PaligemmaSampleGenerator directly from src/.
"""

import argparse
import json
import os
import random
from typing import List, Optional

from chexanatomy.paligemma_generator import PaligemmaSampleGenerator


def generate_from_image_info(image_info_path: str, num_samples: int = 100, 
                           task_types: Optional[List[str]] = None) -> List:
    """
    CLI FUNCTION: Generate training samples from image_info JSON file.
    
    This function provides a convenient CLI interface for batch generation from 
    preprocessed image_info.json files. It wraps the core PaligemmaSampleGenerator
    for command-line usage and returns samples for pipeline integration.
    
    Args:
        image_info_path: Path to image_info file (.npz or .json) containing:
                        - img_array: Image data
                        - structure_info: Dictionary of anatomical structures with tokens
                        - case: Case identifier
        num_samples: Total number of samples to generate across all task types
        task_types: List of task types to include. If None, generates samples for all 4 tasks:
                   ['detection', 'segmentation', 'bbox_token_identification', 'mask_token_identification']
        
    Returns:
        List[TrainingSample]: Training samples ready for pipeline feeding.
                             Each sample contains (prefix, suffix, image, task_type, structure)
                             
    CLI Usage Example:
        ```bash
        python scripts/paligemma_training_generator.py \\
            --image-info /data/processed/image_info.json \\
            --samples 500
        ```
        
    Pipeline Integration:
        ```python
        samples = generate_from_image_info('/path/to/image_info.json', 1000)
        for sample in samples:
            model.train_step(sample.prefix, sample.suffix, sample.image)
        ```
    """
    print(f"Loading image info from: {image_info_path}")
    generator = PaligemmaSampleGenerator(image_info_path)
    
    available_structures = generator.get_available_structures()
    print(f"Found {len(available_structures)} structures: {available_structures}")
    
    if task_types is None:
        task_types = ["detection", "segmentation", "bbox_token_identification", "mask_token_identification", "orientation_identification"]
    
    print(f"Generating {num_samples} samples for tasks: {task_types}")
    
    # Generate samples using the current single-sample approach
    samples = []
    for i in range(num_samples):
        task_type = random.choice(task_types)
        sample = generator.generate_sample(task_type)
        if sample:
            samples.append(sample)
    
    print(f"Generated {len(samples)} training samples for pipeline")
    
    return samples


def generate_from_case_directory(case_dir: str, num_samples: int = 50) -> List:
    """
    Generate training samples directly from case directory.
    
    Args:
        case_dir: Directory containing ct.png and structure masks
        num_samples: Number of samples to generate
        
    Returns:
        List of TrainingSample objects for pipeline feeding
    """
    from chexanatomy.anatomy import get_bounding_box, get_bounding_box_token, get_segmentation_token
    from PIL import Image
    import numpy as np
    
    print(f"Processing case directory: {case_dir}")
    
    # Load main image
    ct_path = os.path.join(case_dir, 'ct.png')
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"ct.png not found in {case_dir}")
    
    ct = np.array(Image.open(ct_path).convert('L'))
    
    # Process structure masks
    png_files = [f for f in os.listdir(case_dir) if f.endswith('.png')]
    structure_files = [f for f in png_files if f != 'ct.png']
    
    structure_info = {}
    for structure_file in structure_files:
        try:
            label = structure_file.replace('.png', '')
            mask_path = os.path.join(case_dir, structure_file)
            mask = np.array(Image.open(mask_path).convert('L'))
            
            bbox = get_bounding_box(mask)
            if bbox is not None:
                bbox_norm = get_bounding_box(mask, normalized=True)
                bbox_token = get_bounding_box_token(mask)
                segmentation_token = get_segmentation_token(mask)
                
                structure_info[label] = {
                    "label": label,
                    "bbox": bbox,
                    "bbox_norm": bbox_norm,
                    "bbox_token": bbox_token,
                    "segmentation_token": segmentation_token,
                }
                
        except Exception as e:
            print(f"Error processing {structure_file}: {e}")
    
    # Create image info
    image_info = {
        "img_array": ct.tolist(),
        "img_shape": list(ct.shape),
        "case": os.path.basename(case_dir),
        "structure_info": structure_info,
        "num_structures": len(structure_info)
    }
    
    print(f"Found {len(structure_info)} structures in case")
    
    # Generate samples
    generator = PaligemmaSampleGenerator(image_info)
    samples = generator.generate_training_batch(num_samples)
    
    print(f"Generated {len(samples)} training samples for pipeline")
    
    return samples


def demo_generation(image_info_path: str):
    """
    Demo the generator with a few samples (for testing).
    
    Args:
        image_info_path: Path to image_info.json file
    """
    generator = PaligemmaSampleGenerator(image_info_path)
    
    print("=== Sample Generation Demo ===")
    print(f"Available structures: {generator.get_available_structures()}")
    
    task_types = ["detection", "segmentation", "bbox_token_identification", "mask_token_identification"]
    
    for i, task in enumerate(task_types, 1):
        sample = generator.generate_sample(task)
        if sample:
            print(f"\n{i}. {task.upper()}")
            print(f"   Prefix: {sample.prefix}")
            print(f"   Suffix: {sample.suffix[:100]}...")
            print(f"   Structure: {sample.structure}")
            print(f"   Image size: {sample.image.size}")


def main():
    parser = argparse.ArgumentParser(
        description='Paligemma Training Data Generator - Pipeline Feeding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from image_info.json for pipeline feeding
  python paligemma_training_generator.py --image_info outputs/image_info.json --num_samples 1000

  # Generate from case directory
  python paligemma_training_generator.py --case_dir /path/to/case --num_samples 100

  # Demo mode (testing)
  python paligemma_training_generator.py --image_info outputs/image_info.json --demo
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_info', help='Path to image_info JSON file')
    input_group.add_argument('--case_dir', help='Case directory with ct.png and masks')
    
    # Generation options
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of training samples to generate')
    parser.add_argument('--task_types', nargs='+',
                       choices=['detection', 'segmentation', 'bbox_token_identification', 
                               'mask_token_identification'],
                       help='Task types to generate (default: all)')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode: generate a few samples for testing')
    
    args = parser.parse_args()
    
    if args.demo:
        if not args.image_info:
            print("Error: --demo requires --image_info")
            return
        demo_generation(args.image_info)
        return
    
    if args.image_info:
        samples = generate_from_image_info(
            args.image_info, 
            args.num_samples,
            args.task_types
        )
    elif args.case_dir:
        samples = generate_from_case_directory(
            args.case_dir,
            args.num_samples
        )
    
    # Print sample statistics
    if samples:
        task_counts = {}
        for sample in samples:
            task_counts[sample.task_type] = task_counts.get(sample.task_type, 0) + 1
        
        print(f"\nTask distribution:")
        for task, count in task_counts.items():
            print(f"  {task}: {count} samples")
        
        print(f"\n✅ {len(samples)} samples ready for pipeline feeding!")
        print("💡 Samples can be directly consumed by your training pipeline")


if __name__ == "__main__":
    main()