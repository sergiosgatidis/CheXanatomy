#!/usr/bin/env python3
"""
Augmented Paligemma Generator Demo

This script demonstrates the new augmentation capability in PaligemmaSampleGenerator.
It shows how to generate training samples with automatic image and token augmentation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from paligemma_generator import PaligemmaSampleGenerator


def visualize_comparison(original_generator, augmented_generator, structure='heart'):
    """Compare original vs augmented samples side by side"""
    
    # Generate samples from both generators
    original_sample = original_generator.generate_sample('detection', structure)
    augmented_sample = augmented_generator.generate_sample('detection', structure)
    
    if not original_sample or not augmented_sample:
        print(f"Could not generate samples for structure: {structure}")
        return
    
    # Get structure info
    original_info = original_generator.get_structure_info(structure)
    augmented_info = augmented_generator.get_structure_info(structure)
    
    if not original_info or not augmented_info:
        print(f"Structure info not available for: {structure}")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(np.array(original_sample.image), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original with bbox
    axes[0, 1].imshow(np.array(original_sample.image), cmap='gray')
    y1, x1, y2, x2 = original_info['bbox']
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f'Original + BBox\\n{original_sample.prefix}')
    axes[0, 1].axis('off')
    
    # Augmented image
    axes[1, 0].imshow(np.array(augmented_sample.image), cmap='gray')
    axes[1, 0].set_title('Augmented Image')
    axes[1, 0].axis('off')
    
    # Augmented with bbox
    axes[1, 1].imshow(np.array(augmented_sample.image), cmap='gray')
    y1_aug, x1_aug, y2_aug, x2_aug = augmented_info['bbox']
    rect_aug = patches.Rectangle((x1_aug, y1_aug), x2_aug-x1_aug, y2_aug-y1_aug,
                               linewidth=2, edgecolor='lime', facecolor='none')
    axes[1, 1].add_patch(rect_aug)
    axes[1, 1].set_title(f'Augmented + BBox\\n{augmented_sample.prefix}')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison details
    print("\\n" + "="*60)
    print("AUGMENTATION COMPARISON")
    print("="*60)
    print(f"Structure: {structure}")
    
    # Check if augmentation was applied
    if augmented_generator.is_augmented():
        aug_params = augmented_generator.get_augmentation_params()
        print(f"Augmentation applied:")
        print(f"  Scale factor: {aug_params['scale_factor']:.3f}")
        print(f"  X offset: {aug_params['x_offset']} pixels")
        print(f"  Y offset: {aug_params['y_offset']} pixels")
    
    print(f"\\nBounding Box Changes:")
    print(f"  Original:  {original_info['bbox']}")
    print(f"  Augmented: {augmented_info['bbox']}")
    
    print(f"\\nToken Changes:")
    print(f"  Original bbox token:  {original_info.get('bbox_token', 'N/A')[:40]}...")
    print(f"  Augmented bbox token: {augmented_info.get('bbox_token', 'N/A')[:40]}...")
    
    if 'segmentation_token' in original_info and 'segmentation_token' in augmented_info:
        orig_seg = original_info['segmentation_token']
        aug_seg = augmented_info['segmentation_token']
        print(f"  Original seg token:   {orig_seg[:40]}...")
        print(f"  Augmented seg token:  {aug_seg[:40]}...")
        print(f"  Seg tokens different: {orig_seg != aug_seg}")


def main():
    # Path to test NPZ file
    npz_path = 'test_npz/train_1.npz'
    
    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        print("Please run the NPZ extraction script first:")
        print("python scripts/extract_npz_files.py --single_case /path/to/case --output_dir test_npz")
        return
    
    print("Creating Paligemma generators...")
    
    # Create generators - one original, one with augmentation
    original_generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=False)
    augmented_generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=True, 
                                                 scale_range=(0.7, 1.0))
    
    # Show available structures
    structures = original_generator.get_available_structures()
    print(f"Available structures: {structures[:10]}...")  # Show first 10
    
    # Test with different structures
    test_structures = ['heart', 'lung_left', 'lung_right']
    
    for structure in test_structures:
        if structure in structures:
            print(f"\\nTesting with structure: {structure}")
            visualize_comparison(original_generator, augmented_generator, structure)
            break
    else:
        # If none of the preferred structures are available, use the first available
        if structures:
            structure = structures[0]
            print(f"\\nUsing first available structure: {structure}")
            visualize_comparison(original_generator, augmented_generator, structure)
    
    # Test different task types
    print("\\n" + "="*60)
    print("TESTING DIFFERENT TASK TYPES WITH AUGMENTATION")
    print("="*60)
    
    test_structure = structures[0] if structures else None
    if test_structure:
        task_types = ['detection', 'segmentation', 'bbox_token_identification', 
                     'mask_token_identification', 'bbox_identification', 'mask_identification', 'orientation_identification']
        
        for task_type in task_types:
            print(f"\\n{task_type.upper()}:")
            sample = augmented_generator.generate_sample(task_type, test_structure)
            if sample:
                print(f"  Prefix: {sample.prefix}")
                print(f"  Suffix: {sample.suffix[:80]}..." if len(sample.suffix) > 80 else f"  Suffix: {sample.suffix}")
            else:
                print(f"  Could not generate sample for {task_type}")


if __name__ == "__main__":
    main()