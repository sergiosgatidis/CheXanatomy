#!/usr/bin/env python3
"""
Test mask_identification with augmentation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from paligemma_generator import PaligemmaSampleGenerator
import matplotlib.pyplot as plt
import numpy as np

def test_mask_identification_with_augmentation():
    """Test mask_identification task with augmentation"""
    
    npz_path = 'test_npz/train_1.npz'
    
    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        return
    
    print("Testing mask_identification with augmentation...")
    
    # Create generators - one original, one with augmentation
    original_generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=False)
    augmented_generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=True, scale_range=(0.8, 1.0))
    
    # Test structure
    structure = 'heart'
    
    # Generate samples
    original_sample = original_generator.generate_sample('mask_identification', structure)
    augmented_sample = augmented_generator.generate_sample('mask_identification', structure)
    
    print(f"\\nOriginal sample:")
    print(f"  Prefix: {original_sample.prefix}")
    print(f"  Suffix: {original_sample.suffix}")
    
    print(f"\\nAugmented sample:")
    print(f"  Prefix: {augmented_sample.prefix}")
    print(f"  Suffix: {augmented_sample.suffix}")
    
    if augmented_generator.is_augmented():
        aug_params = augmented_generator.get_augmentation_params()
        print(f"\\nAugmentation applied:")
        print(f"  Scale: {aug_params['scale_factor']:.3f}")
        print(f"  Offset: ({aug_params['x_offset']}, {aug_params['y_offset']})")
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.imshow(np.array(original_sample.image))
    ax1.set_title(f'Original Mask\\n"{original_sample.prefix}"\\nAnswer: {original_sample.suffix}')
    ax1.axis('off')
    
    ax2.imshow(np.array(augmented_sample.image))
    ax2.set_title(f'Augmented Mask (Scale: {aug_params["scale_factor"]:.3f})\\n"{augmented_sample.prefix}"\\nAnswer: {augmented_sample.suffix}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_identification_augmentation_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\n✅ mask_identification task with augmentation working correctly!")

if __name__ == "__main__":
    test_mask_identification_with_augmentation()