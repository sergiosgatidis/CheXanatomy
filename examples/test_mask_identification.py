#!/usr/bin/env python3
"""
Test script for the new mask_identification task
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from paligemma_generator import PaligemmaSampleGenerator
import matplotlib.pyplot as plt
import numpy as np

def test_mask_identification_task():
    """Test the new mask_identification task"""
    
    # Path to test NPZ file
    npz_path = 'test_npz/train_1.npz'
    
    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        return
    
    print("Testing mask_identification task...")
    
    # Create generator without augmentation for consistent testing
    generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=False)
    
    # Get available structures
    structures = generator.get_available_structures()
    print(f"Available structures: {len(structures)} total")
    
    # Test with a few different structures that commonly have segmentation masks
    test_structures = ['heart', 'lung_left', 'lung_right', 'aorta']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    successful_samples = 0
    
    for i, structure in enumerate(test_structures):
        if structure not in structures:
            print(f"Structure {structure} not available, skipping...")
            continue
            
        if i >= 4:  # Only show first 4
            break
            
        # Generate mask_identification sample
        sample = generator.generate_mask_identification_sample(structure)
        
        if sample:
            print(f"\\n{successful_samples+1}. Structure: {structure}")
            print(f"   Prefix: {sample.prefix}")
            print(f"   Suffix: {sample.suffix}")
            print(f"   Task type: {sample.task_type}")
            
            # Display the image with segmentation mask
            axes[successful_samples].imshow(np.array(sample.image))
            axes[successful_samples].set_title(f'{structure.title()}\\n"{sample.prefix}"\\nAnswer: {sample.suffix}')
            axes[successful_samples].axis('off')
            successful_samples += 1
        else:
            print(f"Could not generate mask_identification sample for {structure}")
    
    # Hide unused subplots
    for i in range(successful_samples, 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_identification_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test all task types including the new one
    print("\\n" + "="*60)
    print("TESTING ALL TASK TYPES INCLUDING NEW mask_identification")
    print("="*60)
    
    all_tasks = ['detection', 'segmentation', 'bbox_token_identification', 
                'mask_token_identification', 'bbox_identification', 'mask_identification', 'orientation_identification']
    
    test_structure = 'heart' if 'heart' in structures else structures[0]
    
    for task in all_tasks:
        print(f"\\n{task.upper()}:")
        
        if task == 'orientation_identification':
            sample = generator.generate_sample(task)
        else:
            sample = generator.generate_sample(task, test_structure)
            
        if sample:
            print(f"  Prefix: {sample.prefix}")
            if len(sample.suffix) > 80:
                print(f"  Suffix: {sample.suffix[:80]}...")
            else:
                print(f"  Suffix: {sample.suffix}")
            print(f"  Task type: {sample.task_type}")
        else:
            print(f"  Could not generate sample")

def test_mask_vs_bbox_comparison():
    """Compare bbox_identification vs mask_identification side by side"""
    
    npz_path = 'test_npz/train_1.npz'
    
    if not os.path.exists(npz_path):
        return
    
    print("\\nComparing bbox_identification vs mask_identification...")
    
    generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=False)
    structure = 'heart'
    
    # Generate both types of samples
    bbox_sample = generator.generate_bbox_identification_sample(structure)
    mask_sample = generator.generate_mask_identification_sample(structure)
    
    if bbox_sample and mask_sample:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(np.array(bbox_sample.image))
        ax1.set_title(f'Bbox Identification\\n"{bbox_sample.prefix}"\\nAnswer: {bbox_sample.suffix}')
        ax1.axis('off')
        
        ax2.imshow(np.array(mask_sample.image))
        ax2.set_title(f'Mask Identification\\n"{mask_sample.prefix}"\\nAnswer: {mask_sample.suffix}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('bbox_vs_mask_identification.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Both bbox and mask identification tasks working!")
    else:
        print("❌ Could not generate both sample types")

if __name__ == "__main__":
    test_mask_identification_task()
    test_mask_vs_bbox_comparison()