#!/usr/bin/env python3
"""
Test script for the new bbox_identification task
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'paligemma_training_data'))

from paligemma_training_sample_generator import PaligemmaSampleGenerator
import matplotlib.pyplot as plt
import numpy as np

def test_bbox_identification_task():
    """Test the new bbox_identification task"""
    
    # Path to test NPZ file
    npz_path = 'test_npz/train_1.npz'
    
    if not os.path.exists(npz_path):
        print(f"NPZ file not found: {npz_path}")
        return
    
    print("Testing bbox_identification task...")
    
    # Create generator without augmentation for consistent testing
    generator = PaligemmaSampleGenerator(npz_path, enable_augmentation=False)
    
    # Get available structures
    structures = generator.get_available_structures()
    print(f"Available structures: {len(structures)} total")
    
    # Test with a few different structures
    test_structures = ['heart', 'lung_left', 'lung_right', 'aorta']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, structure in enumerate(test_structures):
        if structure not in structures:
            print(f"Structure {structure} not available, skipping...")
            continue
            
        if i >= 4:  # Only show first 4
            break
            
        # Generate bbox_identification sample
        sample = generator.generate_bbox_identification_sample(structure)
        
        if sample:
            print(f"\\n{i+1}. Structure: {structure}")
            print(f"   Prefix: {sample.prefix}")
            print(f"   Suffix: {sample.suffix}")
            print(f"   Task type: {sample.task_type}")
            
            # Display the image with bounding box
            axes[i].imshow(np.array(sample.image))
            axes[i].set_title(f'{structure.title()}\\n"{sample.prefix}"\\nAnswer: {sample.suffix}')
            axes[i].axis('off')
        else:
            print(f"Could not generate sample for {structure}")
    
    plt.tight_layout()
    plt.savefig('bbox_identification_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test all task types including the new one
    print("\\n" + "="*60)
    print("TESTING ALL TASK TYPES INCLUDING NEW bbox_identification")
    print("="*60)
    
    all_tasks = ['detection', 'segmentation', 'bbox_token_identification', 
                'mask_token_identification', 'bbox_identification', 'orientation_identification']
    
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

if __name__ == "__main__":
    test_bbox_identification_task()