#!/usr/bin/env python3
"""
Demo of the Paligemma Sample Generator (faithful to original design)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'paligemma_training_data'))

from paligemma_training_sample_generator import PaligemmaSampleGenerator


def demo_paligemma_generator():
    """Demo the Paligemma generator with sample data"""
    
    # Sample image info matching the original format
    sample_image_info = {
        "img_array": [[100, 150, 200] * 256] * 256,  # Sample grayscale image
        "img_shape": [256, 256],
        "case": "sample_case",
        "structure_info": {
            "heart": {
                "label": "heart",
                "bbox": [100, 80, 180, 160],
                "bbox_norm": [0.39, 0.31, 0.70, 0.63],
                "bbox_token": "<loc0400><loc0312><loc0703><loc0625>",
                "segmentation_token": "<seg045><seg023><seg089><seg067><seg123><seg045><seg078><seg091><seg045><seg034><seg056><seg089><seg023><seg067><seg089><seg045>"
            },
            "lung_left": {
                "label": "lung_left", 
                "bbox": [50, 40, 120, 200],
                "bbox_norm": [0.20, 0.16, 0.47, 0.78],
                "bbox_token": "<loc0195><loc0156><loc0469><loc0781>",
                "segmentation_token": "<seg066><seg023><seg089><seg067><seg123><seg045><seg078><seg091><seg045><seg034><seg056><seg089><seg023><seg067><seg089><seg066>"
            }
        },
        "num_structures": 2
    }
    
    # Initialize generator (matches original constructor)
    print("🚀 Initializing PaligemmaSampleGenerator (original design)...")
    generator = PaligemmaSampleGenerator(sample_image_info)
    
    print(f"Available structures: {generator.get_available_structures()}")
    
    # Test the original 4 core tasks + new visual identification tasks
    original_tasks = ['detection', 'segmentation', 'bbox_token_identification', 'mask_token_identification', 'bbox_identification', 'mask_identification']
    
    print("\n📋 Original Task Samples (faithful to task_definitions.py):")
    print("=" * 65)
    
    for i, task in enumerate(original_tasks, 1):
        sample = generator.generate_sample(task, 'heart')  # Test with heart
        if sample:
            print(f"{i}. {task.upper()}:")
            print(f"   Prefix: {sample.prefix}")
            print(f"   Suffix: {sample.suffix[:50]}..." if len(sample.suffix) > 50 else f"   Suffix: {sample.suffix}")
            print()
    
    # NEW: Test augmentation capability
    print("\\n🎲 Testing NEW Augmentation Capability:")
    print("=" * 65)
    
    # Create generator with augmentation enabled
    print("Creating augmented generator...")
    augmented_generator = PaligemmaSampleGenerator(sample_image_info, enable_augmentation=True, scale_range=(0.8, 1.0))
    
    if augmented_generator.is_augmented():
        aug_params = augmented_generator.get_augmentation_params()
        print(f"✅ Augmentation applied successfully!")
        print(f"   Scale factor: {aug_params['scale_factor']:.3f}")
        print(f"   Offset: ({aug_params['x_offset']}, {aug_params['y_offset']}) pixels")
        
        # Compare original vs augmented samples
        print("\\n🔍 Comparing Original vs Augmented Samples:")
        print("-" * 50)
        
        orig_sample = generator.generate_sample('detection', 'heart')
        aug_sample = augmented_generator.generate_sample('detection', 'heart')
        
        print(f"Original Detection:")
        print(f"   Prefix: {orig_sample.prefix}")
        print(f"   Suffix: {orig_sample.suffix}")
        
        print(f"\\nAugmented Detection:")
        print(f"   Prefix: {aug_sample.prefix}")
        print(f"   Suffix: {aug_sample.suffix}")
        
        # Show bbox changes
        orig_info = generator.get_structure_info('heart')
        aug_info = augmented_generator.get_structure_info('heart')
        
        print(f"\\n📐 Bounding Box Changes:")
        print(f"   Original:  {orig_info['bbox']}")
        print(f"   Augmented: {aug_info['bbox']}")
        print(f"   Token changed: {orig_info['bbox_token'] != aug_info['bbox_token']}")
    else:
        print("❌ Augmentation was not applied")
    
    print("\\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_paligemma_generator()