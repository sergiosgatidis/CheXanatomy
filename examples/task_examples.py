#!/usr/bin/env python3
# ==============================================================================
# PALIGEMMA TASK EXAMPLES - DEMONSTRATION SCRIPT
# ==============================================================================
# 
# PURPOSE: Demonstrate all 5 core task types with concrete examples.
#          Shows prompt-response formats and anatomical name variations.
#
# USES: src/paligemma_generator.py (core library)
# COMPANIONS:
#   - src/paligemma_generator.py (the main library for pipeline integration)
#   - scripts/paligemma_training_generator.py (CLI for batch generation)
#
# RUN THIS FIRST: See examples before implementing your training pipeline
# COMMAND: python examples/task_examples.py
#
# NOTE: Uses sample data. For real training, use CLI script or import library.
# ==============================================================================

"""Paligemma Task Examples - Demonstration Script

Demonstration script showing examples of all 5 core task types supported by the
PaligemmaSampleGenerator. This script provides concrete examples of how each task
type generates training data for Paligemma models.

PURPOSE:
--------
- Demonstrate all 5 core task types with real examples
- Show anatomical name variations and token generation
- Provide reference implementation for understanding task formats
- Validate that the generator produces correct prompt/response pairs

TASK EXAMPLES SHOWN:
-------------------
1. Detection Task:
   Input:  'detect Cor'
   Output: '<loc0391><loc0293><loc0684><loc0586> Cor'
   
2. Segmentation Task:
   Input:  'segment Cor'
   Output: '<seg045><seg023><seg089>... Cor'
   
3. Bbox Token Identification:
   Input:  'caption <loc0430><loc0156><loc0508><loc0391>'
   Output: 'Breastbone'
   
4. Mask Token Identification:
   Input:  'caption <seg066><seg123><seg089>...'
   Output: 'Left Lung'
   
5. Orientation Identification:
   Input:  'How is this slice oriented?'
   Output: 'axial'

ANATOMICAL VARIATIONS:
---------------------
Demonstrates the 60+ anatomical name variations used in training:
- Primary names: 'heart', 'lung_left', 'spine', 'sternum'
- Medical variants: 'Cor', 'Left Lung', 'Vertebral Column', 'Breastbone'
- Alternative names: 'cardiac silhouette', 'pulmonary field', etc.

USAGE:
------
```bash
python examples/task_examples.py
```

RELATIONSHIP TO OTHER MODULES:
------------------------------
- Uses: src/paligemma_generator.py (core PaligemmaSampleGenerator class)
- Complements: scripts/paligemma_training_generator.py (CLI interface)
- Role: Educational and validation tool for understanding task generation

NOTE: This script uses sample data for demonstration. For real training,
      use the CLI script or import the core library directly.
"""

from chextrain.paligemma_generator import PaligemmaSampleGenerator


def show_task_examples():
    """
    DEMONSTRATION FUNCTION: Generate and display examples of all 5 core task types.
    
    This function provides concrete examples of how the PaligemmaSampleGenerator creates
    training data for each task type. It uses sample data to demonstrate the prompt-response
    format and shows the anatomical name variations used in training.
    
    What This Function Demonstrates:
    --------------------------------
    1. Task Format Examples: Shows exact input/output format for each task type
    2. Token Generation: Demonstrates bbox and segmentation token creation
    3. Name Variations: Shows how anatomical names are varied for robustness
    4. Real Implementation: Uses the actual PaligemmaSampleGenerator class
    
    Task Examples Generated:
    ------------------------
    - Detection: 'detect [structure]' → '<bbox_tokens> [structure_name]'
    - Segmentation: 'segment [structure]' → '<seg_tokens> [structure_name]'  
    - Bbox ID: 'caption <bbox_tokens>' → '[structure_name]'
    - Mask ID: 'caption <seg_tokens>' → '[structure_name]'
    - Orientation ID: '[orientation_question]' → '[orientation]'
    
    Sample Output Format:
    ---------------------
    Task: detection | Structure: heart
    Input (prefix): 'detect Cor'
    Output (suffix): '<loc0391><loc0293><loc0684><loc0586> Cor'
    
    Usage:
    ------
    Run this script to see examples before implementing your training pipeline:
    ```bash
    python examples/task_examples.py
    ```
    
    For Real Training:
    ------------------
    This uses mock data. For actual training, use:
    - scripts/paligemma_training_generator.py (CLI)
    - Direct import: from chexanatomy.paligemma_generator import PaligemmaSampleGenerator
    """
    
    # Sample image info with multiple structures
    sample_image_info = {
        "img_array": [[100, 150, 200] * 512] * 512,  # Sample chest X-ray
        "img_shape": [512, 512],
        "case": "example_case",
        "orientation": "ax",  # axial orientation
        "structure_info": {
            "heart": {
                "label": "heart",
                "bbox": [200, 150, 350, 300],
                "bbox_norm": [0.39, 0.29, 0.68, 0.59],
                "bbox_token": "<loc0391><loc0293><loc0684><loc0586>",
                "segmentation_token": "<seg045><seg023><seg089><seg067><seg123><seg045><seg078><seg091><seg045><seg034><seg056><seg089><seg023><seg067><seg089><seg045>"
            },
            "lung_left": {
                "label": "lung_left", 
                "bbox": [80, 50, 200, 400],
                "bbox_norm": [0.16, 0.10, 0.39, 0.78],
                "bbox_token": "<loc0156><loc0098><loc0391><loc0781>",
                "segmentation_token": "<seg066><seg123><seg089><seg034><seg091><seg045><seg078><seg023><seg067><seg089><seg056><seg034><seg091><seg078><seg089><seg066>"
            },
            "spine": {
                "label": "spine",
                "bbox": [240, 20, 280, 480],
                "bbox_norm": [0.47, 0.04, 0.55, 0.94],
                "bbox_token": "<loc0469><loc0039><loc0547><loc0938>",
                "segmentation_token": "<seg078><seg034><seg123><seg091><seg045><seg067><seg089><seg078><seg056><seg023><seg091><seg034><seg067><seg089><seg045><seg078>"
            },
            "sternum": {
                "label": "sternum",
                "bbox": [220, 80, 260, 200],
                "bbox_norm": [0.43, 0.16, 0.51, 0.39],
                "bbox_token": "<loc0430><loc0156><loc0508><loc0391>",
                "segmentation_token": "<seg091><seg045><seg078><seg023><seg067><seg034><seg089><seg091><seg078><seg056><seg045><seg023><seg091><seg067><seg089><seg045>"
            }
        },
        "num_structures": 4
    }
    
    # Initialize generator
    generator = PaligemmaSampleGenerator(sample_image_info)
    
    print("🏥 CHEST X-RAY VLM TRAINING TASKS")
    print("=" * 60)
    print(f"Available structures: {generator.get_available_structures()}")
    print()
    
    # Task examples
    tasks = [
        ("detection", "heart"),
        ("detection", "lung_left"), 
        ("segmentation", "heart"),
        ("segmentation", "spine"),
        ("bbox_token_identification", "sternum"),
        ("bbox_token_identification", "heart"),
        ("mask_token_identification", "lung_left"),
        ("mask_token_identification", "spine"),
        ("orientation_identification", None),  # No structure needed for orientation
        ("orientation_identification", None)   # Show multiple orientation examples
    ]

    print("📋 TASK EXAMPLES:")
    print("-" * 60)
    
    for i, (task_type, structure) in enumerate(tasks, 1):
        sample = generator.generate_sample(task_type, structure)
        if sample:
            print(f"\n{i}. {task_type.upper()} - {structure}")
            print(f"   📝 Prompt:   '{sample.prefix}'")
            print(f"   ✅ Response: '{sample.suffix}'")
            print(f"   🎯 Task:     {sample.task_type}")
            print(f"   🔍 Structure: {sample.structure}")
            print(f"   🖼️  Image:     {sample.image.size} RGB")

    print("\n" + "=" * 60)
    print("🎯 TASK BREAKDOWN:")
    print("-" * 60)
    
    print("\n1️⃣  DETECTION TASKS:")
    print("   • Purpose: Find anatomical structures in the image")
    print("   • Format:  'detect [structure]' → '[bbox_token] [structure]'")
    print("   • Example: 'detect heart' → '<loc0391>...<loc0586> heart'")
    
    print("\n2️⃣  SEGMENTATION TASKS:")
    print("   • Purpose: Segment anatomical structures")
    print("   • Format:  'segment [structure]' → '[seg_token] [structure]'")
    print("   • Example: 'segment heart' → '<seg045>...<seg045> heart'")
    
    print("\n3️⃣  BBOX TOKEN IDENTIFICATION:")
    print("   • Purpose: Identify structure from location tokens")
    print("   • Format:  'caption [bbox_token]' → '[structure]'")
    print("   • Example: 'caption <loc0391>...' → 'heart'")
    
    print("\n4️⃣  MASK TOKEN IDENTIFICATION:")
    print("   • Purpose: Identify structure from segmentation tokens")
    print("   • Format:  'caption [seg_token]' → '[structure]'")
    print("   • Example: 'caption <seg045>...' → 'heart'")
    
    print("\n5️⃣  ORIENTATION IDENTIFICATION:")
    print("   • Purpose: Identify image orientation from visual features")
    print("   • Format:  '[orientation_question]' → '[orientation]'")
    print("   • Example: 'How is this slice oriented?' → 'axial'")
    
    print("\n💡 ANATOMICAL NAME VARIATIONS:")
    print("-" * 30)
    structures_shown = ["heart", "lung_left", "spine"]
    for struct in structures_shown:
        names = generator.anatomic_names.get(struct, [struct])
        print(f"   {struct}: {names}")
    
    print(f"\n✅ Ready for Paligemma training pipeline!")


if __name__ == "__main__":
    show_task_examples()
