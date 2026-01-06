"""
Image Info Generator - Simplified Version
Based on the original image_info.py

Functions for processing chest X-ray cases and generating training data
"""
import os
import numpy as np
import json
from PIL import Image
from typing import Dict, List, Optional

from chextrain.anatomy import get_bounding_box_token, get_segmentation_token, get_bounding_box


def get_image_info(case_dir: str, export_json: bool = False, json_path: str = None, flip: bool = False) -> dict:
    """
    Returns detailed information about the chest X-ray case, including array, metadata, and structure-specific info.
    Optionally exports the data to a JSON file.

    Args:
        case_dir (str): Directory containing the case files (ct.png and structure masks)
        export_json (bool): If True, exports the case information to a JSON file. Default is False.
        json_path (str): The path to save the JSON file if export_json is True. Default is "image_info.json".
        flip (bool): If True, flips the image horizontally. Default is False.

    Returns:
        dict: A dictionary containing detailed case information.
    """

    # Initialize dictionary to store structure information
    structure_info = {}

    # Get all PNG files in the directory
    png_files = [f for f in os.listdir(case_dir) if f.endswith('.png')]
    png_files.sort()
    
    # Remove ct.png from the list of structure files
    structure_files = [f for f in png_files if f != 'ct.png']

    # Load main image (ct.png)
    ct_path = os.path.join(case_dir, 'ct.png')
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"Main image 'ct.png' not found in {case_dir}")
    
    ct = Image.open(ct_path).convert('L')  # Convert to grayscale
    ct = np.array(ct)  # Convert to numpy array
    
    if flip:
        ct = np.flip(ct, axis=1)

    # Process each structure mask
    for structure_file in structure_files:
        try:
            # Extract label from filename
            label = structure_file.replace('.png', '')
            
            # Load mask
            mask_path = os.path.join(case_dir, structure_file)
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)  # Convert to numpy array
            
            if flip:
                mask = np.flip(mask, axis=1)
    
            # Get the bounding box coordinates (not normalized)
            bbox = get_bounding_box(mask)

            # Get the bounding box coordinates (normalized)
            bbox_norm = get_bounding_box(mask, normalized=True)
            
            # Get the bounding box token
            bbox_token = get_bounding_box_token(mask)

            # Get the segmentation token
            segmentation_token = get_segmentation_token(mask)

            # Store all retrieved information in the structure_info dictionary
            if bbox is not None:  # Only include structures that have valid bounding boxes
                structure_info[label] = {
                    "label": label,
                    "bbox": bbox,
                    "bbox_norm": bbox_norm, 
                    "bbox_token": bbox_token,
                    "segmentation_token": segmentation_token,
                }
                
        except Exception as e:
            print(f"Error processing {structure_file}: {e}")
            continue

    # Compile all information into a dictionary
    image_info = {
        "img_array": ct.tolist(),  # Convert to list for JSON serialization
        "img_shape": ct.shape,
        "case": os.path.basename(case_dir),
        "structure_info": structure_info,
        "num_structures": len(structure_info)
    }

    if export_json:
        # Default to outputs directory if no path specified
        if json_path is None:
            os.makedirs(os.path.join('..', 'outputs'), exist_ok=True)
            json_path = os.path.join('..', 'outputs', 'image_info.json')
        
        # Save to JSON file
        with open(json_path, 'w') as f:
            json.dump(image_info, f, indent=2)
        print(f"Image info saved to {json_path}")

    return image_info


def generate_training_examples(image_info: dict) -> List[dict]:
    """
    Generate training examples from processed image information
    
    Args:
        image_info: Dictionary containing processed image and structure information
        
    Returns:
        List of training examples
    """
    examples = []
    case_name = image_info['case']
    structure_info = image_info['structure_info']
    
    # 1. Localization tasks - where is structure X?
    for structure_name, info in structure_info.items():
        if info['bbox_token']:
            examples.append({
                'case': case_name,
                'prompt': f"Where is the {structure_name}?",
                'response': f"The {structure_name} is located at: {info['bbox_token']}",
                'structure': structure_name,
                'type': 'localization',
                'bbox': info['bbox'],
                'bbox_norm': info['bbox_norm']
            })
    
    # 2. Segmentation tasks - segment structure X
    for structure_name, info in structure_info.items():
        if info['segmentation_token']:
            examples.append({
                'case': case_name,
                'prompt': f"Segment the {structure_name}",
                'response': info['segmentation_token'],
                'structure': structure_name,
                'type': 'segmentation',
                'bbox': info['bbox'],
                'bbox_norm': info['bbox_norm']
            })
    
    # 3. Detection tasks - is structure X present?
    for structure_name in structure_info.keys():
        examples.append({
            'case': case_name,
            'prompt': f"Is the {structure_name} visible?",
            'response': f"Yes, the {structure_name} is visible.",
            'structure': structure_name,
            'type': 'detection'
        })
    
    # 4. Structure listing
    structures_list = ', '.join(structure_info.keys())
    examples.append({
        'case': case_name,
        'prompt': "What anatomical structures are visible?",
        'response': f"The visible structures include: {structures_list}",
        'type': 'structure_listing',
        'num_structures': len(structure_info)
    })
    
    # 5. General description
    examples.append({
        'case': case_name,
        'prompt': "Describe this chest X-ray image.",
        'response': f"This chest X-ray shows {len(structure_info)} anatomical structures including: {structures_list}",
        'type': 'description',
        'num_structures': len(structure_info)
    })
    
    return examples


def process_multiple_cases(cases_dir: str, output_dir: str = "../outputs/training_data"):
    """
    Process multiple cases and generate training data
    
    Args:
        cases_dir: Directory containing multiple case subdirectories
        output_dir: Directory to save the training data
    """
    # Find all case directories
    case_dirs = []
    for item in os.listdir(cases_dir):
        item_path = os.path.join(cases_dir, item)
        if os.path.isdir(item_path):
            case_dirs.append(item_path)
    
    if not case_dirs:
        print(f"No case directories found in {cases_dir}")
        return
    
    print(f"Processing {len(case_dirs)} cases...")
    
    all_examples = []
    all_case_info = {}
    
    # Process each case
    for case_dir in case_dirs:
        try:
            case_name = os.path.basename(case_dir)
            print(f"Processing case: {case_name}")
            
            # Get image info
            image_info = get_image_info(case_dir)
            all_case_info[case_name] = image_info
            
            # Generate training examples
            examples = generate_training_examples(image_info)
            all_examples.extend(examples)
            
            print(f"  - Generated {len(examples)} examples from {image_info['num_structures']} structures")
            
        except Exception as e:
            print(f"Error processing {case_dir}: {e}")
            continue
    
    # Save training data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training examples
    training_path = os.path.join(output_dir, 'training_examples.json')
    with open(training_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    # Save detailed case information
    cases_path = os.path.join(output_dir, 'case_information.json')
    with open(cases_path, 'w') as f:
        json.dump(all_case_info, f, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated {len(all_examples)} training examples from {len(case_dirs)} cases")
    print(f"📁 Output saved to: {output_dir}")
    print(f"   - training_examples.json ({len(all_examples)} examples)")
    print(f"   - case_information.json ({len(all_case_info)} cases)")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Process chest X-ray cases and generate training data')
    parser.add_argument('--input_dir', required=True, help='Directory containing case subdirectories')
    parser.add_argument('--output_dir', default='../outputs/training_data', help='Output directory')
    parser.add_argument('--single_case', help='Process single case directory instead of multiple')
    
    args = parser.parse_args()
    
    if args.single_case:
        # Process single case
        if not os.path.exists(args.single_case):
            print(f"Error: Case directory {args.single_case} does not exist")
            exit(1)
        
        image_info = get_image_info(args.single_case, export_json=True)
        examples = generate_training_examples(image_info)
        
        print(f"Generated {len(examples)} examples from case {os.path.basename(args.single_case)}")
    else:
        # Process multiple cases
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            exit(1)
        
        process_multiple_cases(args.input_dir, args.output_dir)