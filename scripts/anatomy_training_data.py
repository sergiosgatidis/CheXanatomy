"""
NPZ Image Info Generator

Function for processing chest X-ray cases and creating NPZ files for Paligemma generator
"""
import os
import numpy as np
import json
from PIL import Image
from typing import Dict, List, Optional

from chexanatomy.anatomy import get_bounding_box_token, get_segmentation_token, get_bounding_box


def get_image_info(case_dir: str, export_npz: bool = False, npz_path: str = None, flip: bool = False) -> dict:
    """
    Returns detailed information about the chest X-ray case, including array, metadata, and structure-specific info.
    Optionally exports the data to a compressed NPZ file.

    Args:
        case_dir (str): Directory containing the case files (ct.png and structure masks)
        export_npz (bool): If True, exports the case information to a compressed NPZ file. Default is False.
        npz_path (str): The path to save the NPZ file if export_npz is True. Default is "image_info.npz".
        flip (bool): If True, flips the image horizontally. Default is False.

    Returns:
        dict: A dictionary containing detailed case information.
    """

    # Detect orientation from directory structure
    # Check if case_dir or parent directories contain orientation indicators
    orientation = "ax"  # default to axial
    
    # Check current directory and parent directories for PA/LR indicators
    path_parts = os.path.normpath(case_dir).split(os.sep)
    for part in path_parts:
        if part.upper() == "PA":
            orientation = "PA"
            break
        elif part.upper() == "LR" or part.upper() == "LAT":
            orientation = "LR"
            break
    
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
        "img_array": ct,  # Keep as numpy array for NPZ
        "img_shape": ct.shape,
        "case": os.path.basename(case_dir),
        "orientation": orientation,  # Add orientation information
        "structure_info": structure_info,
        "num_structures": len(structure_info)
    }

    if export_npz:
        # Default to outputs directory if no path specified
        if npz_path is None:
            os.makedirs(os.path.join('..', 'outputs'), exist_ok=True)
            npz_path = os.path.join('..', 'outputs', 'image_info.npz')
        
        # Save arrays and metadata to NPZ
        np.savez_compressed(
            npz_path,
            img_array=image_info["img_array"],
            metadata={
                key: value for key, value in image_info.items()
                if key not in ["img_array"]
            }
        )
        print(f"Image info saved to {npz_path}")

    return image_info


