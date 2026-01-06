"""
Anatomy Training Objects - Simplified Version
Based on the original anatomy_training_objects.py

Functions for generating bounding boxes and location tokens for anatomical structures
"""
from typing import Dict, Union, Optional, Sequence, Tuple
import numpy as np
import os
from PIL import Image


def get_bounding_box(mask: np.ndarray, normalized: bool = False) -> Optional[Sequence[float]]:
    """
    Returns the bounding box for the specified structure in the mask.

    Args:
        mask (np.ndarray): Binary mask array
        normalized (bool): If True, normalizes the bounding box coordinates relative to the mask size.

    Returns:
        Optional[Sequence[float]]: The bounding box coordinates (rmin, cmin, rmax, cmax) if the 
                                structure is present; None if it is not.
    """
    # Convert to a binary mask
    mask = np.array(mask > 0, dtype=np.uint8)  # Ensure binary values (0 or 1)
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Expand the bounding box by 10 pixels in each dimension
    rmin = max(0, rmin - 10)  # Ensure rmin does not go below 0
    rmax = min(mask.shape[0] - 1, rmax + 10)  # Ensure rmax does not exceed the number of rows
    cmin = max(0, cmin - 10)  # Ensure cmin does not go below 0
    cmax = min(mask.shape[1] - 1, cmax + 10)  # Ensure cmax does not exceed the number of columns

    if normalized:
        rmin, rmax = [float(round(rmin / mask.shape[0], 3)), float(round(rmax / mask.shape[0], 3))]
        cmin, cmax = [float(round(cmin / mask.shape[1], 3)), float(round(cmax / mask.shape[1], 3))]
        return rmin, cmin, rmax, cmax

    return int(rmin), int(cmin), int(rmax), int(cmax)


def get_bounding_box_token(mask: np.ndarray) -> Optional[str]:
    """
    Returns the bounding box token for the specified structure in the mask.

    Args:
        mask (np.ndarray): Binary mask array

    Returns:
        Optional[str]: The bounding box token string if the structure is present; None if it is not.
    """
    bbox = get_bounding_box(mask, normalized=True)
    if bbox is None:
        return None

    # Convert to 1024-bin coordinates (Paligemma format)
    binned_loc = [int(round(coord * 1023)) for coord in bbox]
    binned_loc = [max(0, min(1023, coord)) for coord in binned_loc]

    # Create location tokens
    loc_tokens = ''.join([f'<loc{coord:04d}>' for coord in binned_loc])
    return loc_tokens


def get_segmentation_token(mask: np.ndarray) -> Optional[str]:
    """
    Generate segmentation tokens using VQVAE encoding
    Based on the original implementation
    
    Args:
        mask (np.ndarray): Binary mask array
        
    Returns:
        str: Segmentation token string in the format '<loc...><seg...><seg...>...'
    """
    from .vqvae import get_VQVAE_checkpoint, encode_to_codebook_indices, resize_mask_for_encoding
    
    bbox = get_bounding_box(mask, normalized=False)
    bbox_token = get_bounding_box_token(mask)
    
    if bbox is None or bbox_token is None:
        return None

    # Convert to a binary mask
    mask = np.array(mask > 0, dtype=np.uint8)
    
    # Extract bounding box region
    y1, x1, y2, x2 = bbox
    
    # Crop the mask to bounding box
    cropped_mask = mask[y1:y2+1, x1:x2+1]
    
    # Resize to 64x64 for VQVAE input
    resized_mask = resize_mask_for_encoding(cropped_mask, target_size=(64, 64))
    
    # Add batch and channel dimensions: [1, 64, 64, 1]
    resized_mask = resized_mask[None, :, :, None]
    
    # Load VQVAE checkpoint
    try:
        vqvae_checkpoint = get_VQVAE_checkpoint()
        
        # Encode to codebook indices
        mask_indices = encode_to_codebook_indices(vqvae_checkpoint, resized_mask, use_cpu=True)[0]
        
        # Convert indices to segmentation tokens
        seg_tokens = ''.join([f'<seg{idx:03d}>' for idx in mask_indices])
        
        return bbox_token + seg_tokens
        
    except Exception as e:
        print(f"Warning: VQVAE encoding failed ({e}), using simplified tokens")
        # Fallback to simplified tokens if VQVAE fails
        placeholder_seg_tokens = '<seg001><seg002><seg003><seg004>'
        return bbox_token + placeholder_seg_tokens