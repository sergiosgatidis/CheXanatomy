"""
Token to Mask Decoder - Simplified Version  
Based on the original token_to_mask_decoder.py

Functions for decoding segmentation tokens back to masks (placeholder implementation)
"""
import re
import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


# This code is derived from https://huggingface.co/spaces/big-vision/paligemma-hf/blob/main/app.py

# VQVAE model path
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vae-oid.npz')


# Regular expression for detecting location and segmentation tokens
_SEGMENT_DETECT_RE = re.compile(
    r'(.*?)' +
    r'<loc(\d{4})>' * 4 + r'\s*' +
    '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
    r'\s*([^;<>]+)? ?(?:; )?',
)


def parse_location_tokens(token_string: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse location tokens from a string
    
    Args:
        token_string: String containing location tokens like '<loc0123><loc0456><loc0789><loc0012>'
        
    Returns:
        Tuple of (rmin, cmin, rmax, cmax) or None if parsing fails
    """
    # Extract location tokens
    loc_pattern = r'<loc(\d{4})>'
    matches = re.findall(loc_pattern, token_string)
    
    if len(matches) != 4:
        return None
    
    # Convert to coordinates
    coords = [int(match) for match in matches]
    
    # Scale from 1024-bin to actual coordinates (assuming 512x512 image)
    rmin = int(coords[0] * 512 / 1023)
    cmin = int(coords[1] * 512 / 1023) 
    rmax = int(coords[2] * 512 / 1023)
    cmax = int(coords[3] * 512 / 1023)
    
    return rmin, cmin, rmax, cmax


def parse_segmentation_tokens(token_string: str) -> Optional[List[int]]:
    """
    Parse segmentation tokens from a string
    
    Args:
        token_string: String containing segmentation tokens like '<seg001><seg002>...'
        
    Returns:
        List of segmentation token indices or None if parsing fails
    """
    # Extract segmentation tokens
    seg_pattern = r'<seg(\d{3})>'
    matches = re.findall(seg_pattern, token_string)
    
    if not matches:
        return None
    
    return [int(match) for match in matches]


def decode_tokens_to_mask(token_string: str, image_size: Tuple[int, int] = (512, 512)) -> Optional[np.ndarray]:
    """
    Decode location and segmentation tokens to a binary mask
    
    Args:
        token_string: String containing location and segmentation tokens
        image_size: Size of the output image (height, width)
        
    Returns:
        Binary mask as numpy array or None if decoding fails
    """
    # Parse location tokens
    bbox = parse_location_tokens(token_string)
    if bbox is None:
        return None
    
    rmin, cmin, rmax, cmax = bbox
    
    # Create binary mask
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # For simplified version, just fill the bounding box
    # In full version, this would use VQVAE decoder with segmentation tokens
    mask[rmin:rmax+1, cmin:cmax+1] = 255
    
    return mask


def create_mask_from_bbox(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Create a simple rectangular mask from bounding box coordinates
    
    Args:
        bbox: Tuple of (rmin, cmin, rmax, cmax)
        image_size: Size of the output image (height, width)
        
    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    rmin, cmin, rmax, cmax = bbox
    
    # Ensure coordinates are within bounds
    rmin = max(0, min(rmin, image_size[0] - 1))
    rmax = max(0, min(rmax, image_size[0] - 1))  
    cmin = max(0, min(cmin, image_size[1] - 1))
    cmax = max(0, min(cmax, image_size[1] - 1))
    
    mask[rmin:rmax+1, cmin:cmax+1] = 255
    return mask


def visualize_tokens_on_image(image_path: str, token_string: str, save_path: str = None):
    """
    Visualize decoded tokens overlaid on the original image
    
    Args:
        image_path: Path to the original image
        token_string: String containing tokens to decode
        save_path: Optional path to save the visualization
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Decode tokens to mask
    mask = decode_tokens_to_mask(token_string, image_array.shape[:2])
    
    if mask is not None:
        # Create overlay
        overlay = image_array.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay
        
        # Blend with original
        blended = (0.7 * image_array + 0.3 * overlay).astype(np.uint8)
        
        # Save or return
        result_image = Image.fromarray(blended)
        
        if save_path:
            result_image.save(save_path)
        
        return result_image
    
    return None