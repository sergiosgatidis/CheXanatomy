"""
Anatomy Training Objects - Simplified Version
Based on the original anatomy_training_objects.py

Functions for generating bounding boxes and location tokens for anatomical structures
"""
from typing import Dict, Union, Optional, Sequence, Tuple
import numpy as np
import os
import random
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
    from VQVAE_encoder_utils import get_VQVAE_checkpoint, encode_to_codebook_indices, resize_mask_for_encoding
    
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
        print(f"Warning: VQVAE encoding failed ({e}), returning None")
        return None


def augment_image_with_random_scale_and_position(image_array: np.ndarray, 
                                                scale_range: Tuple[float, float] = (0.7, 1.0),
                                                background_value: float = 0.0) -> Tuple[np.ndarray, Dict[str, Union[float, int]]]:
    """
    Augment an image by randomly scaling it within a specified range and then randomly 
    positioning the scaled image within the original frame size. No image information is cut out.
    
    Args:
        image_array (np.ndarray): Input 2D image array (H, W)
        scale_range (Tuple[float, float]): Range of scaling factors (min_scale, max_scale). 
                                         Default is (0.7, 1.0)
        background_value (float): Value to use for padding/background. Default is 0.0
        
    Returns:
        Tuple[np.ndarray, Dict]: 
            - Augmented 2D image array with the same shape as input
            - Dictionary containing transformation parameters:
                - 'scale_factor': The scaling factor applied
                - 'x_offset': Horizontal offset in pixels
                - 'y_offset': Vertical offset in pixels
        
    Example:
        >>> img = np.random.rand(256, 256)
        >>> augmented_img, transform_params = augment_image_with_random_scale_and_position(img, (0.8, 1.0))
        >>> augmented_img.shape  # Same as original: (256, 256)
        >>> transform_params  # {'scale_factor': 0.85, 'x_offset': 12, 'y_offset': 8}
    """
    if len(image_array.shape) != 2:
        raise ValueError("Image array must be 2D (H, W)")
    
    # Store original dimensions
    height, width = image_array.shape
    
    # Generate random scale factor
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    
    # Calculate new dimensions after scaling
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Resize the image using PIL for better quality
    pil_image = Image.fromarray(image_array.astype(np.uint8))
    scaled_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    scaled_array = np.array(scaled_image).astype(image_array.dtype)
    
    # Create output array filled with background value
    output_array = np.full((height, width), background_value, dtype=image_array.dtype)
    
    # Calculate maximum possible offset to ensure no cropping
    max_y_offset = height - new_height
    max_x_offset = width - new_width
    
    # Generate random offsets (can be 0 if scaled image is same size or larger than original)
    y_offset = random.randint(0, max_y_offset) if max_y_offset > 0 else 0
    x_offset = random.randint(0, max_x_offset) if max_x_offset > 0 else 0
    
    # Place the scaled image in the output array
    end_y = min(y_offset + new_height, height)
    end_x = min(x_offset + new_width, width)
    scaled_height = end_y - y_offset
    scaled_width = end_x - x_offset
    
    output_array[y_offset:end_y, x_offset:end_x] = scaled_array[:scaled_height, :scaled_width]
    
    # Create transformation parameters dictionary
    transform_params = {
        'scale_factor': scale_factor,
        'x_offset': x_offset,
        'y_offset': y_offset
    }
    
    return output_array, transform_params


def augment_bounding_box(bbox_coords: Sequence[float], 
                        transform_params: Dict[str, Union[float, int]],
                        image_shape: Tuple[int, int]) -> Sequence[float]:
    """
    Augment bounding box coordinates using transformation parameters from image augmentation.
    Automatically detects if coordinates are normalized and maintains the same format.
    
    Args:
        bbox_coords (Sequence[float]): Bounding box coordinates in format (y_min, x_min, y_max, x_max)
        transform_params (Dict): Transformation parameters from augment_image_with_random_scale_and_position
                               containing 'scale_factor', 'x_offset', 'y_offset'
        image_shape (Tuple[int, int]): Original image shape (height, width) for normalization reference
        
    Returns:
        Sequence[float]: Augmented bounding box coordinates in same format as input
        
    Example:
        >>> bbox = (0.1, 0.2, 0.8, 0.9)  # normalized coordinates
        >>> params = {'scale_factor': 0.8, 'x_offset': 10, 'y_offset': 5}
        >>> augmented_bbox = augment_bounding_box(bbox, params, (512, 512))
        >>> # Returns normalized coordinates after transformation
        
        >>> bbox = (50, 100, 400, 450)  # pixel coordinates  
        >>> augmented_bbox = augment_bounding_box(bbox, params, (512, 512))
        >>> # Returns pixel coordinates after transformation
    """
    if len(bbox_coords) != 4:
        raise ValueError("Bounding box must have 4 coordinates: (y_min, x_min, y_max, x_max)")
    
    y_min, x_min, y_max, x_max = bbox_coords
    height, width = image_shape
    
    # Auto-detect if coordinates are normalized (all values <= 1.0)
    is_normalized = all(coord <= 1.0 for coord in bbox_coords)
    
    # Convert to pixel coordinates if normalized
    if is_normalized:
        y_min_px = y_min * height
        x_min_px = x_min * width
        y_max_px = y_max * height
        x_max_px = x_max * width
    else:
        y_min_px, x_min_px, y_max_px, x_max_px = y_min, x_min, y_max, x_max
    
    # Apply transformation: first scale, then translate
    scale_factor = transform_params['scale_factor']
    x_offset = transform_params['x_offset']
    y_offset = transform_params['y_offset']
    
    # Scale the bounding box coordinates
    y_min_scaled = y_min_px * scale_factor
    x_min_scaled = x_min_px * scale_factor
    y_max_scaled = y_max_px * scale_factor
    x_max_scaled = x_max_px * scale_factor
    
    # Apply offsets
    y_min_transformed = y_min_scaled + y_offset
    x_min_transformed = x_min_scaled + x_offset
    y_max_transformed = y_max_scaled + y_offset
    x_max_transformed = x_max_scaled + x_offset
    
    # Clamp to image boundaries
    y_min_transformed = max(0, min(height - 1, y_min_transformed))
    x_min_transformed = max(0, min(width - 1, x_min_transformed))
    y_max_transformed = max(0, min(height - 1, y_max_transformed))
    x_max_transformed = max(0, min(width - 1, x_max_transformed))
    
    # Ensure valid bounding box (min < max)
    if y_min_transformed >= y_max_transformed:
        y_max_transformed = y_min_transformed + 1
    if x_min_transformed >= x_max_transformed:
        x_max_transformed = x_min_transformed + 1
    
    # Convert back to original format
    if is_normalized:
        return (
            float(y_min_transformed / height),
            float(x_min_transformed / width),
            float(y_max_transformed / height),
            float(x_max_transformed / width)
        )
    else:
        return (
            int(y_min_transformed),
            int(x_min_transformed),
            int(y_max_transformed),
            int(x_max_transformed)
        )


def update_bounding_box_token(bbox_coords: Sequence[float], 
                             image_shape: Tuple[int, int]) -> str:
    """
    Generate updated bounding box tokens from augmented bounding box coordinates.
    
    Args:
        bbox_coords (Sequence[float]): Bounding box coordinates (y_min, x_min, y_max, x_max)
                                     Can be normalized (0-1) or pixel coordinates
        image_shape (Tuple[int, int]): Original image shape (height, width)
        
    Returns:
        str: Updated bounding box token string in Paligemma format
        
    Example:
        >>> bbox = (0.1, 0.2, 0.8, 0.9)  # normalized coordinates
        >>> token = update_bounding_box_token(bbox, (512, 512))
        >>> token  # '<loc0102><loc0204><loc0818><loc0921>'
    """
    if len(bbox_coords) != 4:
        raise ValueError("Bounding box must have 4 coordinates: (y_min, x_min, y_max, x_max)")
    
    y_min, x_min, y_max, x_max = bbox_coords
    height, width = image_shape
    
    # Auto-detect if coordinates are normalized (all values <= 1.0)
    is_normalized = all(coord <= 1.0 for coord in bbox_coords)
    
    # Convert to normalized coordinates if they're in pixels
    if is_normalized:
        norm_coords = bbox_coords
    else:
        norm_coords = (
            y_min / height,
            x_min / width, 
            y_max / height,
            x_max / width
        )
    
    # Convert to 1024-bin coordinates (Paligemma format)
    binned_loc = [int(round(coord * 1023)) for coord in norm_coords]
    binned_loc = [max(0, min(1023, coord)) for coord in binned_loc]
    
    # Create location tokens
    loc_tokens = ''.join([f'<loc{coord:04d}>' for coord in binned_loc])
    return loc_tokens


def update_segmentation_token(original_seg_token: str, 
                             new_bbox_token: str) -> str:
    """
    Update segmentation token by replacing the bounding box tokens with new bbox tokens
    while keeping the segmentation tokens unchanged.
    
    Args:
        original_seg_token (str): Original segmentation token string containing both 
                                 bbox and seg tokens
        new_bbox_token (str): New bounding box token string (already corrected)
        
    Returns:
        str: Updated segmentation token with new bbox tokens and original seg tokens
        
    Example:
        >>> original = '<loc0393><loc0353><loc0769><loc0788><seg001><seg002><seg003>'
        >>> new_bbox = '<loc0204><loc0102><loc0716><loc0818>'
        >>> updated = update_segmentation_token(original, new_bbox)
        >>> updated  # '<loc0204><loc0102><loc0716><loc0818><seg001><seg002><seg003>'
    """
    if not original_seg_token:
        return None
    
    # Find where the segmentation tokens start (after the 4 location tokens)
    # Each location token is 9 characters: '<loc0000>'
    # So first 4 tokens are 36 characters
    bbox_token_length = 4 * 9  # 4 location tokens × 9 characters each
    
    if len(original_seg_token) <= bbox_token_length:
        # If original token only contains bbox tokens, return just the new bbox tokens
        return new_bbox_token
    
    # Extract the seg tokens part (everything after the first 4 loc tokens)
    seg_tokens_part = original_seg_token[bbox_token_length:]
    
    # Combine new bbox tokens with original seg tokens
    updated_token = new_bbox_token + seg_tokens_part
    
    return updated_token


def bounding_box_token_to_coordinates(bbox_token: str, 
                                    image_shape: Tuple[int, int],
                                    normalized: bool = False) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert bounding box tokens to coordinate values.
    
    Args:
        bbox_token (str): Bounding box token string (e.g., '<loc0393><loc0353><loc0769><loc0788>')
        image_shape (Tuple[int, int]): Image shape (height, width) for pixel coordinate conversion
        normalized (bool): If True, return normalized coordinates (0-1), otherwise pixel coordinates
        
    Returns:
        Optional[Tuple[float, float, float, float]]: Bounding box coordinates (y_min, x_min, y_max, x_max)
                                                   None if parsing fails
        
    Example:
        >>> token = '<loc0393><loc0353><loc0769><loc0788>'
        >>> coords = bounding_box_token_to_coordinates(token, (512, 512), normalized=True)
        >>> coords  # (0.384, 0.345, 0.752, 0.771)
    """
    import re
    
    # Extract location tokens using regex
    loc_pattern = r'<loc(\d{4})>'
    matches = re.findall(loc_pattern, bbox_token)
    
    if len(matches) != 4:
        return None
    
    try:
        # Convert to integers and then normalize (Paligemma uses 1024 bins: 0-1023)
        binned_coords = [int(match) for match in matches]
        
        # Convert from 1024-bin coordinates to normalized coordinates (0-1)
        norm_coords = [coord / 1023.0 for coord in binned_coords]
        
        if normalized:
            return tuple(norm_coords)
        else:
            # Convert to pixel coordinates
            height, width = image_shape
            y_min, x_min, y_max, x_max = norm_coords
            
            pixel_coords = (
                y_min * height,
                x_min * width, 
                y_max * height,
                x_max * width
            )
            
            # Round to integers for pixel coordinates
            return tuple(int(round(coord)) for coord in pixel_coords)
            
    except (ValueError, IndexError):
        return None


def coordinates_to_bounding_box_token(coordinates: Tuple[float, float, float, float],
                                    image_shape: Optional[Tuple[int, int]] = None) -> str:
    """
    Convert coordinate values to bounding box tokens.
    
    Args:
        coordinates (Tuple[float, float, float, float]): Bounding box coordinates (y_min, x_min, y_max, x_max)
        image_shape (Optional[Tuple[int, int]]): Image shape (height, width) required if coordinates are in pixels
        
    Returns:
        str: Bounding box token string in Paligemma format
        
    Note:
        Automatically detects if input coordinates are normalized (all values <= 1.0) or pixel coordinates.
        For pixel coordinates, image_shape must be provided.
        
    Example:
        >>> coords = (0.384, 0.345, 0.752, 0.771)  # normalized
        >>> token = coordinates_to_bounding_box_token(coords)
        >>> token  # '<loc0393><loc0353><loc0769><loc0788>'
        
        >>> coords = (196, 177, 385, 395)  # pixel coordinates
        >>> token = coordinates_to_bounding_box_token(coords, (512, 512))
        >>> token  # '<loc0383><loc0346><loc0752><loc0772>'
    """
    if len(coordinates) != 4:
        raise ValueError("Coordinates must have 4 values: (y_min, x_min, y_max, x_max)")
    
    y_min, x_min, y_max, x_max = coordinates
    
    # Auto-detect if coordinates are normalized (all values <= 1.0)
    is_normalized = all(coord <= 1.0 for coord in coordinates)
    
    if is_normalized:
        norm_coords = coordinates
    else:
        # Convert pixel coordinates to normalized coordinates
        if image_shape is None:
            raise ValueError("image_shape must be provided when using pixel coordinates")
        
        height, width = image_shape
        norm_coords = (
            y_min / height,
            x_min / width,
            y_max / height, 
            x_max / width
        )
    
    # Convert to 1024-bin coordinates (Paligemma format)
    binned_coords = [int(round(coord * 1023)) for coord in norm_coords]
    
    # Ensure coordinates are within valid range [0, 1023]
    binned_coords = [max(0, min(1023, coord)) for coord in binned_coords]
    
    # Create location tokens
    loc_tokens = ''.join([f'<loc{coord:04d}>' for coord in binned_coords])
    
    return loc_tokens