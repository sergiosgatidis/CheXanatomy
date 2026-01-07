"""
Token to Mask Decoder - Enhanced Version  
Based on the original token_to_mask_decoder.py

Functions for decoding segmentation tokens back to masks with full VQVAE support

# This code is derived from https://huggingface.co/spaces/big-vision/paligemma-hf/blob/main/app.py

"""
import re
import os
import functools
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# Try to import JAX dependencies for full VQVAE support
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX/Flax not available. Using simplified decoder.")


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


# VQVAE Functions for full segmentation mask reconstruction
def _get_params(checkpoint):
    """Convert PyTorch checkpoint to Flax params."""
    def transp(kernel):
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        return {
            'bias': checkpoint[name + '.bias'],
            'kernel': transp(checkpoint[name + '.weight']),
        }

    def resblock(name):
        return {
            'Conv_0': conv(name + '.0'),
            'Conv_1': conv(name + '.2'),
            'Conv_2': conv(name + '.4'),
        }

    return {
        '_embeddings': checkpoint['_vq_vae._embedding'],
        'Conv_0': conv('decoder.0'),
        'ResBlock_0': resblock('decoder.2.net'),
        'ResBlock_1': resblock('decoder.3.net'),
        'ConvTranspose_0': conv('decoder.4'),
        'ConvTranspose_1': conv('decoder.6'),
        'ConvTranspose_2': conv('decoder.8'),
        'ConvTranspose_3': conv('decoder.10'),
        'Conv_1': conv('decoder.12'),
    }


def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    """Get quantized values from codebook indices."""
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings


@functools.cache
def _get_reconstruct_masks():
    """Reconstructs masks from codebook indices."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX/Flax required for full VQVAE decoder support")

    class ResBlock(nn.Module):
        features: int

        @nn.compact
        def __call__(self, x):
            original_x = x
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
            return x + original_x

    class Decoder(nn.Module):
        """Upscales quantized vectors to mask."""

        @nn.compact
        def __call__(self, x):
            num_res_blocks = 2
            dim = 128
            num_upsample_layers = 4

            x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
            x = nn.relu(x)

            for _ in range(num_res_blocks):
                x = ResBlock(features=dim)(x)

            for _ in range(num_upsample_layers):
                x = nn.ConvTranspose(
                    features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding=2,
                    transpose_kernel=True,
                )(x)
                x = nn.relu(x)
                dim //= 2

            x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)

            return x

    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params['_embeddings']
        )
        return Decoder().apply({'params': params}, quantized)

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"VQVAE model not found at {_MODEL_PATH}")

    with open(_MODEL_PATH, 'rb') as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend='cpu')


def extract_objs(text, width, height, unique_labels=False):
    """
    Returns objects for a string with "<loc>" and "<seg>" tokens.
    
    Args:
        text: Input string containing location and segmentation tokens
        width: Image width
        height: Image height
        unique_labels: Whether to ensure unique labels
        
    Returns:
        List of objects with content, bounding boxes, masks, and names
    """
    objs = []
    seen = set()
    
    while text:
        m = _SEGMENT_DETECT_RE.match(text)
        if not m:
            break
        gs = list(m.groups())
        before = gs.pop(0)
        name = gs.pop()
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        
        y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))
        seg_indices = gs[4:20]
        
        if seg_indices[0] is None:
            mask = None
        else:
            try:
                seg_indices = np.array([int(x) for x in seg_indices], dtype=np.int32)
                m64, = _get_reconstruct_masks()(seg_indices[None])[..., 0]
                m64 = np.clip(np.array(m64) * 0.5 + 0.5, 0, 1)
                m64 = Image.fromarray((m64 * 255).astype('uint8'))
                mask = np.zeros([height, width])
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = np.array(m64.resize([x2 - x1, y2 - y1])) / 255.0
            except Exception as e:
                print(f"Warning: VQVAE reconstruction failed ({e}), using bbox mask")
                # Fallback to simple bounding box mask
                mask = np.zeros([height, width])
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = 1.0

        content = m.group()
        if before:
            objs.append(dict(content=before))
            content = content[len(before):]
        while unique_labels and name in seen:
            name = (name or '') + "'"
        seen.add(name)
        objs.append(dict(
            content=content, xyxy=(x1, y1, x2, y2), mask=mask, name=name))
        text = text[len(before) + len(content):]

    if text:
        objs.append(dict(content=text))

    return objs