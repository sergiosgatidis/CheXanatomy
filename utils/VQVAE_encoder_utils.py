# Derived in part from:
# https://huggingface.co/spaces/big-vision/paligemma-hf/blob/main/app.py
#
# Original work:
# Copyright 2024 Big Vision Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for CheXanatomy:
# - Adapted for local VQ-VAE encoder utilities.
# - Modified from the original implementation.
# - See /LICENSES/Apache-2.0.txt for the full license text.
"""
Segmentation utilities.

Real VQVAE implementation for mask encoding using TensorFlow.
"""
import numpy as np
from PIL import Image
import os
import tensorflow as tf


# Constants from original code
NUM_DOWNSAMPLE_LAYERS = 4
NUM_RES_BLOCKS = 2


def get_VQVAE_checkpoint(model_path: str = None):
    """
    Load VQVAE checkpoint from disk
    
    Args:
        model_path: Path to the VQVAE model file
        
    Returns:
        dict: Model weights loaded from NPZ file
    """
    if model_path is None:
        # Default path in our project
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vae-oid.npz')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"VQVAE model not found at {model_path}")
    
    return dict(np.load(model_path))


def encode_to_codebook_indices(checkpoint, masks, use_cpu: bool = False):
    """
    Encode a batch of binary segmentation masks into 16 tokens each.
    Based on the original TensorFlow implementation.

    Args:
        checkpoint: model weights from PyTorch model.
        masks: Must be in range `[0..1]`, and of shape `[None, 64, 64, 1]`.
        use_cpu: Boolean flag to determine if operations should run on CPU.

    Returns:
        A numpy array of shape `[None, 16]` with elements in `range(128)`.
    """
    device = '/CPU:0' if use_cpu else '/GPU:0'

    with tf.device(device):
        # We require that the input masks are already resized to 64x64.
        x = tf.ensure_shape(masks, [None, 64, 64, 1])
        x = _norm(x, device)

        for n in range(NUM_DOWNSAMPLE_LAYERS):
            x = _conv_tf(
                checkpoint, x, strides=2, padding='SAME', layer_name=f'encoder.{2*n}', device=device
            )
            x = tf.nn.relu(x)

        for n in range(NUM_RES_BLOCKS):
            x = _resblock_tf(checkpoint, x, layer_name=f'encoder.{8+n}.net', device=device)

        x = _conv_tf(
            checkpoint, x, strides=1, padding='SAME', layer_name='encoder.10', device=device
        )

        return _get_codebook_indices(checkpoint, x, device)


def _norm(x, device):
    """Normalize input to [-1, 1] range"""
    with tf.device(device):
        return 2.0 * (x - 0.5)


def _conv_tf(checkpoint, x, strides, padding, layer_name, device):
    """TensorFlow convolution implementation"""
    with tf.device(device):
        kernel = checkpoint[layer_name + '.weight']
        kernel = np.transpose(kernel, (2, 3, 1, 0))
        bias = checkpoint[layer_name + '.bias']
        return tf.nn.conv2d(x, kernel, strides=strides, padding=padding) + bias


def _resblock_tf(checkpoint, x, layer_name, device):
    """Apply a residual block of the mask encoder."""
    with tf.device(device):
        original_x = x
        x = _conv_tf(
            checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.0', device=device
        )
        x = tf.nn.relu(x)
        x = _conv_tf(
            checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.2', device=device
        )
        x = tf.nn.relu(x)
        x = _conv_tf(
            checkpoint, x, padding='SAME', strides=1, layer_name=layer_name + '.4', device=device
        )
        return x + original_x


def _get_codebook_indices(checkpoint, encoder_output, device):
    """Get the closest codebook indices for the encoder output"""
    with tf.device(device):
        embeddings = checkpoint['_vq_vae._embedding']
        flat_input = tf.reshape(encoder_output, [-1, embeddings.shape[1]])
        distances = (
            tf.reduce_sum(flat_input**2, axis=1, keepdims=True)
            + tf.reduce_sum(embeddings**2, axis=1)
            - 2 * tf.matmul(flat_input, embeddings.T)
        )
        indices = tf.argmin(distances, axis=1)
        return tf.reshape(indices, [-1, 16])


def resize_mask_for_encoding(mask: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Resize mask to the size expected by VQVAE encoder
    
    Args:
        mask: Input mask
        target_size: Target size (height, width)
        
    Returns:
        Resized mask as numpy array
    """
    # Convert to PIL for resizing
    if len(mask.shape) == 2:
        pil_mask = Image.fromarray(mask.astype(np.uint8))
    else:
        pil_mask = Image.fromarray(mask[:, :, 0].astype(np.uint8))
    
    # Resize
    resized_pil = pil_mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    resized_array = np.array(resized_pil)
    
    # Ensure binary and normalize to [0, 1]
    resized_array = (resized_array > 0).astype(np.float32)
    
    return resized_array