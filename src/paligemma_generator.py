"""
Paligemma Sample Generator for CheXanatomy - Core Library Module

This module provides the core PaligemmaSampleGenerator class for generating training samples
for Paligemma models using chest X-ray anatomy data. This is a faithful recreation of the 
original task_definitions.py script, simplified and focused on the 5 core tasks.

PURPOSE:
--------
- Core library for generating Paligemma training samples from chest X-ray data
- Designed for direct integration into training pipelines (no file I/O)
- Provides single sample generation for flexible pipeline feeding
- Maintains exact compatibility with original task_definitions.py structure

TASK TYPES:
-----------
1. detection: Generate 'detect [structure]' prompts with bbox token responses
2. segmentation: Generate 'segment [structure]' prompts with mask token responses  
3. bbox_token_identification: Generate bbox token prompts with structure name responses
4. mask_token_identification: Generate mask token prompts with structure name responses\n5. orientation_identification: Generate 'How is this slice oriented?' prompts with orientation responses

USAGE PATTERN:
--------------
```python
from chexanatomy.paligemma_generator import PaligemmaSampleGenerator

# Initialize with image info
generator = PaligemmaSampleGenerator('/path/to/image_info.json')

# Generate single sample
sample = generator.generate_sample()
# Returns TrainingSample(prefix, suffix, image, task_type, structure)

# Use in training pipeline
for sample in training_loop:
    model.train_step(sample.prefix, sample.suffix, sample.image)
```

RELATIONSHIP TO OTHER SCRIPTS:
------------------------------
- scripts/paligemma_training_generator.py: CLI interface using this core library
- examples/task_examples.py: Demonstration script showing usage examples
- This module: Core library containing the generator class implementation

Based on original task_definitions.py - simplified and pipeline-focused.
"""

import numpy as np
from PIL import Image
import random
import json
import os
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from chexanatomy.anatomy import (get_bounding_box_token, get_segmentation_token, get_bounding_box,
                                augment_image_with_random_scale_and_position, augment_bounding_box,
                                update_bounding_box_token, update_segmentation_token)


@dataclass
class TrainingSample:
    """Container for a training sample matching Paligemma format"""
    prefix: str
    suffix: str
    image: Image.Image
    task_type: str
    structure: Optional[str] = None


class PaligemmaSampleGenerator:
    """
    Sample generator for training Paligemma with chest X-ray anatomy tasks.
    
    Generates samples for core tasks:
    - detection: Find anatomical structures  
    - segmentation: Segment anatomical structures
    - bbox_token_identification: Identify structure from bbox token
    - mask_token_identification: Identify structure from mask token
    - bbox_identification: Identify structure from visual bounding box overlay
    - mask_identification: Identify structure from visual segmentation mask overlay
    - orientation_identification: Identify image orientation (axial/sagittal/coronal)    """
    
    def __init__(self, image_info: Union[str, Dict], enable_augmentation: bool = False,
                 scale_range: tuple = (0.7, 1.0), background_value: float = 0.0):
        """
        Initialize the sample generator.
        
        Args:
            image_info: Path to NPZ file, JSON file, or dict containing image information.
                       NPZ files are preferred for smaller file sizes.
            enable_augmentation: If True, apply random augmentation to images and update
                               all related coordinates and tokens accordingly.
            scale_range: Range for random scaling during augmentation (min_scale, max_scale).
                        Only used if enable_augmentation=True.
            background_value: Background fill value for augmented images.
                            Only used if enable_augmentation=True.
        """
        self.enable_augmentation = enable_augmentation
        self.scale_range = scale_range
        self.background_value = background_value
        
        if isinstance(image_info, str):
            self.img_info = self._load_img_info(image_info)
        else:
            self.img_info = image_info
            
        # Apply augmentation if enabled and we loaded from file
        if self.enable_augmentation and isinstance(image_info, str):
            self._apply_augmentation()
            
        self.anatomic_names = self._get_anatomic_names()
    
    def _apply_augmentation(self):
        """
        Apply augmentation to the loaded image data and update all related tokens.
        
        This method:
        1. Augments the image array with random scaling and positioning
        2. Updates all bounding box coordinates (pixel and normalized)
        3. Updates all bounding box tokens
        4. Updates all segmentation tokens to maintain alignment
        """
        if "img_array" not in self.img_info or "structure_info" not in self.img_info:
            print("Warning: Cannot apply augmentation - missing required image data")
            return
        
        # Get original image
        original_image = np.array(self.img_info["img_array"])
        
        # Apply image augmentation
        augmented_image, transform_params = augment_image_with_random_scale_and_position(
            original_image,
            scale_range=self.scale_range,
            background_value=self.background_value
        )
        
        # Update the image array
        self.img_info["img_array"] = augmented_image
        
        # Update all structure information
        for structure_name, structure_info in self.img_info["structure_info"].items():
            # Update pixel coordinates
            if "bbox" in structure_info:
                original_bbox = structure_info["bbox"]
                augmented_bbox = augment_bounding_box(
                    original_bbox, transform_params, original_image.shape
                )
                structure_info["bbox"] = augmented_bbox
            
            # Update normalized coordinates
            if "bbox_norm" in structure_info:
                original_bbox_norm = structure_info["bbox_norm"]
                augmented_bbox_norm = augment_bounding_box(
                    original_bbox_norm, transform_params, original_image.shape
                )
                structure_info["bbox_norm"] = augmented_bbox_norm
            
            # Update bounding box tokens
            if "bbox_token" in structure_info and "bbox" in structure_info:
                new_bbox_token = update_bounding_box_token(
                    structure_info["bbox"], original_image.shape
                )
                structure_info["bbox_token"] = new_bbox_token
            
            # Update segmentation tokens
            if "segmentation_token" in structure_info and "bbox_token" in structure_info:
                original_seg_token = structure_info["segmentation_token"]
                new_seg_token = update_segmentation_token(
                    original_seg_token, structure_info["bbox_token"]
                )
                structure_info["segmentation_token"] = new_seg_token
        
        # Store transformation parameters for reference
        self.img_info["augmentation_params"] = transform_params
        
        print(f"Applied augmentation: scale={transform_params['scale_factor']:.3f}, "
              f"offset=({transform_params['x_offset']}, {transform_params['y_offset']})")
    
    def _load_img_info(self, path: str) -> dict:
        """
        Load image information from NPZ or JSON file.
        
        Args:
            path: Path to the image information file (.npz or .json)
            
        Returns:
            dict: The reconstructed img_info dictionary
            
        Raises:
            ValueError: If the file format is not supported
        """
        file_extension = os.path.splitext(path)[1].lower()
        
        if file_extension == ".npz":
            data = np.load(path, allow_pickle=True)
            # Reconstruct the original dictionary structure
            img_info = {
                "img_array": data["img_array"],
                **data["metadata"].item()
            }
            return img_info
        elif file_extension == ".json":
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only .npz and .json files are supported.")
    
    def _get_anatomic_names(self) -> Dict[str, List[str]]:
        """Get anatomical name variations for structures"""
        return {
            'bones': ["Bones", "Skeleton", "Osseous Structures"],
            'sacrum': ["Sacrum", "Sacral Bone", "Os Sacrum"],
            'skull': ["Skull", "Cranium"],
            'sternum': ["Sternum", "Breastbone"],
            'lung_right': ["Right Lung", "Pulmo Dextra"],
            'lung_left': ["Left Lung", "Pulmo Sinistra"],
            'lung_lower_lobe_right': ["Right Lower Lung Lobe", "Right Lower Lobe", "RLL"],
            'lung_middle_lobe_right': ["Right Middle Lung Lobe", "Right Middle Lobe", "RML"],
            'lung_upper_lobe_right': ["Right Upper Lung Lobe", "Right Upper Lobe", "RUL"],
            'lung_lower_lobe_left': ["Left Lower Lung Lobe", "Left Lower Lobe", "LLL"],
            'lung_upper_lobe_left': ["Left Upper Lung Lobe", "Left Upper Lobe", "LUL"],
            'vertebrae_body': ["Spine", "Vertebral Column"],
            'heart': ["Heart", "Cor"],
            'pulmonary_artery': ["Pulmonary Artery", "Pulmonary Trunk"],
            'lung_trachea_bronchia': ["Tracheobronchial tree"],
            'trachea': ["Trachea", "Windpipe"],
            'bronchia': ["Bronchi", "Bronchial tubes"],
            'aorta': ["Aorta"],
            'body': ["Skeleton", "Bones"],
            'brachiocephalic_vein_left': ["Left Brachiocephalic Vein", "Left Innominate Vein"],
            'brachiocephalic_vein_right': ["Right Brachiocephalic Vein", "Right Innominate Vein"],
            'clavicula_left': ["Left Clavicle", "Left Collarbone"],
            'clavicula_right': ["Right Clavicle", "Right Collarbone"],
            'colon': ["Colon", "Large Bowel", "Large Intestine"],
            'costal_cartilages': ["Costal Cartilages", "Rib Cartilages"],
            'esophagus': ["Esophagus", "Oesophagus", "Food Pipe"],
            'humerus_left': ["Left Humerus", "Left Upper Arm Bone"],
            'humerus_right': ["Right Humerus", "Right Upper Arm Bone"],
            'inferior_vena_cava': ["Inferior Vena Cava", "IVC"],
            'intervertebral_discs': ["Intervertebral Discs", "Spinal discs"],
            'liver': ["Liver", "Hepar"],
            'spleen': ["Spleen", "Lien", "Splenic Organ"],
            'pulmonary_vein': ["Pulmonary Vein", "Pulmonary Veins"],
            'heart_atrium_left': ["Left Atrial Appendage", "Left Atrial Auricle"],
            'heart_atrium_right': ["Right Atrium"],
            'heart_ventricle_left': ["Left Ventricle"],
            'heart_ventricle_right': ["Right Ventricle"],
            'lung_vessels': ["Pulmonary vessels"],
            'scapula_left': ["Left Scapula", "Left Shoulder Blade"],
            'scapula_right': ["Right Scapula", "Right Shoulder Blade"],
            'stomach': ["Stomach", "Gaster"],
            'subcutaneous_fat': ["Subcutaneous Fat"],
            'superior_vena_cava': ["Superior Vena Cava", "SVC"],
            'torso_fat': ["Torso Fat", "Body Fat", "Visceral Fat"],
        }
    
    def get_structure_info(self, structure: str) -> Optional[dict]:
        """
        Retrieve structure information from image info.
        
        Args:
            structure: The structure label (e.g., 'heart', 'lung_left')
            
        Returns:
            Structure information dict or None if not found
        """
        return self.img_info["structure_info"].get(structure)
    
    def get_available_structures(self) -> List[str]:
        """Get list of structures available in current image"""
        return list(self.img_info.get('structure_info', {}).keys())
    
    def is_augmented(self) -> bool:
        """Check if the current image data has been augmented."""
        return "augmentation_params" in self.img_info
    
    def get_augmentation_params(self) -> Optional[dict]:
        """Get the augmentation parameters if image was augmented."""
        return self.img_info.get("augmentation_params")
    
    def generate_sample(self, task_name: str, structure: Optional[str] = None) -> Optional[TrainingSample]:
        """
        Generate a single training sample for the specified task and structure.
        
        This is the core method for generating training data. It creates prompt-response 
        pairs for different anatomical understanding tasks using chest X-ray images.
        Designed for direct integration into training pipelines.
        
        Args:
            task_name: The task type to generate. Must be one of:
                      - "detection": Generate 'detect [structure]' → bbox token + structure
                      - "segmentation": Generate 'segment [structure]' → mask token + structure  
                      - "bbox_token_identification": Generate bbox tokens → structure name
                      - "mask_token_identification": Generate mask tokens → structure name
                      - "bbox_identification": Generate image with bbox overlay → structure name
                      - "mask_identification": Generate image with mask overlay → structure name
                      - "orientation_identification": Generate 'How is this slice oriented?' → orientation
            structure: Anatomical structure to focus on. If None, randomly selects from 
                      available structures in the image. Not used for orientation_identification.
            
        Returns:
            TrainingSample: Container with prompt, response, image, and metadata for pipeline feeding.
                           Returns None if task_name is not implemented.
                           
        Pipeline Integration Example:
            ```python
            for epoch in training_epochs:
                for structure in available_structures:
                    sample = generator.generate_sample('detection', structure)
                    if sample:
                        loss = model.forward(sample.prefix, sample.suffix, sample.image)
                        optimizer.step(loss)
            ```
        """
        task_methods = {
            "detection": self.generate_detection_sample,
            "segmentation": self.generate_segmentation_sample,
            "bbox_token_identification": self.generate_bbox_token_identification_sample,
            "mask_token_identification": self.generate_mask_token_identification_sample,
            "bbox_identification": self.generate_bbox_identification_sample,
            "mask_identification": self.generate_mask_identification_sample,
            "orientation_identification": self.generate_orientation_identification_sample,
        }
        
        task_name = task_name.lower()
        
        if task_name not in task_methods:
            return None
        
        # Some tasks don't require anatomical structure (e.g., orientation_identification)
        if task_name in ["orientation_identification"]:
            return task_methods[task_name]()
        
        # Select random structure if not specified
        if structure is None:
            available = self.get_available_structures()
            if not available:
                return None
            structure = random.choice(available)
        
        try:
            return task_methods[task_name](structure)
        except Exception as e:
            print(f"Error generating {task_name} task for {structure}: {e}")
            return None
    
    def generate_detection_sample(self, structure: str) -> TrainingSample:
        """
        Generate a detection sample.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with detection task
        """
        img_array = np.array(self.img_info["img_array"])
        image = self._array_to_rgb_image(img_array)
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        structure_info = self.get_structure_info(structure)
        
        if not structure_info:
            # Structure not found, return empty suffix
            prefix = f"detect {structure_name}"
            suffix = ""
            return TrainingSample(prefix=prefix, suffix=suffix, image=image, task_type="detection", structure=structure)
        
        prefix = f"detect {structure_name}"
        suffix = f"{structure_info['bbox_token']} {structure_name}"
        return TrainingSample(prefix=prefix, suffix=suffix, image=image, task_type="detection", structure=structure)
    
    def generate_segmentation_sample(self, structure: str) -> TrainingSample:
        """
        Generate a segmentation sample.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with segmentation task
        """
        img_array = np.array(self.img_info["img_array"])
        image = self._array_to_rgb_image(img_array)
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        structure_info = self.get_structure_info(structure)
        
        if not structure_info:
            # Structure not found, return empty suffix
            prefix = f"segment {structure_name}"
            suffix = ""
            return TrainingSample(prefix=prefix, suffix=suffix, image=image, task_type="segmentation", structure=structure)
        
        prefix = f"segment {structure_name}"
        suffix = f"{structure_info['segmentation_token']} {structure_name}"
        return TrainingSample(prefix=prefix, suffix=suffix, image=image, task_type="segmentation", structure=structure)
    
    def generate_bbox_token_identification_sample(self, structure: str) -> Optional[TrainingSample]:
        """
        Generate a bounding box token identification sample.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with bbox identification task or None
        """
        structure_info = self.get_structure_info(structure)
        if not structure_info or "bbox_token" not in structure_info:
            return None
        
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        prefix = f"caption {structure_info['bbox_token']}"
        suffix = structure_name
        
        img_array = np.array(self.img_info["img_array"])
        image = self._array_to_rgb_image(img_array)
        
        return TrainingSample(prefix=prefix, suffix=suffix, image=image, 
                            task_type="bbox_token_identification", structure=structure)
    
    def generate_mask_token_identification_sample(self, structure: str) -> Optional[TrainingSample]:
        """
        Generate a mask token identification sample.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with mask identification task or None
        """
        structure_info = self.get_structure_info(structure)
        if not structure_info or "segmentation_token" not in structure_info:
            return None
        
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        prefix = f"caption {structure_info['segmentation_token']}"
        suffix = structure_name
        
        img_array = np.array(self.img_info["img_array"])
        image = self._array_to_rgb_image(img_array)
        
        return TrainingSample(prefix=prefix, suffix=suffix, image=image,
                            task_type="mask_token_identification", structure=structure)
    
    def generate_bbox_identification_sample(self, structure: str) -> Optional[TrainingSample]:
        """
        Generate a bbox identification sample with bounding box drawn on image.
        
        This task creates an image with a bounding box overlay and asks the model
        to identify which structure is highlighted by the bounding box.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with bbox identification task or None
        """
        structure_info = self.get_structure_info(structure)
        if not structure_info or "bbox" not in structure_info:
            return None
        
        # Get the image and draw bounding box
        img_array = np.array(self.img_info["img_array"])
        image_with_bbox = self._draw_bounding_box_on_image(img_array, structure_info["bbox"])
        
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        
        # Various ways to ask about the highlighted structure
        prefix_variations = [
            "Which structure is labeled by the bounding box?",
            "What structure is highlighted by the bounding box?", 
            "Identify the structure marked by the bounding box.",
            "What anatomical structure is outlined in the image?",
            "Which structure is indicated by the red box?",
            "What structure is shown in the bounding box?"
        ]
        prefix = random.choice(prefix_variations)
        suffix = structure_name
        
        return TrainingSample(prefix=prefix, suffix=suffix, image=image_with_bbox,
                            task_type="bbox_identification", structure=structure)
    
    def generate_mask_identification_sample(self, structure: str) -> Optional[TrainingSample]:
        """
        Generate a mask identification sample with segmentation mask drawn on image.
        
        This task creates an image with a segmentation mask overlay and asks the model
        to identify which structure is highlighted by the mask.
        
        Args:
            structure: The structure label
            
        Returns:
            TrainingSample with mask identification task or None
        """
        structure_info = self.get_structure_info(structure)
        if not structure_info or "segmentation_token" not in structure_info:
            return None
        
        # Get the image and draw segmentation mask
        img_array = np.array(self.img_info["img_array"])
        image_with_mask = self._draw_segmentation_mask_on_image(img_array, structure_info["segmentation_token"])
        
        if image_with_mask is None:
            return None
        
        structure_name = random.choice(self.anatomic_names.get(structure, [structure.replace('_', ' ')]))
        
        # Various ways to ask about the highlighted structure
        prefix_variations = [
            "Which structure is highlighted by the segmentation mask?",
            "What structure is shown by the colored overlay?", 
            "Identify the structure marked by the segmentation mask.",
            "What anatomical structure is segmented in the image?",
            "Which structure is indicated by the colored region?",
            "What structure is shown by the mask overlay?"
        ]
        prefix = random.choice(prefix_variations)
        suffix = structure_name
        
        return TrainingSample(prefix=prefix, suffix=suffix, image=image_with_mask,
                            task_type="mask_identification", structure=structure)
    
    def generate_orientation_identification_sample(self) -> TrainingSample:
        """
        Generate an orientation identification sample.
        
        This task asks the model to identify the orientation of the medical image slice
        (axial, sagittal, or coronal). It doesn't require any anatomical structure input.
        
        Returns:
            TrainingSample with orientation identification task
        """
        img_array = np.array(self.img_info["img_array"])
        image = self._array_to_rgb_image(img_array)
        
        # Get orientation from image info, defaulting to "axial" if not present
        orientation = self.img_info.get("orientation", "ax")
        
        # Map orientation codes to full names
        orientation_suffix = {
            "ax": "axial",
            "sag": "sagittal", 
            "cor": "coronal",
            "pa": "posteroanterior",
            "lr": "lateral",
            "axial": "axial",
            "sagittal": "sagittal",
            "coronal": "coronal"
        }
        orientation_label = orientation_suffix.get(orientation.lower(), "axial")
        
        # Various ways to ask about orientation
        prefix_variations = [
            "How is this slice oriented?",
            "What is the orientation of this slice?", 
            "Identify the orientation of this slice.",
            "Determine the orientation of this slice.",
            "What anatomical plane is this image taken in?",
            "Which orientation does this slice represent?"
        ]
        prefix = random.choice(prefix_variations)
        suffix = orientation_label
        
        return TrainingSample(prefix=prefix, suffix=suffix, image=image,
                            task_type="orientation_identification", structure=None)
    
    def _draw_bounding_box_on_image(self, img_array: np.ndarray, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Draw a bounding box on the image and return as PIL Image.
        
        Args:
            img_array: The original image array
            bbox: Bounding box coordinates (y_min, x_min, y_max, x_max)
            
        Returns:
            PIL Image with bounding box drawn
        """
        from PIL import ImageDraw
        
        # Convert to RGB image first
        image = self._array_to_rgb_image(img_array)
        
        # Create a copy for drawing
        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        
        # Extract bbox coordinates (y_min, x_min, y_max, x_max)
        y_min, x_min, y_max, x_max = bbox
        
        # Randomize appearance
        colors = ['red', 'lime', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'white']
        color = random.choice(colors)
        
        # Random line width (thickness)
        line_width = random.randint(2, 6)
        
        # Draw bounding box rectangle with random color and thickness
        box_coords = [x_min, y_min, x_max, y_max]  # PIL uses (x_min, y_min, x_max, y_max)
        
        # Draw multiple lines to make it thick and visible
        for i in range(line_width):
            draw.rectangle(
                [box_coords[0] + i, box_coords[1] + i, box_coords[2] - i, box_coords[3] - i],
                outline=color, fill=None
            )
        
        return image_with_bbox
    
    def _draw_segmentation_mask_on_image(self, img_array: np.ndarray, segmentation_token: str) -> Optional[Image.Image]:
        """
        Draw a segmentation mask on the image and return as PIL Image.
        
        Args:
            img_array: The original image array
            segmentation_token: Segmentation token string
            
        Returns:
            PIL Image with segmentation mask drawn, or None if mask extraction fails
        """
        try:
            # Import decoder to extract mask from segmentation tokens
            from decoder import extract_objs
            
            # Get image dimensions
            height, width = img_array.shape[:2]
            
            # Extract segmentation objects from tokens
            segmentation_objects = extract_objs(segmentation_token, width, height)
            
            # Find the first object with a mask
            mask = None
            for obj in segmentation_objects:
                if 'mask' in obj and obj['mask'] is not None:
                    mask = obj['mask']
                    break
            
            if mask is None:
                return None
            
            # Convert to RGB image first
            image = self._array_to_rgb_image(img_array)
            
            # Convert image to numpy array for mask overlay
            img_np = np.array(image)
            
            # Randomize mask appearance
            colors = [
                [255, 0, 255],    # Magenta
                [0, 255, 255],    # Cyan
                [255, 255, 0],    # Yellow
                [0, 255, 0],      # Green
                [255, 0, 0],      # Red
                [0, 0, 255],      # Blue
                [255, 165, 0],    # Orange
                [128, 0, 128],    # Purple
            ]
            
            # Random color selection
            mask_color = random.choice(colors)
            
            # Random transparency (alpha between 0.3 and 0.7)
            alpha = random.uniform(0.3, 0.5)
            
            # Create colored mask overlay with random color
            mask_overlay = np.zeros_like(img_np)
            mask_overlay[:, :, 0] = mask_color[0]  # Red channel
            mask_overlay[:, :, 1] = mask_color[1]  # Green channel  
            mask_overlay[:, :, 2] = mask_color[2]  # Blue channel
            
            # Apply mask (only where mask > threshold)
            mask_threshold = 0.3
            mask_binary = mask > mask_threshold
            
            # Blend the mask overlay with the original image using random alpha
            img_np = img_np.astype(float)
            
            for c in range(3):  # For each color channel
                img_np[:, :, c] = np.where(
                    mask_binary,
                    (1 - alpha) * img_np[:, :, c] + alpha * mask_overlay[:, :, c],
                    img_np[:, :, c]
                )
            
            # Convert back to PIL Image
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            image_with_mask = Image.fromarray(img_np)
            
            return image_with_mask
            
        except Exception as e:
            print(f"Warning: Failed to extract segmentation mask ({e}), skipping mask identification task")
            return None
    
    def _array_to_rgb_image(self, array: np.ndarray) -> Image.Image:
        """
        Convert a slice array to a normalized RGB image.
        
        Args:
            array: Input image array
            
        Returns:
            RGB Image
        """
        img_array_normalized = self.normalize_image_array(array)
        return Image.fromarray(img_array_normalized).convert("RGB")
    
    def normalize_image_array(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize an image array to the 0–255 range for visualization.
        
        Args:
            img_array: The original image array
            
        Returns:
            Normalized image array in 0–255 range as np.uint8
        """
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array_normalized = np.zeros_like(img_array, dtype=np.uint8)
        return img_array_normalized


def save_training_samples(samples: List[TrainingSample], output_path: str):
    """
    Save training samples to JSON file (utility function - not used in pipeline mode).
    
    Args:
        samples: List of TrainingSample objects
        output_path: Path to save JSON file
    """
    data = []
    for sample in samples:
        # Convert PIL Image to array for serialization
        img_array = np.array(sample.image.convert('L')).tolist()
        
        data.append({
            'prefix': sample.prefix,
            'suffix': sample.suffix,
            'task_type': sample.task_type,
            'structure': sample.structure,
            'image_array': img_array
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(samples)} training samples to {output_path}")