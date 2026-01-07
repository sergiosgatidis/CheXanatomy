from setuptools import setup, find_packages

setup(
    name="chexanatomy",
    version="0.2.0",
    description="Chest X-ray Vision-Language Model Training Pipeline",
    long_description="A comprehensive training data generation pipeline for chest X-ray analysis using Paligemma vision-language models, featuring anatomical structure detection, segmentation token generation, and advanced data augmentation.",
    author="Sergios Gatidis",
    url="https://github.com/sergiosgatidis/cheXanatomy",
    package_dir={
        "chexanatomy.utils": "utils", 
        "chexanatomy.paligemma_training_data": "paligemma_training_data",
        "chexanatomy.training_file_generation": "training_file_generation",
        "chexanatomy.training": "training"
    },
    packages=[
        "chexanatomy.utils", 
        "chexanatomy.paligemma_training_data",
        "chexanatomy.training_file_generation",
        "chexanatomy.training"
    ],
    install_requires=[
        # Core dependencies
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "tensorflow>=2.8.0",
        
        # Data processing & augmentation
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        
        # Visualization & analysis
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        
        # Training & ML
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        
        # Data handling & utilities
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "PyYAML>=6.0",
        
        # Image & file I/O
        "imageio>=2.15.0",
        "h5py>=3.6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "pytest>=6.2.0",
        ],
        "training": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="chest-xray, vision-language-model, paligemma, medical-imaging, deep-learning",
)