from setuptools import setup, find_packages

setup(
    name="chexanatomy",
    version="0.1.0",
    description="Chest X-ray VLM Training Data Generator",
    author="Sergios Gatidis",
    package_dir={"chexanatomy.utils": "utils", 
                 "chexanatomy.paligemma_training_data": "paligemma_training_data",
                 "chexanatomy.training_file_generation": "training_file_generation",
                 "chexanatomy.training": "training"},
    packages=["chexanatomy.utils", 
              "chexanatomy.paligemma_training_data",
              "chexanatomy.training_file_generation",
              "chexanatomy.training"],
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "tqdm>=4.64.0",
        "PyYAML>=6.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.9",
)