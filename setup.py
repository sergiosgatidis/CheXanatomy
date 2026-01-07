from setuptools import setup, find_packages

setup(
    name="chexanatomy",
    version="0.1.0",
    description="Chest X-ray VLM Training Data Generator",
    author="Sergios Gatidis",
    package_dir={"chexanatomy": "src"},
    packages=["chexanatomy"],
    install_requires=[
        "tensorflow>=2.8.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
    ],
    python_requires=">=3.9",
)