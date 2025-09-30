from setuptools import setup, find_packages

setup(
    name="macaque_tracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python==4.8.1.78",
        "ultralytics==8.0.196",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "scikit-learn==1.3.0",
        "scipy==1.11.1",
        "jupyter==1.0.0",
        "ipywidgets==8.0.7",
        "tqdm==4.65.0",
        "Pillow==10.0.0",
        "moviepy==1.0.3",
        "deep-sort-realtime==1.3.2",
        "facenet-pytorch==2.5.3",
        "umap-learn==0.5.4",
        "hdbscan==0.8.33",
        "plotly==5.15.0",
        "ipykernel==6.25.0"
    ],
    description="Macaque individual tracking from camera trap videos",
    author="Primate Biologist",
    python_requires=">=3.8",
)