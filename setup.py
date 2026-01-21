from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lemon",
    version="0.1.0",
    author="remokasu",
    description="PyTorch-like numerical computation/tensor/autograd library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/remokasu/numlib",
    packages=find_packages(include=["lemon", "lemon.*", "src", "src.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gpu": ["cupy>=9.0.0"],
        "plot": ["matplotlib>=3.0.0", "scikit-learn>=0.24.0"],
        "onnx": ["onnx>=1.10.0", "onnxruntime>=1.10.0"],
        "all": [
            "cupy>=9.0.0",
            "matplotlib>=3.0.0",
            "scikit-learn>=0.24.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.10.0",
        ],
    },
)
