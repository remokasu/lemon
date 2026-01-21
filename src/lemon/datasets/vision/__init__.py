"""
vision - Computer vision datasets

This module provides loaders for common computer vision datasets.

Available Datasets
------------------
- MNIST: Handwritten digits (28x28 grayscale)
- FashionMNIST: Fashion products (28x28 grayscale)
- CIFAR10: Natural images in 10 classes (32x32 RGB)
- CIFAR100: Natural images in 100 classes (32x32 RGB)
"""

from lemon.datasets.vision.mnist import MNIST
from lemon.datasets.vision.fashion_mnist import FashionMNIST
from lemon.datasets.vision.cifar10 import CIFAR10
from lemon.datasets.vision.cifar100 import CIFAR100

__all__ = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
]
