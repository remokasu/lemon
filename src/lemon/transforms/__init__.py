"""
transforms - Data transformation utilities

This module provides transforms for data preprocessing and augmentation.

Modules
-------
- vision: Image transforms (Normalize, ToTensor, RandomHorizontalFlip, RandomNoise)
- compose: Transform composition (Compose, Lambda)

Quick Access
------------
Import transforms directly:
    from lemon.transforms.vision import Normalize, ToTensor
    from lemon.transforms.compose import Compose, Lambda

Or import entire modules:
    from lemon.transforms import vision, compose
"""

from lemon.transforms import vision
from lemon.transforms import compose

# Convenience imports for common transforms
from lemon.transforms.compose import Compose, Lambda
from lemon.transforms.vision import Normalize, ToTensor, RandomNoise, RandomHorizontalFlip

__all__ = [
    "vision",
    "compose",
    "Compose",
    "Lambda",
    "Normalize",
    "ToTensor",
    "RandomNoise",
    "RandomHorizontalFlip",
]
