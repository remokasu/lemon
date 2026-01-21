"""
data - Core data abstractions for neural networks

This module provides fundamental data handling components for neural network training.

Key Features
------------
- Dataset base classes (Dataset, SupervisedDataSet)
- DataLoader for batching and iteration
- Dataset utilities (TensorDataset, ConcatDataset, Subset, AutoencoderDataset)
- Random splitting functionality
- Sampler base class (for future extensions)
"""

from lemon.nnlib.data.dataset import Dataset, SupervisedDataSet
from lemon.nnlib.data.dataloader import DataLoader
from lemon.nnlib.data.tensor_dataset import TensorDataset
from lemon.nnlib.data.concat_dataset import ConcatDataset
from lemon.nnlib.data.subset import Subset
from lemon.nnlib.data.autoencoder_dataset import AutoencoderDataset
from lemon.nnlib.data.utils import random_split, random_sample, split_dataset
from lemon.nnlib.data.sampler import Sampler

__all__ = [
    "Dataset",
    "SupervisedDataSet",
    "DataLoader",
    "TensorDataset",
    "ConcatDataset",
    "Subset",
    "AutoencoderDataset",
    "random_split",
    "random_sample",
    "split_dataset",
    "Sampler",
]
