"""
Dataset utilities

Utility functions for working with datasets.
"""

from typing import List, Optional
import lemon.numlib as nm
from lemon.nnlib.data.dataset import Dataset
from lemon.nnlib.data.subset import Subset


def random_split(dataset: Dataset, lengths: List[int], seed: Optional[int] = None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths

    Parameters
    ----------
    dataset : Dataset
        Dataset to be split
    lengths : List[int]
        Lengths of splits to be produced
    seed : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    List[Dataset]
        List of Subset datasets

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST
    >>> train_dataset = MNIST(root="./data", train=True)
    >>> train_set, val_set = random_split(train_dataset, [50000, 10000])
    >>> print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    Train: 50000, Val: 10000

    Notes
    -----
    The sum of lengths should equal the length of the dataset
    """
    # Validate lengths
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Sum of lengths ({sum(lengths)}) does not equal dataset length ({len(dataset)})"
        )

    # Get array module (numpy or cupy)
    xp = nm.get_array_module(nm.zeros(1)._data)

    # Generate random permutation
    if seed is not None:
        rng = xp.random.RandomState(seed)
        indices = rng.permutation(len(dataset))
    else:
        indices = xp.random.permutation(len(dataset))

    # Split indices
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset : offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length

    return subsets


def random_sample(dataset: Dataset, n: int, seed: Optional[int] = None):
    """
    Randomly sample n items from a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to sample from
    n : int
        Number of samples to select
    seed : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    Dataset
        Subset dataset with n randomly selected samples

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST
    >>> train_dataset = MNIST(root="./data", train=True)
    >>> small_dataset = random_sample(train_dataset, n=1000, seed=42)
    >>> print(len(small_dataset))
    1000

    Notes
    -----
    If n > len(dataset), all samples are returned
    """
    dataset_size = len(dataset)

    # If n is larger than dataset size, return full dataset
    if n >= dataset_size:
        return dataset

    # Get array module (numpy or cupy)
    xp = nm.get_array_module(nm.zeros(1)._data)

    # Generate random indices
    if seed is not None:
        rng = xp.random.RandomState(seed)
        indices = rng.choice(dataset_size, size=n, replace=False)
    else:
        indices = xp.random.choice(dataset_size, size=n, replace=False)

    return Subset(dataset, indices)


def split_dataset(
    dataset: Dataset,
    ratios: Optional[List[float]] = None,
    sizes: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """
    Split dataset into multiple subsets by ratios or sizes

    More convenient wrapper around random_split that supports both
    ratio-based and size-based splitting.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split
    ratios : List[float], optional
        List of ratios that sum to 1.0 (e.g., [0.7, 0.15, 0.15])
        Either ratios or sizes must be provided, not both
    sizes : List[int], optional
        List of exact sizes for each split
        Either ratios or sizes must be provided, not both
    seed : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    List[Dataset]
        List of Subset datasets

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST
    >>> dataset = MNIST(root="./data", train=True)
    >>>
    >>> # Split by ratios (70% train, 15% val, 15% test)
    >>> train, val, test = split_dataset(dataset, ratios=[0.7, 0.15, 0.15])
    >>>
    >>> # Split by exact sizes
    >>> train, val, test = split_dataset(dataset, sizes=[50000, 5000, 5000])
    >>>
    >>> # Simple train/test split
    >>> train, test = split_dataset(dataset, ratios=[0.8, 0.2], seed=42)

    Notes
    -----
    When using ratios, the actual sizes are computed and may be off by ±1
    due to rounding. The last split gets any remaining samples.
    """
    if ratios is None and sizes is None:
        raise ValueError("Either ratios or sizes must be provided")

    if ratios is not None and sizes is not None:
        raise ValueError("Cannot specify both ratios and sizes")

    dataset_size = len(dataset)

    if ratios is not None:
        # Validate ratios
        if not all(0 < r < 1 for r in ratios):
            raise ValueError("All ratios must be between 0 and 1")

        ratio_sum = sum(ratios)
        if not (0.99 <= ratio_sum <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

        # Convert ratios to sizes
        sizes = []
        remaining = dataset_size
        for i, ratio in enumerate(ratios[:-1]):
            size = int(dataset_size * ratio)
            sizes.append(size)
            remaining -= size
        # Last split gets remaining samples (handles rounding)
        sizes.append(remaining)

    else:  # sizes is not None
        # Validate sizes
        if sum(sizes) != dataset_size:
            raise ValueError(
                f"Sum of sizes ({sum(sizes)}) must equal dataset size ({dataset_size})"
            )

    # Use existing random_split function
    return random_split(dataset, sizes, seed=seed)
