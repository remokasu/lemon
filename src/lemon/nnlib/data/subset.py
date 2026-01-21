"""
Subset

Dataset representing a subset of a larger dataset at specified indices.
"""

import lemon.numlib as nm
from lemon.nnlib.data.dataset import Dataset


class Subset(Dataset):
    """
    Subset of a dataset at specified indices

    Parameters
    ----------
    dataset : Dataset
        The whole dataset
    indices : array_like
        Indices in the whole dataset selected for subset

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST
    >>> dataset = MNIST(root="./data", train=True)
    >>> subset = Subset(dataset, [0, 1, 2, 3, 4])
    >>> print(len(subset))
    5
    """

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        # Convert to list for consistent indexing
        xp = nm.get_array_module(nm.zeros(1)._data)
        if isinstance(indices, (xp.ndarray if hasattr(xp, "ndarray") else list)):
            self.indices = [int(i) for i in indices]
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
