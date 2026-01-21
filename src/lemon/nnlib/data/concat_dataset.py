"""
ConcatDataset

Dataset for concatenating multiple datasets into one.
"""

from typing import List
from lemon.nnlib.data.dataset import Dataset


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets

    This class is useful for combining multiple datasets into one.

    Parameters
    ----------
    datasets : List[Dataset]
        List of datasets to concatenate

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST, FashionMNIST
    >>> mnist = MNIST(root='./data', train=True)
    >>> fashion = FashionMNIST(root='./data', train=True)
    >>> combined = ConcatDataset([mnist, fashion])
    >>> print(len(combined))  # 120000 (60000 + 60000)
    >>> X, y = combined[0]  # First sample from mnist
    >>> X, y = combined[60000]  # First sample from fashion_mnist

    Notes
    -----
    All datasets should return samples in the same format (e.g., (X, y) pairs)
    """

    def __init__(self, datasets: List[Dataset]):
        if len(datasets) == 0:
            raise ValueError("datasets should not be empty")

        self.datasets = datasets
        self.cumulative_sizes = self._cumsum([len(d) for d in datasets])

    @staticmethod
    def _cumsum(sequence):
        """Calculate cumulative sum"""
        r, s = [], 0
        for e in sequence:
            r.append(s + e)
            s += e
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                dataset_idx = i
                break

        # Calculate index within that dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]
