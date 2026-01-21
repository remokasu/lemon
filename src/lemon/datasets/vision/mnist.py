"""
MNIST dataset

The MNIST database of handwritten digits.
"""

import os
from lemon.datasets.vision._base import _MNISTLikeDataset


class MNIST(_MNISTLikeDataset):
    """
    MNIST dataset loader

    Automatically downloads MNIST if not present.

    Parameters
    ----------
    root : str
        Root directory where dataset exists or will be downloaded (default: './data')
    train : bool, optional
        If True, creates dataset from training set, otherwise from test set (default: True)
    download : bool, optional
        If True, downloads the dataset if it doesn't exist (default: False)
    transform : callable, optional
        A function/transform to apply to the data (default: None)
    flatten : bool, optional
        If True, returns flattened images (784,). If False, returns (1, 28, 28) for CNN (default: False)

    Examples
    --------
    >>> from lemon.datasets.vision import MNIST
    >>> train_dataset = MNIST(root='./data', train=True, download=True)
    >>> test_dataset = MNIST(root='./data', train=False)
    >>> X, y = train_dataset[0]
    >>> print(X.shape)  # (1, 28, 28) or (784,) if flatten=True
    >>> print(y)  # 5 (for example)
    """

    @property
    def urls(self):
        """MNIST download URLs with mirror support"""
        return [
            # PyTorch mirror (most reliable)
            {
                "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
                "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            },
            # Original Yann LeCun site
            {
                "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            },
        ]

    @property
    def filenames(self):
        """Required filenames for MNIST dataset"""
        return [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]

    def __init__(self, root="./data", train=True, download=False, transform=None, flatten=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.flatten = flatten

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "MNIST dataset not found. You can use download=True to download it"
            )

        # Load data
        if self.train:
            data_file = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            label_file = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            data_file = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            label_file = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")

        self.data = self._load_images(data_file, flatten=self.flatten)
        self.targets = self._load_labels(label_file)
