"""
Fashion-MNIST dataset

A dataset of Zalando's article images.
"""

import os
from lemon.datasets.vision._base import _MNISTLikeDataset


class FashionMNIST(_MNISTLikeDataset):
    """
    Fashion-MNIST dataset loader

    A dataset of Zalando's article images with 10 classes.
    Same format as MNIST (28x28 grayscale images).

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
    >>> from lemon.datasets.vision import FashionMNIST
    >>> train_dataset = FashionMNIST(root='./data', train=True, download=True)
    >>> test_dataset = FashionMNIST(root='./data', train=False)
    >>> X, y = train_dataset[0]
    >>> print(X.shape)  # (1, 28, 28) or (784,) if flatten=True
    >>> print(y)  # 0-9 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

    Notes
    -----
    Classes: 0=T-shirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
             5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
    """

    @property
    def urls(self):
        """Fashion-MNIST download URLs"""
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        return [
            {
                "train_images": base_url + "train-images-idx3-ubyte.gz",
                "train_labels": base_url + "train-labels-idx1-ubyte.gz",
                "test_images": base_url + "t10k-images-idx3-ubyte.gz",
                "test_labels": base_url + "t10k-labels-idx1-ubyte.gz",
            }
        ]

    @property
    def filenames(self):
        """Required filenames for Fashion-MNIST dataset"""
        return [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]

    def __init__(self, root="./data", train=True, download=False, transform=None, flatten=False):
        self.root = os.path.join(root, "fashion_mnist")
        self.train = train
        self.transform = transform
        self.flatten = flatten

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Fashion-MNIST dataset not found. You can use download=True to download it"
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
