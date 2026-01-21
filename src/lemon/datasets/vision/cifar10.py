"""
CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes.
"""

import os
import tarfile
import urllib.request
import pickle
import lemon.numlib as nm
from lemon.datasets.vision._base import _DownloadableDataset


class CIFAR10(_DownloadableDataset):
    """
    CIFAR-10 dataset loader

    32x32 color images in 10 classes, with 6000 images per class.

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
        If True, returns flattened arrays (3072,). If False, returns (3, 32, 32) (default: True)

    Examples
    --------
    >>> from lemon.datasets.vision import CIFAR10
    >>> train_dataset = CIFAR10(root='./data', train=True, download=True)
    >>> test_dataset = CIFAR10(root='./data', train=False)
    >>> X, y = train_dataset[0]
    >>> print(X.shape)  # (3072,) or (3, 32, 32) depending on flatten parameter
    >>> print(y)  # 0-9 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

    Notes
    -----
    Classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
             5=dog, 6=frog, 7=horse, 8=ship, 9=truck
    """

    @property
    def urls(self):
        """CIFAR-10 download URL"""
        return [
            {"cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"}
        ]

    @property
    def filenames(self):
        """Required filenames for CIFAR-10 dataset"""
        return ["cifar-10-python.tar.gz"]

    def __init__(
        self, root="./data", train=True, download=False, transform=None, flatten=False
    ):
        self.base_root = root
        self.root = os.path.join(root, "cifar-10-batches-py")
        self.train = train
        self.transform = transform
        self.flatten = flatten

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        # Load data
        self.data, self.targets = self._load_data()

    def _check_exists(self) -> bool:
        """Check if dataset files exist"""
        return os.path.exists(self.root) and os.path.isdir(self.root)

    def download(self):
        """Download and extract CIFAR-10 dataset"""
        if self._check_exists():
            return

        os.makedirs(self.base_root, exist_ok=True)

        filename = "cifar-10-python.tar.gz"
        filepath = os.path.join(self.base_root, filename)
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        if not os.path.exists(filepath):
            print(f"Downloading CIFAR-10...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    with open(filepath, "wb") as f:
                        f.write(response.read())
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise RuntimeError(f"Failed to download CIFAR-10: {e}")

        # Extract tar file
        print("Extracting files...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=self.base_root)
        print("Download complete!")

    def _load_data(self):
        """Load CIFAR-10 data from pickle files"""

        xp = nm.get_array_module(nm.zeros(1)._data)

        if self.train:
            # Load training batches
            data_list = []
            targets_list = []
            for i in range(1, 6):
                file_path = os.path.join(self.root, f"data_batch_{i}")
                with open(file_path, "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                    data_list.append(batch[b"data"])
                    targets_list.extend(batch[b"labels"])

            data = xp.concatenate(data_list, axis=0)
            targets = xp.array(targets_list, dtype=xp.int64)
        else:
            # Load test batch
            file_path = os.path.join(self.root, "test_batch")
            with open(file_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
                data = xp.array(batch[b"data"])
                targets = xp.array(batch[b"labels"], dtype=xp.int64)

        # Normalize to [0, 1]
        data = data.astype(xp.float32) / 255.0

        # Reshape if not flattening: (N, 3072) -> (N, 3, 32, 32)
        if not self.flatten:
            data = data.reshape(-1, 3, 32, 32)

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y
