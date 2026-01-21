"""
Iris dataset

The classic Iris flower dataset for classification.
"""

import os
import urllib.request
import lemon.numlib as nm
from lemon.datasets.vision._base import _DownloadableDataset


class Iris(_DownloadableDataset):
    """
    Iris dataset loader

    Classic dataset for classification with 4 features and 3 classes.

    Parameters
    ----------
    root : str
        Root directory where dataset exists or will be downloaded (default: './data')
    download : bool, optional
        If True, downloads the dataset if it doesn't exist (default: False)
    transform : callable, optional
        A function/transform to apply to the data (default: None)

    Examples
    --------
    >>> from lemon.datasets.tabular import Iris
    >>> dataset = Iris(root='./data', download=True)
    >>> X, y = dataset[0]
    >>> print(X.shape)  # (4,)
    >>> print(y)  # 0, 1, or 2 (setosa, versicolor, virginica)

    Notes
    -----
    Features: sepal length, sepal width, petal length, petal width
    Classes: 0=setosa, 1=versicolor, 2=virginica
    Total samples: 150 (50 per class)
    """

    @property
    def urls(self):
        """Iris dataset download URL"""
        return [
            {
                "iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            }
        ]

    @property
    def filenames(self):
        """Required filenames for Iris dataset"""
        return ["iris.data"]

    def __init__(self, root="./data", download=False, transform=None):
        self.root = os.path.join(root, "iris")
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        # Load data
        self.data, self.targets = self._load_data()

    def _check_exists(self) -> bool:
        """Check if dataset file exists"""
        return os.path.exists(os.path.join(self.root, "iris.data"))

    def download(self):
        """Download Iris dataset"""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        filepath = os.path.join(self.root, "iris.data")

        print("Downloading Iris dataset...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(filepath, "wb") as f:
                    f.write(response.read())
            print("Successfully downloaded iris.data")
        except Exception as e:
            print(f"Error downloading iris.data: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            raise RuntimeError(f"Failed to download Iris dataset: {e}")

        print("Download complete!")

    def _load_data(self):
        """Load Iris data from CSV file"""
        xp = nm.get_array_module(nm.zeros(1)._data)

        filepath = os.path.join(self.root, "iris.data")

        data_list = []
        targets_list = []
        class_mapping = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
        }

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split(",")
                if len(parts) == 5:
                    features = [float(x) for x in parts[:4]]
                    label = class_mapping[parts[4]]
                    data_list.append(features)
                    targets_list.append(label)

        data = xp.array(data_list, dtype=xp.float32)
        targets = xp.array(targets_list, dtype=xp.int64)

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y
