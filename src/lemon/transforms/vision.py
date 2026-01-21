"""
Vision transforms

Transforms for image data preprocessing and augmentation.
"""

import lemon.numlib as nm


class Normalize:
    """
    Normalize data with mean and standard deviation

    Parameters
    ----------
    mean : float or array_like
        Mean for normalization
    std : float or array_like
        Standard deviation for normalization

    Examples
    --------
    >>> from lemon.transforms.vision import Normalize
    >>> from lemon.datasets.vision import MNIST
    >>> transform = Normalize(mean=0.5, std=0.5)
    >>> dataset = MNIST(root='./data', train=True, transform=transform)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        xp = nm.get_array_module(x if not hasattr(x, "_data") else x._data)
        mean = xp.array(self.mean, dtype=xp.float32)
        std = xp.array(self.std, dtype=xp.float32)
        return (x - mean) / std

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ToTensor:
    """
    Convert numpy array or NumType to Tensor

    Examples
    --------
    >>> from lemon.transforms.vision import ToTensor
    >>> from lemon.datasets.vision import MNIST
    >>> transform = ToTensor()
    >>> dataset = MNIST(root='./data', train=True, transform=transform)
    """

    def __call__(self, x):
        if hasattr(x, "_data"):
            return x
        return nm.tensor(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomNoise:
    """
    Add random Gaussian noise to data

    Parameters
    ----------
    std : float
        Standard deviation of the noise

    Examples
    --------
    >>> from lemon.transforms.vision import RandomNoise
    >>> from lemon.datasets.vision import MNIST
    >>> transform = RandomNoise(std=0.01)
    >>> dataset = MNIST(root='./data', train=True, transform=transform)
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        xp = nm.get_array_module(x if not hasattr(x, "_data") else x._data)
        noise = xp.random.normal(0, self.std, x.shape).astype(xp.float32)
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with probability p

    Parameters
    ----------
    p : float
        Probability of flipping (default: 0.5)

    Examples
    --------
    >>> from lemon.transforms.vision import RandomHorizontalFlip
    >>> transform = RandomHorizontalFlip(p=0.5)
    >>> # For images shaped as (height, width) or (channels, height, width)

    Notes
    -----
    Assumes images are in shape (H, W) or (C, H, W)
    For flattened images, reshape first
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        xp = nm.get_array_module(x if not hasattr(x, "_data") else x._data)

        if xp.random.random() < self.p:
            # Get data array
            data = x._data if hasattr(x, "_data") else x

            # Flip along the last axis (width)
            flipped = xp.flip(data, axis=-1)

            # Return in same format as input
            if hasattr(x, "_data"):
                return nm.tensor(flipped)
            return flipped
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
