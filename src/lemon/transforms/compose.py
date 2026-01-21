"""
Transform composition utilities

Provides utilities for composing multiple transforms and applying custom functions.
"""


class Compose:
    """
    Compose multiple transforms together

    Parameters
    ----------
    transforms : List[callable]
        List of transform functions to compose

    Examples
    --------
    >>> from lemon.transforms import Compose
    >>> from lemon.transforms.vision import Normalize
    >>> transform = Compose([
    ...     Normalize(mean=0.5, std=0.5),
    ... ])
    >>> from lemon.datasets.vision import MNIST
    >>> dataset = MNIST(root='./data', train=True, transform=transform)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Lambda:
    """
    Apply a custom lambda function as a transform

    Parameters
    ----------
    lambd : callable
        Lambda function to apply

    Examples
    --------
    >>> from lemon.transforms import Lambda
    >>> transform = Lambda(lambda x: x * 2)
    >>> transform = Lambda(lambda x: x.reshape(28, 28))
    """

    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, x):
        return self.lambd(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
