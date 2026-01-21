from lemon.nnlib.module import Module
import lemon.numlib as nm


class Flatten(Module):
    """
    Flatten multi-dimensional data to 1D

    Examples
    --------
    >>> transform = nl.Flatten()
    >>> # For images: (28, 28) -> (784,)
    """

    def forward(self, x):  # __call__ではなくforward
        batch_size = x.shape[0]
        return nm.reshape(x, batch_size, -1)  # バッチ次元を保持

    def __repr__(self):
        return f"{self.__class__.__name__}()"
