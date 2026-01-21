from lemon.nnlib import Module
import lemon.numlib as nm


def tanh(x):
    """
    Tanh activation function

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Uses numlib.tanh which already supports autograd.

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> y = tanh(x)
    >>> # y ≈ [-0.762, 0.0, 0.762]
    """
    return nm.tanh(x)


class Tanh(Module):
    """
    Tanh activation module (wrapper for Sequential)

    For direct usage, prefer tanh(x).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh(x)

    def __repr__(self):
        return "Tanh()"
