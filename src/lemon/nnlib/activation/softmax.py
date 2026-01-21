from lemon.nnlib import Module
import lemon.numlib as nm


def softmax(x, axis=-1):
    """
    Softmax activation function

    softmax(x_i) = exp(x_i) / Σ exp(x_j)

    Parameters
    ----------
    x : Tensor
        Input tensor
    axis : int, optional
        Axis to apply softmax (default: -1)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([[1.0, 2.0, 3.0]])
    >>> y = softmax(x)
    >>> # y ≈ [[0.09, 0.24, 0.67]]
    >>> nm.sum(y, axis=-1)  # Sum is 1
    """
    x_max = nm.amax(x, axis=axis, keepdims=True)
    exp_x = nm.exp(x - x_max)
    return exp_x / nm.sum(exp_x, axis=axis, keepdims=True)


class Softmax(Module):
    """
    Softmax activation module (wrapper for Sequential)

    For direct usage, prefer softmax(x, axis=-1).

    Parameters
    ----------
    axis : int, optional
        Axis to apply softmax (default: -1)
    """

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return softmax(x, axis=self.axis)

    def __repr__(self):
        return f"Softmax(axis={self.axis})"
