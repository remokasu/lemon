from lemon.nnlib import Module
import lemon.numlib as nm
from lemon.nnlib.activation.softplus import softplus


def mish(x):
    """
    Mish activation function

    Mish(x) = x * tanh(softplus(x))
           = x * tanh(log(1 + exp(x)))

    ONNX operator: Mish
    Reference: https://onnx.ai/onnx/operators/onnx__Mish.html

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
    >>> y = mish(x)
    >>> # y ≈ [-0.303, 0.0, 0.865]

    Notes
    -----
    Mish is a smooth, non-monotonic activation function that often
    outperforms ReLU and Swish in deep networks.
    """
    return x * nm.tanh(softplus(x))


class Mish(Module):
    """
    Mish activation module (ONNX-compliant)

    For direct usage, prefer mish(x).

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Mish(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

    def __repr__(self):
        return "Mish()"
