from lemon.nnlib import Module
import lemon.numlib as nm


def softsign(x):
    """
    Softsign activation function

    Softsign(x) = x / (1 + |x|)

    ONNX operator: Softsign
    Reference: https://onnx.ai/onnx/operators/onnx__Softsign.html

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
    >>> y = softsign(x)
    >>> # y = [-0.5, 0.0, 0.5]
    """
    return x / (1 + nm.abs(x))


class Softsign(Module):
    """
    Softsign activation module (ONNX-compliant)

    For direct usage, prefer softsign(x).

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Softsign(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softsign(x)

    def __repr__(self):
        return "Softsign()"
