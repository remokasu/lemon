from lemon.nnlib import Module
import lemon.numlib as nm


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function

    ReLU(x) = max(0, x)

    ONNX operator: Relu
    Reference: https://onnx.ai/onnx/operators/onnx__Relu.html

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
    >>> y = relu(x)
    >>> # y = [0.0, 0.0, 1.0]
    """
    return nm.maximum(x, 0)


class Relu(Module):
    """
    Relu activation module (ONNX-compliant)

    This is a thin wrapper around the relu() function.
    For direct usage, prefer relu(x).

    ONNX operator: Relu
    Reference: https://onnx.ai/onnx/operators/onnx__Relu.html

    Examples
    --------
    >>> # In Sequential
    >>> model = Sequential(
    ...     Linear(10, 20),
    ...     Relu(),
    ...     Linear(20, 5)
    ... )
    >>>
    >>> # Direct functional usage (recommended)
    >>> y = relu(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)

    def __repr__(self):
        return "Relu()"
