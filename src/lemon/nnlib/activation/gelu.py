from lemon.nnlib import Module
import lemon.numlib as nm
import numpy as np


def gelu(x):
    """
    GeLU (Gaussian Error Linear Unit) activation function

    GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    This is the tanh approximation used in BERT and GPT models.

    ONNX operator: Gelu
    Reference: https://onnx.ai/onnx/operators/onnx__Gelu.html

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
    >>> y = gelu(x)
    >>> # y ≈ [-0.158, 0.0, 0.841]

    Notes
    -----
    GELU is widely used in Transformer models (BERT, GPT, etc.).
    It provides smooth, non-monotonic activation with better gradient flow
    compared to ReLU.
    """
    return 0.5 * x * (1 + nm.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Gelu(Module):
    """
    Gelu activation module (ONNX-compliant)

    For direct usage, prefer gelu(x).

    ONNX operator: Gelu
    Reference: https://onnx.ai/onnx/operators/onnx__Gelu.html

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Gelu(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

    def __repr__(self):
        return "Gelu()"
