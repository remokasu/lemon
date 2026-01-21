from lemon.nnlib import Module
import lemon.numlib as nm


def elu(x, alpha=1.0):
    """
    ELU (Exponential Linear Unit) activation function

    ELU(x) = x if x > 0 else alpha * (exp(x) - 1)

    ONNX operator: Elu
    Reference: https://onnx.ai/onnx/operators/onnx__Elu.html

    Parameters
    ----------
    x : Tensor
        Input tensor
    alpha : float, optional
        Alpha parameter (default: 1.0)

    Returns
    -------
    Tensor
        Output tensor
    """
    return nm.where(x > 0, x, alpha * (nm.exp(x) - 1))


class Elu(Module):
    """
    Elu activation module (ONNX-compliant)

    ONNX operator: Elu
    Reference: https://onnx.ai/onnx/operators/onnx__Elu.html

    Parameters
    ----------
    alpha : float, optional
        Alpha parameter (default: 1.0)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Elu(alpha=1.0),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return elu(x, alpha=self.alpha)

    def __repr__(self):
        return f"Elu(alpha={self.alpha})"
