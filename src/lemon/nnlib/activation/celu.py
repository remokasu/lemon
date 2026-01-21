from lemon.nnlib import Module
import lemon.numlib as nm


def celu(x, alpha=1.0):
    """
    CELU (Continuously differentiable ELU) activation function

    CELU(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))

    ONNX operator: Celu
    Reference: https://onnx.ai/onnx/operators/onnx__Celu.html

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

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> y = celu(x)
    """
    return nm.maximum(0.0, x) + nm.minimum(0.0, alpha * (nm.exp(x / alpha) - 1))


class Celu(Module):
    """
    CELU activation module (ONNX-compliant)

    Parameters
    ----------
    alpha : float, optional
        Alpha parameter (default: 1.0)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Celu(alpha=1.0),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return celu(x, alpha=self.alpha)

    def __repr__(self):
        return f"Celu(alpha={self.alpha})"
