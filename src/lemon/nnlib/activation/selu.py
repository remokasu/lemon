from lemon.nnlib import Module
import lemon.numlib as nm


def selu(x, alpha=1.67326324, gamma=1.05070098):
    """
    SELU (Scaled Exponential Linear Unit) activation function

    SELU(x) = gamma * (x if x > 0 else alpha * (exp(x) - 1))

    ONNX operator: Selu
    Reference: https://onnx.ai/onnx/operators/onnx__Selu.html

    Parameters
    ----------
    x : Tensor
        Input tensor
    alpha : float, optional
        SELU alpha parameter (default: 1.67326324)
    gamma : float, optional
        SELU gamma parameter (default: 1.05070098)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> y = selu(x)
    >>> # y ≈ [-1.111, 0.0, 1.051]

    Notes
    -----
    SELU enables self-normalizing properties in neural networks.
    Use with lecun_normal initialization for best results.
    """
    return gamma * nm.where(x > 0, x, alpha * (nm.exp(x) - 1))


class Selu(Module):
    """
    SELU activation module (ONNX-compliant)

    For direct usage, prefer selu(x).

    Parameters
    ----------
    alpha : float, optional
        SELU alpha parameter (default: 1.67326324)
    gamma : float, optional
        SELU gamma parameter (default: 1.05070098)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.Selu(),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=1.67326324, gamma=1.05070098):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x):
        return selu(x, alpha=self.alpha, gamma=self.gamma)

    def __repr__(self):
        return f"Selu(alpha={self.alpha}, gamma={self.gamma})"
