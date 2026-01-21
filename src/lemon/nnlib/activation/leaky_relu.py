from lemon.nnlib import Module
import lemon.numlib as nm


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function

    LeakyReLU(x) = max(0, x) + alpha * min(0, x)
                 = x if x > 0 else alpha * x

    ONNX operator: LeakyRelu
    Reference: https://onnx.ai/onnx/operators/onnx__LeakyRelu.html

    Parameters
    ----------
    x : Tensor
        Input tensor
    alpha : float, optional
        Slope for negative values (default: 0.01)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> y = nl.leaky_relu(x, alpha=0.01)
    >>> # y = [-0.01, 0.0, 1.0]

    Notes
    -----
    Addresses dying ReLU problem by allowing small negative values.
    """
    return nm.maximum(alpha * x, x)


class LeakyRelu(Module):
    """
    LeakyRelu activation module (ONNX-compliant)

    ONNX operator: LeakyRelu
    Reference: https://onnx.ai/onnx/operators/onnx__LeakyRelu.html

    Parameters
    ----------
    alpha : float, optional
        Slope for negative values (default: 0.01)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.LeakyRelu(alpha=0.01),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return leaky_relu(x, alpha=self.alpha)

    def __repr__(self):
        return f"LeakyRelu(alpha={self.alpha})"
