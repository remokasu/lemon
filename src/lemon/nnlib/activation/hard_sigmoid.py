from lemon.nnlib import Module
import lemon.numlib as nm


def hard_sigmoid(x, alpha=0.2, beta=0.5):
    """
    HardSigmoid activation function

    HardSigmoid(x) = max(0, min(1, alpha * x + beta))

    ONNX operator: HardSigmoid
    Reference: https://onnx.ai/onnx/operators/onnx__HardSigmoid.html

    Parameters
    ----------
    x : Tensor
        Input tensor
    alpha : float, optional
        Slope parameter (default: 0.2)
    beta : float, optional
        Bias parameter (default: 0.5)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-3.0, 0.0, 3.0])
    >>> y = hard_sigmoid(x)
    >>> # y = [0.0, 0.5, 1.0]
    """
    return nm.maximum(0.0, nm.minimum(1.0, alpha * x + beta))


class HardSigmoid(Module):
    """
    HardSigmoid activation module (ONNX-compliant)

    Parameters
    ----------
    alpha : float, optional
        Slope parameter (default: 0.2)
    beta : float, optional
        Bias parameter (default: 0.5)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.HardSigmoid(alpha=0.2, beta=0.5),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return hard_sigmoid(x, alpha=self.alpha, beta=self.beta)

    def __repr__(self):
        return f"HardSigmoid(alpha={self.alpha}, beta={self.beta})"
