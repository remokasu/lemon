from lemon.nnlib import Module
import lemon.numlib as nm


def thresholded_relu(x, alpha=1.0):
    """
    ThresholdedRelu activation function

    ThresholdedRelu(x) = x if x > alpha else 0

    ONNX operator: ThresholdedRelu
    Reference: https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html

    Parameters
    ----------
    x : Tensor
        Input tensor
    alpha : float, optional
        Threshold value (default: 1.0)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([0.5, 1.0, 1.5])
    >>> y = thresholded_relu(x, alpha=1.0)
    >>> # y = [0.0, 0.0, 1.5]
    """
    return nm.where(x > alpha, x, 0.0)


class ThresholdedRelu(Module):
    """
    ThresholdedRelu activation module (ONNX-compliant)

    Parameters
    ----------
    alpha : float, optional
        Threshold value (default: 1.0)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.Linear(10, 20),
    ...     nl.ThresholdedRelu(alpha=1.0),
    ...     nl.Linear(20, 5)
    ... )
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return thresholded_relu(x, alpha=self.alpha)

    def __repr__(self):
        return f"ThresholdedRelu(alpha={self.alpha})"
