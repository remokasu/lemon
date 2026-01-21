"""
Mean Squared Error metric
"""

import lemon.numlib as nm
from .metric import Metric


class MSE(Metric):
    """
    Mean Squared Error metric for regression

    Calculates the mean squared difference between predictions and targets.

    Examples
    --------
    >>> metric = MSE()
    >>> y_pred = nm.tensor([1.0, 2.0, 3.0])
    >>> y_true = nm.tensor([1.1, 2.2, 2.8])
    >>> mse = metric(y_pred, y_true)
    >>> print(mse)  # 0.0233
    """

    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = diff * diff
        return float(nm.mean(squared_diff).item())

    def name(self):
        return "mse"
