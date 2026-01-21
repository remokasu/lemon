"""
Mean Absolute Error metric
"""

import lemon.numlib as nm
from .metric import Metric


class MAE(Metric):
    """
    Mean Absolute Error metric for regression

    Calculates the mean absolute difference between predictions and targets.

    Examples
    --------
    >>> metric = MAE()
    >>> y_pred = nm.tensor([1.0, 2.0, 3.0])
    >>> y_true = nm.tensor([1.1, 2.2, 2.8])
    >>> mae = metric(y_pred, y_true)
    >>> print(mae)  # 0.1667
    """

    def __call__(self, y_pred, y_true):
        diff = nm.abs(y_pred - y_true)
        return float(nm.mean(diff).item())

    def name(self):
        return "mae"
