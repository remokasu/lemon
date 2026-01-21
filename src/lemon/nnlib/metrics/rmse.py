"""
Root Mean Squared Error metric
"""

import lemon.numlib as nm
from .metric import Metric


class RMSE(Metric):
    """
    Root Mean Squared Error metric for regression

    Calculates the square root of mean squared difference.

    Examples
    --------
    >>> metric = RMSE()
    >>> y_pred = nm.tensor([1.0, 2.0, 3.0])
    >>> y_true = nm.tensor([1.1, 2.2, 2.8])
    >>> rmse = metric(y_pred, y_true)
    >>> print(rmse)  # 0.1528
    """

    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = diff * diff
        mse = nm.mean(squared_diff)
        return float(nm.sqrt(mse).item())

    def name(self):
        return "rmse"
