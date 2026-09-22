"""
Top-K accuracy metric for multiclass classification
"""

import lemon.numlib as nm
from .metric import Metric


class TopKAccuracy(Metric):
    """
    Top-K Accuracy metric for multiclass classification

    Checks if true label is in top K predictions.
    Returns accuracy as a percentage (0-100).

    Parameters
    ----------
    k : int
        Number of top predictions to consider (default: 5)

    Examples
    --------
    >>> metric = TopKAccuracy(k=2)
    >>> y_pred = nm.tensor([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
    >>> y_true = nm.tensor([1, 0])
    >>> acc = metric(y_pred, y_true)
    >>> print(acc)  # 100.0
    """

    def __init__(self, k=5):
        self.k = k

    def __call__(self, y_pred, y_true):
        # 評価は計算グラフの外の処理なので、生の配列で計算する
        pred = y_pred._data if isinstance(y_pred, nm.NumType) else y_pred
        xp = nm.get_array_module(pred)
        pred = xp.asarray(pred)
        true = y_true._data if isinstance(y_true, nm.NumType) else y_true
        true = xp.asarray(true).reshape(-1).astype(int)

        # Top-k indices along the class axis
        top_k_indices = xp.argsort(pred, axis=-1)[:, -self.k :]

        # Check if true label is in top k
        correct = (top_k_indices == true[:, None]).any(axis=1)
        return float(correct.mean()) * 100.0

    def name(self):
        return f"top{self.k}_accuracy"

    def format(self, value):
        """Format top-k accuracy as percentage"""
        return f"{self.name()}={value:5.2f}%"
