"""
Accuracy metric for multiclass classification
"""

import lemon.numlib as nm
from .metric import Metric


class Accuracy(Metric):
    """
    Accuracy metric for multiclass classification

    Uses argmax to find predicted class and compares with true labels.
    Returns accuracy as a percentage (0-100).

    Examples
    --------
    >>> metric = Accuracy()
    >>> y_pred = nm.tensor([[0.1, 0.9], [0.8, 0.2]])
    >>> y_true = nm.tensor([1, 0])
    >>> acc = metric(y_pred, y_true)
    >>> print(acc)  # 100.0
    """

    def __call__(self, y_pred, y_true):
        pred_labels = y_pred.argmax(axis=-1)
        # Flatten y_true to avoid broadcasting issues
        y_true_flat = y_true.reshape(-1) if len(y_true.shape) > 1 else y_true
        correct = nm.sum(pred_labels == y_true_flat)
        return float(correct.item() / len(y_true_flat)) * 100.0

    def name(self):
        return "accuracy"

    def format(self, value):
        """Format accuracy as percentage"""
        return f"{self.name()}={value:5.2f}%"
