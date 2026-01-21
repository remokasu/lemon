"""
Binary accuracy metric for binary classification
"""

import lemon.numlib as nm
from .metric import Metric


class BinaryAccuracy(Metric):
    """
    Accuracy metric for binary classification with sigmoid output

    Uses 0.5 threshold to convert probabilities to binary predictions.
    Returns accuracy as a percentage (0-100).

    Examples
    --------
    >>> metric = BinaryAccuracy()
    >>> y_pred = nm.tensor([[0.1], [0.9], [0.8], [0.2]])  # Sigmoid outputs
    >>> y_true = nm.tensor([0, 1, 1, 0])
    >>> acc = metric(y_pred, y_true)
    >>> print(acc)  # 100.0
    """

    def __call__(self, y_pred, y_true):
        # Threshold at 0.5 and convert to binary (0 or 1)
        pred_labels = (y_pred >= 0.5).astype(int).reshape(-1)
        # Flatten y_true to avoid broadcasting issues
        y_true_flat = y_true.reshape(-1) if len(y_true.shape) > 1 else y_true
        # Cast y_true to int for comparison
        y_true_int = y_true_flat.astype(int)
        correct = nm.sum(pred_labels == y_true_int)
        return float(correct.item() / len(y_true_flat)) * 100.0

    def name(self):
        return "accuracy"

    def format(self, value):
        """Format accuracy as percentage"""
        return f"{self.name()}={value:5.2f}%"
