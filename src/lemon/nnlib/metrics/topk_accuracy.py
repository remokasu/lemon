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
        # Get top k predictions
        batch_size = y_pred.shape[0]
        # Simple approach: sort and take top k indices
        top_k_indices = nm.argsort(y_pred, axis=-1)[:, -self.k :]

        # Flatten y_true
        y_true_flat = y_true.reshape(-1) if len(y_true.shape) > 1 else y_true

        # Check if true label is in top k
        correct = 0
        for i in range(batch_size):
            if int(y_true_flat[i].item()) in [
                int(idx.item()) for idx in top_k_indices[i]
            ]:
                correct += 1

        return float(correct / batch_size) * 100.0

    def name(self):
        return f"top{self.k}_accuracy"

    def format(self, value):
        """Format top-k accuracy as percentage"""
        return f"{self.name()}={value:5.2f}%"
