from lemon.nnlib.module import Module
import lemon.numlib as nm
from lemon.nnlib.activation import softmax


# Cross Entropy Loss with Softmax
def softmax_cross_entropy(y_pred, y_true, reduction="mean"):
    """
    Cross Entropy loss (with softmax)

    CE = -mean(y_true * log(softmax(y_pred)))

    Parameters
    ----------
    y_pred : Tensor
        Raw logits (before softmax), shape (batch_size, num_classes)
    y_true : Tensor
        Target class indices (shape: batch_size) or one-hot encoded (shape: batch_size, num_classes)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Returns
    -------
    Tensor
        Loss value
    """
    # Apply softmax
    probs = softmax(y_pred, axis=-1)

    if y_true._data.ndim == 1:
        # Class indices: convert to one-hot
        num_classes = y_pred.shape[-1]
        y_true_onehot = nm.one_hot(y_true, num_classes=num_classes)
    else:
        # Already one-hot encoded
        y_true_onehot = y_true

    # Cross entropy: -sum(y_true * log(probs))
    eps = 1e-10

    log_probs = nm.log(probs + eps)
    loss_per_sample = -nm.sum(y_true_onehot * log_probs, axis=-1)

    if reduction == "mean":
        return nm.mean(loss_per_sample)
    elif reduction == "sum":
        return nm.sum(loss_per_sample)
    elif reduction == "none":
        return loss_per_sample
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'"
        )


# Cross Entropy Loss Module
class CrossEntropyLoss(Module):
    """
    Cross Entropy loss module

    Parameters
    ----------
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = CrossEntropyLoss()
    >>> loss = criterion(y_pred, y_true)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return softmax_cross_entropy(y_pred, y_true, reduction=self.reduction)

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
