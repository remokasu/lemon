import lemon.numlib as nm
from lemon.nnlib.module import Module


# Binary Cross Entropy Loss
def binary_cross_entropy(y_pred, y_true, reduction="mean"):
    """
    Binary Cross Entropy loss

    BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Parameters
    ----------
    y_pred : Tensor
        Predicted probabilities (after sigmoid), values in [0, 1]
    y_true : Tensor
        Target values (0 or 1)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Returns
    -------
    Tensor
        Loss value
    """
    eps = 1e-10
    y_pred_clamped = y_pred
    log_pred = nm.log(y_pred_clamped + eps)
    log_one_minus_pred = nm.log(1 - y_pred_clamped + eps)

    loss = -(y_true * log_pred + (1 - y_true) * log_one_minus_pred)

    if reduction == "mean":
        return nm.mean(loss)
    elif reduction == "sum":
        return nm.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'"
        )


# Binary Cross Entropy Loss Module
class BCELoss(Module):
    """
    Binary Cross Entropy loss module

    Parameters
    ----------
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = BCELoss()
    >>> loss = criterion(y_pred, y_true)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return binary_cross_entropy(y_pred, y_true, reduction=self.reduction)

    def __repr__(self):
        return f"BCELoss(reduction='{self.reduction}')"
