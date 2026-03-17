import lemon.numlib as nm
from lemon.nnlib.module import Module


def focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25, reduction="mean"):
    """
    Focal loss for binary classification

    Addresses class imbalance by down-weighting easy examples and
    focusing training on hard negatives.

        FL = -alpha * (1 - p_t)^gamma * log(p_t)

        where p_t = y_pred if y_true == 1 else 1 - y_pred

    Parameters
    ----------
    y_pred : Tensor
        Predicted probabilities (after sigmoid), values in [0, 1]
    y_true : Tensor
        Target values (0 or 1)
    gamma : float, optional
        Focusing parameter. Higher values down-weight easy examples more (default: 2.0)
    alpha : float, optional
        Weighting factor for positive class (default: 0.25)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Returns
    -------
    Tensor
        Loss value
    """
    eps = 1e-10
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    loss = -alpha_t * (1 - p_t) ** gamma * nm.log(p_t + eps)

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


class FocalLoss(Module):
    """
    Focal loss module for binary classification

    Designed for class-imbalanced datasets. Down-weights easy examples
    so the model focuses on hard, misclassified examples.

    Parameters
    ----------
    gamma : float, optional
        Focusing parameter (default: 2.0)
    alpha : float, optional
        Weighting factor for positive class (default: 0.25)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = FocalLoss(gamma=2.0, alpha=0.25)
    >>> loss = criterion(y_pred, y_true)
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return focal_loss(
            y_pred,
            y_true,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, reduction='{self.reduction}')"
