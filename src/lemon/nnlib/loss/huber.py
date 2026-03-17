import lemon.numlib as nm
from lemon.nnlib.module import Module


def huber_loss(y_pred, y_true, delta=1.0, reduction="mean"):
    """
    Huber loss (Smooth L1 loss)

    Combines MSE for small errors and MAE for large errors,
    making it robust to outliers.

        L = 0.5 * (y_pred - y_true)^2              if |error| <= delta
        L = delta * (|error| - 0.5 * delta)         if |error| >  delta

    Parameters
    ----------
    y_pred : Tensor
        Predicted values
    y_true : Tensor
        Target values
    delta : float, optional
        Threshold between MSE and MAE regions (default: 1.0)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Returns
    -------
    Tensor
        Loss value
    """
    error = y_pred - y_true
    abs_error = nm.abs(error)
    quadratic = 0.5 * error**2
    linear = delta * (abs_error - 0.5 * delta)
    loss = nm.where(abs_error <= delta, quadratic, linear)

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


class HuberLoss(Module):
    """
    Huber loss module (Smooth L1 loss)

    Robust regression loss that is less sensitive to outliers than MSELoss.
    Uses MSE for small errors and MAE for large errors.

    Parameters
    ----------
    delta : float, optional
        Threshold between MSE and MAE regions (default: 1.0)
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = HuberLoss(delta=1.0)
    >>> loss = criterion(y_pred, y_true)
    """

    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return huber_loss(y_pred, y_true, delta=self.delta, reduction=self.reduction)

    def __repr__(self):
        return f"HuberLoss(delta={self.delta}, reduction='{self.reduction}')"
