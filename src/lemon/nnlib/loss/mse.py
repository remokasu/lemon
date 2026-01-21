from lemon.nnlib.module import Module
import lemon.numlib as nm


# Mean Squared Error Loss
def mean_squared_error(y_pred, y_true, reduction="mean"):
    """
    Mean Squared Error loss

    MSE = mean((y_pred - y_true)^2)

    Parameters
    ----------
    y_pred : Tensor
        Predicted values
    y_true : Tensor
        Target values
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Returns
    -------
    Tensor
        Loss value

    Examples
    --------
    >>> y_pred = nm.tensor([2.5, 0.0, 2.0])
    >>> y_true = nm.tensor([3.0, -0.5, 2.0])
    >>> loss = mean_squared_error(y_pred, y_true)
    >>> # loss ≈ 0.167
    """
    diff = y_pred - y_true
    squared_diff = diff**2

    if reduction == "mean":
        return nm.mean(squared_diff)
    elif reduction == "sum":
        return nm.sum(squared_diff)
    elif reduction == "none":
        return squared_diff
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'"
        )


# Mean Squared Error Loss Module
class MSELoss(Module):
    """
    Mean Squared Error loss module

    Parameters
    ----------
    reduction : str, optional
        'mean', 'sum', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = MSELoss()
    >>> loss = criterion(y_pred, y_true)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return mean_squared_error(y_pred, y_true, reduction=self.reduction)

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"
