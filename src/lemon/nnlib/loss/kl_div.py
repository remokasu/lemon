import lemon.numlib as nm
from lemon.nnlib.module import Module


def kl_div_loss(y_pred, y_true, reduction="mean"):
    """
    KL Divergence loss

    Measures how one probability distribution diverges from another.

        KL(y_true || y_pred) = y_true * (log(y_true) - log(y_pred))

    Note: y_pred should be log-probabilities (output of log_softmax).
          y_true should be probabilities (not log).

    Parameters
    ----------
    y_pred : Tensor
        Log-probabilities (e.g. output of log_softmax)
    y_true : Tensor
        Target probabilities (not log)
    reduction : str, optional
        'mean', 'sum', or 'batchmean' (default: 'mean')
        'batchmean' divides by batch size (PyTorch convention)

    Returns
    -------
    Tensor
        Loss value
    """
    eps = 1e-10
    loss = y_true * (nm.log(y_true + eps) - y_pred)

    if reduction == "mean":
        return nm.mean(loss)
    elif reduction == "sum":
        return nm.sum(loss)
    elif reduction == "batchmean":
        batch_size = y_pred.shape[0]
        return nm.sum(loss) / batch_size
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', 'batchmean', or 'none'"
        )


class KLDivLoss(Module):
    """
    KL Divergence loss module

    Useful for VAEs, knowledge distillation, and any task involving
    probability distribution matching.

    Parameters
    ----------
    reduction : str, optional
        'mean', 'sum', 'batchmean', or 'none' (default: 'mean')

    Examples
    --------
    >>> criterion = KLDivLoss()
    >>> log_probs = nm.log_softmax(y_pred, axis=-1)
    >>> loss = criterion(log_probs, y_true_probs)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return kl_div_loss(y_pred, y_true, reduction=self.reduction)

    def __repr__(self):
        return f"KLDivLoss(reduction='{self.reduction}')"
