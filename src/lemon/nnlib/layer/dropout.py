import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.train_control import train


def dropout(x, p=0.5, training=True):
    """
    Dropout (functional API)

    Randomly zeros some elements of the input tensor with probability p
    during training. Uses inverted dropout for scaling.

    Parameters
    ----------
    x : Tensor
        Input tensor
    p : float
        Dropout probability (default: 0.5)
    training : bool
        Training mode (default: True)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.randn(10, 20)
    >>> y = dropout(x, p=0.5, training=True)
    """
    return nm.random_mask(x, p=p, training=training)


class Dropout(Module):
    """
    Dropout layer

    Randomly zeros some elements with probability p during training.

    Parameters
    ----------
    p : float
        Dropout probability (default: 0.5)

    Examples
    --------
    >>> model = Sequential(
    ...     Linear(784, 128),
    ...     ReLU(),
    ...     Dropout(0.5),
    ...     Linear(128, 10)
    ... )
    >>>
    >>> # Training mode
    >>> with train.on:
    ...     y = model(x)  # Dropout is active
    >>>
    >>> # Evaluation mode
    >>> with train.off:
    ...     y = model(x)  # Dropout is inactive
    """

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        return dropout(x, p=self.p, training=train.is_on())

    def __repr__(self):
        return f"Dropout(p={self.p})"
