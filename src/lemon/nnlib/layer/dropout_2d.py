import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.train_control import train


def dropout_2d(x, p=0.5, training=True):
    """
    Dropout2d (functional API with autograd support)

    Randomly zeros entire channels of the input tensor with probability p
    during training. This is commonly used in CNNs.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)
    p : float
        Dropout probability (default: 0.5)
    training : bool
        Training mode (default: True)

    Returns
    -------
    Tensor
        Output tensor with same shape as input

    Examples
    --------
    >>> x = nm.randn(4, 16, 28, 28, requires_grad=True)
    >>> y = dropout_2d(x, p=0.5, training=True)
    >>> y.shape
    (4, 16, 28, 28)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    Unlike regular dropout which zeros individual elements,
    dropout_2d zeros entire channels. This is more effective for
    convolutional layers where adjacent pixels are strongly correlated.
    """
    return nm.random_mask_channel(x, p=p, training=training)


class Dropout2d(Module):
    """
    Dropout2d layer

    Randomly zeros entire channels with probability p during training.
    This is specifically designed for convolutional layers.

    Parameters
    ----------
    p : float
        Dropout probability (default: 0.5)

    Examples
    --------
    >>> import numlib as nm
    >>> import lemon as lm
    >>> model = nl.Sequential(
    ...     nl.Conv2d(3, 16, 3, padding=1),
    ...     nl.Relu(),
    ...     nl.Dropout2d(0.5),
    ...     nl.Conv2d(16, 32, 3, padding=1)
    ... )
    >>>
    >>> # Training mode
    >>> with train.on:
    ...     y = model(x)  # Dropout2d is active
    >>>
    >>> # Evaluation mode
    >>> with train.off:
    ...     y = model(x)  # Dropout2d is inactive

    Notes
    -----
    Dropout2d is preferred over regular Dropout for convolutional layers
    because it drops entire feature maps, which is more effective given
    the spatial correlation in images.
    """

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        """
        Forward pass of dropout_2d

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """
        return dropout_2d(x, p=self.p, training=train.is_on())

    def __repr__(self):
        return f"Dropout2d(p={self.p})"
