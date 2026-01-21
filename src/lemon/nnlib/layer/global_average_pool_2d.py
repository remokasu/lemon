import lemon.numlib as nm
from lemon.nnlib.module import Module


def global_average_pool_2d(x):
    """
    Global average pooling 2D (functional API with autograd support)

    Applies global average pooling over the spatial dimensions (H, W).
    This is equivalent to adaptive_avg_pool_2d(x, output_size=1).

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)

    Returns
    -------
    Tensor
        Output tensor of shape (N, C, 1, 1)

    Examples
    --------
    >>> x = nm.randn(4, 512, 7, 7, requires_grad=True)
    >>> y = global_average_pool_2d(x)
    >>> y.shape
    (4, 512, 1, 1)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    Global average pooling is commonly used as an alternative to fully
    connected layers in classification networks. It reduces overfitting
    and has no parameters to learn.
    Uses numlib's mean function which supports autograd.
    """
    # Simply use mean over spatial dimensions
    # nm.mean supports autograd
    result = nm.mean(x, axis=(2, 3), keepdims=True)
    return result


class GlobalAveragePooling2d(Module):
    """
    Global average pooling layer

    Applies global average pooling over the spatial dimensions.
    Reduces each feature map to a single value by averaging.

    Examples
    --------
    >>> import numlib as nm
    >>> import lemon as lm
    >>> # Commonly used to replace flatten + dense layers
    >>> pool = nl.GlobalAveragePooling2d()
    >>> x = nm.randn(32, 512, 7, 7)
    >>> y = pool(x)
    >>> y.shape
    (32, 512, 1, 1)

    >>> # Often followed by a single fully connected layer
    >>> model = nl.Sequential(
    ...     nl.Conv2d(3, 64, 3, padding=1),
    ...     nl.Relu(),
    ...     nl.GlobalAveragePooling2d(),
    ...     nl.Flatten(),
    ...     nl.Linear(64, 10)
    ... )

    >>> # With gradient computation
    >>> x = nm.randn(4, 64, 8, 8, requires_grad=True)
    >>> pool = nl.GlobalAveragePooling2d()
    >>> y = pool(x)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    Global average pooling has several advantages:
    - No parameters to learn (no overfitting)
    - Works with any input size
    - Naturally corresponds to categories (one value per feature map)
    - Used in networks like ResNet, Inception, MobileNet
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of global average pooling

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, 1, 1)
        """
        return global_average_pool_2d(x)

    def __repr__(self):
        return "GlobalAveragePooling2d()"
