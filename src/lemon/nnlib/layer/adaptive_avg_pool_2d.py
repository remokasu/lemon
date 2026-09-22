from lemon.nnlib.module import Module
import lemon.numlib as nm
from lemon.nnlib.layer.avg_pool_2d import avg_pool_2d


def adaptive_avg_pool_2d(x, output_size):
    """
    2D Adaptive average pooling (functional API with autograd support)

    Applies a 2D adaptive average pooling over an input signal.
    The output size is specified, and the pooling kernel and stride are
    automatically computed.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (N, C, H, W)
    output_size : int or tuple
        Target output size (H_out, W_out)

    Returns
    -------
    Tensor
        Output tensor of shape (N, C, H_out, W_out)

    Examples
    --------
    >>> x = nm.randn(4, 512, 7, 7, requires_grad=True)
    >>> y = adaptive_avg_pool_2d(x, output_size=1)
    >>> y.shape
    (4, 512, 1, 1)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    This is commonly used before fully connected layers in classification networks
    to handle variable input sizes. Uses numlib's mean function which supports autograd.
    """
    N, C, H, W = x.shape

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    H_out, W_out = output_size

    # Calculate stride and kernel size
    stride_h = H // H_out
    stride_w = W // W_out
    kernel_h = H - (H_out - 1) * stride_h
    kernel_w = W - (W_out - 1) * stride_w

    # 窓の大きさと stride を決めたら、平均プーリングと同じ計算（勾配も入力に戻る）
    return avg_pool_2d(x, (kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)


class AdaptiveAvgPool2d(Module):
    """
    2D Adaptive average pooling layer

    Applies a 2D adaptive average pooling over an input signal.
    The output size is specified, and the pooling parameters are
    automatically computed to achieve the desired output size.

    Parameters
    ----------
    output_size : int or tuple
        Target output size (H_out, W_out)

    Examples
    --------
    >>> import numlib as nm
    >>> import lemon as lm
    >>> # Commonly used to flatten feature maps to 1x1
    >>> pool = nl.AdaptiveAvgPool2d(output_size=1)
    >>> x = nm.randn(32, 512, 7, 7)
    >>> y = pool(x)
    >>> y.shape
    (32, 512, 1, 1)

    >>> # Can specify different output size
    >>> pool2 = nl.AdaptiveAvgPool2d(output_size=(2, 2))
    >>> x = nm.randn(16, 256, 14, 14)
    >>> y = pool2(x)
    >>> y.shape
    (16, 256, 2, 2)

    >>> # With gradient computation
    >>> x = nm.randn(4, 64, 8, 8, requires_grad=True)
    >>> pool = nl.AdaptiveAvgPool2d(1)
    >>> y = pool(x)
    >>> loss = nm.sum(y)
    >>> loss.backward()
    >>> # x.grad is computed

    Notes
    -----
    This layer is commonly used in modern CNNs (ResNet, VGG, etc.) to handle
    variable input sizes and prepare features for classification layers.
    """

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        """
        Forward pass of adaptive average pooling

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W)

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H_out, W_out)
        """
        return adaptive_avg_pool_2d(x, self.output_size)

    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"
