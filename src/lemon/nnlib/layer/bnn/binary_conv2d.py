import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.activation.sign import sign
from lemon.nnlib.layer.conv_2d import conv_2d


def binary_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Binary 2D convolution

    Binarizes weights with sign function (STE) then applies convolution.

    Parameters
    ----------
    x : Tensor
        Input tensor (N, C_in, H, W)
    weight : Tensor
        Real-valued weight tensor (C_out, C_in/groups, kernel_h, kernel_w)
    bias : Tensor, optional
        Bias tensor (C_out,)
    stride, padding, dilation, groups : same as conv_2d

    Returns
    -------
    Tensor
        Output tensor (N, C_out, H_out, W_out)
    """
    w_b = sign(weight)
    return conv_2d(x, w_b, bias=bias, stride=stride, padding=padding,
                   dilation=dilation, groups=groups)


class BinaryConv2d(Module):
    """
    Binary 2D Convolutional layer (BNN)

    Stores real-valued weights for gradient accumulation.
    During forward pass, weights are binarized to {-1, +1} via sign function.
    Gradients flow back through the Straight-Through Estimator.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Convolution kernel size
    stride : int or tuple, optional
        Stride (default: 1)
    padding : int or tuple, optional
        Padding (default: 0)
    dilation : int or tuple, optional
        Dilation (default: 1)
    groups : int, optional
        Groups (default: 1)
    bias : bool, optional
        If True, adds a learnable real-valued bias (default: True)

    Examples
    --------
    >>> model = nl.Sequential(
    ...     nl.BinaryConv2d(3, 32, kernel_size=3, padding=1),
    ...     nl.Sign(),
    ...     nl.BinaryConv2d(32, 64, kernel_size=3, padding=1),
    ...     nl.Sign(),
    ...     nl.Flatten(),
    ...     nl.BinaryLinear(64 * 8 * 8, 10),
    ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.kernel_h, self.kernel_w = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        fan_in = (in_channels // groups) * self.kernel_h * self.kernel_w
        std = (2.0 / fan_in) ** 0.5
        self.weight = Parameter(
            nm.randn(out_channels, in_channels // groups, self.kernel_h, self.kernel_w) * std
        )

        if bias:
            self.bias = Parameter(nm.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        return binary_conv2d(
            x,
            self.weight.data,
            bias=self.bias.data if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self):
        return (
            f"BinaryConv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None})"
        )
