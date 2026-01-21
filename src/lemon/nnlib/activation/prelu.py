import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter


def prelu(x, weight):
    """
    PReLU (Parametric ReLU) activation function

    PReLU(x) = x if x >= 0 else weight * x

    Parameters
    ----------
    x : Tensor
        Input tensor
    weight : Tensor or Parameter
        Learnable weight parameter (slope for negative values)

    Returns
    -------
    Tensor
        Output tensor

    Examples
    --------
    >>> x = nm.tensor([-1.0, 0.0, 1.0])
    >>> weight = nm.tensor([0.25])
    >>> y = nl.prelu(x, weight)
    """
    # weightがParameterの場合、.dataを取り出す
    if hasattr(weight, "data"):
        weight = weight.data

    # weightの形状に応じてreshape（勾配を維持したまま）
    if weight.shape == (1,):
        # Single parameter: reshape to () for proper broadcasting
        slope = nm.reshape(weight, ())
    elif x._data.ndim == 4:
        # 4D input (N, C, H, W): reshape weight to (1, C, 1, 1)
        slope = nm.reshape(weight, (1, -1, 1, 1))
    elif x._data.ndim == 2:
        # 2D input (N, C): reshape weight to (1, C)
        slope = nm.reshape(weight, (1, -1))
    else:
        slope = weight

    return nm.where(x >= 0, x, slope * x)


class PRelu(Module):
    """
    PReLU (Parametric ReLU) activation module

    Parameters
    ----------
    num_parameters : int, optional
        Number of parameters (default: 1)
    init : float, optional
        Initial value for slope (default: 0.25)
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.slope = Parameter(nm.ones((num_parameters,)) * init)

    def forward(self, x):
        # self.slopeをそのまま渡す（Parameterとして）
        return prelu(x, self.slope)

    def __repr__(self):
        return f"PReLU(num_parameters={self.num_parameters})"
