"""
Weight initialization functions for neural network layers.

All functions follow the in-place modification pattern (name ends with '_').
"""

import lemon.numlib as nm
from lemon.nnlib.parameter import Parameter
from typing import Union
import math


def _calculate_fan_in_and_fan_out(tensor):
    """
    Calculate fan_in and fan_out for a tensor.

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor. Shape can be:
        - (fan_in, fan_out) for Linear layers
        - (out_channels, in_channels, kH, kW) for Conv layers

    Returns
    -------
    tuple
        (fan_in, fan_out)
    """
    dimensions = len(tensor.shape)

    if dimensions < 2:
        raise ValueError(f"Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions. Got {dimensions}")

    if dimensions == 2:
        # Linear layer: (fan_in, fan_out)
        fan_in = tensor.shape[0]
        fan_out = tensor.shape[1]
    else:
        # Conv layer: (out_channels, in_channels, kH, kW, ...)
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            for s in tensor.shape[2:]:
                receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    """
    Calculate fan value based on mode.

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor
    mode : str
        'fan_in', 'fan_out', or 'fan_avg'

    Returns
    -------
    int
        Fan value
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        return fan_in
    elif mode == 'fan_out':
        return fan_out
    else:  # fan_avg
        return (fan_in + fan_out) / 2


def _calculate_gain(nonlinearity, param=None):
    """
    Calculate gain value for different nonlinearities.

    Parameters
    ----------
    nonlinearity : str
        Nonlinearity name ('relu', 'leaky_relu', 'tanh', 'sigmoid', 'linear')
    param : float, optional
        Parameter for leaky_relu (negative slope)

    Returns
    -------
    float
        Gain value
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


# ============================================================================
# Kaiming (He) Initialization - for ReLU networks
# ============================================================================

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming uniform initialization (He initialization).

    Fills the tensor with values according to the method described in
    "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    - He, K. et al. (2015).

    The resulting tensor will have values sampled from U(-bound, bound) where:
    bound = gain * sqrt(3 / fan)

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor to initialize
    a : float, optional
        Negative slope of the rectifier used after this layer (only for 'leaky_relu')
    mode : str, optional
        'fan_in' (default), 'fan_out', or 'fan_avg'
    nonlinearity : str, optional
        Nonlinearity name ('relu', 'leaky_relu', etc.)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.kaiming_uniform_(fc.weight, nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    # In-place modification
    # uniform(a, b) = a + (b - a) * rand()
    if isinstance(tensor, Parameter):
        tensor.data = -bound + (2 * bound) * nm.rand(*tensor.shape)
    else:
        # Direct tensor modification
        new_data = -bound + (2 * bound) * nm.rand(*tensor.shape)
        tensor.data = new_data

    return tensor


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming normal initialization (He initialization).

    Fills the tensor with values according to the method described in
    "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    - He, K. et al. (2015).

    The resulting tensor will have values sampled from N(0, std^2) where:
    std = gain / sqrt(fan)

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor to initialize
    a : float, optional
        Negative slope of the rectifier used after this layer (only for 'leaky_relu')
    mode : str, optional
        'fan_in' (default), 'fan_out', or 'fan_avg'
    nonlinearity : str, optional
        Nonlinearity name ('relu', 'leaky_relu', etc.)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.kaiming_normal_(fc.weight, nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    # In-place modification
    if isinstance(tensor, Parameter):
        tensor.data = nm.randn(*tensor.shape) * std
    else:
        new_data = nm.randn(*tensor.shape) * std
        tensor.data = new_data

    return tensor


# ============================================================================
# Xavier (Glorot) Initialization - for Tanh/Sigmoid networks
# ============================================================================

def xavier_uniform_(tensor, gain=1.0):
    """
    Xavier uniform initialization (Glorot initialization).

    Fills the tensor with values according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    - Glorot, X. & Bengio, Y. (2010).

    The resulting tensor will have values sampled from U(-a, a) where:
    a = gain * sqrt(6 / (fan_in + fan_out))

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor to initialize
    gain : float, optional
        Scaling factor (default: 1.0)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.xavier_uniform_(fc.weight)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std

    # In-place modification
    # uniform(a, b) = a + (b - a) * rand()
    if isinstance(tensor, Parameter):
        tensor.data = -bound + (2 * bound) * nm.rand(*tensor.shape)
    else:
        new_data = -bound + (2 * bound) * nm.rand(*tensor.shape)
        tensor.data = new_data

    return tensor


def xavier_normal_(tensor, gain=1.0):
    """
    Xavier normal initialization (Glorot initialization).

    Fills the tensor with values according to the method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    - Glorot, X. & Bengio, Y. (2010).

    The resulting tensor will have values sampled from N(0, std^2) where:
    std = gain * sqrt(2 / (fan_in + fan_out))

    Parameters
    ----------
    tensor : Tensor or Parameter
        Weight tensor to initialize
    gain : float, optional
        Scaling factor (default: 1.0)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.xavier_normal_(fc.weight)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    # In-place modification
    if isinstance(tensor, Parameter):
        tensor.data = nm.randn(*tensor.shape) * std
    else:
        new_data = nm.randn(*tensor.shape) * std
        tensor.data = new_data

    return tensor


# ============================================================================
# Basic initialization functions
# ============================================================================

def zeros_(tensor):
    """
    Fill the tensor with zeros.

    Parameters
    ----------
    tensor : Tensor or Parameter
        Tensor to initialize

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.zeros_(fc.bias)
    """
    if isinstance(tensor, Parameter):
        tensor.data = nm.zeros(tensor.shape)
    else:
        tensor.data = nm.zeros(tensor.shape)

    return tensor


def ones_(tensor):
    """
    Fill the tensor with ones.

    Parameters
    ----------
    tensor : Tensor or Parameter
        Tensor to initialize

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.ones_(fc.weight)
    """
    if isinstance(tensor, Parameter):
        tensor.data = nm.ones(tensor.shape)
    else:
        tensor.data = nm.ones(tensor.shape)

    return tensor


def constant_(tensor, val):
    """
    Fill the tensor with a constant value.

    Parameters
    ----------
    tensor : Tensor or Parameter
        Tensor to initialize
    val : float
        Value to fill

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.constant_(fc.bias, 0.1)
    """
    if isinstance(tensor, Parameter):
        tensor.data = nm.ones(tensor.shape) * val
    else:
        tensor.data = nm.ones(tensor.shape) * val

    return tensor


def normal_(tensor, mean=0.0, std=1.0):
    """
    Fill the tensor with values from a normal distribution N(mean, std^2).

    Parameters
    ----------
    tensor : Tensor or Parameter
        Tensor to initialize
    mean : float, optional
        Mean of the normal distribution (default: 0.0)
    std : float, optional
        Standard deviation (default: 1.0)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.normal_(fc.weight, mean=0, std=0.01)
    """
    if isinstance(tensor, Parameter):
        tensor.data = nm.randn(*tensor.shape) * std + mean
    else:
        tensor.data = nm.randn(*tensor.shape) * std + mean

    return tensor


def uniform_(tensor, a=0.0, b=1.0):
    """
    Fill the tensor with values from a uniform distribution U(a, b).

    Parameters
    ----------
    tensor : Tensor or Parameter
        Tensor to initialize
    a : float, optional
        Lower bound (default: 0.0)
    b : float, optional
        Upper bound (default: 1.0)

    Returns
    -------
    Tensor or Parameter
        The initialized tensor

    Examples
    --------
    >>> import lemon.nnlib as nl
    >>> fc = nl.Linear(100, 200)
    >>> nl.init.uniform_(fc.weight, -0.1, 0.1)
    """
    # uniform(a, b) = a + (b - a) * rand()
    if isinstance(tensor, Parameter):
        tensor.data = a + (b - a) * nm.rand(*tensor.shape)
    else:
        tensor.data = a + (b - a) * nm.rand(*tensor.shape)

    return tensor
