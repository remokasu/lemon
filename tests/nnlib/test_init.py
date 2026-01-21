"""
Test cases for weight initialization functions (nnlib.init)
"""

import pytest
import numpy as np

import lemon.numlib as nm
import lemon.nnlib as nl


def test_zeros_():
    """Test zeros_ initialization"""
    fc = nl.Linear(10, 20)
    nl.init.zeros_(fc.weight)

    assert np.allclose(nm.as_numpy(fc.weight.data), 0.0)


def test_ones_():
    """Test ones_ initialization"""
    fc = nl.Linear(10, 20)
    nl.init.ones_(fc.weight)

    assert np.allclose(nm.as_numpy(fc.weight.data), 1.0)


def test_constant_():
    """Test constant_ initialization"""
    fc = nl.Linear(10, 20)
    nl.init.constant_(fc.weight, 0.5)

    assert np.allclose(nm.as_numpy(fc.weight.data), 0.5)


def test_normal_():
    """Test normal_ initialization"""
    fc = nl.Linear(100, 200)
    nl.init.normal_(fc.weight, mean=0.0, std=0.01)

    weight_np = nm.as_numpy(fc.weight.data)
    mean = np.mean(weight_np)
    std = np.std(weight_np)

    # Check mean is close to 0
    assert abs(mean) < 0.01

    # Check std is close to 0.01 (with some tolerance)
    assert abs(std - 0.01) < 0.005


def test_uniform_():
    """Test uniform_ initialization"""
    fc = nl.Linear(100, 200)
    nl.init.uniform_(fc.weight, a=-0.1, b=0.1)

    weight_np = nm.as_numpy(fc.weight.data)

    # Check all values are within bounds
    assert np.all(weight_np >= -0.1)
    assert np.all(weight_np <= 0.1)

    # Check distribution is roughly uniform
    mean = np.mean(weight_np)
    assert abs(mean) < 0.02  # Should be close to 0


def test_kaiming_uniform_():
    """Test Kaiming uniform initialization for ReLU"""
    fc = nl.Linear(100, 200)
    nl.init.kaiming_uniform_(fc.weight, nonlinearity='relu')

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected bound
    fan_in = 100
    gain = np.sqrt(2.0)  # For ReLU
    std = gain / np.sqrt(fan_in)
    bound = np.sqrt(3.0) * std

    # Check all values are within expected bounds
    assert np.all(weight_np >= -bound)
    assert np.all(weight_np <= bound)

    # Check mean is close to 0
    mean = np.mean(weight_np)
    assert abs(mean) < 0.05


def test_kaiming_normal_():
    """Test Kaiming normal initialization for ReLU"""
    fc = nl.Linear(100, 200)
    nl.init.kaiming_normal_(fc.weight, nonlinearity='relu')

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected std
    fan_in = 100
    gain = np.sqrt(2.0)  # For ReLU
    expected_std = gain / np.sqrt(fan_in)

    # Check mean is close to 0
    mean = np.mean(weight_np)
    assert abs(mean) < 0.05

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.05


def test_kaiming_with_leaky_relu():
    """Test Kaiming initialization with LeakyReLU"""
    fc = nl.Linear(100, 200)
    negative_slope = 0.1
    nl.init.kaiming_normal_(fc.weight, a=negative_slope, nonlinearity='leaky_relu')

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected std
    fan_in = 100
    gain = np.sqrt(2.0 / (1 + negative_slope ** 2))
    expected_std = gain / np.sqrt(fan_in)

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.05


def test_xavier_uniform_():
    """Test Xavier uniform initialization"""
    fc = nl.Linear(100, 200)
    nl.init.xavier_uniform_(fc.weight)

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected bound
    fan_in, fan_out = 100, 200
    gain = 1.0
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std

    # Check all values are within expected bounds
    assert np.all(weight_np >= -bound)
    assert np.all(weight_np <= bound)

    # Check mean is close to 0
    mean = np.mean(weight_np)
    assert abs(mean) < 0.05


def test_xavier_normal_():
    """Test Xavier normal initialization"""
    fc = nl.Linear(100, 200)
    nl.init.xavier_normal_(fc.weight)

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected std
    fan_in, fan_out = 100, 200
    gain = 1.0
    expected_std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    # Check mean is close to 0
    mean = np.mean(weight_np)
    assert abs(mean) < 0.05

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.05


def test_xavier_with_gain():
    """Test Xavier initialization with custom gain"""
    fc = nl.Linear(100, 200)
    custom_gain = 2.0
    nl.init.xavier_normal_(fc.weight, gain=custom_gain)

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected std with custom gain
    fan_in, fan_out = 100, 200
    expected_std = custom_gain * np.sqrt(2.0 / (fan_in + fan_out))

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.1


def test_initialization_on_parameter():
    """Test that initialization works on Parameter objects"""
    from lemon.nnlib.parameter import Parameter

    param = Parameter(nm.zeros((10, 20)))
    nl.init.kaiming_normal_(param)

    param_np = nm.as_numpy(param.data)

    # Check it's no longer all zeros
    assert not np.allclose(param_np, 0.0)

    # Check shape is preserved
    assert param.shape == (10, 20)


def test_fan_calculation_for_conv():
    """Test fan calculation for convolution-like tensors"""
    # Simulate Conv2d weight: (out_channels, in_channels, kH, kW)
    from lemon.nnlib.parameter import Parameter

    weight = Parameter(nm.zeros((32, 16, 3, 3)))
    nl.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')

    weight_np = nm.as_numpy(weight.data)

    # For Conv: fan_in = in_channels * kH * kW = 16 * 3 * 3 = 144
    fan_in = 16 * 3 * 3
    gain = np.sqrt(2.0)
    expected_std = gain / np.sqrt(fan_in)

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.05


def test_mode_fan_out():
    """Test initialization with fan_out mode"""
    fc = nl.Linear(100, 200)
    nl.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')

    weight_np = nm.as_numpy(fc.weight.data)

    # Calculate expected std with fan_out
    fan_out = 200
    gain = np.sqrt(2.0)
    expected_std = gain / np.sqrt(fan_out)

    # Check std is close to expected value
    std = np.std(weight_np)
    assert abs(std - expected_std) < 0.05


def test_invalid_mode():
    """Test that invalid mode raises error"""
    from lemon.nnlib.parameter import Parameter

    weight = Parameter(nm.zeros((10, 20)))

    with pytest.raises(ValueError, match="Mode.*not supported"):
        nl.init.kaiming_normal_(weight, mode='invalid_mode')


def test_invalid_nonlinearity():
    """Test that invalid nonlinearity raises error"""
    from lemon.nnlib.parameter import Parameter

    weight = Parameter(nm.zeros((10, 20)))

    with pytest.raises(ValueError, match="Unsupported nonlinearity"):
        nl.init.kaiming_normal_(weight, nonlinearity='invalid_activation')


def test_1d_tensor_error():
    """Test that 1D tensor raises error"""
    from lemon.nnlib.parameter import Parameter

    weight = Parameter(nm.zeros((10,)))

    with pytest.raises(ValueError, match="fewer than 2 dimensions"):
        nl.init.kaiming_normal_(weight)


def test_real_world_scenario():
    """Test real-world usage scenario"""
    # Create a simple network
    fc1 = nl.Linear(784, 256)
    fc2 = nl.Linear(256, 128)
    fc3 = nl.Linear(128, 10)

    # Initialize with Kaiming for ReLU
    nl.init.kaiming_normal_(fc1.weight, nonlinearity='relu')
    nl.init.kaiming_normal_(fc2.weight, nonlinearity='relu')
    nl.init.xavier_normal_(fc3.weight)  # Last layer with Softmax

    # Initialize biases to zero
    nl.init.zeros_(fc1.bias)
    nl.init.zeros_(fc2.bias)
    nl.init.zeros_(fc3.bias)

    # Check that weights are initialized properly
    assert not np.allclose(nm.as_numpy(fc1.weight.data), 0.0)
    assert not np.allclose(nm.as_numpy(fc2.weight.data), 0.0)
    assert not np.allclose(nm.as_numpy(fc3.weight.data), 0.0)

    # Check that biases are zero
    assert np.allclose(nm.as_numpy(fc1.bias.data), 0.0)
    assert np.allclose(nm.as_numpy(fc2.bias.data), 0.0)
    assert np.allclose(nm.as_numpy(fc3.bias.data), 0.0)

    # Test forward pass doesn't explode
    x = nm.randn(32, 784)
    y1 = nl.relu(fc1(x))
    y2 = nl.relu(fc2(y1))
    y3 = fc3(y2)

    # Check output is reasonable (not NaN, not huge)
    y3_np = nm.as_numpy(y3)
    assert not np.any(np.isnan(y3_np))
    assert np.all(np.abs(y3_np) < 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
