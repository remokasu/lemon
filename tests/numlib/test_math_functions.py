import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestExponentialFunctions:
    """Test exponential and logarithmic functions"""

    def test_exp_scalar(self):
        """Test exp of scalar"""
        x = nm.Real(1.0)
        result = nm.exp(x)
        assert abs(float(result._data) - np.e) < 1e-6

    def test_exp_vector(self):
        """Test exp of vector"""
        v = nm.Vector([0, 1, 2])
        result = nm.exp(v)
        expected = np.exp([0, 1, 2])
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)

    def test_exp_backward(self):
        """Test backward pass for exp"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.exp(x)
        y.backward()
        # d(exp(x))/dx = exp(x)
        assert abs(float(x.grad._data) - np.e) < 1e-6

    def test_log_scalar(self):
        """Test log of scalar"""
        x = nm.Real(np.e)
        result = nm.log(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_log_vector(self):
        """Test log of vector"""
        v = nm.Vector([1, np.e, np.e**2])
        result = nm.log(v)
        expected = [0, 1, 2]
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)

    def test_log_backward(self):
        """Test backward pass for log"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.log(x)
        y.backward()
        # d(log(x))/dx = 1/x
        assert abs(float(x.grad._data) - 0.5) < 1e-6

    def test_log2_scalar(self):
        """Test log2 of scalar"""
        x = nm.Real(8.0)
        result = nm.log2(x)
        assert abs(float(result._data) - 3.0) < 1e-6

    def test_log2_backward(self):
        """Test backward pass for log2"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.log2(x)
        y.backward()
        # d(log2(x))/dx = 1/(x*ln(2))
        expected = 1 / (2 * np.log(2))
        assert abs(float(x.grad._data) - expected) < 1e-6

    def test_log10_scalar(self):
        """Test log10 of scalar"""
        x = nm.Real(100.0)
        result = nm.log10(x)
        assert abs(float(result._data) - 2.0) < 1e-6

    def test_log10_backward(self):
        """Test backward pass for log10"""
        x = nm.Real(10.0, requires_grad=True)
        y = nm.log10(x)
        y.backward()
        # d(log10(x))/dx = 1/(x*ln(10))
        expected = 1 / (10 * np.log(10))
        assert abs(float(x.grad._data) - expected) < 1e-6

    def test_sqrt_scalar(self):
        """Test sqrt of scalar"""
        x = nm.Real(4.0)
        result = nm.sqrt(x)
        assert abs(float(result._data) - 2.0) < 1e-6

    def test_sqrt_vector(self):
        """Test sqrt of vector"""
        v = nm.Vector([1, 4, 9, 16])
        result = nm.sqrt(v)
        expected = [1, 2, 3, 4]
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)

    def test_sqrt_backward(self):
        """Test backward pass for sqrt"""
        x = nm.Real(4.0, requires_grad=True)
        y = nm.sqrt(x)
        y.backward()
        # d(sqrt(x))/dx = 1/(2*sqrt(x)) = 1/4
        assert abs(float(x.grad._data) - 0.25) < 1e-6


class TestTrigonometricFunctions:
    """Test trigonometric functions"""

    def test_sin_scalar(self):
        """Test sin of scalar"""
        x = nm.Real(np.pi / 2)
        result = nm.sin(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_sin_vector(self):
        """Test sin of vector"""
        v = nm.Vector([0, np.pi / 2, np.pi])
        result = nm.sin(v)
        expected = [0, 1, 0]
        np.testing.assert_array_almost_equal(
            result._data.flatten(), expected, decimal=6
        )

    def test_sin_backward(self):
        """Test backward pass for sin"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.sin(x)
        y.backward()
        # d(sin(x))/dx = cos(x) = cos(0) = 1
        assert abs(float(x.grad._data) - 1.0) < 1e-6

    def test_cos_scalar(self):
        """Test cos of scalar"""
        x = nm.Real(0.0)
        result = nm.cos(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_cos_vector(self):
        """Test cos of vector"""
        v = nm.Vector([0, np.pi / 2, np.pi])
        result = nm.cos(v)
        expected = [1, 0, -1]
        np.testing.assert_array_almost_equal(
            result._data.flatten(), expected, decimal=6
        )

    def test_cos_backward(self):
        """Test backward pass for cos"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.cos(x)
        y.backward()
        # d(cos(x))/dx = -sin(x) = -sin(0) = 0
        assert abs(float(x.grad._data)) < 1e-6

    def test_tan_scalar(self):
        """Test tan of scalar"""
        x = nm.Real(np.pi / 4)
        result = nm.tan(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_tan_backward(self):
        """Test backward pass for tan"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.tan(x)
        y.backward()
        # d(tan(x))/dx = 1/cos^2(x) = 1/1 = 1
        assert abs(float(x.grad._data) - 1.0) < 1e-6

    def test_asin_scalar(self):
        """Test asin of scalar"""
        x = nm.Real(0.5)
        result = nm.asin(x)
        expected = np.pi / 6
        assert abs(float(result._data) - expected) < 1e-6

    def test_asin_backward(self):
        """Test backward pass for asin"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.asin(x)
        y.backward()
        # d(asin(x))/dx = 1/sqrt(1-x^2)
        expected = 1 / np.sqrt(1 - 0.5**2)
        assert abs(float(x.grad._data) - expected) < 1e-6

    def test_acos_scalar(self):
        """Test acos of scalar"""
        x = nm.Real(0.5)
        result = nm.acos(x)
        expected = np.pi / 3
        assert abs(float(result._data) - expected) < 1e-6

    def test_acos_backward(self):
        """Test backward pass for acos"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.acos(x)
        y.backward()
        # d(acos(x))/dx = -1/sqrt(1-x^2)
        expected = -1 / np.sqrt(1 - 0.5**2)
        assert abs(float(x.grad._data) - expected) < 1e-6

    def test_atan_scalar(self):
        """Test atan of scalar"""
        x = nm.Real(1.0)
        result = nm.atan(x)
        expected = np.pi / 4
        assert abs(float(result._data) - expected) < 1e-6

    def test_atan_backward(self):
        """Test backward pass for atan"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.atan(x)
        y.backward()
        # d(atan(x))/dx = 1/(1+x^2) = 1/2
        assert abs(float(x.grad._data) - 0.5) < 1e-6

    def test_atan2_scalar(self):
        """Test atan2 (two-argument arctangent)"""
        y = nm.Real(1.0)
        x = nm.Real(1.0)
        result = nm.atan2(y, x)
        expected = np.pi / 4
        assert abs(float(result._data) - expected) < 1e-6


class TestHyperbolicFunctions:
    """Test hyperbolic functions"""

    def test_sinh_scalar(self):
        """Test sinh of scalar"""
        x = nm.Real(0.0)
        result = nm.sinh(x)
        assert abs(float(result._data)) < 1e-6

    def test_sinh_backward(self):
        """Test backward pass for sinh"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.sinh(x)
        y.backward()
        # d(sinh(x))/dx = cosh(x) = cosh(0) = 1
        assert abs(float(x.grad._data) - 1.0) < 1e-6

    def test_cosh_scalar(self):
        """Test cosh of scalar"""
        x = nm.Real(0.0)
        result = nm.cosh(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_cosh_backward(self):
        """Test backward pass for cosh"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.cosh(x)
        y.backward()
        # d(cosh(x))/dx = sinh(x) = sinh(0) = 0
        assert abs(float(x.grad._data)) < 1e-6

    def test_tanh_scalar(self):
        """Test tanh of scalar"""
        x = nm.Real(0.0)
        result = nm.tanh(x)
        assert abs(float(result._data)) < 1e-6

    def test_tanh_backward(self):
        """Test backward pass for tanh"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.tanh(x)
        y.backward()
        # d(tanh(x))/dx = 1 - tanh^2(x) = 1 - 0 = 1
        assert abs(float(x.grad._data) - 1.0) < 1e-6

    def test_asinh_scalar(self):
        """Test asinh of scalar"""
        x = nm.Real(0.0)
        result = nm.asinh(x)
        assert abs(float(result._data)) < 1e-6

    def test_asinh_backward(self):
        """Test backward pass for asinh"""
        x = nm.Real(0.0, requires_grad=True)
        y = nm.asinh(x)
        y.backward()
        # d(asinh(x))/dx = 1/sqrt(x^2+1) = 1
        assert abs(float(x.grad._data) - 1.0) < 1e-6

    def test_acosh_scalar(self):
        """Test acosh of scalar"""
        x = nm.Real(1.0)
        result = nm.acosh(x)
        assert abs(float(result._data)) < 1e-6

    def test_acosh_backward(self):
        """Test backward pass for acosh"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.acosh(x)
        y.backward()
        # d(acosh(x))/dx = 1/sqrt(x^2-1)
        expected = 1 / np.sqrt(2**2 - 1)
        assert abs(float(x.grad._data) - expected) < 1e-6

    def test_atanh_scalar(self):
        """Test atanh of scalar"""
        x = nm.Real(0.0)
        result = nm.atanh(x)
        assert abs(float(result._data)) < 1e-6

    def test_atanh_backward(self):
        """Test backward pass for atanh"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.atanh(x)
        y.backward()
        # d(atanh(x))/dx = 1/(1-x^2)
        expected = 1 / (1 - 0.5**2)
        assert abs(float(x.grad._data) - expected) < 1e-6


class TestMathEdgeCases:
    """Test edge cases for mathematical functions"""

    def test_exp_zero(self):
        """Test exp(0) = 1"""
        x = nm.Real(0.0)
        result = nm.exp(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_log_one(self):
        """Test log(1) = 0"""
        x = nm.Real(1.0)
        result = nm.log(x)
        assert abs(float(result._data)) < 1e-6

    def test_sqrt_zero(self):
        """Test sqrt(0) = 0"""
        x = nm.Real(0.0)
        result = nm.sqrt(x)
        assert abs(float(result._data)) < 1e-6

    def test_sin_zero(self):
        """Test sin(0) = 0"""
        x = nm.Real(0.0)
        result = nm.sin(x)
        assert abs(float(result._data)) < 1e-6

    def test_cos_zero(self):
        """Test cos(0) = 1"""
        x = nm.Real(0.0)
        result = nm.cos(x)
        assert abs(float(result._data) - 1.0) < 1e-6

    def test_tan_zero(self):
        """Test tan(0) = 0"""
        x = nm.Real(0.0)
        result = nm.tan(x)
        assert abs(float(result._data)) < 1e-6

    def test_exp_large_value(self):
        """Test exp with large value"""
        x = nm.Real(10.0)
        result = nm.exp(x)
        expected = np.exp(10.0)
        assert abs(float(result._data) - expected) / expected < 1e-6

    def test_log_small_value(self):
        """Test log with small positive value"""
        x = nm.Real(1e-10)
        result = nm.log(x)
        expected = np.log(1e-10)
        assert abs(float(result._data) - expected) < 1e-4
