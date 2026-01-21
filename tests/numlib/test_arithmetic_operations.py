import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestAddition:
    """Test addition operation"""

    def test_add_scalars(self):
        """Test adding two scalars"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = x + y
        assert float(z._data) == 5.0

    def test_add_vectors(self):
        """Test adding two vectors"""
        x = nm.Vector([1, 2, 3])
        y = nm.Vector([4, 5, 6])
        z = x + y
        np.testing.assert_array_equal(z._data.flatten(), [5, 7, 9])

    def test_add_matrices(self):
        """Test adding two matrices"""
        x = nm.Matrix([[1, 2], [3, 4]])
        y = nm.Matrix([[5, 6], [7, 8]])
        z = x + y
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(z._data, expected)

    def test_add_scalar_to_vector(self):
        """Test adding scalar to vector (broadcasting)"""
        x = nm.Vector([1, 2, 3])
        y = nm.Real(10.0)
        z = x + y
        np.testing.assert_array_equal(z._data.flatten(), [11, 12, 13])

    def test_add_vector_to_scalar(self):
        """Test adding vector to scalar (broadcasting)"""
        x = nm.Real(10.0)
        y = nm.Vector([1, 2, 3])
        z = x + y
        np.testing.assert_array_equal(z._data.flatten(), [11, 12, 13])

    def test_add_python_int_to_scalar(self):
        """Test adding Python int to scalar"""
        x = nm.Real(2.5)
        z = x + 3
        assert float(z._data) == 5.5

    def test_add_python_int_to_scalar_reverse(self):
        """Test adding scalar to Python int (radd)"""
        x = nm.Real(2.5)
        z = 3 + x
        assert float(z._data) == 5.5

    def test_add_different_shapes_broadcasts(self):
        """Test addition with different shapes broadcasts correctly"""
        x = nm.Matrix([[1, 2, 3]])  # (1, 3)
        y = nm.Vector([10, 20, 30])  # (3, 1)
        z = x + y
        expected = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        np.testing.assert_array_equal(z._data, expected)

    def test_add_requires_grad_true(self):
        """Test addition with requires_grad=True"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x + y
        assert z.requires_grad is True

    def test_add_requires_grad_one_false(self):
        """Test addition when only one requires grad"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=False)
        z = x + y
        assert z.requires_grad is True

    def test_add_requires_grad_both_false(self):
        """Test addition when both don't require grad"""
        x = nm.Real(2.0, requires_grad=False)
        y = nm.Real(3.0, requires_grad=False)
        z = x + y
        assert z.requires_grad is False

    def test_add_backward_scalars(self):
        """Test backward pass for scalar addition"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x + y
        z.backward()
        assert float(x.grad._data) == 1.0
        assert float(y.grad._data) == 1.0

    def test_add_backward_vectors(self):
        """Test backward pass for vector addition"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        y = nm.Vector([4, 5, 6], requires_grad=True)
        z = x + y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, 1, 1])
        np.testing.assert_array_equal(y.grad._data.flatten(), [1, 1, 1])

    def test_add_backward_broadcast(self):
        """Test backward pass with broadcasting"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        y = nm.Real(10.0, requires_grad=True)
        z = x + y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, 1, 1])
        assert float(y.grad._data) == 3.0  # Sum of gradients

    def test_add_function(self):
        """Test nm.add function"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = nm.add(x, y)
        assert float(z._data) == 5.0

    def test_add_autograd_off(self):
        """Test addition with autograd off"""
        with nm.autograd.off:
            x = nm.Real(2.0)
            y = nm.Real(3.0)
            z = x + y
            assert z.requires_grad is False


class TestSubtraction:
    """Test subtraction operation"""

    def test_sub_scalars(self):
        """Test subtracting two scalars"""
        x = nm.Real(5.0)
        y = nm.Real(3.0)
        z = x - y
        assert float(z._data) == 2.0

    def test_sub_vectors(self):
        """Test subtracting two vectors"""
        x = nm.Vector([5, 7, 9])
        y = nm.Vector([1, 2, 3])
        z = x - y
        np.testing.assert_array_equal(z._data.flatten(), [4, 5, 6])

    def test_sub_matrices(self):
        """Test subtracting two matrices"""
        x = nm.Matrix([[5, 6], [7, 8]])
        y = nm.Matrix([[1, 2], [3, 4]])
        z = x - y
        expected = np.array([[4, 4], [4, 4]])
        np.testing.assert_array_equal(z._data, expected)

    def test_sub_scalar_from_vector(self):
        """Test subtracting scalar from vector"""
        x = nm.Vector([10, 20, 30])
        y = nm.Real(5.0)
        z = x - y
        np.testing.assert_array_equal(z._data.flatten(), [5, 15, 25])

    def test_sub_vector_from_scalar(self):
        """Test subtracting vector from scalar"""
        x = nm.Real(10.0)
        y = nm.Vector([1, 2, 3])
        z = x - y
        np.testing.assert_array_equal(z._data.flatten(), [9, 8, 7])

    def test_sub_python_int(self):
        """Test subtracting Python int"""
        x = nm.Real(10.0)
        z = x - 3
        assert float(z._data) == 7.0

    def test_sub_python_int_reverse(self):
        """Test subtracting from Python int (rsub)"""
        x = nm.Real(3.0)
        z = 10 - x
        assert float(z._data) == 7.0

    def test_sub_backward_scalars(self):
        """Test backward pass for scalar subtraction"""
        x = nm.Real(5.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x - y
        z.backward()
        assert float(x.grad._data) == 1.0
        assert float(y.grad._data) == -1.0

    def test_sub_backward_vectors(self):
        """Test backward pass for vector subtraction"""
        x = nm.Vector([5, 7, 9], requires_grad=True)
        y = nm.Vector([1, 2, 3], requires_grad=True)
        z = x - y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, 1, 1])
        np.testing.assert_array_equal(y.grad._data.flatten(), [-1, -1, -1])

    def test_sub_backward_broadcast(self):
        """Test backward pass with broadcasting"""
        x = nm.Vector([10, 20, 30], requires_grad=True)
        y = nm.Real(5.0, requires_grad=True)
        z = x - y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, 1, 1])
        assert float(y.grad._data) == -3.0

    def test_sub_function(self):
        """Test nm.sub function"""
        x = nm.Real(5.0)
        y = nm.Real(3.0)
        z = nm.sub(x, y)
        assert float(z._data) == 2.0


class TestMultiplication:
    """Test multiplication operation"""

    def test_mul_scalars(self):
        """Test multiplying two scalars"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = x * y
        assert float(z._data) == 6.0

    def test_mul_vectors_elementwise(self):
        """Test element-wise multiplication of vectors"""
        x = nm.Vector([1, 2, 3])
        y = nm.Vector([4, 5, 6])
        z = x * y
        np.testing.assert_array_equal(z._data.flatten(), [4, 10, 18])

    def test_mul_matrices_elementwise(self):
        """Test element-wise multiplication of matrices"""
        x = nm.Matrix([[1, 2], [3, 4]])
        y = nm.Matrix([[5, 6], [7, 8]])
        z = x * y
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(z._data, expected)

    def test_mul_scalar_by_vector(self):
        """Test multiplying scalar by vector"""
        x = nm.Real(2.0)
        y = nm.Vector([1, 2, 3])
        z = x * y
        np.testing.assert_array_equal(z._data.flatten(), [2, 4, 6])

    def test_mul_vector_by_scalar(self):
        """Test multiplying vector by scalar"""
        x = nm.Vector([1, 2, 3])
        y = nm.Real(2.0)
        z = x * y
        np.testing.assert_array_equal(z._data.flatten(), [2, 4, 6])

    def test_mul_python_int(self):
        """Test multiplying by Python int"""
        x = nm.Real(2.5)
        z = x * 3
        assert float(z._data) == 7.5

    def test_mul_python_int_reverse(self):
        """Test multiplying Python int by scalar (rmul)"""
        x = nm.Real(2.5)
        z = 3 * x
        assert float(z._data) == 7.5

    def test_mul_backward_scalars(self):
        """Test backward pass for scalar multiplication"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x * y
        z.backward()
        assert float(x.grad._data) == 3.0  # dz/dx = y
        assert float(y.grad._data) == 2.0  # dz/dy = x

    def test_mul_backward_vectors(self):
        """Test backward pass for element-wise vector multiplication"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        y = nm.Vector([4, 5, 6], requires_grad=True)
        z = x * y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [4, 5, 6])
        np.testing.assert_array_equal(y.grad._data.flatten(), [1, 2, 3])

    def test_mul_backward_scalar_by_vector(self):
        """Test backward pass for scalar * vector"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Vector([1, 2, 3], requires_grad=True)
        z = x * y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        assert float(x.grad._data) == 6.0  # sum([1, 2, 3])
        np.testing.assert_array_equal(y.grad._data.flatten(), [2, 2, 2])

    def test_mul_function(self):
        """Test nm.mul function"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = nm.mul(x, y)
        assert float(z._data) == 6.0


class TestDivision:
    """Test division operation"""

    def test_div_scalars(self):
        """Test dividing two scalars"""
        x = nm.Real(6.0)
        y = nm.Real(2.0)
        z = x / y
        assert float(z._data) == 3.0

    def test_div_vectors_elementwise(self):
        """Test element-wise division of vectors"""
        x = nm.Vector([12, 15, 18])
        y = nm.Vector([3, 5, 6])
        z = x / y
        np.testing.assert_array_equal(z._data.flatten(), [4, 3, 3])

    def test_div_matrices_elementwise(self):
        """Test element-wise division of matrices"""
        x = nm.Matrix([[12, 15], [18, 21]])
        y = nm.Matrix([[3, 5], [6, 7]])
        z = x / y
        expected = np.array([[4, 3], [3, 3]])
        np.testing.assert_array_equal(z._data, expected)

    def test_div_vector_by_scalar(self):
        """Test dividing vector by scalar"""
        x = nm.Vector([10, 20, 30])
        y = nm.Real(10.0)
        z = x / y
        np.testing.assert_array_equal(z._data.flatten(), [1, 2, 3])

    def test_div_scalar_by_vector(self):
        """Test dividing scalar by vector"""
        x = nm.Real(12.0)
        y = nm.Vector([2, 3, 4])
        z = x / y
        np.testing.assert_array_equal(z._data.flatten(), [6, 4, 3])

    def test_div_python_int(self):
        """Test dividing by Python int"""
        x = nm.Real(10.0)
        z = x / 2
        assert float(z._data) == 5.0

    def test_div_python_int_reverse(self):
        """Test dividing Python int by scalar (rdiv)"""
        x = nm.Real(2.0)
        z = 10 / x
        assert float(z._data) == 5.0

    def test_div_by_zero_gives_inf(self):
        """Test division by zero gives infinity"""
        x = nm.Real(1.0)
        y = nm.Real(0.0)
        z = x / y
        assert np.isinf(z._data)

    def test_div_backward_scalars(self):
        """Test backward pass for scalar division"""
        x = nm.Real(6.0, requires_grad=True)
        y = nm.Real(2.0, requires_grad=True)
        z = x / y
        z.backward()
        assert float(x.grad._data) == 0.5  # dz/dx = 1/y = 1/2
        assert float(y.grad._data) == -1.5  # dz/dy = -x/y^2 = -6/4

    def test_div_backward_vectors(self):
        """Test backward pass for element-wise vector division"""
        x = nm.Vector([12, 15, 18], requires_grad=True)
        y = nm.Vector([3, 5, 6], requires_grad=True)
        z = x / y
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        expected_x_grad = [1 / 3, 1 / 5, 1 / 6]
        np.testing.assert_array_almost_equal(x.grad._data.flatten(), expected_x_grad)
        expected_y_grad = [-12 / 9, -15 / 25, -18 / 36]
        np.testing.assert_array_almost_equal(y.grad._data.flatten(), expected_y_grad)

    def test_div_function(self):
        """Test nm.div function"""
        x = nm.Real(6.0)
        y = nm.Real(2.0)
        z = nm.div(x, y)
        assert float(z._data) == 3.0


class TestFloorDivision:
    """Test floor division operation"""

    def test_floordiv_scalars(self):
        """Test floor division of scalars"""
        x = nm.Real(7.0)
        y = nm.Real(2.0)
        z = x // y
        assert float(z._data) == 3.0

    def test_floordiv_vectors(self):
        """Test floor division of vectors"""
        x = nm.Vector([7, 8, 9])
        y = nm.Vector([2, 3, 4])
        z = x // y
        np.testing.assert_array_equal(z._data.flatten(), [3, 2, 2])

    def test_floordiv_python_int(self):
        """Test floor division by Python int"""
        x = nm.Real(7.5)
        z = x // 2
        assert float(z._data) == 3.0

    def test_floordiv_python_int_reverse(self):
        """Test floor division of Python int by scalar"""
        x = nm.Real(2.0)
        z = 7 // x
        assert float(z._data) == 3.0

    def test_floordiv_negative(self):
        """Test floor division with negative numbers"""
        x = nm.Real(-7.0)
        y = nm.Real(2.0)
        z = x // y
        assert float(z._data) == -4.0

    def test_floordiv_no_gradient(self):
        """Test floor division does not compute gradients"""
        x = nm.Real(7.0, requires_grad=True)
        y = nm.Real(2.0, requires_grad=True)
        z = x // y
        # Floor division is not differentiable
        assert z.requires_grad is False

    def test_floordiv_function(self):
        """Test nm.floordiv function"""
        x = nm.Real(7.0)
        y = nm.Real(2.0)
        z = nm.floordiv(x, y)
        assert float(z._data) == 3.0


class TestModulo:
    """Test modulo operation"""

    def test_mod_scalars(self):
        """Test modulo of scalars"""
        x = nm.Real(7.0)
        y = nm.Real(3.0)
        z = x % y
        assert float(z._data) == 1.0

    def test_mod_vectors(self):
        """Test modulo of vectors"""
        x = nm.Vector([7, 8, 9])
        y = nm.Vector([3, 3, 4])
        z = x % y
        np.testing.assert_array_equal(z._data.flatten(), [1, 2, 1])

    def test_mod_python_int(self):
        """Test modulo by Python int"""
        x = nm.Real(7.0)
        z = x % 3
        assert float(z._data) == 1.0

    def test_mod_python_int_reverse(self):
        """Test modulo of Python int by scalar"""
        x = nm.Real(3.0)
        z = 7 % x
        assert float(z._data) == 1.0

    def test_mod_negative(self):
        """Test modulo with negative numbers"""
        x = nm.Real(-7.0)
        y = nm.Real(3.0)
        z = x % y
        assert float(z._data) == 2.0  # Python-style modulo

    def test_mod_no_gradient(self):
        """Test modulo does not compute gradients"""
        x = nm.Real(7.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x % y
        assert z.requires_grad is False

    def test_mod_function(self):
        """Test nm.mod function"""
        x = nm.Real(7.0)
        y = nm.Real(3.0)
        z = nm.mod(x, y)
        assert float(z._data) == 1.0


class TestPower:
    """Test power operation"""

    def test_pow_scalars(self):
        """Test power of scalars"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = x**y
        assert float(z._data) == 8.0

    def test_pow_vectors_elementwise(self):
        """Test element-wise power of vectors"""
        x = nm.Vector([2, 3, 4])
        y = nm.Vector([2, 2, 2])
        z = x**y
        np.testing.assert_array_equal(z._data.flatten(), [4, 9, 16])

    def test_pow_vector_by_scalar(self):
        """Test raising vector to scalar power"""
        x = nm.Vector([2, 3, 4])
        z = x**2
        np.testing.assert_array_equal(z._data.flatten(), [4, 9, 16])

    def test_pow_scalar_to_vector(self):
        """Test raising scalar to vector power"""
        x = nm.Real(2.0)
        y = nm.Vector([1, 2, 3])
        z = x**y
        np.testing.assert_array_equal(z._data.flatten(), [2, 4, 8])

    def test_pow_python_int(self):
        """Test raising to Python int power"""
        x = nm.Real(2.0)
        z = x**3
        assert float(z._data) == 8.0

    def test_pow_python_int_reverse(self):
        """Test raising Python int to scalar power"""
        x = nm.Real(3.0)
        z = 2**x
        assert float(z._data) == 8.0

    def test_pow_square(self):
        """Test squaring (x**2)"""
        x = nm.Real(5.0)
        z = x**2
        assert float(z._data) == 25.0

    def test_pow_cube(self):
        """Test cubing (x**3)"""
        x = nm.Real(3.0)
        z = x**3
        assert float(z._data) == 27.0

    def test_pow_sqrt(self):
        """Test square root (x**0.5)"""
        x = nm.Real(4.0)
        z = x**0.5
        assert float(z._data) == 2.0

    def test_pow_inverse(self):
        """Test inverse (x**-1)"""
        x = nm.Real(4.0)
        z = x**-1
        assert float(z._data) == 0.25

    def test_pow_zero_exponent(self):
        """Test anything to power 0 is 1"""
        x = nm.Real(5.0)
        z = x**0
        assert float(z._data) == 1.0

    def test_pow_one_exponent(self):
        """Test anything to power 1 is itself"""
        x = nm.Real(5.0)
        z = x**1
        assert float(z._data) == 5.0

    def test_pow_backward_scalars(self):
        """Test backward pass for scalar power"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x**y
        z.backward()
        # dz/dx = y * x^(y-1) = 3 * 2^2 = 12
        assert float(x.grad._data) == 12.0
        # dz/dy = x^y * ln(x) = 8 * ln(2) ≈ 5.545
        assert abs(float(y.grad._data) - 8 * np.log(2)) < 1e-5

    def test_pow_backward_python_int_exponent(self):
        """Test backward pass with Python int exponent"""
        x = nm.Real(2.0, requires_grad=True)
        z = x**3
        z.backward()
        # dz/dx = 3 * x^2 = 3 * 4 = 12
        assert float(x.grad._data) == 12.0

    def test_pow_backward_square(self):
        """Test backward pass for squaring"""
        x = nm.Real(5.0, requires_grad=True)
        z = x**2
        z.backward()
        # dz/dx = 2 * x = 10
        assert float(x.grad._data) == 10.0

    def test_pow_function(self):
        """Test nm.pow function"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = nm.pow(x, y)
        assert float(z._data) == 8.0


class TestNegation:
    """Test negation operation"""

    def test_neg_scalar(self):
        """Test negating scalar"""
        x = nm.Real(5.0)
        z = -x
        assert float(z._data) == -5.0

    def test_neg_vector(self):
        """Test negating vector"""
        x = nm.Vector([1, -2, 3])
        z = -x
        np.testing.assert_array_equal(z._data.flatten(), [-1, 2, -3])

    def test_neg_matrix(self):
        """Test negating matrix"""
        x = nm.Matrix([[1, -2], [3, -4]])
        z = -x
        expected = np.array([[-1, 2], [-3, 4]])
        np.testing.assert_array_equal(z._data, expected)

    def test_neg_backward_scalar(self):
        """Test backward pass for scalar negation"""
        x = nm.Real(5.0, requires_grad=True)
        z = -x
        z.backward()
        assert float(x.grad._data) == -1.0

    def test_neg_backward_vector(self):
        """Test backward pass for vector negation"""
        x = nm.Vector([1, 2, 3], requires_grad=True)
        z = -x
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [-1, -1, -1])

    def test_neg_function(self):
        """Test nm.neg function"""
        x = nm.Real(5.0)
        z = nm.neg(x)
        assert float(z._data) == -5.0

    def test_double_negation(self):
        """Test double negation returns original"""
        x = nm.Real(5.0)
        z = -(-x)
        assert float(z._data) == 5.0


class TestAbsoluteValue:
    """Test absolute value operation"""

    def test_abs_positive_scalar(self):
        """Test abs of positive scalar"""
        x = nm.Real(5.0)
        z = abs(x)
        assert float(z._data) == 5.0

    def test_abs_negative_scalar(self):
        """Test abs of negative scalar"""
        x = nm.Real(-5.0)
        z = abs(x)
        assert float(z._data) == 5.0

    def test_abs_zero(self):
        """Test abs of zero"""
        x = nm.Real(0.0)
        z = abs(x)
        assert float(z._data) == 0.0

    def test_abs_vector(self):
        """Test abs of vector"""
        x = nm.Vector([1, -2, 3, -4])
        z = abs(x)
        np.testing.assert_array_equal(z._data.flatten(), [1, 2, 3, 4])

    def test_abs_matrix(self):
        """Test abs of matrix"""
        x = nm.Matrix([[1, -2], [-3, 4]])
        z = abs(x)
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(z._data, expected)

    def test_abs_backward_positive(self):
        """Test backward pass for abs with positive value"""
        x = nm.Real(5.0, requires_grad=True)
        z = abs(x)
        z.backward()
        assert float(x.grad._data) == 1.0  # sign(5) = 1

    def test_abs_backward_negative(self):
        """Test backward pass for abs with negative value"""
        x = nm.Real(-5.0, requires_grad=True)
        z = abs(x)
        z.backward()
        assert float(x.grad._data) == -1.0  # sign(-5) = -1

    def test_abs_backward_zero(self):
        """Test backward pass for abs at zero"""
        x = nm.Real(0.0, requires_grad=True)
        z = abs(x)
        z.backward()
        assert float(x.grad._data) == 0.0  # sign(0) = 0

    def test_abs_backward_vector(self):
        """Test backward pass for abs of vector"""
        x = nm.Vector([1, -2, 3], requires_grad=True)
        z = abs(x)
        grad_output = nm.Vector([1, 1, 1])
        z.backward(grad_output)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, -1, 1])


class TestInPlaceOperations:
    """Test in-place arithmetic operations"""

    def test_iadd_scalar(self):
        """Test in-place addition for scalar"""
        x = nm.Real(5.0)
        x += 3
        assert float(x._data) == 8.0

    def test_isub_scalar(self):
        """Test in-place subtraction for scalar"""
        x = nm.Real(5.0)
        x -= 3
        assert float(x._data) == 2.0

    def test_imul_scalar(self):
        """Test in-place multiplication for scalar"""
        x = nm.Real(5.0)
        x *= 3
        assert float(x._data) == 15.0

    def test_itruediv_scalar(self):
        """Test in-place true division for scalar"""
        x = nm.Real(6.0)
        x /= 2
        assert float(x._data) == 3.0

    def test_ifloordiv_scalar(self):
        """Test in-place floor division for scalar"""
        x = nm.Real(7.0)
        x //= 2
        assert float(x._data) == 3.0

    def test_imod_scalar(self):
        """Test in-place modulo for scalar"""
        x = nm.Real(7.0)
        x %= 3
        assert float(x._data) == 1.0

    def test_ipow_scalar(self):
        """Test in-place power for scalar"""
        x = nm.Real(2.0)
        x **= 3
        assert float(x._data) == 8.0

    def test_iadd_vector(self):
        """Test in-place addition for vector"""
        x = nm.Vector([1, 2, 3])
        x += nm.Vector([4, 5, 6])
        np.testing.assert_array_equal(x._data.flatten(), [5, 7, 9])

    def test_itruediv_integer_raises(self):
        """Test in-place true division raises for Integer"""
        x = nm.Integer(10)
        with pytest.raises(TypeError, match="In-place true division not supported"):
            x /= 2


class TestMixedTypeArithmetic:
    """Test arithmetic with mixed types"""

    def test_real_plus_integer(self):
        """Test Real + Integer"""
        r = nm.Real(3.14)
        i = nm.Integer(2)
        z = r + i
        assert isinstance(z, nm.Real)
        assert abs(float(z._data) - 5.14) < 1e-6

    def test_complex_plus_real(self):
        """Test Complex + Real"""
        c = nm.Complex(1 + 2j)
        r = nm.Real(3.0)
        z = c + r
        assert isinstance(z, nm.Complex)
        assert complex(z._data) == (4 + 2j)

    def test_vector_plus_scalar(self):
        """Test Vector + Scalar (broadcasting)"""
        v = nm.Vector([1, 2, 3])
        s = nm.Real(10.0)
        z = v + s
        np.testing.assert_array_equal(z._data.flatten(), [11, 12, 13])

    def test_matrix_times_scalar(self):
        """Test Matrix * Scalar (broadcasting)"""
        m = nm.Matrix([[1, 2], [3, 4]])
        s = nm.Real(2.0)
        z = m * s
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(z._data, expected)


class TestEdgeCases:
    """Test edge cases in arithmetic operations"""

    def test_zero_times_anything(self):
        """Test 0 * x = 0"""
        x = nm.Real(5.0)
        z = x * 0
        assert float(z._data) == 0.0

    def test_one_times_anything(self):
        """Test 1 * x = x"""
        x = nm.Real(5.0)
        z = x * 1
        assert float(z._data) == 5.0

    def test_anything_plus_zero(self):
        """Test x + 0 = x"""
        x = nm.Real(5.0)
        z = x + 0
        assert float(z._data) == 5.0

    def test_anything_minus_itself(self):
        """Test x - x = 0"""
        x = nm.Real(5.0)
        z = x - x
        assert float(z._data) == 0.0

    def test_inf_plus_inf(self):
        """Test inf + inf = inf"""
        x = nm.Real(float("inf"))
        y = nm.Real(float("inf"))
        z = x + y
        assert np.isinf(z._data)

    def test_nan_propagation(self):
        """Test NaN propagates through operations"""
        x = nm.Real(float("nan"))
        y = nm.Real(5.0)
        z = x + y
        assert np.isnan(z._data)
