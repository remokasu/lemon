import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestSum:
    """Test sum reduction"""

    def test_sum_vector_all(self):
        """Test sum of entire vector"""
        v = nm.Vector([1, 2, 3, 4])
        result = nm.sum(v)
        assert float(result._data) == 10.0

    def test_sum_matrix_all(self):
        """Test sum of entire matrix"""
        m = nm.Matrix([[1, 2], [3, 4]])
        result = nm.sum(m)
        assert float(result._data) == 10.0

    def test_sum_matrix_axis0(self):
        """Test sum along axis 0 (columns)"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.sum(m, axis=0)
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result._data, expected)

    def test_sum_matrix_axis1(self):
        """Test sum along axis 1 (rows)"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.sum(m, axis=1)
        expected = np.array([6, 15])
        np.testing.assert_array_equal(result._data, expected)

    def test_sum_keepdims_true(self):
        """Test sum with keepdims=True"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.sum(m, axis=1, keepdims=True)
        assert result.shape == (2, 1)
        expected = np.array([[6], [15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_sum_keepdims_false(self):
        """Test sum with keepdims=False (default)"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.sum(m, axis=1, keepdims=False)
        assert result.shape == (2,)

    def test_sum_backward_vector(self):
        """Test backward pass for vector sum"""
        v = nm.Vector([1, 2, 3], requires_grad=True)
        result = nm.sum(v)
        result.backward()
        # Gradient should be all ones
        np.testing.assert_array_equal(v.grad._data.flatten(), [1, 1, 1])

    def test_sum_backward_matrix(self):
        """Test backward pass for matrix sum"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        result = nm.sum(m)
        result.backward()
        # Gradient should be all ones
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_sum_backward_axis0(self):
        """Test backward pass for sum along axis 0"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        result = nm.sum(m, axis=0)
        grad_output = nm.Tensor([1, 1, 1])
        result.backward(grad_output)
        # Gradient should broadcast to original shape
        expected = np.array([[1, 1, 1], [1, 1, 1]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_sum_backward_axis1(self):
        """Test backward pass for sum along axis 1"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        result = nm.sum(m, axis=1)
        grad_output = nm.Tensor([1, 2])
        result.backward(grad_output)
        # Gradient should broadcast to original shape
        expected = np.array([[1, 1, 1], [2, 2, 2]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_sum_method(self):
        """Test sum as method"""
        v = nm.Vector([1, 2, 3])
        result = v.sum()
        assert float(result._data) == 6.0


class TestMean:
    """Test mean reduction"""

    def test_mean_vector_all(self):
        """Test mean of entire vector"""
        v = nm.Vector([1, 2, 3, 4])
        result = nm.mean(v)
        assert float(result._data) == 2.5

    def test_mean_matrix_all(self):
        """Test mean of entire matrix"""
        m = nm.Matrix([[1, 2], [3, 4]])
        result = nm.mean(m)
        assert float(result._data) == 2.5

    def test_mean_matrix_axis0(self):
        """Test mean along axis 0"""
        m = nm.Matrix([[1, 2, 3], [5, 6, 7]])
        result = nm.mean(m, axis=0)
        expected = np.array([3, 4, 5])
        np.testing.assert_array_equal(result._data, expected)

    def test_mean_matrix_axis1(self):
        """Test mean along axis 1"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.mean(m, axis=1)
        expected = np.array([2, 5])
        np.testing.assert_array_equal(result._data, expected)

    def test_mean_keepdims_true(self):
        """Test mean with keepdims=True"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.mean(m, axis=1, keepdims=True)
        assert result.shape == (2, 1)
        expected = np.array([[2], [5]])
        np.testing.assert_array_equal(result._data, expected)

    def test_mean_backward_vector(self):
        """Test backward pass for vector mean"""
        v = nm.Vector([1, 2, 3], requires_grad=True)
        result = nm.mean(v)
        result.backward()
        # Gradient should be 1/n for each element
        expected = [1 / 3, 1 / 3, 1 / 3]
        np.testing.assert_array_almost_equal(v.grad._data.flatten(), expected)

    def test_mean_backward_matrix(self):
        """Test backward pass for matrix mean"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        result = nm.mean(m)
        result.backward()
        # Gradient should be 1/4 for each element
        expected = np.array([[0.25, 0.25], [0.25, 0.25]])
        np.testing.assert_array_almost_equal(m.grad._data, expected)

    def test_mean_backward_axis0(self):
        """Test backward pass for mean along axis 0"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        result = nm.mean(m, axis=0)
        grad_output = nm.Tensor([1, 1, 1])
        result.backward(grad_output)
        # Gradient should be 1/2 (2 rows) for each element
        expected = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        np.testing.assert_array_almost_equal(m.grad._data, expected)

    def test_mean_backward_axis1(self):
        """Test backward pass for mean along axis 1"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        result = nm.mean(m, axis=1)
        grad_output = nm.Tensor([1, 1])
        result.backward(grad_output)
        # Gradient should be 1/3 (3 columns) for each element
        expected = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
        np.testing.assert_array_almost_equal(m.grad._data, expected)

    def test_mean_method(self):
        """Test mean as method"""
        v = nm.Vector([2, 4, 6])
        result = v.mean()
        assert float(result._data) == 4.0


class TestMaximum:
    """Test maximum operation"""

    def test_maximum_scalars(self):
        """Test maximum of two scalars"""
        x = nm.Real(3.0)
        y = nm.Real(5.0)
        result = nm.maximum(x, y)
        assert float(result._data) == 5.0

    def test_maximum_vectors(self):
        """Test element-wise maximum of vectors"""
        v1 = nm.Vector([1, 5, 3])
        v2 = nm.Vector([4, 2, 6])
        result = nm.maximum(v1, v2)
        expected = np.array([4, 5, 6])
        np.testing.assert_array_equal(result._data.flatten(), expected)

    def test_maximum_matrices(self):
        """Test element-wise maximum of matrices"""
        m1 = nm.Matrix([[1, 2], [5, 6]])
        m2 = nm.Matrix([[3, 1], [4, 7]])
        result = nm.maximum(m1, m2)
        expected = np.array([[3, 2], [5, 7]])
        np.testing.assert_array_equal(result._data, expected)

    def test_maximum_broadcast(self):
        """Test maximum with broadcasting"""
        v = nm.Vector([1, 2, 3])
        s = nm.Real(2.5)
        result = nm.maximum(v, s)
        expected = np.array([2.5, 2.5, 3])
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)

    def test_maximum_negative_values(self):
        """Test maximum with negative values"""
        x = nm.Real(-3.0)
        y = nm.Real(-1.0)
        result = nm.maximum(x, y)
        assert float(result._data) == -1.0


class TestMinimum:
    """Test minimum operation"""

    def test_minimum_scalars(self):
        """Test minimum of two scalars"""
        x = nm.Real(3.0)
        y = nm.Real(5.0)
        result = nm.minimum(x, y)
        assert float(result._data) == 3.0

    def test_minimum_vectors(self):
        """Test element-wise minimum of vectors"""
        v1 = nm.Vector([1, 5, 3])
        v2 = nm.Vector([4, 2, 6])
        result = nm.minimum(v1, v2)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result._data.flatten(), expected)

    def test_minimum_matrices(self):
        """Test element-wise minimum of matrices"""
        m1 = nm.Matrix([[1, 2], [5, 6]])
        m2 = nm.Matrix([[3, 1], [4, 7]])
        result = nm.minimum(m1, m2)
        expected = np.array([[1, 1], [4, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_minimum_broadcast(self):
        """Test minimum with broadcasting"""
        v = nm.Vector([1, 2, 3])
        s = nm.Real(2.5)
        result = nm.minimum(v, s)
        expected = np.array([1, 2, 2.5])
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)

    def test_minimum_negative_values(self):
        """Test minimum with negative values"""
        x = nm.Real(-3.0)
        y = nm.Real(-1.0)
        result = nm.minimum(x, y)
        assert float(result._data) == -3.0


class TestTensorReductionMethods:
    """Test reduction methods on Tensor class"""

    def test_max_method_vector(self):
        """Test max method on vector"""
        v = nm.Vector([1, 5, 3, 2])
        result = v.max()
        assert float(result._data) == 5.0

    def test_max_method_matrix(self):
        """Test max method on matrix"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.max()
        assert float(result._data) == 6.0

    def test_max_method_axis0(self):
        """Test max method along axis 0"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.max(axis=0)
        expected = np.array([6, 5, 4])
        np.testing.assert_array_equal(result._data, expected)

    def test_max_method_axis1(self):
        """Test max method along axis 1"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.max(axis=1)
        expected = np.array([3, 6])
        np.testing.assert_array_equal(result._data, expected)

    def test_max_method_keepdims(self):
        """Test max method with keepdims"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.max(axis=1, keepdims=True)
        assert result.shape == (2, 1)
        expected = np.array([[3], [6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_min_method_vector(self):
        """Test min method on vector"""
        v = nm.Vector([1, 5, 3, 2])
        result = v.min()
        assert float(result._data) == 1.0

    def test_min_method_matrix(self):
        """Test min method on matrix"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.min()
        assert float(result._data) == 1.0

    def test_min_method_axis0(self):
        """Test min method along axis 0"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.min(axis=0)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result._data, expected)

    def test_argmax_method_vector(self):
        """Test argmax method on vector"""
        v = nm.Vector([1, 5, 3, 2])
        result = v.argmax()
        assert result == 1  # Index of maximum value

    def test_argmax_method_matrix(self):
        """Test argmax method on matrix (flattened)"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.argmax()
        assert result == 3  # Index 3 in flattened array

    def test_argmax_method_axis0(self):
        """Test argmax method along axis 0"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.argmax(axis=0)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result._data, expected)

    def test_argmin_method_vector(self):
        """Test argmin method on vector"""
        v = nm.Vector([1, 5, 3, 2])
        result = v.argmin()
        assert result == 0  # Index of minimum value

    def test_argmin_method_matrix(self):
        """Test argmin method on matrix (flattened)"""
        m = nm.Matrix([[1, 2, 3], [6, 5, 4]])
        result = m.argmin()
        assert result == 0  # Index 0 in flattened array

    def test_all_method_true(self):
        """Test all method when all elements are true"""
        v = nm.Vector([1, 2, 3])
        result = v.all()
        assert result is True

    def test_all_method_false(self):
        """Test all method when some element is false"""
        v = nm.Vector([1, 0, 3])
        result = v.all()
        assert result is False

    def test_any_method_true(self):
        """Test any method when at least one element is true"""
        v = nm.Vector([0, 0, 1])
        result = v.any()
        assert result is True

    def test_any_method_false(self):
        """Test any method when all elements are false"""
        v = nm.Vector([0, 0, 0])
        result = v.any()
        assert result is False


class TestReductionEdgeCases:
    """Test edge cases for reduction operations"""

    def test_sum_empty_array(self):
        """Test sum of empty array"""
        v = nm.Tensor(np.array([]))
        result = nm.sum(v)
        assert float(result._data) == 0.0

    def test_mean_single_element(self):
        """Test mean of single element"""
        v = nm.Vector([5])
        result = nm.mean(v)
        assert float(result._data) == 5.0

    def test_sum_all_zeros(self):
        """Test sum of all zeros"""
        v = nm.Vector([0, 0, 0])
        result = nm.sum(v)
        assert float(result._data) == 0.0

    def test_sum_negative_values(self):
        """Test sum with negative values"""
        v = nm.Vector([-1, -2, -3])
        result = nm.sum(v)
        assert float(result._data) == -6.0

    def test_mean_large_values(self):
        """Test mean with large values"""
        v = nm.Vector([1e10, 2e10, 3e10])
        result = nm.mean(v)
        assert float(result._data) == 2e10
