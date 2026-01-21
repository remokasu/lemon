import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import unittest
import numpy as np

from lemon.numlib import *


class TestVector(unittest.TestCase):
    """Test cases for Vector operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.v1 = vector([1.0, 2.0, 3.0])
        self.v2 = vector([4.0, 5.0, 6.0])
        self.np_v1 = np.array([[1.0], [2.0], [3.0]])  # (3, 1)形状
        self.np_v2 = np.array([[4.0], [5.0], [6.0]])  # (3, 1)形状

    def test_vector_creation(self):
        """Test vector creation"""
        v = vector([1, 2, 3])
        self.assertEqual(v.shape, (3, 1))  # 修正: (3,) -> (3, 1)
        self.assertEqual(v.ndim, 2)  # 修正: 1 -> 2
        np.testing.assert_array_equal(v._data, np.array([[1], [2], [3]]))

    def test_vector_from_scalar(self):
        """Test vector creation from scalar"""
        v = vector([5.0])
        self.assertEqual(v.shape, (1, 1))  # 修正: (1,) -> (1, 1)
        self.assertEqual(v._data[0, 0], 5.0)  # 修正: v._data[0] -> v._data[0, 0]

    def test_vector_addition(self):
        """Test vector addition"""
        result = self.v1 + self.v2
        expected = self.np_v1 + self.np_v2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_subtraction(self):
        """Test vector subtraction"""
        result = self.v1 - self.v2
        expected = self.np_v1 - self.np_v2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_multiplication(self):
        """Test element-wise vector multiplication"""
        result = self.v1 * self.v2
        expected = self.np_v1 * self.np_v2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_division(self):
        """Test element-wise vector division"""
        result = self.v1 / self.v2
        expected = self.np_v1 / self.np_v2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_scalar_multiplication(self):
        """Test vector-scalar multiplication"""
        result = self.v1 * 2.0
        expected = self.np_v1 * 2.0
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_dot_product(self):
        """Test vector dot product (inner product)"""
        result = dot(self.v1, self.v2)
        expected = np.dot(self.np_v1.flatten(), self.np_v2.flatten())
        self.assertAlmostEqual(result._data.item(), expected, places=5)

    def test_vector_outer_product(self):
        """Test vector outer product"""
        rv = self.v1.T  # RowVector
        result = self.v1 @ rv  # Vector @ RowVector -> Matrix
        expected = self.np_v1 @ self.np_v1.T
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_transpose(self):
        """Test vector transpose"""
        v = vector([1, 2, 3])
        vT = v.T
        self.assertIsInstance(vT, RowVector)
        np.testing.assert_array_equal(vT._data, v._data.T)

    def test_vector_double_transpose(self):
        """Test double transpose returns to original"""
        v = vector([1, 2, 3])
        vTT = v.T.T
        self.assertIsInstance(vTT, Vector)
        np.testing.assert_array_equal(vTT._data, v._data)

    def test_vector_matrix_multiplication(self):
        """Test vector-matrix multiplication"""
        M = matrix([[1, 2], [3, 4], [5, 6]])
        v = vector([1, 2, 3])

        # Row vector @ Matrix -> Row vector
        result = v.T @ M  # (1, 3) @ (3, 2) -> (1, 2)
        expected = np.array([[1, 2, 3]]) @ np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_vector_multiplication(self):
        """Test matrix-vector multiplication"""
        M = matrix([[1, 2, 3], [4, 5, 6]])
        v = vector([1, 2, 3])

        # Matrix @ Vector -> Vector
        result = M @ v  # (2, 3) @ (3, 1) -> (2, 1)
        expected = np.array([[1, 2, 3], [4, 5, 6]]) @ np.array([[1], [2], [3]])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_norm(self):
        """Test vector norm (L2 norm)"""
        v = vector([3.0, 4.0])
        norm = sqrt(sum(v * v))
        expected = np.linalg.norm(np.array([3.0, 4.0]))
        self.assertAlmostEqual(norm._data.item(), expected, places=5)

    def test_vector_gradient(self):
        """Test gradient computation for vectors"""
        v = vector([1.0, 2.0, 3.0])
        y = sum(v**2)
        y.backward()

        # Gradient of sum(v^2) is 2*v
        expected_grad = 2.0 * np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_almost_equal(v.grad._data, expected_grad)

    def test_vector_gradient_chain(self):
        """Test gradient computation through chain of operations"""
        v = vector([1.0, 2.0, 3.0])
        y = sum(exp(v * 2.0))
        y.backward()

        # Gradient should be: 2 * exp(2*v)
        expected_grad = 2.0 * np.exp(2.0 * np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_array_almost_equal(v.grad._data, expected_grad, decimal=5)

    def test_vector_indexing(self):
        """Test vector indexing"""
        v = vector([1, 2, 3, 4, 5])

        # Single element
        self.assertEqual(v[2, 0]._data, 3)  # 修正: v[2] -> v[2, 0]

        # Slice
        result = v[1:4]
        np.testing.assert_array_equal(result._data, np.array([[2], [3], [4]]))

    def test_vector_negative(self):
        """Test vector negation"""
        v = vector([1.0, 2.0, 3.0])
        result = -v
        expected = -np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_abs(self):
        """Test absolute value of vector"""
        v = vector([-1.0, 2.0, -3.0])
        result = abs(v)
        expected = np.abs(np.array([[-1.0], [2.0], [-3.0]]))
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_power(self):
        """Test vector power operation"""
        v = vector([1.0, 2.0, 3.0])
        result = v**2
        expected = np.array([[1.0], [2.0], [3.0]]) ** 2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_vector_sum(self):
        """Test vector sum"""
        v = vector([1.0, 2.0, 3.0])
        result = v.sum()
        expected = np.sum(np.array([[1.0], [2.0], [3.0]]))
        self.assertAlmostEqual(result._data.item(), expected, places=5)

    def test_vector_mean(self):
        """Test vector mean"""
        v = vector([1.0, 2.0, 3.0])
        result = v.mean()
        expected = np.mean(np.array([[1.0], [2.0], [3.0]]))
        self.assertAlmostEqual(result._data.item(), expected, places=5)

    def test_vector_reshape(self):
        """Test vector reshape"""
        v = vector([1, 2, 3, 4, 5, 6])
        result = v.reshape(2, 3)
        expected = np.array([[1], [2], [3], [4], [5], [6]]).reshape(2, 3)
        np.testing.assert_array_equal(result._data, expected)

    def test_vector_concatenate(self):
        """Test vector concatenation"""
        v1 = vector([1, 2, 3])
        v2 = vector([4, 5, 6])
        result = concatenate([v1, v2], axis=0)  # 軸を明示
        expected = np.concatenate(
            [np.array([[1], [2], [3]]), np.array([[4], [5], [6]])], axis=0
        )
        np.testing.assert_array_equal(result._data, expected)

    def test_vector_mathematical_functions(self):
        """Test mathematical functions on vectors"""
        v = vector([1.0, 2.0, 3.0])

        # exp
        result_exp = exp(v)
        expected_exp = np.exp(np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_array_almost_equal(result_exp._data, expected_exp)

        # log
        result_log = log(v)
        expected_log = np.log(np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_array_almost_equal(result_log._data, expected_log)

        # sin
        result_sin = sin(v)
        expected_sin = np.sin(np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_array_almost_equal(result_sin._data, expected_sin)


class TestMatrix(unittest.TestCase):
    """Test cases for Matrix operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.m1 = matrix([[1.0, 2.0], [3.0, 4.0]])
        self.m2 = matrix([[5.0, 6.0], [7.0, 8.0]])
        self.np_m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.np_m2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    def test_matrix_creation(self):
        """Test matrix creation"""
        m = matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m.ndim, 2)
        np.testing.assert_array_equal(m._data, np.array([[1, 2], [3, 4]]))

    def test_matrix_from_1d(self):
        """Test matrix creation from 1D array"""
        m = matrix([1, 2, 3, 4])
        self.assertEqual(m.shape, (1, 4))

    def test_matrix_addition(self):
        """Test matrix addition"""
        result = self.m1 + self.m2
        expected = self.np_m1 + self.np_m2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_subtraction(self):
        """Test matrix subtraction"""
        result = self.m1 - self.m2
        expected = self.np_m1 - self.np_m2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_multiplication_elementwise(self):
        """Test element-wise matrix multiplication"""
        result = self.m1 * self.m2
        expected = self.np_m1 * self.np_m2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_multiplication_matmul(self):
        """Test matrix multiplication (matmul)"""
        result = self.m1 @ self.m2
        expected = self.np_m1 @ self.np_m2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_transpose(self):
        """Test matrix transpose"""
        m = matrix([[1, 2, 3], [4, 5, 6]])
        result = m.T
        expected = np.array([[1, 2, 3], [4, 5, 6]]).T
        np.testing.assert_array_equal(result._data, expected)

    def test_matrix_scalar_operations(self):
        """Test matrix scalar operations"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])

        # Multiplication
        result = m * 2.0
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) * 2.0
        np.testing.assert_array_almost_equal(result._data, expected)

        # Addition
        result = m + 1.0
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) + 1.0
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_indexing(self):
        """Test matrix indexing"""
        m = matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Single element
        self.assertEqual(m[1, 1]._data, 5)

        # Row slice
        result = m[0, :]
        np.testing.assert_array_equal(result._data, np.array([1, 2, 3]))

        # Column slice
        result = m[:, 1]
        np.testing.assert_array_equal(result._data, np.array([2, 5, 8]))

    def test_matrix_trace(self):
        """Test matrix trace"""
        m = matrix([[1, 2], [3, 4]])
        # Trace = sum of diagonal elements
        trace = m[0, 0] + m[1, 1]
        expected = np.trace(np.array([[1, 2], [3, 4]]))
        self.assertAlmostEqual(trace._data, expected, places=5)

    def test_matrix_determinant_2x2(self):
        """Test 2x2 matrix determinant calculation"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        # det = ad - bc
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        expected = np.linalg.det(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertAlmostEqual(det._data, expected, places=5)

    def test_matrix_gradient(self):
        """Test gradient computation for matrices"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        y = sum(m**2)
        y.backward()

        # Gradient of sum(m^2) is 2*m
        expected_grad = 2.0 * np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(m.grad, expected_grad)

    def test_matrix_chain_operations(self):
        """Test chain of matrix operations"""
        m1 = matrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = matrix([[2.0, 0.0], [0.0, 2.0]])

        result = (m1 @ m2) @ m1.T

        np_m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        np_m2 = np.array([[2.0, 0.0], [0.0, 2.0]])
        expected = (np_m1 @ np_m2) @ np_m1.T

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_reshape(self):
        """Test matrix reshape"""
        m = matrix([[1, 2, 3], [4, 5, 6]])
        result = m.reshape(3, 2)
        expected = np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
        np.testing.assert_array_equal(result._data, expected)

    def test_matrix_sum_operations(self):
        """Test matrix sum operations"""
        m = matrix([[1, 2, 3], [4, 5, 6]])

        # Sum all elements
        result_all = m.sum()
        expected_all = np.sum(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertAlmostEqual(result_all._data, expected_all, places=5)

        # Sum along axis 0 (columns)
        result_axis0 = m.sum(axis=0)
        expected_axis0 = np.sum(np.array([[1, 2, 3], [4, 5, 6]]), axis=0)
        np.testing.assert_array_almost_equal(result_axis0._data, expected_axis0)

        # Sum along axis 1 (rows)
        result_axis1 = m.sum(axis=1)
        expected_axis1 = np.sum(np.array([[1, 2, 3], [4, 5, 6]]), axis=1)
        np.testing.assert_array_almost_equal(result_axis1._data, expected_axis1)

    def test_matrix_mean_operations(self):
        """Test matrix mean operations"""
        m = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Mean all elements
        result_all = m.mean()
        expected_all = np.mean(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertAlmostEqual(result_all._data, expected_all, places=5)

        # Mean along axis 0
        result_axis0 = m.mean(axis=0)
        expected_axis0 = np.mean(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), axis=0)
        np.testing.assert_array_almost_equal(result_axis0._data, expected_axis0)

    def test_matrix_mathematical_functions(self):
        """Test mathematical functions on matrices"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])

        # exp
        result_exp = exp(m)
        expected_exp = np.exp(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_almost_equal(result_exp._data, expected_exp)

        # log
        result_log = log(m)
        expected_log = np.log(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_almost_equal(result_log._data, expected_log)

    def test_matrix_power(self):
        """Test matrix power operation (element-wise)"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        result = m**2
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) ** 2
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_negative(self):
        """Test matrix negation"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        result = -m
        expected = -np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_division(self):
        """Test matrix division"""
        m1 = matrix([[4.0, 6.0], [8.0, 10.0]])
        m2 = matrix([[2.0, 3.0], [4.0, 5.0]])
        result = m1 / m2
        expected = np.array([[4.0, 6.0], [8.0, 10.0]]) / np.array(
            [[2.0, 3.0], [4.0, 5.0]]
        )
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_matrix_identity(self):
        """Test identity matrix creation"""
        I = eye(3)
        expected = np.eye(3)
        np.testing.assert_array_equal(I._data, expected)

    def test_matrix_zeros_ones(self):
        """Test zeros and ones matrix creation"""
        zeros = zeros_matrix((2, 3))
        ones = ones_matrix((2, 3))

        np.testing.assert_array_equal(zeros._data, np.zeros((2, 3)))
        np.testing.assert_array_equal(ones._data, np.ones((2, 3)))

    def test_matrix_gradient_chain(self):
        """Test gradient computation through chain of matrix operations"""
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        y = sum(exp(m * 2.0))
        y.backward()

        # Gradient should be: 2 * exp(2*m)
        expected_grad = 2.0 * np.exp(2.0 * np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_almost_equal(m.grad, expected_grad, decimal=5)


class TestTensor(unittest.TestCase):
    """Test cases for Tensor operations (3D and higher)"""

    def setUp(self):
        """Set up test fixtures"""
        self.t3d = tensor(np.random.randn(2, 3, 4))
        self.t4d = tensor(np.random.randn(2, 3, 4, 5))
        self.t5d = tensor(np.random.randn(2, 3, 4, 5, 6))

    def test_tensor_3d_creation(self):
        """Test 3D tensor creation"""
        t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(t.shape, (2, 2, 2))
        self.assertEqual(t.ndim, 3)

    def test_tensor_4d_creation(self):
        """Test 4D tensor creation"""
        t = tensor(np.ones((2, 3, 4, 5)))
        self.assertEqual(t.shape, (2, 3, 4, 5))
        self.assertEqual(t.ndim, 4)

    def test_tensor_5d_creation(self):
        """Test 5D tensor creation"""
        t = tensor(np.ones((2, 3, 4, 5, 6)))
        self.assertEqual(t.shape, (2, 3, 4, 5, 6))
        self.assertEqual(t.ndim, 5)

    def test_tensor_3d_addition(self):
        """Test 3D tensor addition"""
        t1 = tensor(np.random.randn(2, 3, 4))
        t2 = tensor(np.random.randn(2, 3, 4))

        result = t1 + t2
        expected = t1._data + t2._data

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_4d_multiplication(self):
        """Test 4D tensor element-wise multiplication"""
        t1 = tensor(np.random.randn(2, 3, 4, 5))
        t2 = tensor(np.random.randn(2, 3, 4, 5))

        result = t1 * t2
        expected = t1._data * t2._data

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_broadcasting_3d(self):
        """Test broadcasting with 3D tensors"""
        t = tensor(np.random.randn(2, 3, 4))
        scalar = real(2.0)

        result = t * scalar
        expected = t._data * 2.0

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_sum_3d(self):
        """Test sum operation on 3D tensor"""
        t = tensor(np.random.randn(2, 3, 4))

        # Sum all elements
        result_all = t.sum()
        expected_all = np.sum(t._data)
        self.assertAlmostEqual(result_all._data, expected_all, places=5)

        # Sum along axis 0
        result_axis0 = t.sum(axis=0)
        expected_axis0 = np.sum(t._data, axis=0)
        np.testing.assert_array_almost_equal(result_axis0._data, expected_axis0)

        # Sum along axis 1
        result_axis1 = t.sum(axis=1)
        expected_axis1 = np.sum(t._data, axis=1)
        np.testing.assert_array_almost_equal(result_axis1._data, expected_axis1)

        # Sum along axis 2
        result_axis2 = t.sum(axis=2)
        expected_axis2 = np.sum(t._data, axis=2)
        np.testing.assert_array_almost_equal(result_axis2._data, expected_axis2)

    def test_tensor_mean_4d(self):
        """Test mean operation on 4D tensor"""
        t = tensor(np.random.randn(2, 3, 4, 5))

        # Mean all elements
        result_all = t.mean()
        expected_all = np.mean(t._data)
        self.assertAlmostEqual(result_all._data, expected_all, places=5)

        # Mean along axis 2
        result_axis2 = t.mean(axis=2)
        expected_axis2 = np.mean(t._data, axis=2)
        np.testing.assert_array_almost_equal(result_axis2._data, expected_axis2)

    def test_tensor_reshape_3d(self):
        """Test reshape operation on 3D tensor"""
        t = tensor(np.random.randn(2, 3, 4))

        # Reshape to 2D
        result = t.reshape(6, 4)
        expected = t._data.reshape(6, 4)
        np.testing.assert_array_almost_equal(result._data, expected)

        # Reshape to different 3D
        result = t.reshape(4, 3, 2)
        expected = t._data.reshape(4, 3, 2)
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_indexing_3d(self):
        """Test indexing on 3D tensor"""
        t = tensor(np.arange(24).reshape(2, 3, 4))

        # Single element
        result = t[1, 2, 3]
        expected = np.arange(24).reshape(2, 3, 4)[1, 2, 3]
        self.assertEqual(result._data, expected)

        # Slice
        result = t[0, :, :]
        expected = np.arange(24).reshape(2, 3, 4)[0, :, :]
        np.testing.assert_array_equal(result._data, expected)

    def test_tensor_transpose_3d(self):
        """Test transpose on 3D tensor"""
        t = tensor(np.random.randn(2, 3, 4))

        # Standard transpose
        result = t.T
        expected = t._data.T
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_mathematical_functions_3d(self):
        """Test mathematical functions on 3D tensors"""
        t = tensor(np.abs(np.random.randn(2, 3, 4)) + 0.1)

        # exp
        result_exp = exp(t)
        expected_exp = np.exp(t._data)
        np.testing.assert_array_almost_equal(result_exp._data, expected_exp)

        # log
        result_log = log(t)
        expected_log = np.log(t._data)
        np.testing.assert_array_almost_equal(result_log._data, expected_log)

        # sin
        result_sin = sin(t)
        expected_sin = np.sin(t._data)
        np.testing.assert_array_almost_equal(result_sin._data, expected_sin)

    def test_tensor_gradient_3d(self):
        """Test gradient computation for 3D tensors"""
        t = tensor(np.random.randn(2, 3, 4))
        y = sum(t**2)
        y.backward()

        # Gradient of sum(t^2) is 2*t
        expected_grad = 2.0 * t._data
        np.testing.assert_array_almost_equal(t.grad, expected_grad)

    def test_tensor_gradient_4d(self):
        """Test gradient computation for 4D tensors"""
        t = tensor(np.random.randn(2, 3, 4, 5))
        y = mean(t**2)
        y.backward()

        # Gradient of mean(t^2) is 2*t / n
        n = t.size
        expected_grad = 2.0 * t._data / n
        np.testing.assert_array_almost_equal(t.grad, expected_grad, decimal=5)

    def test_tensor_chain_operations_3d(self):
        """Test chain of operations on 3D tensors"""
        t = tensor(np.random.randn(2, 3, 4))
        result = sum(exp(t * 2.0))
        result.backward()

        # Gradient should be: 2 * exp(2*t)
        expected_grad = 2.0 * np.exp(2.0 * t._data)
        np.testing.assert_array_almost_equal(t.grad, expected_grad, decimal=5)

    def test_tensor_batch_matmul(self):
        """Test batch matrix multiplication"""
        # (batch, n, m) @ (batch, m, p) = (batch, n, p)
        t1 = tensor(np.random.randn(3, 4, 5))
        t2 = tensor(np.random.randn(3, 5, 6))

        result = bmm(t1, t2)
        expected = t1._data @ t2._data

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_concatenate_3d(self):
        """Test concatenation of 3D tensors"""
        t1 = tensor(np.random.randn(2, 3, 4))
        t2 = tensor(np.random.randn(2, 3, 4))

        # Concatenate along axis 0
        result = concatenate([t1, t2], axis=0)
        expected = np.concatenate([t1._data, t2._data], axis=0)
        np.testing.assert_array_almost_equal(result._data, expected)

        # Concatenate along axis 1
        result = concatenate([t1, t2], axis=1)
        expected = np.concatenate([t1._data, t2._data], axis=1)
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_stack(self):
        """Test stacking of tensors"""
        t1 = tensor(np.random.randn(3, 4))
        t2 = tensor(np.random.randn(3, 4))

        result = stack([t1, t2], axis=0)
        expected = np.stack([t1._data, t2._data], axis=0)

        np.testing.assert_array_almost_equal(result._data, expected)
        self.assertEqual(result.shape, (2, 3, 4))

    def test_tensor_einsum(self):
        """Test Einstein summation"""
        # Matrix multiplication using einsum
        a = tensor(np.random.randn(3, 4))
        b = tensor(np.random.randn(4, 5))

        result = einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", a._data, b._data)

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_tensordot(self):
        """Test tensor dot product"""
        a = tensor(np.random.randn(3, 4, 5))
        b = tensor(np.random.randn(5, 6))

        result = tensordot(a, b, axes=1)
        expected = np.tensordot(a._data, b._data, axes=1)

        np.testing.assert_array_almost_equal(result._data, expected)

    def test_tensor_5d_operations(self):
        """Test operations on 5D tensors"""
        t1 = tensor(np.random.randn(2, 3, 4, 5, 6))
        t2 = tensor(np.random.randn(2, 3, 4, 5, 6))

        # Addition
        result_add = t1 + t2
        expected_add = t1._data + t2._data
        np.testing.assert_array_almost_equal(result_add._data, expected_add)

        # Multiplication
        result_mul = t1 * t2
        expected_mul = t1._data * t2._data
        np.testing.assert_array_almost_equal(result_mul._data, expected_mul)

        # Sum
        result_sum = t1.sum()
        expected_sum = np.sum(t1._data)
        self.assertAlmostEqual(result_sum._data, expected_sum, places=5)

    def test_tensor_complex_operations(self):
        """Test complex chain of tensor operations"""
        t = tensor(np.random.randn(2, 3, 4))

        # Complex operation: (t^2 + t) * exp(t)
        result = (t**2 + t) * exp(t)
        result_sum = sum(result)
        result_sum.backward()

        # Verify gradient exists and has correct shape
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_tensor_keepdims(self):
        """Test keepdims parameter in reduction operations"""
        t = tensor(np.random.randn(2, 3, 4))

        # Sum with keepdims
        result = t.sum(axis=1, keepdims=True)
        expected = np.sum(t._data, axis=1, keepdims=True)

        np.testing.assert_array_almost_equal(result._data, expected)
        self.assertEqual(result.shape, (2, 1, 4))

        # Mean with keepdims
        result = t.mean(axis=2, keepdims=True)
        expected = np.mean(t._data, axis=2, keepdims=True)

        np.testing.assert_array_almost_equal(result._data, expected)
        self.assertEqual(result.shape, (2, 3, 1))


class TestIntegration(unittest.TestCase):
    """Integration tests for Vector, Matrix, and Tensor interactions"""

    def test_vector_matrix_tensor_interaction(self):
        """Test interaction between vectors, matrices, and tensors"""
        v = vector([1.0, 2.0, 3.0])
        m = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = tensor(np.random.randn(2, 3, 4))

        # Vector-matrix multiplication
        result1 = m @ v
        self.assertIsInstance(result1, Vector)

        # Matrix-vector multiplication
        result2 = v.T @ m.T
        self.assertIsInstance(result2, RowVector)

    def test_broadcasting_between_types(self):
        """Test broadcasting between different types"""
        v = vector([1.0, 2.0, 3.0]).T
        m = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Vector broadcast to matrix
        result = m + v
        expected = m._data + v._data
        np.testing.assert_array_almost_equal(result._data, expected)

    def test_mixed_type_gradients(self):
        """Test gradient computation with mixed types"""
        v = vector([1.0, 2.0, 3.0])
        m = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Matrix @ Vector の場合をテスト
        # m: (2, 3), v: (3,) -> result: (2,)
        result_vec = m @ v
        loss = sum(result_vec**2)
        loss.backward()

        # Both should have gradients
        self.assertIsNotNone(v.grad)
        self.assertIsNotNone(m.grad)

        # Verify gradient shapes
        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(m.grad.shape, m.shape)

    def test_complex_computational_graph(self):
        """Test complex computational graph with multiple types"""
        v1 = vector([1.0, 2.0])
        v2 = vector([3.0, 4.0])
        m = matrix([[1.0, 2.0], [3.0, 4.0]])

        # Build complex graph
        result1 = m @ v1  # (2, 2) @ (2,) = (2,)
        result2 = m.T @ v2  # (2, 2) @ (2,) = (2,)
        final = sum(result1 * result2)

        final.backward()

        # All should have gradients
        self.assertIsNotNone(v1.grad)
        self.assertIsNotNone(v2.grad)
        self.assertIsNotNone(m.grad)

    def test_batch_processing(self):
        """Test batch processing with tensors"""
        # Simplified batch processing test
        batch_size = 4
        batch_results = []

        for i in range(batch_size):
            x = vector(np.random.randn(10))
            W = matrix(np.random.randn(10, 5))

            result = W.T @ x
            batch_results.append(result)

        # Verify all have correct shape
        for result in batch_results:
            self.assertEqual(result.shape, (5, 1))

    def test_loss_computation_integration(self):
        """Test complete forward and backward pass - simplified"""
        # Matrix @ Vector のシンプルなケース
        x = vector([1.0, 2.0, 3.0])
        W = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Forward pass
        # W: (2, 3), x: (3,) -> y: (2,)
        y = W @ x
        loss = sum(y**2)

        # Backward pass
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(W.grad)

        # Verify gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(W.grad.shape, W.shape)

    def test_type_preservation(self):
        """Test that types are preserved through operations"""
        v = vector([1.0, 2.0, 3.0])
        m = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Vector operations should return vectors
        v_result = v + v
        self.assertIsInstance(v_result, Vector)

        # Matrix operations should return matrices
        m_result = m @ m.T
        self.assertIsInstance(m_result, Matrix)

    def test_dimensional_consistency(self):
        """Test dimensional consistency across operations"""
        # Create tensors of different dimensions
        t2 = matrix(np.random.randn(3, 4))
        t3 = tensor(np.random.randn(2, 3, 4))
        t4 = tensor(np.random.randn(5, 2, 3, 4))

        # Verify shapes are maintained
        self.assertEqual(t2.shape, (3, 4))
        self.assertEqual(t3.shape, (2, 3, 4))
        self.assertEqual(t4.shape, (5, 2, 3, 4))

        # Verify operations maintain shapes
        result2 = t2 * 2
        result3 = t3 * 2
        result4 = t4 * 2

        self.assertEqual(result2.shape, (3, 4))
        self.assertEqual(result3.shape, (2, 3, 4))
        self.assertEqual(result4.shape, (5, 2, 3, 4))

    def test_matrix_vector_gradient_detailed(self):
        """Detailed test for Matrix @ Vector gradient computation"""
        # 簡単なケースで手計算で検証
        m = matrix([[1.0, 2.0], [3.0, 4.0]])
        v = vector([5.0, 6.0])

        # Forward: m @ v = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        result = m @ v
        np.testing.assert_array_almost_equal(result._data, np.array([[17.0], [39.0]]))

        # Loss = sum(result) = 17 + 39 = 56
        loss = sum(result)

        # Backward
        loss.backward()

        # Gradient of m should be outer([1, 1], v) = [[5, 6], [5, 6]]
        expected_m_grad = np.array([[5.0, 6.0], [5.0, 6.0]])
        np.testing.assert_array_almost_equal(m.grad, expected_m_grad)

        # Gradient of v should be m^T @ [1, 1] = [1+3, 2+4] = [4, 6]
        expected_v_grad = np.array([[4.0], [6.0]])
        np.testing.assert_array_almost_equal(v.grad, expected_v_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
