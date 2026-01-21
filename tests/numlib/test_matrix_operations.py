import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestMatmul:
    """Test matrix multiplication (matmul)"""

    def test_matmul_matrices(self):
        """Test matrix @ matrix"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        C = nm.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(C._data, expected)

    def test_matmul_matrix_vector(self):
        """Test matrix @ vector"""
        A = nm.Matrix([[1, 2], [3, 4]])
        v = nm.Vector([5, 6])
        result = nm.matmul(A, v)
        assert isinstance(result, nm.Vector)
        assert result.shape == (2, 1)
        expected = np.array([[17], [39]])
        np.testing.assert_array_equal(result._data, expected)

    def test_matmul_vector_rowvector(self):
        """Test vector @ rowvector (outer product)"""
        v = nm.Vector([1, 2, 3])
        rv = nm.RowVector([4, 5])
        result = nm.matmul(v, rv)
        assert isinstance(result, nm.Matrix)
        assert result.shape == (3, 2)
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_matmul_rowvector_vector(self):
        """Test rowvector @ vector (inner product)"""
        rv = nm.RowVector([1, 2, 3])
        v = nm.Vector([4, 5, 6])
        result = nm.matmul(rv, v)
        assert isinstance(result, nm.Scalar)
        assert float(result._data) == 32.0

    def test_matmul_rowvector_matrix(self):
        """Test rowvector @ matrix"""
        rv = nm.RowVector([1, 2])
        A = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.matmul(rv, A)
        assert isinstance(result, nm.RowVector)
        assert result.shape == (1, 3)
        expected = np.array([[9, 12, 15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_matmul_backward_matrices(self):
        """Test backward pass for matrix multiplication"""
        A = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        B = nm.Matrix([[5, 6], [7, 8]], requires_grad=True)
        C = nm.matmul(A, B)
        grad_output = nm.Matrix([[1, 1], [1, 1]])
        C.backward(grad_output)

        # dL/dA = grad_output @ B.T
        expected_A_grad = np.array([[11, 15], [11, 15]])
        np.testing.assert_array_equal(A.grad._data, expected_A_grad)

        # dL/dB = A.T @ grad_output
        expected_B_grad = np.array([[4, 4], [6, 6]])
        np.testing.assert_array_equal(B.grad._data, expected_B_grad)

    def test_matmul_backward_matrix_vector(self):
        """Test backward pass for matrix @ vector"""
        A = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        v = nm.Vector([5, 6], requires_grad=True)
        result = nm.matmul(A, v)
        grad_output = nm.Vector([1, 1])
        result.backward(grad_output)

        # dL/dA = grad_output @ v.T
        expected_A_grad = np.array([[5, 6], [5, 6]])
        np.testing.assert_array_equal(A.grad._data, expected_A_grad)

        # dL/dv = A.T @ grad_output
        expected_v_grad = np.array([[4], [6]])
        np.testing.assert_array_equal(v.grad._data, expected_v_grad)

    def test_matmul_dimension_mismatch_raises(self):
        """Test matmul with incompatible dimensions raises error"""
        A = nm.Matrix([[1, 2, 3]])  # (1, 3)
        B = nm.Matrix([[4, 5], [6, 7]])  # (2, 2)
        with pytest.raises((ValueError, Exception)):
            nm.matmul(A, B)


class TestDot:
    """Test dot product"""

    def test_dot_vectors(self):
        """Test dot product of two vectors"""
        v1 = nm.Vector([1, 2, 3])
        v2 = nm.Vector([4, 5, 6])
        result = nm.dot(v1, v2)
        assert isinstance(result, nm.Scalar)
        assert float(result._data) == 32.0  # 1*4 + 2*5 + 3*6

    def test_dot_rowvector_vector(self):
        """Test dot product of rowvector and vector"""
        rv = nm.RowVector([1, 2, 3])
        v = nm.Vector([4, 5, 6])
        result = nm.dot(rv, v)
        assert float(result._data) == 32.0

    def test_dot_1d_arrays(self):
        """Test dot product of 1D tensors"""
        t1 = nm.Tensor([1, 2, 3])
        t2 = nm.Tensor([4, 5, 6])
        result = nm.dot(t1, t2)
        assert float(result._data) == 32.0

    def test_dot_backward_vectors(self):
        """Test backward pass for dot product"""
        v1 = nm.Vector([1, 2, 3], requires_grad=True)
        v2 = nm.Vector([4, 5, 6], requires_grad=True)
        result = nm.dot(v1, v2)
        result.backward()

        # dL/dv1 = v2, dL/dv2 = v1
        np.testing.assert_array_equal(v1.grad._data.flatten(), [4, 5, 6])
        np.testing.assert_array_equal(v2.grad._data.flatten(), [1, 2, 3])

    def test_dot_matrices_2d(self):
        """Test dot product with 2D matrices"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        result = nm.dot(A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result._data, expected)


class TestBMM:
    """Test batch matrix multiplication"""

    def test_bmm_basic(self):
        """Test basic batch matrix multiplication"""
        # Batch of 2 matrices: (2, 2, 3) @ (2, 3, 2) = (2, 2, 2)
        A = nm.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        B = nm.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
        result = nm.bmm(A, B)
        assert result.shape == (2, 2, 2)

    def test_bmm_batch_size_mismatch_raises(self):
        """Test bmm with different batch sizes raises error"""
        A = nm.Tensor([[[1, 2], [3, 4]]])  # batch=1
        B = nm.Tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]])  # batch=2
        with pytest.raises(ValueError, match="Batch size mismatch"):
            nm.bmm(A, B)

    def test_bmm_dimension_mismatch_raises(self):
        """Test bmm with incompatible inner dimensions raises error"""
        A = nm.Tensor([[[1, 2, 3]], [[4, 5, 6]]])  # (2, 1, 3)
        B = nm.Tensor([[[7, 8]], [[9, 10]]])  # (2, 2, 2) - mismatch!
        with pytest.raises(ValueError, match="Inner dimensions mismatch"):
            nm.bmm(A, B)

    def test_bmm_not_3d_raises(self):
        """Test bmm with non-3D tensor raises error"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        with pytest.raises(ValueError, match="must be a 3D tensor"):
            nm.bmm(A, B)


class TestEinsum:
    """Test Einstein summation"""

    def test_einsum_matrix_multiplication(self):
        """Test einsum for matrix multiplication"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        result = nm.einsum("ij,jk->ik", A, B)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result._data, expected)

    def test_einsum_trace(self):
        """Test einsum for matrix trace"""
        A = nm.Matrix([[1, 2], [3, 4]])
        result = nm.einsum("ii->", A)
        assert float(result._data) == 5.0  # 1 + 4

    def test_einsum_transpose(self):
        """Test einsum for transpose"""
        A = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.einsum("ij->ji", A)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_einsum_outer_product(self):
        """Test einsum for outer product"""
        v1 = nm.Vector([1, 2, 3])
        v2 = nm.Vector([4, 5])
        result = nm.einsum("i,j->ij", v1, v2)
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_einsum_element_wise_product(self):
        """Test einsum for element-wise product"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        result = nm.einsum("ij,ij->ij", A, B)
        expected = np.array([[5, 12], [21, 32]])
        np.testing.assert_array_equal(result._data, expected)


class TestTensordot:
    """Test tensor dot product"""

    def test_tensordot_default_axes(self):
        """Test tensordot with default axes=2"""
        A = nm.Matrix([[1, 2], [3, 4]])
        B = nm.Matrix([[5, 6], [7, 8]])
        result = nm.tensordot(A, B)
        # Default axes=2 means sum over last 2 axes of A and first 2 of B
        assert result.shape == ()

    def test_tensordot_axes_1(self):
        """Test tensordot with axes=1"""
        A = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        B = nm.Matrix([[7, 8], [9, 10], [11, 12]])
        result = nm.tensordot(A, B, axes=1)
        expected = np.tensordot(A._data, B._data, axes=1)
        np.testing.assert_array_equal(result._data, expected)

    def test_tensordot_specified_axes(self):
        """Test tensordot with specified axes tuple"""
        A = nm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        B = nm.Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])  # (2, 2, 2)
        result = nm.tensordot(A, B, axes=([0, 1], [0, 1]))
        expected = np.tensordot(A._data, B._data, axes=([0, 1], [0, 1]))
        np.testing.assert_array_equal(result._data, expected)
