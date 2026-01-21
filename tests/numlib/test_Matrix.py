import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestMatrix:
    """Test Matrix class"""

    def test_matrix_creation_2d(self):
        """Test creating matrix from 2D array"""
        m = nm.Matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)
        assert m.ndim == 2

    def test_matrix_creation_from_1d_array(self):
        """Test creating matrix from 1D array creates row matrix"""
        m = nm.Matrix([1, 2, 3])
        assert m.shape == (1, 3)

    def test_matrix_invalid_3d_raises(self):
        """Test creating matrix from 3D array raises error"""
        with pytest.raises(ValueError, match="Matrix must be 2-dimensional"):
            nm.Matrix([[[1, 2], [3, 4]]])

    def test_matrix_transpose(self):
        """Test matrix transpose"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        m_t = m.T
        assert isinstance(m_t, nm.Matrix)
        assert m_t.shape == (3, 2)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(m_t._data, expected)

    def test_matrix_matmul_matrix(self):
        """Test matrix @ matrix"""
        m1 = nm.Matrix([[1, 2], [3, 4]])
        m2 = nm.Matrix([[5, 6], [7, 8]])
        result = m1 @ m2
        assert isinstance(result, nm.Matrix)
        assert result.shape == (2, 2)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result._data, expected)

    def test_matrix_matmul_vector(self):
        """Test matrix @ vector"""
        m = nm.Matrix([[1, 2], [3, 4]])
        v = nm.Vector([5, 6])
        result = m @ v
        assert isinstance(result, nm.Vector)
        assert result.shape == (2, 1)

    def test_matrix_matmul_rowvector_raises(self):
        """Test matrix @ rowvector raises error"""
        m = nm.Matrix([[1, 2], [3, 4]])
        rv = nm.RowVector([5, 6])
        with pytest.raises(ValueError, match="Cannot multiply matrix with row vector"):
            m @ rv

    def test_rowvector_matmul_matrix(self):
        """Test rowvector @ matrix"""
        rv = nm.RowVector([1, 2])
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = rv @ m
        assert isinstance(result, nm.RowVector)
        assert result.shape == (1, 3)

    def test_vector_matmul_matrix_raises(self):
        """Test vector @ matrix raises error"""
        v = nm.Vector([1, 2])
        m = nm.Matrix([[1, 2], [3, 4]])
        with pytest.raises(
            ValueError,
            match="Cannot perform matrix multiplication between incompatible types",
        ):
            v @ m

    def test_matrix_indexing(self):
        """Test matrix indexing"""
        m = nm.Matrix([[1, 2], [3, 4]])
        assert float(m[0, 0]._data) == 1.0
        assert float(m[1, 1]._data) == 4.0

    def test_matrix_factory_function(self):
        """Test matrix factory function"""
        m = nm.matrix([[1, 2], [3, 4]])
        assert isinstance(m, nm.Matrix)

    def test_matrix_alias(self):
        """Test mat alias"""
        m = nm.mat([[1, 2], [3, 4]])
        assert isinstance(m, nm.Matrix)
