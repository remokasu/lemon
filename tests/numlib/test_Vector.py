import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestVector:
    """Test Vector class (column vector)"""

    def test_vector_creation_from_list(self):
        """Test creating vector from list"""
        v = nm.Vector([1, 2, 3])
        assert v.shape == (3, 1)
        assert v.ndim == 2

    def test_vector_creation_from_1d_array(self):
        """Test creating vector from 1D array"""
        arr = np.array([1, 2, 3])
        v = nm.Vector(arr)
        assert v.shape == (3, 1)

    def test_vector_creation_from_scalar(self):
        """Test creating vector from scalar"""
        v = nm.Vector(5)
        assert v.shape == (1, 1)

    def test_vector_creation_from_2d_column(self):
        """Test creating vector from 2D column array"""
        arr = np.array([[1], [2], [3]])
        v = nm.Vector(arr)
        assert v.shape == (3, 1)

    def test_vector_creation_from_2d_row_transposes(self):
        """Test creating vector from 2D row array transposes it"""
        arr = np.array([[1, 2, 3]])
        v = nm.Vector(arr)
        assert v.shape == (3, 1)

    def test_vector_invalid_shape_raises(self):
        """Test creating vector from invalid shape raises error"""
        with pytest.raises(ValueError, match="Vector must have shape"):
            nm.Vector([[1, 2], [3, 4]])

    def test_vector_transpose_to_rowvector(self):
        """Test transposing vector creates rowvector"""
        v = nm.Vector([1, 2, 3])
        rv = v.T
        assert isinstance(rv, nm.RowVector)
        assert rv.shape == (1, 3)

    def test_vector_matmul_vector_raises(self):
        """Test vector @ vector raises error"""
        v1 = nm.Vector([1, 2, 3])
        v2 = nm.Vector([4, 5, 6])
        with pytest.raises(
            ValueError,
            match="Cannot perform matrix multiplication between incompatible types",
        ):
            v1 @ v2

    def test_vector_matmul_rowvector_outer_product(self):
        """Test vector @ rowvector gives outer product (matrix)"""
        v = nm.Vector([1, 2, 3])
        rv = nm.RowVector([4, 5])
        result = v @ rv
        assert isinstance(result, nm.Matrix)
        assert result.shape == (3, 2)
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_vector_matmul_matrix_raises(self):
        """Test vector @ matrix raises error"""
        v = nm.Vector([1, 2])
        m = nm.Matrix([[1, 2], [3, 4]])
        with pytest.raises(
            ValueError,
            match="Cannot perform matrix multiplication between incompatible types",
        ):
            v @ m

    def test_rowvector_matmul_vector(self):
        """Test rowvector @ vector gives inner product (scalar)"""
        rv = nm.RowVector([1, 2, 3])
        v = nm.Vector([4, 5, 6])
        result = rv @ v
        assert isinstance(result, nm.Scalar)
        assert float(result._data) == 32.0  # 1*4 + 2*5 + 3*6

    def test_matrix_matmul_vector(self):
        """Test matrix @ vector gives vector"""
        m = nm.Matrix([[1, 2], [3, 4]])
        v = nm.Vector([5, 6])
        result = m @ v
        assert isinstance(result, nm.Vector)
        assert result.shape == (2, 1)
        expected = np.array([[17], [39]])
        np.testing.assert_array_equal(result._data, expected)

    def test_vector_factory_function(self):
        """Test vector factory function"""
        v = nm.vector([1, 2, 3])
        assert isinstance(v, nm.Vector)

    def test_vector_alias(self):
        """Test vec alias"""
        v = nm.vec([1, 2, 3])
        assert isinstance(v, nm.Vector)
