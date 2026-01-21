import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestRowVector:
    """Test RowVector class"""

    def test_rowvector_creation_from_list(self):
        """Test creating rowvector from list"""
        rv = nm.RowVector([1, 2, 3])
        assert rv.shape == (1, 3)
        assert rv.ndim == 2

    def test_rowvector_creation_from_1d_array(self):
        """Test creating rowvector from 1D array"""
        arr = np.array([1, 2, 3])
        rv = nm.RowVector(arr)
        assert rv.shape == (1, 3)

    def test_rowvector_creation_from_scalar(self):
        """Test creating rowvector from scalar"""
        rv = nm.RowVector(5)
        assert rv.shape == (1, 1)

    def test_rowvector_creation_from_2d_row(self):
        """Test creating rowvector from 2D row array"""
        arr = np.array([[1, 2, 3]])
        rv = nm.RowVector(arr)
        assert rv.shape == (1, 3)

    def test_rowvector_creation_from_2d_column_transposes(self):
        """Test creating rowvector from 2D column array transposes it"""
        arr = np.array([[1], [2], [3]])
        rv = nm.RowVector(arr)
        assert rv.shape == (1, 3)

    def test_rowvector_invalid_shape_raises(self):
        """Test creating rowvector from invalid shape raises error"""
        with pytest.raises(ValueError, match="RowVector must have shape"):
            nm.RowVector([[1, 2], [3, 4]])

    def test_rowvector_transpose_to_vector(self):
        """Test transposing rowvector creates vector"""
        rv = nm.RowVector([1, 2, 3])
        v = rv.T
        assert isinstance(v, nm.Vector)
        assert v.shape == (3, 1)

    def test_rowvector_matmul_rowvector_raises(self):
        """Test rowvector @ rowvector raises error"""
        rv1 = nm.RowVector([1, 2, 3])
        rv2 = nm.RowVector([4, 5, 6])
        with pytest.raises(
            ValueError, match="Cannot multiply row vector with row vector"
        ):
            rv1 @ rv2

    def test_rowvector_matmul_matrix(self):
        """Test rowvector @ matrix gives rowvector"""
        rv = nm.RowVector([1, 2])
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = rv @ m
        assert isinstance(result, nm.RowVector)
        assert result.shape == (1, 3)
        expected = np.array([[9, 12, 15]])
        np.testing.assert_array_equal(result._data, expected)

    def test_vector_matmul_rowvector_outer_product(self):
        """Test vector @ rowvector gives outer product (matrix)"""
        v = nm.Vector([1, 2, 3])
        rv = nm.RowVector([4, 5])
        result = v @ rv
        assert isinstance(result, nm.Matrix)
        assert result.shape == (3, 2)

    def test_matrix_matmul_rowvector_raises(self):
        """Test matrix @ rowvector raises error"""
        m = nm.Matrix([[1, 2], [3, 4]])
        rv = nm.RowVector([5, 6])
        with pytest.raises(ValueError, match="Cannot multiply matrix with row vector"):
            m @ rv

    def test_rowvector_factory_function(self):
        """Test rowvector factory function"""
        rv = nm.rowvector([1, 2, 3])
        assert isinstance(rv, nm.RowVector)

    def test_rowvector_alias(self):
        """Test rowvec alias"""
        rv = nm.rowvec([1, 2, 3])
        assert isinstance(rv, nm.RowVector)
