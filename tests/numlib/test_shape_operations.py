import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestReshape:
    """Test reshape operation"""

    def test_reshape_vector_to_matrix(self):
        """Test reshaping vector to matrix"""
        v = nm.Vector([1, 2, 3, 4, 5, 6])
        m = nm.reshape(v, 2, 3)
        assert m.shape == (2, 3)
        assert isinstance(m, nm.Matrix)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(m._data, expected)

    def test_reshape_matrix_to_vector(self):
        """Test reshaping matrix to vector"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        v = nm.reshape(m, 6, 1)
        assert v.shape == (6, 1)
        assert isinstance(v, nm.Vector)

    def test_reshape_with_tuple(self):
        """Test reshape with tuple argument"""
        v = nm.Vector([1, 2, 3, 4])
        m = nm.reshape(v, (2, 2))
        assert m.shape == (2, 2)

    def test_reshape_with_minus_one(self):
        """Test reshape with -1 (auto-calculate dimension)"""
        v = nm.Vector([1, 2, 3, 4, 5, 6])
        m = nm.reshape(v, 2, -1)
        assert m.shape == (2, 3)

    def test_reshape_flatten(self):
        """Test reshape to flatten"""
        m = nm.Matrix([[1, 2], [3, 4]])
        v = nm.reshape(m, -1)
        assert v.shape == (4,)

    def test_reshape_backward(self):
        """Test backward pass for reshape"""
        x = nm.Vector([1, 2, 3, 4], requires_grad=True)
        y = nm.reshape(x, 2, 2)
        grad_output = nm.Matrix([[1, 1], [1, 1]])
        y.backward(grad_output)
        # Gradient should have original shape
        assert x.grad.shape == (4, 1)
        np.testing.assert_array_equal(x.grad._data.flatten(), [1, 1, 1, 1])

    def test_reshape_method(self):
        """Test reshape as method"""
        v = nm.Vector([1, 2, 3, 4])
        m = v.reshape(2, 2)
        assert m.shape == (2, 2)

    def test_reshape_incompatible_size_raises(self):
        """Test reshape with incompatible size raises error"""
        v = nm.Vector([1, 2, 3])
        with pytest.raises((ValueError, Exception)):
            nm.reshape(v, 2, 2)  # 3 elements can't be reshaped to 2x2


class TestFlatten:
    """Test flatten operation"""

    def test_flatten_matrix(self):
        """Test flattening matrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        v = nm.flatten(m)
        assert v.shape == (6,)
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(v._data, expected)

    def test_flatten_3d_tensor(self):
        """Test flattening 3D tensor"""
        t = nm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        v = nm.flatten(t)
        assert v.shape == (8,)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(v._data, expected)

    def test_flatten_already_1d(self):
        """Test flattening already 1D tensor"""
        v = nm.Tensor([1, 2, 3, 4])
        result = nm.flatten(v)
        assert result.shape == (4,)

    def test_flatten_method(self):
        """Test flatten as method"""
        m = nm.Matrix([[1, 2], [3, 4]])
        v = m.flatten()
        assert v.shape == (4,)

    def test_ravel_alias(self):
        """Test ravel as alias for flatten"""
        m = nm.Matrix([[1, 2], [3, 4]])
        v1 = m.ravel()
        v2 = m.flatten()
        np.testing.assert_array_equal(v1._data, v2._data)


class TestTranspose:
    """Test transpose operation"""

    def test_transpose_matrix(self):
        """Test transposing matrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        m_t = nm.transpose(m)
        assert m_t.shape == (3, 2)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(m_t._data, expected)

    def test_transpose_vector_to_rowvector(self):
        """Test transposing vector to rowvector"""
        v = nm.Vector([1, 2, 3])
        rv = nm.transpose(v)
        assert isinstance(rv, nm.RowVector)
        assert rv.shape == (1, 3)

    def test_transpose_rowvector_to_vector(self):
        """Test transposing rowvector to vector"""
        rv = nm.RowVector([1, 2, 3])
        v = nm.transpose(rv)
        assert isinstance(v, nm.Vector)
        assert v.shape == (3, 1)

    def test_transpose_with_axes(self):
        """Test transpose with specified axes"""
        t = nm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        t_t = nm.transpose(t, (2, 0, 1))
        assert t_t.shape == (2, 2, 2)

    def test_transpose_backward_matrix(self):
        """Test backward pass for matrix transpose"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        m_t = nm.transpose(m)
        grad_output = nm.Matrix([[1, 2], [3, 4]])
        m_t.backward(grad_output)
        # Gradient should be transposed back
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_transpose_backward_vector(self):
        """Test backward pass for vector transpose"""
        v = nm.Vector([1, 2, 3], requires_grad=True)
        rv = nm.transpose(v)
        grad_output = nm.RowVector([1, 2, 3])
        rv.backward(grad_output)
        # Gradient should be transposed back to vector
        assert isinstance(v.grad, nm.Vector)
        np.testing.assert_array_equal(v.grad._data.flatten(), [1, 2, 3])

    def test_transpose_method(self):
        """Test transpose as method"""
        m = nm.Matrix([[1, 2], [3, 4]])
        m_t = m.transpose()
        assert m_t.shape == (2, 2)

    def test_transpose_double(self):
        """Test double transpose returns original"""
        m = nm.Matrix([[1, 2], [3, 4]])
        m_tt = nm.transpose(nm.transpose(m))
        np.testing.assert_array_equal(m._data, m_tt._data)


class TestIndexing:
    """Test indexing and slicing operations"""

    def test_getitem_scalar_index(self):
        """Test getting single element from matrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        element = m[0, 0]
        assert float(element._data) == 1.0

    def test_getitem_row_slice(self):
        """Test getting row from matrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        row = m[0, :]
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(row._data, expected)

    def test_getitem_column_slice(self):
        """Test getting column from matrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        col = m[:, 0]
        expected = np.array([1, 4])
        np.testing.assert_array_equal(col._data, expected)

    def test_getitem_submatrix(self):
        """Test getting submatrix"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub = m[0:2, 1:3]
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(sub._data, expected)

    def test_getitem_vector_index(self):
        """Test getting element from vector"""
        v = nm.Vector([1, 2, 3])
        element = v[1, 0]
        assert float(element._data) == 2.0

    def test_getitem_negative_index(self):
        """Test negative indexing"""
        v = nm.Vector([1, 2, 3])
        element = v[-1, 0]
        assert float(element._data) == 3.0

    def test_getitem_backward(self):
        """Test backward pass for indexing"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        element = m[0, 1]
        element.backward()
        # Gradient should be 1 at indexed location, 0 elsewhere
        expected = np.array([[0, 1, 0], [0, 0, 0]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_getitem_slice_backward(self):
        """Test backward pass for slice"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        row = m[0, :]
        grad_output = nm.Tensor([1, 2, 3])
        row.backward(grad_output)
        # Gradient should be in first row
        expected = np.array([[1, 2, 3], [0, 0, 0]])
        np.testing.assert_array_equal(m.grad._data, expected)

    def test_setitem_scalar(self):
        """Test setting single element"""
        m = nm.Matrix([[1, 2], [3, 4]])
        m[0, 0] = 10
        assert float(m._data[0, 0]) == 10.0

    def test_setitem_row(self):
        """Test setting entire row"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        m[0, :] = nm.Tensor([7, 8, 9])
        expected = np.array([[7, 8, 9], [4, 5, 6]])
        np.testing.assert_array_equal(m._data, expected)


class TestOtherShapeOperations:
    """Test other shape-related operations"""

    def test_squeeze_remove_single_dims(self):
        """Test squeeze removes dimensions of size 1"""
        t = nm.Tensor([[[1, 2, 3]]])  # (1, 1, 3)
        result = t.squeeze()
        assert result.shape == (3,)

    def test_squeeze_specific_axis(self):
        """Test squeeze with specific axis"""
        t = nm.Tensor([[[1], [2], [3]]])  # (1, 3, 1)
        result = t.squeeze(axis=0)
        assert result.shape == (3, 1)

    def test_squeeze_no_effect(self):
        """Test squeeze has no effect when no singleton dimensions"""
        m = nm.Matrix([[1, 2], [3, 4]])
        result = m.squeeze()
        assert result.shape == (2, 2)

    def test_copy_method(self):
        """Test copy creates independent copy"""
        m = nm.Matrix([[1, 2], [3, 4]])
        m_copy = m.copy()
        m_copy._data[0, 0] = 10
        # Original should be unchanged
        assert m._data[0, 0] == 1

    def test_astype_method(self):
        """Test astype converts dtype"""
        m = nm.Matrix([[1, 2], [3, 4]])  # default float64
        m_int = m.astype(np.int32)
        assert m_int.dtype == np.int32

    def test_clip_method(self):
        """Test clip limits values"""
        v = nm.Vector([1, 2, 3, 4, 5])
        result = v.clip(min=2, max=4)
        expected = np.array([2, 2, 3, 4, 4])
        np.testing.assert_array_equal(result._data.flatten(), expected)

    def test_round_method(self):
        """Test round method"""
        v = nm.Vector([1.2, 2.7, 3.5])
        result = v.round()
        expected = np.array([1, 3, 4])
        np.testing.assert_array_equal(result._data.flatten(), expected)

    def test_round_with_decimals(self):
        """Test round with decimal places"""
        v = nm.Vector([1.234, 2.567])
        result = v.round(decimals=1)
        expected = np.array([1.2, 2.6])
        np.testing.assert_array_almost_equal(result._data.flatten(), expected)


class TestShapeEdgeCases:
    """Test edge cases for shape operations"""

    def test_reshape_to_scalar(self):
        """Test reshaping single element to scalar"""
        v = nm.Vector([5])
        s = nm.reshape(v, ())
        assert s.shape == ()

    def test_transpose_1d_array(self):
        """Test transpose of 1D array"""
        t = nm.Tensor([1, 2, 3])
        t_t = nm.transpose(t)
        # 1D transpose should return original shape
        assert t_t.shape == t.shape

    def test_flatten_scalar(self):
        """Test flattening scalar"""
        s = nm.Real(5.0)
        v = nm.flatten(s)
        assert v.shape == (1,)

    def test_reshape_preserve_data(self):
        """Test reshape preserves data order"""
        v = nm.Vector([1, 2, 3, 4, 5, 6])
        m = nm.reshape(v, 2, 3)
        # Check row-major order (C-style)
        assert float(m[0, 0]._data) == 1
        assert float(m[0, 1]._data) == 2
        assert float(m[0, 2]._data) == 3
        assert float(m[1, 0]._data) == 4
