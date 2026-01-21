import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestTensor:
    """Test Tensor class"""

    def test_tensor_creation_1d(self):
        """Test creating 1D tensor"""
        x = nm.Tensor([1, 2, 3])
        assert x.shape == (3,)
        assert x.ndim == 1

    def test_tensor_creation_2d(self):
        """Test creating 2D tensor"""
        x = nm.Tensor([[1, 2], [3, 4]])
        assert x.shape == (2, 2)
        assert x.ndim == 2

    def test_tensor_creation_3d(self):
        """Test creating 3D tensor"""
        x = nm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert x.shape == (2, 2, 2)
        assert x.ndim == 3

    def test_tensor_with_dtype(self):
        """Test tensor creation with dtype"""
        x = nm.Tensor([1, 2, 3], dtype=np.float32)
        assert x.dtype == np.float32

    def test_tensor_rank_property(self):
        """Test rank property (alias for ndim)"""
        x = nm.Tensor([[[1, 2], [3, 4]]])
        assert x.rank == 3
        assert x.rank == x.ndim

    def test_tensor_T_property(self):
        """Test T property for tensor"""
        x = nm.Tensor([[1, 2], [3, 4]])
        x_t = x.T
        assert x_t.shape == (2, 2)
        np.testing.assert_array_equal(x_t._data, [[1, 3], [2, 4]])

    def test_tensor_real_imag_real_valued(self):
        """Test real and imag properties for real-valued tensor"""
        x = nm.Tensor([1, 2, 3])
        assert isinstance(x.real, nm.Tensor)
        assert isinstance(x.imag, nm.Tensor)
        np.testing.assert_array_equal(x.real._data, [1, 2, 3])
        np.testing.assert_array_equal(x.imag._data, [0, 0, 0])

    def test_tensor_real_imag_complex_valued(self):
        """Test real and imag properties for complex-valued tensor"""
        x = nm.Tensor([1 + 2j, 3 + 4j])
        np.testing.assert_array_equal(x.real._data, [1, 3])
        np.testing.assert_array_equal(x.imag._data, [2, 4])

    def test_tensor_factory_function(self):
        """Test tensor factory function"""
        x = nm.tensor([1, 2, 3])
        assert isinstance(x, nm.Tensor)

    def test_tensor_alias(self):
        """Test ten alias"""
        x = nm.ten([1, 2, 3])
        assert isinstance(x, nm.Tensor)
