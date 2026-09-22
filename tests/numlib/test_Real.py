import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestReal:
    """Test Real scalar type"""

    def test_real_creation_default(self):
        """Test creating Real with default settings (float64)"""
        x = nm.Real(3.14)
        assert x.shape == ()
        assert x.kind == 64

    def test_real_creation_float16(self):
        """Test creating 16-bit float"""
        x = nm.Real(3.14, kind=16)
        assert x.kind == 16

    def test_real_creation_float32(self):
        """Test creating 32-bit float"""
        x = nm.Real(3.14, kind=32)
        assert x.kind == 32

    def test_real_creation_float64(self):
        """Test creating 64-bit float"""
        x = nm.Real(3.14, kind=64)
        assert x.kind == 64

    def test_real_requires_grad_default(self):
        """Test Real respects autograd state by default"""
        nm.autograd.enable()
        x = nm.Real(3.14)
        assert x.requires_grad is True

        nm.autograd.disable()
        y = nm.Real(3.14)
        assert y.requires_grad is False

        nm.autograd.enable()

    def test_real_priority(self):
        """Test Real has priority 3"""
        assert nm.Real._priority == 3

    def test_real_factory_functions(self):
        """Test real factory functions"""
        x = nm.real(3.14)
        assert isinstance(x, nm.Real)

        x16 = nm.real16(3.14)
        assert x16.kind == 16

        x32 = nm.real32(3.14)
        assert x32.kind == 32

        x64 = nm.real64(3.14)
        assert x64.kind == 64


class TestRealKindMatchesData:
    """Real の kind と中身の dtype が必ず一致する"""

    needs_float128 = pytest.mark.skipif(
        not hasattr(np, "float128"), reason="float128 is not available"
    )

    @needs_float128
    @pytest.mark.parametrize(
        "data",
        [1.5, np.array(1.5), np.array(1.5, dtype=np.float32)],
        ids=["python-float", "ndarray-f64", "ndarray-f32"],
    )
    def test_real128_converts_to_float128(self, data):
        x = nm.real128(data)
        assert x.kind == 128
        assert x.dtype == np.float128

    @needs_float128
    def test_real128_from_real(self):
        x = nm.real128(nm.real(1.5))
        assert (x.kind, x.dtype) == (128, np.float128)

    @pytest.mark.parametrize("kind", [16, 32, 64])
    def test_ndarray_input_is_cast_to_kind(self, kind):
        x = nm.Real(np.array(1.5, dtype=np.float64), kind=kind)
        assert x.dtype == np.dtype(f"float{kind}")

    @pytest.mark.parametrize(
        "data", [1.5, np.array(1.5)], ids=["python-float", "ndarray"]
    )
    def test_invalid_kind_raises(self, data):
        with pytest.raises(ValueError, match="Real kind must be"):
            nm.Real(data, kind=8)
