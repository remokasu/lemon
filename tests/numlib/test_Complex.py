import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestComplex:
    """Test Complex scalar type"""

    def test_complex_creation_from_complex(self):
        """Test creating Complex from complex number"""
        x = nm.Complex(1 + 2j)
        assert x.shape == ()
        assert x.kind == 128

    def test_complex_creation_from_two_args(self):
        """Test creating Complex from two arguments"""
        x = nm.Complex(1, 2)
        assert complex(x._data) == 1 + 2j

    def test_complex_creation_complex64(self):
        """Test creating 64-bit complex"""
        x = nm.Complex(1 + 2j, kind=64)
        assert x.kind == 64

    def test_complex_creation_complex128(self):
        """Test creating 128-bit complex"""
        x = nm.Complex(1 + 2j, kind=128)
        assert x.kind == 128

    def test_complex_real_property(self):
        """Test real property returns Real"""
        x = nm.Complex(3 + 4j)
        real_part = x.real
        assert isinstance(real_part, nm.Real)
        assert float(real_part._data) == 3.0

    def test_complex_imag_property(self):
        """Test imag property returns Real"""
        x = nm.Complex(3 + 4j)
        imag_part = x.imag
        assert isinstance(imag_part, nm.Real)
        assert float(imag_part._data) == 4.0

    def test_complex_requires_grad_default(self):
        """Test Complex respects autograd state by default"""
        nm.autograd.enable()
        x = nm.Complex(1 + 2j)
        assert x.requires_grad is True

        nm.autograd.disable()
        y = nm.Complex(1 + 2j)
        assert y.requires_grad is False

        nm.autograd.enable()

    def test_complex_priority(self):
        """Test Complex has priority 4 (highest)"""
        assert nm.Complex._priority == 4

    def test_complex_factory_functions(self):
        """Test complex factory functions"""
        x = nm.cmplx(1 + 2j)
        assert isinstance(x, nm.Complex)

        x64 = nm.cmplx64(1 + 2j)
        assert x64.kind == 64

        x128 = nm.cmplx128(1 + 2j)
        assert x128.kind == 128
