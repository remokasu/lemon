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
