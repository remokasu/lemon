import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestBoolean:
    """Test Boolean scalar type"""

    def test_boolean_creation_from_bool(self):
        """Test creating Boolean from bool"""
        x = nm.Boolean(True)
        assert x.shape == ()
        assert bool(x._data) is True

    def test_boolean_creation_from_int(self):
        """Test creating Boolean from int"""
        x = nm.Boolean(1)
        assert bool(x._data) is True
        y = nm.Boolean(0)
        assert bool(y._data) is False

    def test_boolean_dtype(self):
        """Test Boolean dtype"""
        x = nm.Boolean(True)
        assert x.dtype.kind == "b"

    def test_boolean_requires_grad_always_false(self):
        """Test Boolean always has requires_grad=False"""
        x = nm.Boolean(True, requires_grad=True)
        assert x.requires_grad is False

    def test_boolean_requires_grad_warning(self):
        """Test Boolean creation with requires_grad=True shows warning"""
        with pytest.warns(UserWarning, match="Boolean type cannot have gradients"):
            nm.Boolean(True, requires_grad=True)

    def test_boolean_priority(self):
        """Test Boolean has lowest priority"""
        assert nm.Boolean._priority == 1

    def test_boolean_factory_function(self):
        """Test boolean factory function"""
        x = nm.boolean(True)
        assert isinstance(x, nm.Boolean)
