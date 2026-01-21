import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestScalar:
    """Test Scalar base class"""

    def test_scalar_is_0d(self):
        """Test scalar is 0-dimensional"""
        x = nm.Real(3.14)
        assert x.shape == ()
        assert x.ndim == 0

    def test_scalar_from_python_int(self):
        """Test scalar from Python int"""
        x = nm.Real(42)
        assert x.shape == ()

    def test_scalar_from_python_float(self):
        """Test scalar from Python float"""
        x = nm.Real(3.14)
        assert x.shape == ()

    def test_scalar_from_0d_array(self):
        """Test scalar from 0D array"""
        arr = np.array(3.14)
        x = nm.Real(arr)
        assert x.shape == ()

    def test_scalar_from_1d_array_reshapes(self):
        """Test scalar from 1D array with single element reshapes to 0D"""
        arr = np.array([3.14])
        x = nm.Real(arr)
        assert x.shape == ()

    def test_scalar_comparison_returns_bool(self):
        """Test scalar comparison returns Python bool"""
        x = nm.Real(3.0)
        y = nm.Real(4.0)
        assert isinstance(x < y, bool)
        assert x < y

    def test_scalar_equality(self):
        """Test scalar equality"""
        x = nm.Real(3.14)
        y = nm.Real(3.14)
        assert x == y

    def test_scalar_inequality(self):
        """Test scalar inequality"""
        x = nm.Real(3.14)
        y = nm.Real(2.71)
        assert x != y

    def test_scalar_less_than(self):
        """Test scalar less than"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        assert x < y
        assert not (y < x)

    def test_scalar_less_equal(self):
        """Test scalar less or equal"""
        x = nm.Real(2.0)
        y = nm.Real(3.0)
        z = nm.Real(2.0)
        assert x <= y
        assert x <= z

    def test_scalar_greater_than(self):
        """Test scalar greater than"""
        x = nm.Real(3.0)
        y = nm.Real(2.0)
        assert x > y
        assert not (y > x)

    def test_scalar_greater_equal(self):
        """Test scalar greater or equal"""
        x = nm.Real(3.0)
        y = nm.Real(2.0)
        z = nm.Real(3.0)
        assert x >= y
        assert x >= z


class TestScalarTypePromotion:
    """Test type promotion between scalar types"""

    def test_integer_plus_real_promotes_to_real(self):
        """Test Integer + Real promotes to Real"""
        i = nm.Integer(2)
        r = nm.Real(3.14)
        result = i + r
        assert isinstance(result, nm.Real)

    def test_real_plus_complex_promotes_to_complex(self):
        """Test Real + Complex promotes to Complex"""
        r = nm.Real(3.14)
        c = nm.Complex(1 + 2j)
        result = r + c
        assert isinstance(result, nm.Complex)

    def test_integer_plus_complex_promotes_to_complex(self):
        """Test Integer + Complex promotes to Complex"""
        i = nm.Integer(2)
        c = nm.Complex(1 + 2j)
        result = i + c
        assert isinstance(result, nm.Complex)

    def test_boolean_plus_integer_promotes_to_integer(self):
        """Test Boolean + Integer promotes to Integer"""
        b = nm.Boolean(True)
        i = nm.Integer(5)
        result = b + i
        assert isinstance(result, nm.Integer)

    def test_promotion_priority_order(self):
        """Test promotion follows priority order: Boolean < Integer < Real < Complex"""
        assert nm.Boolean._priority < nm.Integer._priority
        assert nm.Integer._priority < nm.Real._priority
        assert nm.Real._priority < nm.Complex._priority

    def test_division_promotes_to_real(self):
        """Test division promotes Integer to Real"""
        i = nm.Integer(7)
        j = nm.Integer(2)
        result = i / j
        assert isinstance(result, nm.Real)
        assert float(result._data) == 3.5
