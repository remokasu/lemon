import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestCastError:
    """Test CastError exception"""

    def test_complex_to_real_raises_cast_error(self):
        """Test casting Complex to Real raises CastError"""
        c = nm.Complex(1 + 2j)
        with pytest.raises(nm.CastError, match="Cannot cast `Complex` to `Real`"):
            nm.Real(c)

    def test_complex_to_integer_raises_cast_error(self):
        """Test casting Complex to Integer raises CastError"""
        c = nm.Complex(1 + 2j)
        with pytest.raises(nm.CastError, match="Cannot cast `Complex` to `Integer`"):
            nm.Integer(c)

    def test_cast_error_message(self):
        """Test CastError has proper message"""
        error = nm.CastError("Complex", "Real")
        assert "Cannot cast `Complex` to `Real`" in str(error)
