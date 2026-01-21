import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


class TestConstantNotInComputationalGraph:
    """Test that constants (literals) are not included in computational graph"""

    def test_literal_not_in_prev_mul(self):
        """Test x * 3 does not include 3 in _prev"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3

        # _prev should only contain x, not the constant 3
        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_not_in_prev_add(self):
        """Test x + 5 does not include 5 in _prev"""
        x = nm.Real(2.0, requires_grad=True)
        y = x + 5

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_not_in_prev_sub(self):
        """Test x - 3 does not include 3 in _prev"""
        x = nm.Real(10.0, requires_grad=True)
        y = x - 3

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_not_in_prev_div(self):
        """Test x / 2 does not include 2 in _prev"""
        x = nm.Real(10.0, requires_grad=True)
        y = x / 2

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_not_in_prev_pow(self):
        """Test x ** 2 does not include 2 in _prev"""
        x = nm.Real(3.0, requires_grad=True)
        y = x**2

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_not_in_prev_pow_float(self):
        """Test x ** 0.5 does not include 0.5 in _prev"""
        x = nm.Real(4.0, requires_grad=True)
        y = x**0.5

        assert len(y._prev) == 1
        assert x in y._prev

    def test_reverse_literal_not_in_prev(self):
        """Test 3 * x does not include 3 in _prev"""
        x = nm.Real(2.0, requires_grad=True)
        y = 3 * x

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_left_operand_sub(self):
        """Test 10 - x does not include 10 in _prev"""
        x = nm.Real(3.0, requires_grad=True)
        y = 10 - x

        assert len(y._prev) == 1
        assert x in y._prev

    def test_literal_left_operand_div(self):
        """Test 12 / x does not include 12 in _prev"""
        x = nm.Real(3.0, requires_grad=True)
        y = 12 / x

        assert len(y._prev) == 1
        assert x in y._prev

    def test_two_variables_both_in_prev(self):
        """Test x + y includes both x and y in _prev"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = x + y

        assert len(z._prev) == 2
        assert x in z._prev
        assert y in z._prev

    def test_variable_plus_literal_then_variable(self):
        """Test (x + 3) + y has correct _prev"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(5.0, requires_grad=True)

        temp = x + 3  # temp._prev should only have x
        assert len(temp._prev) == 1
        assert x in temp._prev

        z = temp + y  # z._prev should have temp and y
        assert len(z._prev) == 2
        assert temp in z._prev
        assert y in z._prev

    def test_literal_converted_has_requires_grad_false(self):
        """Test that literals are converted with requires_grad=False"""
        x = nm.Real(2.0, requires_grad=True)

        # Manually convert a literal as the function would
        literal = nm._auto_convert(3, requires_grad=False)

        assert literal.requires_grad is False
        assert isinstance(literal, nm.NumType)


class TestGradientComputationWithLiterals:
    """Test that gradients are computed correctly when literals are involved"""

    def test_gradient_mul_literal(self):
        """Test gradient of x * 3"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        y.backward()

        # dy/dx = 3
        assert x.grad is not None
        assert float(x.grad._data) == 3.0

    def test_gradient_add_literal(self):
        """Test gradient of x + 5"""
        x = nm.Real(2.0, requires_grad=True)
        y = x + 5
        y.backward()

        # dy/dx = 1
        assert x.grad is not None
        assert float(x.grad._data) == 1.0

    def test_gradient_sub_literal(self):
        """Test gradient of x - 3"""
        x = nm.Real(10.0, requires_grad=True)
        y = x - 3
        y.backward()

        # dy/dx = 1
        assert x.grad is not None
        assert float(x.grad._data) == 1.0

    def test_gradient_literal_sub_variable(self):
        """Test gradient of 10 - x"""
        x = nm.Real(3.0, requires_grad=True)
        y = 10 - x
        y.backward()

        # dy/dx = -1
        assert x.grad is not None
        assert float(x.grad._data) == -1.0

    def test_gradient_div_literal(self):
        """Test gradient of x / 2"""
        x = nm.Real(10.0, requires_grad=True)
        y = x / 2
        y.backward()

        # dy/dx = 1/2 = 0.5
        assert x.grad is not None
        assert float(x.grad._data) == 0.5

    def test_gradient_literal_div_variable(self):
        """Test gradient of 12 / x"""
        x = nm.Real(3.0, requires_grad=True)
        y = 12 / x
        y.backward()

        # dy/dx = -12/x^2 = -12/9 = -4/3
        assert x.grad is not None
        np.testing.assert_almost_equal(float(x.grad._data), -12.0 / 9.0)

    def test_gradient_pow_literal(self):
        """Test gradient of x ** 2"""
        x = nm.Real(3.0, requires_grad=True)
        y = x**2
        y.backward()

        # dy/dx = 2*x = 2*3 = 6
        assert x.grad is not None
        assert float(x.grad._data) == 6.0

    def test_gradient_pow_literal_float(self):
        """Test gradient of x ** 0.5"""
        x = nm.Real(4.0, requires_grad=True)
        y = x**0.5
        y.backward()

        # dy/dx = 0.5 * x^(-0.5) = 0.5 / sqrt(4) = 0.5 / 2 = 0.25
        assert x.grad is not None
        assert float(x.grad._data) == 0.25

    def test_gradient_complex_expression(self):
        """Test gradient of (x * 3 + 5) * 2"""
        x = nm.Real(2.0, requires_grad=True)
        y = (x * 3 + 5) * 2
        y.backward()

        # y = (3x + 5) * 2 = 6x + 10
        # dy/dx = 6
        assert x.grad is not None
        assert float(x.grad._data) == 6.0


class TestMatrixOperationsWithLiterals:
    """Test matrix operations with literals"""

    def test_vector_mul_literal_not_in_prev(self):
        """Test vector * 3 does not include 3 in _prev"""
        v = nm.Vector([1, 2, 3], requires_grad=True)
        result = v * 3

        assert len(result._prev) == 1
        assert v in result._prev

    def test_matrix_mul_literal_not_in_prev(self):
        """Test matrix * 2 does not include 2 in _prev"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        result = m * 2

        assert len(result._prev) == 1
        assert m in result._prev

    def test_vector_gradient_with_literal(self):
        """Test gradient of vector * 3"""
        v = nm.Vector([1, 2, 3], requires_grad=True)
        result = v * 3

        grad_output = nm.Vector([1, 1, 1])
        result.backward(grad_output)

        assert v.grad is not None
        np.testing.assert_array_equal(v.grad._data.flatten(), [3, 3, 3])

    def test_matrix_gradient_with_literal(self):
        """Test gradient of matrix * 2"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        result = m * 2

        grad_output = nm.Matrix([[1, 1], [1, 1]])
        result.backward(grad_output)

        assert m.grad is not None
        np.testing.assert_array_equal(m.grad._data, [[2, 2], [2, 2]])


class TestAutogradOffWithLiterals:
    """Test behavior when autograd is off"""

    def test_literal_with_autograd_off(self):
        """Test that with autograd off, literals still work correctly"""
        with nm.autograd.off:
            x = nm.Real(2.0)
            y = x * 3

            assert y.requires_grad is False
            # _prev should be empty when autograd is off
            assert len(y._prev) == 0

    def test_no_gradient_tracking_autograd_off(self):
        """Test no gradient tracking when autograd is off"""
        x = nm.Real(2.0, requires_grad=True)

        with nm.autograd.off:
            y = x * 3
            assert y.requires_grad is False
            assert len(y._prev) == 0


class TestMixedOperationsLiteralsAndVariables:
    """Test mixed operations with both literals and variables"""

    def test_x_plus_literal_mul_y(self):
        """Test (x + 3) * y"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(4.0, requires_grad=True)

        temp = x + 3
        assert len(temp._prev) == 1
        assert x in temp._prev

        z = temp * y
        assert len(z._prev) == 2
        assert temp in z._prev
        assert y in z._prev

        z.backward()

        # z = (x + 3) * y = xy + 3y
        # dz/dx = y = 4
        # dz/dy = x + 3 = 5
        assert float(x.grad._data) == 4.0
        assert float(y.grad._data) == 5.0

    def test_literal_mul_x_plus_y(self):
        """Test 3 * x + y"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(5.0, requires_grad=True)

        temp = 3 * x
        assert len(temp._prev) == 1
        assert x in temp._prev

        z = temp + y
        assert len(z._prev) == 2

        z.backward()

        # z = 3x + y
        # dz/dx = 3
        # dz/dy = 1
        assert float(x.grad._data) == 3.0
        assert float(y.grad._data) == 1.0

    def test_nested_literals(self):
        """Test (x * 2 + 3) * 4 + 5"""
        x = nm.Real(1.0, requires_grad=True)

        y1 = x * 2
        assert len(y1._prev) == 1

        y2 = y1 + 3
        assert len(y2._prev) == 1

        y3 = y2 * 4
        assert len(y3._prev) == 1

        y4 = y3 + 5
        assert len(y4._prev) == 1

        y4.backward()

        # y4 = ((x * 2 + 3) * 4 + 5) = (2x + 3) * 4 + 5 = 8x + 12 + 5 = 8x + 17
        # dy4/dx = 8
        assert float(x.grad._data) == 8.0


class TestDotMatmulWithLiterals:
    """Test dot and matmul operations with literals"""

    def test_matmul_matrix_literal(self):
        """Test matrix @ (literal * identity) behavior"""
        m = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        identity = nm.Matrix([[1, 0], [0, 1]], requires_grad=False)

        result = m @ identity

        # Only m should be in _prev
        assert len(result._prev) == 1
        assert m in result._prev

    def test_dot_vector_with_literal_scalar(self):
        """Test that dot product gradient works with literals"""
        v1 = nm.Vector([1, 2, 3], requires_grad=True)
        v2 = nm.Vector([4, 5, 6], requires_grad=False)  # Like a constant

        result = nm.dot(v1, v2)

        # Only v1 should be in _prev
        assert len(result._prev) == 1
        assert v1 in result._prev


class TestPowSpecialCases:
    """Test pow function special cases with literals"""

    def test_pow_special_case_2(self):
        """Test x ** 2 uses optimized path"""
        x = nm.Real(3.0, requires_grad=True)
        y = x**2

        # Should use special case optimization
        assert len(y._prev) == 1
        assert x in y._prev

        y.backward()
        assert float(x.grad._data) == 6.0

    def test_pow_special_case_3(self):
        """Test x ** 3 uses optimized path"""
        x = nm.Real(2.0, requires_grad=True)
        y = x**3

        assert len(y._prev) == 1
        y.backward()
        # dy/dx = 3 * x^2 = 3 * 4 = 12
        assert float(x.grad._data) == 12.0

    def test_pow_special_case_sqrt(self):
        """Test x ** 0.5 uses sqrt optimization"""
        x = nm.Real(4.0, requires_grad=True)
        y = x**0.5

        assert len(y._prev) == 1
        y.backward()
        # dy/dx = 0.5 * x^(-0.5) = 0.5 / 2 = 0.25
        assert float(x.grad._data) == 0.25

    def test_pow_special_case_inverse(self):
        """Test x ** -1 uses 1/x optimization"""
        x = nm.Real(2.0, requires_grad=True)
        y = x**-1

        assert len(y._prev) == 1
        y.backward()
        # dy/dx = -1 * x^(-2) = -1/4 = -0.25
        assert float(x.grad._data) == -0.25

    def test_pow_special_case_identity(self):
        """Test x ** 1 returns copy"""
        x = nm.Real(5.0, requires_grad=True)
        y = x**1

        assert len(y._prev) == 1
        y.backward()
        assert float(x.grad._data) == 1.0

    def test_pow_special_case_zero(self):
        """Test x ** 0 returns 1"""
        x = nm.Real(5.0, requires_grad=True)
        y = x**0

        # x^0 = 1, gradient should be 0
        # But we don't track gradient for constants
        assert float(y._data) == 1.0


class TestMemoryEfficiency:
    """Test that the fix reduces memory usage in computational graph"""

    def test_long_chain_with_literals(self):
        """Test that long chain with literals doesn't bloat _prev"""
        x = nm.Real(1.0, requires_grad=True)

        # Create a long chain: x * 2 * 3 * 4 * 5
        result = x
        for i in range(2, 6):
            result = result * i
            # Each intermediate result should only have 1 element in _prev
            assert len(result._prev) == 1

        result.backward()

        # Gradient should be 2 * 3 * 4 * 5 = 120
        assert float(x.grad._data) == 120.0

    def test_wide_computation_graph(self):
        """Test wide computation graph with many literals"""
        x = nm.Real(1.0, requires_grad=True)

        # Create many branches
        branches = []
        for i in range(10):
            branch = x * (i + 1)
            assert len(branch._prev) == 1
            branches.append(branch)

        # Sum all branches
        result = branches[0]
        for branch in branches[1:]:
            result = result + branch

        result.backward()

        # Gradient should be sum(1..10) = 55
        assert float(x.grad._data) == 55.0


class TestEdgeCasesWithConstants:
    """Test edge cases involving constants"""

    def test_zero_literal(self):
        """Test operations with 0 literal"""
        x = nm.Real(5.0, requires_grad=True)

        y1 = x + 0
        assert len(y1._prev) == 1

        y2 = x * 0
        assert len(y2._prev) == 1

        y1.backward()
        assert float(x.grad._data) == 1.0

        x.zero_grad()
        y2.backward()
        assert float(x.grad._data) == 0.0

    def test_one_literal(self):
        """Test operations with 1 literal"""
        x = nm.Real(5.0, requires_grad=True)

        y1 = x * 1
        assert len(y1._prev) == 1

        y2 = x / 1
        assert len(y2._prev) == 1

        y1.backward()
        assert float(x.grad._data) == 1.0

        x.zero_grad()
        y2.backward()
        assert float(x.grad._data) == 1.0

    def test_negative_literal(self):
        """Test operations with negative literals"""
        x = nm.Real(5.0, requires_grad=True)

        y = x * -3
        assert len(y._prev) == 1

        y.backward()
        assert float(x.grad._data) == -3.0

    def test_float_literal(self):
        """Test operations with float literals"""
        x = nm.Real(2.0, requires_grad=True)

        y = x * 3.14
        assert len(y._prev) == 1

        y.backward()
        np.testing.assert_almost_equal(float(x.grad._data), 3.14)


class TestBackwardCompatibility:
    """Test that existing functionality still works"""

    def test_two_variables_backward(self):
        """Test backward with two variables (no literals)"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)

        z = x * y
        assert len(z._prev) == 2

        z.backward()

        assert float(x.grad._data) == 3.0
        assert float(y.grad._data) == 2.0

    def test_complex_backward_no_literals(self):
        """Test complex backward without literals"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)
        z = nm.Real(4.0, requires_grad=True)

        result = (x * y) + z
        result.backward()

        assert float(x.grad._data) == 3.0
        assert float(y.grad._data) == 2.0
        assert float(z.grad._data) == 1.0
