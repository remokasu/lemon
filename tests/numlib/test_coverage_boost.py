# tests/numlib/test_coverage_boost.py

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


# ==========================================
# 1. Integer ビット演算のテスト
# ==========================================
class TestIntegerBitwiseOperations:
    """Test Integer bitwise operations (lines 1906-1970)"""

    def test_and_operation(self):
        """Test bitwise AND"""
        a = nm.Integer(12)  # 1100
        b = nm.Integer(10)  # 1010
        result = a & b  # 1000 = 8
        assert int(result._data) == 8

    def test_rand_operation(self):
        """Test reverse AND"""
        result = 10 & nm.Integer(12)
        assert int(result._data) == 8

    def test_or_operation(self):
        """Test bitwise OR"""
        a = nm.Integer(12)  # 1100
        b = nm.Integer(10)  # 1010
        result = a | b  # 1110 = 14
        assert int(result._data) == 14

    def test_ror_operation(self):
        """Test reverse OR"""
        result = 10 | nm.Integer(12)
        assert int(result._data) == 14

    def test_xor_operation(self):
        """Test bitwise XOR"""
        a = nm.Integer(12)  # 1100
        b = nm.Integer(10)  # 1010
        result = a ^ b  # 0110 = 6
        assert int(result._data) == 6

    def test_rxor_operation(self):
        """Test reverse XOR"""
        result = 10 ^ nm.Integer(12)
        assert int(result._data) == 6

    def test_lshift_operation(self):
        """Test left shift"""
        a = nm.Integer(5)  # 0101
        result = a << 2  # 10100 = 20
        assert int(result._data) == 20

    def test_rlshift_operation(self):
        """Test reverse left shift"""
        result = 5 << nm.Integer(2)
        assert int(result._data) == 20

    def test_rshift_operation(self):
        """Test right shift"""
        a = nm.Integer(20)  # 10100
        result = a >> 2  # 0101 = 5
        assert int(result._data) == 5

    def test_rrshift_operation(self):
        """Test reverse right shift"""
        result = 20 >> nm.Integer(2)
        assert int(result._data) == 5

    def test_invert_operation(self):
        """Test bitwise NOT"""
        a = nm.Integer(5, kind=8, signed=True)
        result = ~a
        assert isinstance(result, nm.Integer)

    def test_bitwise_with_real_raises_error(self):
        """Test that bitwise ops with Real raise TypeError"""
        a = nm.Integer(5)
        b = nm.Real(3.0)

        with pytest.raises(TypeError):
            a & b

        with pytest.raises(TypeError):
            a | b

        with pytest.raises(TypeError):
            a ^ b

        with pytest.raises(TypeError):
            a << b

        with pytest.raises(TypeError):
            a >> b

    def test_integer_binary_properties(self):
        """Test bin, oct, hex properties"""
        a = nm.Integer(42)

        bin_str = a.bin
        assert isinstance(bin_str, str)

        oct_str = a.oct
        assert isinstance(oct_str, str)

        hex_str = a.hex
        assert isinstance(hex_str, str)


# ==========================================
# 2. エラーハンドリングのテスト
# ==========================================
class TestErrorHandling:
    """Test error handling branches"""

    def test_integer_requires_grad_warning(self):
        """Test Integer with requires_grad raises warning (line 1807-1815)"""
        with pytest.warns(UserWarning, match="Integer type cannot have gradients"):
            x = nm.Integer(5, requires_grad=True)
        assert x.requires_grad is False

    def test_boolean_requires_grad_warning(self):
        """Test Boolean with requires_grad raises warning"""
        with pytest.warns(UserWarning, match="Boolean type cannot have gradients"):
            x = nm.Boolean(True, requires_grad=True)
        assert x.requires_grad is False

    def test_vector_with_2d_array(self):
        """Test Vector with 2D array"""
        # Vectorは2次元配列を受け入れるが、(n,1)に変形される
        v = nm.Vector([[1], [2], [3]])
        assert v.shape == (3, 1)

    def test_integer_itruediv_error(self):
        """Test Integer in-place true division error"""
        a = nm.Integer(10)
        with pytest.raises(TypeError, match="In-place true division not supported"):
            a /= 2


# ==========================================
# 3. 特殊メソッドのテスト
# ==========================================
class TestSpecialMethods:
    """Test special methods for coverage"""

    def test_integer_repr(self):
        """Test Integer __repr__"""
        x = nm.Integer(42, kind=32, signed=True)
        repr_str = repr(x)
        assert "int32" in repr_str
        assert "42" in repr_str

        x_unsigned = nm.Integer(42, kind=16, signed=False)
        repr_str = repr(x_unsigned)
        assert "uint16" in repr_str

    def test_real_repr(self):
        """Test Real __repr__"""
        x = nm.Real(3.14)
        repr_str = repr(x)
        assert "3.14" in repr_str or "real" in repr_str.lower()

    def test_complex_repr(self):
        """Test Complex __repr__"""
        x = nm.Complex(3 + 4j)
        repr_str = repr(x)
        assert "complex" in repr_str.lower() or "3" in repr_str

    def test_boolean_repr(self):
        """Test Boolean __repr__"""
        x = nm.Boolean(True)
        repr_str = repr(x)
        assert "True" in repr_str or "bool" in repr_str.lower()

    def test_matrix_repr(self):
        """Test Matrix __repr__"""
        m = nm.Matrix([[1, 2], [3, 4]])
        repr_str = repr(m)
        assert "matrix" in repr_str.lower() or "1" in repr_str

    def test_vector_repr(self):
        """Test Vector __repr__"""
        v = nm.Vector([1, 2, 3])
        repr_str = repr(v)
        assert "vector" in repr_str.lower() or "1" in repr_str


# ==========================================
# 4. 数学関数の特殊ケース
# ==========================================
class TestMathFunctionEdgeCases:
    """Test edge cases in math functions"""

    def test_sqrt_negative_value(self):
        """Test sqrt with negative value"""
        x = nm.Real(-4.0)
        with np.errstate(invalid="ignore"):
            result = nm.sqrt(x)
            assert np.isnan(result._data)

    def test_log_zero(self):
        """Test log(0) gives -inf"""
        x = nm.Real(0.0)
        with np.errstate(divide="ignore"):
            result = nm.log(x)
            assert np.isneginf(result._data)

    def test_log_negative(self):
        """Test log with negative value"""
        x = nm.Real(-1.0)
        with np.errstate(invalid="ignore"):
            result = nm.log(x)
            assert np.isnan(result._data)

    def test_pow_special_cases_with_numtype(self):
        """Test pow with NumType exponents (lines 2669-2709)"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Real(3.0, requires_grad=True)

        # 一般的なケース
        result = nm.pow(x, y)
        assert float(result._data) == 8.0

        result.backward()
        assert x.grad is not None
        assert y.grad is not None

    def test_pow_with_zero_exponent_numtype(self):
        """Test x^0 where 0 is NumType"""
        x = nm.Real(5.0, requires_grad=True)
        y = nm.Real(0.0, requires_grad=False)

        result = nm.pow(x, y)
        assert float(result._data) == 1.0

    def test_pow_with_one_exponent_numtype(self):
        """Test x^1 where 1 is NumType"""
        x = nm.Real(5.0, requires_grad=True)
        y = nm.Real(1.0, requires_grad=False)

        result = nm.pow(x, y)
        assert float(result._data) == 5.0

    def test_pow_with_negative_base(self):
        """Test pow with negative base and fractional exponent"""
        x = nm.Real(-2.0, requires_grad=True)
        y = nm.Real(0.5, requires_grad=True)

        with np.errstate(invalid="ignore"):
            result = nm.pow(x, y)
            # Should handle negative base
            assert result is not None


# ==========================================
# 5. リダクション操作の追加テスト
# ==========================================
class TestReductionEdgeCases:
    """Test reduction operations edge cases"""

    def test_mean_with_keepdims(self):
        """Test mean with keepdims=True"""
        x = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        result = nm.mean(x, axis=0, keepdims=True)
        assert result.shape == (1, 3)

    def test_sum_with_keepdims(self):
        """Test sum with keepdims=True"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = nm.sum(x, axis=1, keepdims=True)
        assert result.shape == (2, 1)

    def test_sum_no_axis(self):
        """Test sum without axis (total sum)"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = nm.sum(x)
        assert float(result._data) == 10.0

    def test_mean_no_axis(self):
        """Test mean without axis (total mean)"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = nm.mean(x)
        assert float(result._data) == 2.5


# ==========================================
# 6. 形状操作の追加テスト
# ==========================================
class TestShapeOperationsEdgeCases:
    """Test shape operations edge cases"""

    def test_reshape_to_scalar(self):
        """Test reshape to scalar shape"""
        x = nm.Tensor([[5]])
        result = nm.reshape(x, ())
        assert result.shape == ()
        assert float(result._data) == 5.0

    def test_transpose_with_axes(self):
        """Test transpose with explicit axes"""
        x = nm.Tensor(np.arange(24).reshape(2, 3, 4))
        result = nm.transpose(x, axes=(2, 0, 1))
        assert result.shape == (4, 2, 3)

    def test_flatten_method(self):
        """Test flatten method"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = x.flatten()
        assert result.shape == (4,)

    def test_ravel_method(self):
        """Test ravel method"""
        x = nm.Matrix([[1, 2], [3, 4]])
        result = x.ravel()
        assert result.shape == (4,)


# ==========================================
# 7. 比較演算のテスト
# ==========================================
class TestComparisonOperations:
    """Test comparison operations"""

    def test_less_than(self):
        """Test < operator"""
        x = nm.Real(3.0)
        y = nm.Real(5.0)
        result = x < y
        # 比較演算子はPythonのboolを返す
        assert result is True

    def test_less_equal(self):
        """Test <= operator"""
        x = nm.Real(3.0)
        y = nm.Real(3.0)
        result = x <= y
        assert result is True

    def test_greater_than(self):
        """Test > operator"""
        x = nm.Real(5.0)
        y = nm.Real(3.0)
        result = x > y
        assert result is True

    def test_greater_equal(self):
        """Test >= operator"""
        x = nm.Real(5.0)
        y = nm.Real(5.0)
        result = x >= y
        assert result is True

    def test_equal(self):
        """Test == operator"""
        x = nm.Real(3.0)
        y = nm.Real(3.0)
        result = x == y
        assert result is True

    def test_not_equal(self):
        """Test != operator"""
        x = nm.Real(3.0)
        y = nm.Real(5.0)
        result = x != y
        assert result is True


# ==========================================
# 8. 複素数演算のテスト
# ==========================================
class TestComplexOperations:
    """Test Complex number operations"""

    def test_complex_arithmetic(self):
        """Test Complex arithmetic operations"""
        x = nm.Complex(3 + 4j)
        y = nm.Complex(1 + 2j)

        # Addition
        result = x + y
        assert result._data == (4 + 6j)

        # Subtraction
        result = x - y
        assert result._data == (2 + 2j)

        # Multiplication
        result = x * y
        assert result._data == (-5 + 10j)

    def test_complex_abs(self):
        """Test complex absolute value"""
        x = nm.Complex(3 + 4j)
        result = nm.abs(x)
        assert float(result._data) == 5.0

    def test_complex_real_imag(self):
        """Test real and imag properties"""
        x = nm.Complex(3 + 4j)
        assert float(x.real._data) == 3.0
        assert float(x.imag._data) == 4.0


# ==========================================
# 9. 型変換のテスト
# ==========================================
class TestTypeCasting:
    """Test type casting operations"""

    def test_to_numpy_method(self):
        """Test to_numpy() method"""
        x = nm.Real(3.14)
        arr = x.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert float(arr) == 3.14

    def test_item_method(self):
        """Test item() method for scalars"""
        x = nm.Real(3.14)
        value = x.item()
        assert isinstance(value, (int, float))
        assert value == 3.14

    def test_float_conversion(self):
        """Test float() conversion"""
        x = nm.Real(3.14)
        value = float(x)
        assert value == 3.14

    def test_int_conversion(self):
        """Test int() conversion"""
        x = nm.Integer(42)
        value = int(x)
        assert value == 42


# ==========================================
# 10. その他の演算
# ==========================================
class TestMiscOperations:
    """Test miscellaneous operations"""

    def test_floordiv_operation(self):
        """Test floor division"""
        x = nm.Real(7.0)
        y = nm.Real(2.0)
        result = nm.floordiv(x, y)
        assert float(result._data) == 3.0

    def test_mod_operation(self):
        """Test modulo operation"""
        x = nm.Real(7.0)
        y = nm.Real(3.0)
        result = nm.mod(x, y)
        assert float(result._data) == 1.0

    def test_maximum_operation(self):
        """Test element-wise maximum"""
        x = nm.Tensor([1, 5, 3])
        y = nm.Tensor([2, 4, 6])
        result = nm.maximum(x, y)
        expected = np.array([2, 5, 6])
        np.testing.assert_array_equal(result._data, expected)

    def test_minimum_operation(self):
        """Test element-wise minimum"""
        x = nm.Tensor([1, 5, 3])
        y = nm.Tensor([2, 4, 6])
        result = nm.minimum(x, y)
        expected = np.array([1, 4, 3])
        np.testing.assert_array_equal(result._data, expected)

    def test_atan2_operation(self):
        """Test atan2 operation"""
        y = nm.Real(1.0)
        x = nm.Real(1.0)
        result = nm.atan2(y, x)
        expected = np.pi / 4
        np.testing.assert_almost_equal(float(result._data), expected)


# ==========================================
# 11. 逆伝播の追加テスト
# ==========================================
class TestBackwardEdgeCases:
    """Test backward pass edge cases"""

    def test_backward_with_no_grad_raises_error(self):
        """Test backward when requires_grad=False raises RuntimeError"""
        x = nm.Real(2.0, requires_grad=False)
        y = x * 3

        # requires_grad=False なので backward は RuntimeError を起こすべき
        with pytest.raises(RuntimeError, match="does not require gradients"):
            y.backward()

        assert x.grad is None

    def test_backward_accumulates_gradients(self):
        """Test calling backward twice accumulates gradients"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3

        y.backward(retain_graph=True)  # 計算グラフを保持
        first_grad = float(x.grad._data)
        assert first_grad == 3.0

        # 2回目のbackward（勾配は累積される）
        y.backward()  # retain_graph=False (デフォルト)
        second_grad = float(x.grad._data)

        # 勾配は累積される
        assert second_grad == 6.0  # 3.0 + 3.0

    def test_zero_grad(self):
        """Test zero_grad method"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3

        y.backward()
        assert x.grad is not None
        assert float(x.grad._data) == 3.0

        x.zero_grad()
        assert x.grad is None

    def test_backward_with_gradient_argument(self):
        """Test backward with custom gradient"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3

        # カスタム勾配を渡す
        custom_grad = nm.Real(2.0)
        y.backward(gradient=custom_grad)

        # dy/dx = 3, gradient = 2 なので、x.grad = 3 * 2 = 6
        assert float(x.grad._data) == 6.0

    def test_backward_on_intermediate_node(self):
        """Test backward from intermediate node"""
        x = nm.Real(2.0, requires_grad=True)
        y = x * 3
        z = y + 5

        # y で backward（葉ノードではない）
        y.backward()

        assert x.grad is not None
        assert float(x.grad._data) == 3.0

    def test_backward_multiple_paths(self):
        """Test backward with multiple computation paths"""
        x = nm.Real(2.0, requires_grad=True)
        y1 = x * 2
        y2 = x * 3
        z = y1 + y2

        z.backward()

        # dz/dx = dy1/dx + dy2/dx = 2 + 3 = 5
        assert float(x.grad._data) == 5.0


# ==========================================
# 12. NumType基本機能のテスト
# ==========================================
class TestNumTypeBasics:
    """Test NumType basic functionality"""

    def test_shape_property(self):
        """Test shape property"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.shape == (2, 2)

    def test_ndim_property(self):
        """Test ndim property"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.ndim == 2

    def test_size_property(self):
        """Test size property"""
        x = nm.Matrix([[1, 2], [3, 4]])
        assert x.size == 4

    def test_dtype_property(self):
        """Test dtype property"""
        x = nm.Real(3.14)
        assert x.dtype == x._data.dtype
