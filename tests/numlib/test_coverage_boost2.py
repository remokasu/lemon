# tests/numlib/test_coverage_boost2.py

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest
from lemon import numlib as nm
import numpy as np


# ==========================================
# 1. 三角関数・双曲線関数の勾配テスト
# ==========================================
class TestTrigonometricGradients:
    """Test gradients of trigonometric functions"""

    def test_sin_backward(self):
        """Test sin backward"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.sin(x)
        y.backward()

        # dy/dx = cos(x) = cos(1.0)
        expected = np.cos(1.0)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_cos_backward(self):
        """Test cos backward"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.cos(x)
        y.backward()

        # dy/dx = -sin(x) = -sin(1.0)
        expected = -np.sin(1.0)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_tan_backward(self):
        """Test tan backward"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.tan(x)
        y.backward()

        # dy/dx = 1/cos^2(x) = sec^2(x)
        expected = 1.0 / (np.cos(0.5) ** 2)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_sinh_backward(self):
        """Test sinh backward"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.sinh(x)
        y.backward()

        # dy/dx = cosh(x)
        expected = np.cosh(1.0)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_cosh_backward(self):
        """Test cosh backward"""
        x = nm.Real(1.0, requires_grad=True)
        y = nm.cosh(x)
        y.backward()

        # dy/dx = sinh(x)
        expected = np.sinh(1.0)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_tanh_backward(self):
        """Test tanh backward"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.tanh(x)
        y.backward()

        # dy/dx = 1 - tanh^2(x)
        expected = 1.0 - np.tanh(0.5) ** 2
        np.testing.assert_almost_equal(float(x.grad._data), expected)


# ==========================================
# 2. 逆三角関数の勾配テスト（正しい関数名）
# ==========================================
class TestInverseTrigGradients:
    """Test gradients of inverse trigonometric functions"""

    def test_asin_backward(self):
        """Test asin backward"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.asin(x)
        y.backward()

        # dy/dx = 1/sqrt(1-x^2)
        expected = 1.0 / np.sqrt(1.0 - 0.5**2)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_acos_backward(self):
        """Test acos backward"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.acos(x)
        y.backward()

        # dy/dx = -1/sqrt(1-x^2)
        expected = -1.0 / np.sqrt(1.0 - 0.5**2)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_atan_backward(self):
        """Test atan backward"""
        x = nm.Real(0.5, requires_grad=True)
        y = nm.atan(x)
        y.backward()

        # dy/dx = 1/(1+x^2)
        expected = 1.0 / (1.0 + 0.5**2)
        np.testing.assert_almost_equal(float(x.grad._data), expected)


# ==========================================
# 3. リダクション操作の詳細テスト
# ==========================================
class TestReductionDetails:
    """Test reduction operations in detail"""

    def test_sum_multiple_axes(self):
        """Test sum with multiple axes"""
        x = nm.Tensor(np.arange(24).reshape(2, 3, 4))
        result = nm.sum(x, axis=(0, 2))
        assert result.shape == (3,)

    def test_mean_multiple_axes(self):
        """Test mean with multiple axes"""
        x = nm.Tensor(np.arange(24).reshape(2, 3, 4))
        result = nm.mean(x, axis=(0, 2))
        assert result.shape == (3,)

    def test_sum_backward(self):
        """Test sum backward pass"""
        x = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        y = nm.sum(x)
        y.backward()

        # 全要素の勾配は1
        expected = np.ones((2, 2))
        np.testing.assert_array_equal(x.grad._data, expected)

    def test_mean_backward(self):
        """Test mean backward pass"""
        x = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        y = nm.mean(x)
        y.backward()

        # 全要素の勾配は1/n
        expected = np.ones((2, 2)) / 4
        np.testing.assert_array_equal(x.grad._data, expected)

    def test_sum_axis_backward(self):
        """Test sum with axis backward"""
        x = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = nm.sum(x, axis=0)

        grad_output = nm.Tensor([1, 2, 3])
        y.backward(grad_output)

        # 各列の勾配がbroadcastされる
        expected = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(x.grad._data, expected)


# ==========================================
# 4. 形状操作の詳細テスト
# ==========================================
class TestShapeOperationsDetails:
    """Test shape operations in detail"""

    def test_concatenate_axis1(self):
        """Test concatenate along axis 1"""
        a = nm.Matrix([[1, 2], [3, 4]])
        b = nm.Matrix([[5, 6], [7, 8]])
        result = nm.concatenate([a, b], axis=1)

        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
        np.testing.assert_array_equal(result._data, expected)

    def test_concatenate_three_tensors(self):
        """Test concatenate with 3 tensors"""
        a = nm.Tensor([1, 2])
        b = nm.Tensor([3, 4])
        c = nm.Tensor([5, 6])
        result = nm.concatenate([a, b, c], axis=0)

        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result._data, expected)

    def test_stack_operation(self):
        """Test stack operation"""
        a = nm.Tensor([1, 2, 3])
        b = nm.Tensor([4, 5, 6])
        result = nm.stack([a, b], axis=0)

        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_stack_axis1(self):
        """Test stack along axis 1"""
        a = nm.Tensor([1, 2, 3])
        b = nm.Tensor([4, 5, 6])
        result = nm.stack([a, b], axis=1)

        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result._data, expected)

    def test_reshape_backward(self):
        """Test reshape backward"""
        x = nm.Tensor([1, 2, 3, 4], requires_grad=True)
        y = nm.reshape(x, (2, 2))

        grad_output = nm.Matrix([[1, 2], [3, 4]])
        y.backward(grad_output)

        # 勾配は元の形状に戻る
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(x.grad._data, expected)

    def test_transpose_backward(self):
        """Test transpose backward"""
        x = nm.Matrix([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = nm.transpose(x)

        grad_output = nm.Matrix([[1, 2], [3, 4], [5, 6]])
        y.backward(grad_output)

        # 勾配も転置される
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_equal(x.grad._data, expected)


# ==========================================
# 5. 行列操作の特殊ケーステスト
# ==========================================
class TestMatrixOperationsSpecialCases:
    """Test special cases in matrix operations"""

    def test_dot_2d_arrays(self):
        """Test dot with 2D arrays"""
        a = nm.Matrix([[1, 2], [3, 4]])
        b = nm.Matrix([[5, 6], [7, 8]])
        result = nm.dot(a, b)

        # 2D配列のdotは行列積と同じ
        expected = np.dot(a._data, b._data)
        np.testing.assert_array_equal(result._data, expected)

    def test_matmul_backward_simple(self):
        """Test matmul backward with simple case"""
        a = nm.Matrix([[1, 2], [3, 4]], requires_grad=True)
        b = nm.Matrix([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b

        # 勾配を計算
        grad_output = nm.Matrix([[1, 1], [1, 1]])
        c.backward(grad_output)

        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_dot_backward(self):
        """Test dot backward"""
        a = nm.Vector([1, 2, 3], requires_grad=True)
        b = nm.Vector([4, 5, 6], requires_grad=True)

        c = nm.dot(a, b)
        c.backward()

        # da/dc = b, db/dc = a
        np.testing.assert_array_equal(a.grad._data.flatten(), [4, 5, 6])
        np.testing.assert_array_equal(b.grad._data.flatten(), [1, 2, 3])


# ==========================================
# 6. 特殊な演算子のテスト
# ==========================================
class TestSpecialOperators:
    """Test special operators"""

    def test_getitem_integer_indexing(self):
        """Test integer indexing"""
        x = nm.Tensor([1, 2, 3, 4, 5])
        result = x[2]
        assert float(result._data) == 3.0

    def test_getitem_slice(self):
        """Test slice indexing"""
        x = nm.Tensor([1, 2, 3, 4, 5])
        result = x[1:4]
        expected = np.array([2, 3, 4])
        np.testing.assert_array_equal(result._data, expected)

    def test_getitem_2d_indexing(self):
        """Test 2D indexing"""
        x = nm.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = x[1, 2]
        assert float(result._data) == 6.0

    def test_getitem_fancy_indexing(self):
        """Test fancy indexing"""
        x = nm.Tensor([1, 2, 3, 4, 5])
        result = x[[0, 2, 4]]
        expected = np.array([1, 3, 5])
        np.testing.assert_array_equal(result._data, expected)

    def test_len_method(self):
        """Test __len__ method"""
        x = nm.Tensor([1, 2, 3, 4, 5])
        assert len(x) == 5

        m = nm.Matrix([[1, 2], [3, 4], [5, 6]])
        assert len(m) == 3  # 最初の次元

    def test_getitem_backward(self):
        """Test getitem backward"""
        x = nm.Tensor([1, 2, 3, 4, 5], requires_grad=True)
        y = x[1:4]

        grad_output = nm.Tensor([10, 20, 30])
        y.backward(grad_output)

        # スライスされた部分のみ勾配が伝播
        expected = np.array([0, 10, 20, 30, 0])
        np.testing.assert_array_equal(x.grad._data, expected)


# ==========================================
# 7. エッジケースと境界値テスト
# ==========================================
class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary values"""

    def test_very_small_values(self):
        """Test operations with very small values"""
        x = nm.Real(1e-10, requires_grad=True)
        y = x * x
        y.backward()

        assert x.grad is not None
        expected = 2 * 1e-10
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_very_large_values(self):
        """Test operations with very large values"""
        x = nm.Real(1e10)
        y = x + 1
        assert float(y._data) > 1e10

    def test_inf_values(self):
        """Test operations with inf"""
        x = nm.Real(np.inf)
        y = nm.Real(1.0)

        result = x + y
        assert np.isinf(result._data)

    def test_nan_propagation(self):
        """Test NaN propagation"""
        x = nm.Real(np.nan)
        y = nm.Real(1.0)

        result = x + y
        assert np.isnan(result._data)

    def test_zero_division_handling(self):
        """Test zero division"""
        x = nm.Real(1.0)
        y = nm.Real(0.0)

        with np.errstate(divide="ignore"):
            result = x / y
            assert np.isinf(result._data)


# ==========================================
# 8. 型プロモーションのテスト
# ==========================================
class TestTypePromotion:
    """Test type promotion rules"""

    def test_integer_real_promotion(self):
        """Test Integer + Real promotes to Real"""
        i = nm.Integer(5)
        r = nm.Real(3.14)

        result = i + r
        assert isinstance(result, nm.Real)

    def test_real_complex_promotion(self):
        """Test Real + Complex promotes to Complex"""
        r = nm.Real(3.14)
        c = nm.Complex(1 + 2j)

        result = r + c
        assert isinstance(result, nm.Complex)

    def test_integer_complex_promotion(self):
        """Test Integer + Complex promotes to Complex"""
        i = nm.Integer(5)
        c = nm.Complex(1 + 2j)

        result = i + c
        assert isinstance(result, nm.Complex)


# ==========================================
# 9. ブロードキャストのテスト
# ==========================================
class TestBroadcasting:
    """Test broadcasting rules"""

    def test_scalar_tensor_broadcast(self):
        """Test scalar + tensor broadcasting"""
        s = nm.Real(5.0)
        t = nm.Tensor([1, 2, 3])

        result = s + t
        expected = np.array([6, 7, 8])
        np.testing.assert_array_equal(result._data, expected)

    def test_matrix_vector_broadcast(self):
        """Test matrix + vector broadcasting"""
        m = nm.Matrix([[1, 2, 3], [4, 5, 6]])
        v = nm.Tensor([10, 20, 30])

        result = m + v
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        np.testing.assert_array_equal(result._data, expected)

    def test_broadcast_backward(self):
        """Test backward with broadcasting"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.Tensor([1, 2, 3], requires_grad=False)

        z = x * y
        grad_output = nm.Tensor([1, 1, 1])
        z.backward(grad_output)

        # x.grad should be sum of y
        expected = 6.0  # 1 + 2 + 3
        np.testing.assert_almost_equal(float(x.grad._data), expected)


# ==========================================
# 10. 指数・対数関数の勾配テスト
# ==========================================
class TestExpLogGradients:
    """Test gradients of exp and log functions"""

    def test_exp_backward(self):
        """Test exp backward"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.exp(x)
        y.backward()

        # dy/dx = exp(x)
        expected = np.exp(2.0)
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_log_backward(self):
        """Test log backward"""
        x = nm.Real(2.0, requires_grad=True)
        y = nm.log(x)
        y.backward()

        # dy/dx = 1/x
        expected = 1.0 / 2.0
        np.testing.assert_almost_equal(float(x.grad._data), expected)

    def test_sqrt_backward(self):
        """Test sqrt backward"""
        x = nm.Real(4.0, requires_grad=True)
        y = nm.sqrt(x)
        y.backward()

        # dy/dx = 1/(2*sqrt(x)) = 1/4
        expected = 1.0 / (2 * np.sqrt(4.0))
        np.testing.assert_almost_equal(float(x.grad._data), expected)


# ==========================================
# 11. 複雑な計算グラフのテスト
# ==========================================
class TestComplexComputationGraphs:
    """Test complex computation graphs"""

    def test_multi_layer_backward(self):
        """Test backward through multiple layers"""
        x = nm.Real(2.0, requires_grad=True)

        # 複雑な計算グラフ
        y1 = x * 3
        y2 = y1 + 5
        y3 = y2**2
        y4 = nm.sin(y3)

        y4.backward()

        assert x.grad is not None
        # 連鎖律により勾配が計算される
        # dy4/dx = cos(y3) * 2*y2 * 3

    def test_multiple_outputs(self):
        """Test graph with multiple outputs"""
        x = nm.Real(2.0, requires_grad=True)

        y1 = x * 2
        y2 = x * 3

        # 両方からbackward
        y1.backward()
        grad_from_y1 = float(x.grad._data)

        x.zero_grad()
        y2.backward()
        grad_from_y2 = float(x.grad._data)

        assert grad_from_y1 == 2.0
        assert grad_from_y2 == 3.0

    def test_shared_intermediate_node(self):
        """Test graph with shared intermediate node"""
        x = nm.Real(2.0, requires_grad=True)

        y = x * 2
        z1 = y + 1
        z2 = y * 3

        result = z1 + z2
        result.backward()

        # dy/dx through both paths
        # dz1/dx = 2, dz2/dx = 6
        # dresult/dx = 2 + 6 = 8
        assert float(x.grad._data) == 8.0
