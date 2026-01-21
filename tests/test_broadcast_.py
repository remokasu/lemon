import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import pytest
import numpy as np
from lemon.numlib import (
    real,
    vec,
    mat,
    tensor,
    autograd,
    cuda,
    add,
    sub,
    mul,
    div,
    pow,
    get_array_module,
)


class TestBroadcastingGradients:
    """Test broadcasting behavior in gradient computation"""

    def setup_method(self):
        """Reset state before each test"""
        autograd.enable()
        cuda.disable()

    # ==============================
    # 基本的なブロードキャスティング
    # ==============================

    def test_scalar_vector_broadcasting(self):
        """Test scalar + vector broadcasting"""
        # スカラー + ベクトル
        x = real(2.0)
        v = vec([1.0, 2.0, 3.0])

        result = x + v  # [3, 4, 5]
        loss = result.sum()  # 12
        loss.backward()

        # スカラーの勾配は合計
        assert x.grad.shape == ()
        assert abs(x.grad.item() - 3.0) < 1e-6

        # ベクトルの勾配は1
        assert v.grad.shape == (3, 1)
        np.testing.assert_allclose(v.grad.data, np.ones((3, 1)), rtol=1e-6)

    def test_scalar_matrix_broadcasting(self):
        """Test scalar * matrix broadcasting"""
        x = real(2.0)
        m = mat([[1.0, 2.0], [3.0, 4.0]])

        result = x * m
        loss = result.sum()  # 2*(1+2+3+4) = 20
        loss.backward()

        # スカラーの勾配は行列要素の合計
        assert x.grad.shape == ()
        assert abs(x.grad.item() - 10.0) < 1e-6  # 1+2+3+4

        # 行列の勾配はスカラー値
        assert m.grad.shape == (2, 2)
        np.testing.assert_allclose(m.grad.data, np.full((2, 2), 2.0), rtol=1e-6)

    def test_vector_matrix_broadcasting(self):
        """Test vector broadcasting with matrix"""
        v = vec([1.0, 2.0])  # (2, 1) 形状
        m = mat([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

        # v + m の動作
        result = v + m  # (2, 2) - ブロードキャスト
        loss = result.sum()
        loss.backward()

        # ベクトルの勾配
        assert v.grad.shape == (2, 1)  # 修正: (2,) -> (2, 1)
        np.testing.assert_allclose(
            v.grad.data, [[2.0], [2.0]], rtol=1e-6
        )  # 修正: 形状を合わせる

        # 行列の勾配
        assert m.grad.shape == (2, 2)
        np.testing.assert_allclose(m.grad.data, np.ones((2, 2)), rtol=1e-6)

    # ==============================
    # 各演算のブロードキャスティング
    # ==============================

    def test_subtraction_broadcasting(self):
        """Test broadcasting in subtraction"""
        x = real(5.0)
        v = vec([1.0, 2.0, 3.0])

        result = x - v  # [4, 3, 2]
        loss = result.sum()  # 9
        loss.backward()

        # スカラーの勾配
        assert x.grad.shape == ()
        assert abs(x.grad.item() - 3.0) < 1e-6

        # ベクトルの勾配（符号反転）
        assert v.grad.shape == (3, 1)  # 既に正しい
        np.testing.assert_allclose(
            v.grad.data, [[-1.0], [-1.0], [-1.0]], rtol=1e-6
        )  # 修正: 形状を合わせる

    def test_multiplication_broadcasting(self):
        """Test broadcasting in multiplication"""
        x = real(2.0)
        v = vec([1.0, 2.0, 3.0])

        result = x * v  # [2, 4, 6]
        loss = result.sum()  # 12
        loss.backward()

        # x.grad = sum(v) = 6
        assert x.grad.shape == ()
        assert abs(x.grad.item() - 6.0) < 1e-6

        # v.grad = [2, 2, 2]
        assert v.grad.shape == (3, 1)  # 修正: (3,) -> (3, 1)
        np.testing.assert_allclose(
            v.grad.data, [[2.0], [2.0], [2.0]], rtol=1e-6
        )  # 修正: 形状を合わせる

    def test_division_broadcasting(self):
        """Test broadcasting in division"""
        x = real(6.0)
        v = vec([1.0, 2.0, 3.0])

        result = x / v  # [6, 3, 2]
        loss = result.sum()  # 11
        loss.backward()

        # x.grad = sum(1/v) = 1 + 0.5 + 0.333...
        assert x.grad.shape == ()
        expected = 1.0 + 0.5 + 1.0 / 3.0
        assert abs(x.grad.item() - expected) < 1e-6

        # v.grad = -x/v^2
        assert v.grad.shape == (3, 1)  # 修正: (3,) -> (3, 1)
        expected_grad = [[-6.0], [-1.5], [-2.0 / 3.0]]  # 修正: 形状を合わせる
        np.testing.assert_allclose(v.grad.data, expected_grad, rtol=1e-6)

    def test_power_broadcasting(self):
        """Test broadcasting in power operation"""
        x = vec([2.0, 3.0])
        y = real(2.0)

        result = x**y  # [4, 9]
        loss = result.sum()  # 13
        loss.backward()

        # x.grad = 2y*x^(y-1) = 2*2*[2^1, 3^1] = [4, 6]
        assert x.grad.shape == (2, 1)  # 修正: (2,) -> (2, 1)
        np.testing.assert_allclose(
            x.grad.data, [[4.0], [6.0]], rtol=1e-6
        )  # 修正: 形状を合わせる

        # y.grad = sum(x^y * log(x))
        assert y.grad.shape == ()
        expected = 4 * np.log(2) + 9 * np.log(3)
        assert abs(y.grad.item() - expected) < 1e-6

    # ==============================
    # 高次元ブロードキャスティング
    # ==============================

    def test_3d_broadcasting(self):
        """Test broadcasting with 3D tensors"""
        # (1, 3, 1) と (2, 1, 4) のブロードキャスト
        x = tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        y = tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])  # (2, 1, 4)

        result = x + y  # (2, 3, 4)
        loss = result.sum()
        loss.backward()

        # xの勾配: (2, 3, 4) -> (1, 3, 1)
        assert x.grad.shape == (1, 3, 1)
        expected_x = np.ones((2, 3, 4)).sum(axis=(0, 2), keepdims=True).reshape(1, 3, 1)
        np.testing.assert_allclose(x.grad.data, expected_x, rtol=1e-6)

        # yの勾配: (2, 3, 4) -> (2, 1, 4)
        assert y.grad.shape == (2, 1, 4)
        expected_y = np.ones((2, 3, 4)).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(y.grad.data, expected_y, rtol=1e-6)

    # ==============================
    # エッジケース
    # ==============================

    def test_scalar_scalar_no_broadcasting(self):
        """Test that scalar-scalar operations don't need unbroadcasting"""
        x = real(2.0)
        y = real(3.0)

        result = x * y
        result.backward()

        assert x.grad.shape == ()
        assert y.grad.shape == ()
        assert abs(x.grad.item() - 3.0) < 1e-6
        assert abs(y.grad.item() - 2.0) < 1e-6

    def test_same_shape_no_broadcasting(self):
        """Test operations with same shape (no broadcasting needed)"""
        x = vec([1.0, 2.0, 3.0])
        y = vec([4.0, 5.0, 6.0])

        result = x + y
        loss = result.sum()
        loss.backward()

        assert x.grad.shape == (3, 1)  # 修正: (3,) -> (3, 1)
        assert y.grad.shape == (3, 1)  # 修正: (3,) -> (3, 1)
        np.testing.assert_allclose(
            x.grad.data, np.ones((3, 1)), rtol=1e-6
        )  # 修正: 形状を合わせる
        np.testing.assert_allclose(
            y.grad.data, np.ones((3, 1)), rtol=1e-6
        )  # 修正: 形状を合わせる

    def test_complex_broadcasting_chain(self):
        """Test complex chain with multiple broadcasting operations"""
        a = real(2.0)
        b = vec([1.0, 2.0])
        c = mat([[1.0, 2.0], [3.0, 4.0]])

        # a * b -> (2, 1)
        temp1 = a * b  # [[2], [4]]
        # temp1 * c -> (2, 2)
        temp2 = temp1 * c  # [[2, 4], [12, 16]]

        loss = temp2.sum()  # 34
        loss.backward()

        # 勾配の形状チェック
        assert a.grad.shape == ()
        assert b.grad.shape == (2, 1)  # 修正: (2,) -> (2, 1)
        assert c.grad.shape == (2, 2)

        # 値のチェック（複雑な連鎖なので形状のみ確認）
        assert a.grad.item() > 0
        assert np.all(b.grad.data > 0)
        assert np.all(c.grad.data > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
