import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import numpy as np
from lemon import numlib as nm


class TestClip:
    """clip関数のテスト"""

    def test_basic_clip(self):
        """基本的なクリッピング"""
        x = nm.tensor([-2, -1, 0, 1, 2])
        y = nm.clip(x, -1, 1)

        expected = np.array([-1, -1, 0, 1, 1])
        np.testing.assert_array_equal(y.data, expected)

    def test_min_only(self):
        """最小値のみ指定"""
        x = nm.tensor([-2, -1, 0, 1, 2])
        y = nm.clip(x, min_val=0)

        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(y.data, expected)

    def test_max_only(self):
        """最大値のみ指定"""
        x = nm.tensor([-2, -1, 0, 1, 2])
        y = nm.clip(x, max_val=0)

        expected = np.array([-2, -1, 0, 0, 0])
        np.testing.assert_array_equal(y.data, expected)

    def test_no_clip(self):
        """クリップなし（Noneを指定）"""
        x = nm.tensor([-2, -1, 0, 1, 2])
        y = nm.clip(x, None, None)

        np.testing.assert_array_equal(y.data, x.data)

    def test_2d_clip(self):
        """2次元テンソルのクリップ"""
        x = nm.tensor([[-2, -1], [0, 1], [2, 3]])
        y = nm.clip(x, 0, 2)

        expected = np.array([[0, 0], [0, 1], [2, 2]])
        np.testing.assert_array_equal(y.data, expected)

    def test_gradient_flow(self):
        """勾配の伝播（範囲内のみ）"""
        x = nm.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = nm.clip(x, -1, 1)
        loss = nm.sum(y)
        loss.backward()

        # クリップされた要素の勾配は0
        expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(x.grad.data, expected)

    def test_gradient_partial_clip(self):
        """部分的なクリップでの勾配"""
        x = nm.tensor([0.5, 1.5, 2.5], requires_grad=True)
        y = nm.clip(x, max_val=2.0)
        loss = nm.sum(y * 2)
        loss.backward()

        # 2.5だけクリップされる
        np.testing.assert_array_equal(x.grad.data, [2.0, 2.0, 0.0])


class TestWhere:
    """where関数のテスト"""

    def test_basic_selection(self):
        """基本的な条件選択"""
        condition = nm.tensor([True, False, True, False])
        x = nm.tensor([1, 2, 3, 4])
        y = nm.tensor([10, 20, 30, 40])

        result = nm.where(condition, x, y)
        expected = np.array([1, 20, 3, 40])
        np.testing.assert_array_equal(result.data, expected)

    def test_numpy_condition(self):
        """NumPy配列を条件として使用"""
        condition = np.array([True, False, True])
        x = nm.tensor([1.0, 2.0, 3.0])
        y = nm.tensor([10.0, 20.0, 30.0])

        result = nm.where(condition, x, y)
        expected = np.array([1.0, 20.0, 3.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_broadcast_scalar(self):
        """スカラーのブロードキャスト"""
        condition = nm.tensor([[True, False], [False, True]])
        x = nm.tensor(1)  # スカラー
        y = nm.tensor([[10, 20], [30, 40]])

        result = nm.where(condition, x, y)
        expected = np.array([[1, 20], [30, 1]])
        np.testing.assert_array_equal(result.data, expected)

    def test_broadcast_arrays(self):
        """配列のブロードキャスト"""
        condition = nm.tensor([True, False, True])
        x = nm.tensor([[1], [2]])  # (2, 1)
        y = nm.tensor([10, 20, 30])  # (3,)

        # ブロードキャストして (2, 3)
        result = nm.where(condition, x, y)
        assert result.shape == (2, 3)

    def test_gradient_flow_x(self):
        """xの勾配（条件True）"""
        condition = nm.tensor([True, False, True])
        x = nm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = nm.tensor([10.0, 20.0, 30.0], requires_grad=False)

        result = nm.where(condition, x, y)
        loss = nm.sum(result)
        loss.backward()

        # Trueの位置のみ勾配
        np.testing.assert_array_equal(x.grad.data, [1.0, 0.0, 1.0])

    def test_gradient_flow_y(self):
        """yの勾配（条件False）"""
        condition = nm.tensor([True, False, True])
        x = nm.tensor([1.0, 2.0, 3.0], requires_grad=False)
        y = nm.tensor([10.0, 20.0, 30.0], requires_grad=True)

        result = nm.where(condition, x, y)
        loss = nm.sum(result)
        loss.backward()

        # Falseの位置のみ勾配
        np.testing.assert_array_equal(y.grad.data, [0.0, 1.0, 0.0])

    def test_gradient_flow_both(self):
        """両方の勾配"""
        condition = nm.tensor([True, False])
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tensor([10.0, 20.0], requires_grad=True)

        result = nm.where(condition, x, y)
        loss = nm.sum(result * 2)
        loss.backward()

        np.testing.assert_array_equal(x.grad.data, [2.0, 0.0])
        np.testing.assert_array_equal(y.grad.data, [0.0, 2.0])
