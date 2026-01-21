import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import numpy as np
from lemon import numlib as nm


class TestExpandDims:
    """expand_dims関数のテスト"""

    def test_axis_positions(self):
        """各軸位置での次元追加"""
        x = nm.tensor([1, 2, 3])

        # 先頭に追加
        y0 = nm.expand_dims(x, 0)
        assert y0.shape == (1, 3)

        # 末尾に追加
        y1 = nm.expand_dims(x, 1)
        assert y1.shape == (3, 1)

        # 負のインデックス
        y_neg = nm.expand_dims(x, -1)
        assert y_neg.shape == (3, 1)

    def test_multi_dimensional(self):
        """多次元テンソルでの次元追加"""
        x = nm.tensor([[1, 2], [3, 4]])  # (2, 2)

        y = nm.expand_dims(x, 1)  # (2, 1, 2)
        assert y.shape == (2, 1, 2)
        np.testing.assert_array_equal(y.data[0, 0], [1, 2])

    def test_gradient_flow(self):
        """勾配の伝播"""
        x = nm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = nm.expand_dims(x, 0)
        loss = nm.sum(y * 2)
        loss.backward()

        np.testing.assert_array_equal(x.grad.data, [2.0, 2.0, 2.0])


class TestSqueeze:
    """squeeze関数のテスト"""

    def test_remove_all_ones(self):
        """サイズ1の全次元を削除"""
        x = nm.tensor([[[1]], [[2]], [[3]]])  # (3, 1, 1)
        y = nm.squeeze(x)

        assert y.shape == (3,)
        np.testing.assert_array_equal(y.data, [1, 2, 3])

    def test_specific_axis(self):
        """特定軸のみ削除"""
        x = nm.tensor([[[1, 2]], [[3, 4]]])  # (2, 1, 2)

        y = nm.squeeze(x, axis=1)
        assert y.shape == (2, 2)
        np.testing.assert_array_equal(y.data, [[1, 2], [3, 4]])

    def test_no_squeeze_needed(self):
        """削除する次元がない場合"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.squeeze(x)

        assert y.shape == x.shape
        np.testing.assert_array_equal(y.data, x.data)

    def test_gradient_flow(self):
        """勾配の伝播"""
        x = nm.tensor([[[1.0]], [[2.0]]], requires_grad=True)
        y = nm.squeeze(x)
        loss = nm.sum(y)
        loss.backward()

        assert x.grad.shape == x.shape
        np.testing.assert_array_equal(x.grad.data.flatten(), [1.0, 1.0])


class TestSplit:
    """split関数のテスト"""

    def test_equal_split(self):
        """均等分割"""
        x = nm.tensor([1, 2, 3, 4, 5, 6])
        parts = nm.split(x, 3)

        assert len(parts) == 3
        for i, part in enumerate(parts):
            expected = [i * 2 + 1, i * 2 + 2]
            np.testing.assert_array_equal(part.data, expected)

    def test_index_split(self):
        """インデックス指定分割"""
        x = nm.tensor([1, 2, 3, 4, 5])
        parts = nm.split(x, [1, 3])

        assert len(parts) == 3
        np.testing.assert_array_equal(parts[0].data, [1])
        np.testing.assert_array_equal(parts[1].data, [2, 3])
        np.testing.assert_array_equal(parts[2].data, [4, 5])

    def test_2d_split(self):
        """2次元での分割"""
        x = nm.tensor([[1, 2, 3], [4, 5, 6]])

        # 列方向で分割
        parts = nm.split(x, 3, axis=1)
        assert len(parts) == 3
        np.testing.assert_array_equal(parts[0].data, [[1], [4]])

    def test_gradient_flow(self):
        """勾配の伝播"""
        x = nm.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        parts = nm.split(x, 2)

        # 異なる部分に異なる演算
        loss = nm.sum(parts[0]) * 2 + nm.sum(parts[1]) * 3
        loss.backward()

        np.testing.assert_array_equal(x.grad.data, [2.0, 2.0, 3.0, 3.0])


class TestTile:
    """tile関数のテスト"""

    def test_1d_tile(self):
        """1次元タイル"""
        x = nm.tensor([1, 2])
        y = nm.tile(x, 3)

        expected = np.array([1, 2, 1, 2, 1, 2])
        np.testing.assert_array_equal(y.data, expected)

    def test_2d_tile(self):
        """2次元タイル"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.tile(x, (2, 3))

        assert y.shape == (4, 6)
        # パターンの確認
        np.testing.assert_array_equal(y.data[:2, :2], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(y.data[2:, :2], [[1, 2], [3, 4]])

    def test_tile_with_expansion(self):
        """次元拡張を伴うタイル"""
        x = nm.tensor([1, 2])  # (2,)
        y = nm.tile(x, (3, 2))  # -> (3, 4)

        assert y.shape == (3, 4)
        expected_row = [1, 2, 1, 2]
        for i in range(3):
            np.testing.assert_array_equal(y.data[i], expected_row)

    def test_gradient_flow_1d(self):
        """1次元での勾配の伝播"""
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tile(x, 3)
        loss = nm.sum(y)
        loss.backward()

        # 各要素が3回使われる
        np.testing.assert_array_equal(x.grad.data, [3.0, 3.0])

    def test_gradient_flow_2d(self):
        """2次元での勾配の伝播"""
        x = nm.tensor([[1.0, 2.0]], requires_grad=True)
        y = nm.tile(x, (3, 2))
        loss = nm.sum(y)
        loss.backward()

        # 各要素が6回使われる (3*2)
        np.testing.assert_array_equal(x.grad.data, [[6.0, 6.0]])

    def test_gradient_flow_with_expansion(self):
        """次元拡張を伴う勾配の伝播"""
        x = nm.tensor([1.0, 2.0], requires_grad=True)  # (2,)
        y = nm.tile(x, (3, 2))  # -> (3, 4)
        loss = nm.sum(y * 2)
        loss.backward()

        # 各要素が6回使われ、2倍される
        np.testing.assert_array_equal(x.grad.data, [12.0, 12.0])
