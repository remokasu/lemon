import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


class TestBroadcastTo:
    """broadcast_to関数のテスト"""

    def test_basic_broadcast(self):
        """基本的なブロードキャスト"""
        x = nm.tensor([1, 2, 3])
        y = nm.broadcast_to(x, (2, 3))

        assert y.shape == (2, 3)
        expected = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(y.data, expected)

    def test_multi_dimensional_broadcast(self):
        """多次元ブロードキャスト"""
        x = nm.tensor([[1, 2]])  # (1, 2)
        y = nm.broadcast_to(x, (3, 4, 2))

        assert y.shape == (3, 4, 2)
        assert np.all(y.data[:, :, 0] == 1)
        assert np.all(y.data[:, :, 1] == 2)

    def test_same_shape_returns_self(self):
        """既に同じ形状の場合は同一オブジェクトを返す"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.broadcast_to(x, (2, 2))
        assert y is x

    def test_scalar_broadcast(self):
        """スカラーのブロードキャスト"""
        x = nm.tensor(5.0)
        y = nm.broadcast_to(x, (3, 3))

        assert y.shape == (3, 3)
        assert np.all(y.data == 5.0)

    def test_gradient_flow(self):
        """勾配の正しい伝播"""
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.broadcast_to(x, (3, 2))
        loss = nm.sum(y)
        loss.backward()

        # 各要素が3回使われるので勾配は3
        np.testing.assert_array_equal(x.grad.data, [3.0, 3.0])

    def test_complex_gradient(self):
        """複雑な形状での勾配"""
        x = nm.tensor([[1.0]], requires_grad=True)  # (1, 1)
        y = nm.broadcast_to(x, (2, 3, 4))
        loss = nm.sum(y * 2)
        loss.backward()

        # 2*3*4=24回使われ、各々2倍されている
        assert x.grad.data[0, 0] == 48.0


class TestSumTo:
    """sum_to関数のテスト"""

    def test_basic_sum(self):
        """基本的な軸に沿った合計"""
        x = nm.tensor([[1, 2, 3], [4, 5, 6]])
        y = nm.sum_to(x, (3,))

        assert y.shape == (3,)
        np.testing.assert_array_equal(y.data, [5, 7, 9])

    def test_same_shape_returns_self(self):
        """既に同じ形状の場合"""
        x = nm.tensor([1, 2, 3])
        y = nm.sum_to(x, (3,))
        assert y is x

    def test_scalar_sum(self):
        """スカラーへの合計"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.sum_to(x, ())

        assert y.shape == ()
        assert y.data == 10

    def test_partial_sum(self):
        """部分的な軸の合計"""
        x = nm.tensor(np.ones((2, 3, 4, 5)))
        y = nm.sum_to(x, (3, 1, 5))

        assert y.shape == (3, 1, 5)
        # 2*4=8個ずつ合計される
        assert np.all(y.data == 8)

    def test_gradient_flow(self):
        """勾配の正しい伝播"""
        x = nm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = nm.sum_to(x, (2,))
        loss = nm.sum(y)
        loss.backward()

        # 勾配は均等に配分
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(x.grad.data, expected)


class TestBroadcastSumInverse:
    """broadcast_toとsum_toが逆操作であることの検証"""

    def test_inverse_operations(self):
        """相互に逆操作"""
        x = nm.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # broadcast -> sum
        y = nm.broadcast_to(x, (4, 3))
        z = nm.sum_to(y, (3,))

        # 4回足される
        expected = x.data * 4
        np.testing.assert_array_almost_equal(z.data, expected)

    def test_gradient_consistency(self):
        """勾配の一貫性"""
        # 元の形状
        x = nm.tensor([[1.0, 2.0]], requires_grad=True)

        # broadcast -> sum -> broadcast
        y = nm.broadcast_to(x, (3, 1, 2))
        z = nm.sum_to(y, (1, 2))
        w = nm.broadcast_to(z, (2, 1, 2))

        loss = nm.sum(w)
        loss.backward()

        # 最終的に6回使われる (3 -> 1 -> 2)
        assert x.grad.data[0, 0] == 6.0
        assert x.grad.data[0, 1] == 6.0
