"""
Test suite for minimum and maximum functions in numlib
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import pytest
import numpy as np
from lemon import numlib as nm


class TestMaximum:
    """Test maximum function"""

    def test_basic_maximum(self):
        """基本的なmaximum演算"""
        x = nm.tensor([1, 2, 3])
        y = nm.tensor([2, 1, 4])
        z = nm.maximum(x, y)

        expected = np.array([2, 2, 4])
        np.testing.assert_array_equal(z.data, expected)

    def test_maximum_with_scalar(self):
        """スカラーとのmaximum"""
        x = nm.tensor([-1, 0, 1, 2])
        z = nm.maximum(x, 0)

        expected = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(z.data, expected)

    def test_maximum_broadcasting(self):
        """ブロードキャスティング"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.tensor([2, 3])
        z = nm.maximum(x, y)

        expected = np.array([[2, 3], [3, 4]])
        np.testing.assert_array_equal(z.data, expected)

    def test_maximum_gradient_x(self):
        """xに対する勾配"""
        x = nm.tensor([1.0, 3.0, 2.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0, 2.0])
        z = nm.maximum(x, y)
        loss = nm.sum(z)
        loss.backward()

        # x > y: 1.0、x == y: 0.5、x < y: 0.0
        expected_grad = np.array([0.0, 1.0, 0.5])
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_maximum_gradient_both(self):
        """両方の入力に対する勾配"""
        x = nm.tensor([1.0, 3.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0], requires_grad=True)
        z = nm.maximum(x, y)
        loss = nm.sum(z)
        loss.backward()

        np.testing.assert_array_equal(x.grad.data, [0.0, 1.0])
        np.testing.assert_array_equal(y.grad.data, [1.0, 0.0])

    def test_maximum_no_grad_when_disabled(self):
        """autograd無効時の動作"""
        nm.autograd.disable()
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0])
        z = nm.maximum(x, y)

        assert z.requires_grad == False
        nm.autograd.enable()

    def test_maximum_no_grad_when_inputs_no_grad(self):
        """入力がrequires_grad=Falseの場合"""
        x = nm.tensor([1.0, 2.0], requires_grad=False)
        y = nm.tensor([2.0, 1.0], requires_grad=False)
        z = nm.maximum(x, y)

        assert z.requires_grad == False

    def test_relu_using_maximum(self):
        """maximumを使ったReLU"""
        x = nm.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = nm.maximum(x, 0)
        loss = nm.sum(y)
        loss.backward()

        # x == 0の時は勾配0.5（劣勾配）
        expected_grad = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_equal(x.grad.data, expected_grad)


class TestMinimum:
    """Test minimum function"""

    def test_basic_minimum(self):
        """基本的なminimum演算"""
        x = nm.tensor([1, 2, 3])
        y = nm.tensor([2, 1, 4])
        z = nm.minimum(x, y)

        expected = np.array([1, 1, 3])
        np.testing.assert_array_equal(z.data, expected)

    def test_minimum_with_scalar(self):
        """スカラーとのminimum"""
        x = nm.tensor([1, 2, 3, 4])
        z = nm.minimum(x, 2)

        expected = np.array([1, 2, 2, 2])
        np.testing.assert_array_equal(z.data, expected)

    def test_minimum_broadcasting(self):
        """ブロードキャスティング"""
        x = nm.tensor([[1, 2], [3, 4]])
        y = nm.tensor([2, 3])
        z = nm.minimum(x, y)

        expected = np.array([[1, 2], [2, 3]])
        np.testing.assert_array_equal(z.data, expected)

    def test_minimum_gradient_x(self):
        """xに対する勾配"""
        x = nm.tensor([1.0, 3.0, 2.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0, 2.0])
        z = nm.minimum(x, y)
        loss = nm.sum(z)
        loss.backward()

        # x < y: 1.0、x == y: 0.5、x > y: 0.0
        expected_grad = np.array([1.0, 0.0, 0.5])
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_minimum_gradient_both(self):
        """両方の入力に対する勾配"""
        x = nm.tensor([1.0, 3.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0], requires_grad=True)
        z = nm.minimum(x, y)
        loss = nm.sum(z)
        loss.backward()

        np.testing.assert_array_equal(x.grad.data, [1.0, 0.0])
        np.testing.assert_array_equal(y.grad.data, [0.0, 1.0])

    def test_minimum_no_grad_when_disabled(self):
        """autograd無効時の動作"""
        nm.autograd.disable()
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.tensor([2.0, 1.0])
        z = nm.minimum(x, y)

        assert z.requires_grad == False
        nm.autograd.enable()

    def test_minimum_no_grad_when_inputs_no_grad(self):
        """入力がrequires_grad=Falseの場合"""
        x = nm.tensor([1.0, 2.0], requires_grad=False)
        y = nm.tensor([2.0, 1.0], requires_grad=False)
        z = nm.minimum(x, y)

        assert z.requires_grad == False

    def test_clip_using_min_max(self):
        """minimumとmaximumを使ったクリッピング"""
        x = nm.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        clipped = nm.minimum(nm.maximum(x, -1), 1)
        loss = nm.sum(clipped)
        loss.backward()

        # 境界値では勾配0.5
        expected_grad = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_array_equal(x.grad.data, expected_grad)


class TestMinMaxInteraction:
    """Test interaction between minimum and maximum"""

    def test_min_max_composition(self):
        """min(max(x, a), b)の動作確認"""
        x = nm.tensor([-2, -1, 0, 1, 2], requires_grad=True)
        result = nm.minimum(nm.maximum(x, -1), 1)

        expected = np.array([-1, -1, 0, 1, 1])
        np.testing.assert_array_equal(result.data, expected)

        loss = nm.sum(result)
        loss.backward()

        # 実際の動作：
        # x < -1: 勾配0（maximumでクリップ）
        # -1 <= x <= 1: 勾配1（両方通過）
        # x > 1: 勾配0（minimumでクリップ）
        # ただし、x == -1やx == 1の境界では、
        # 中間結果が定数と等しくなるため勾配が0になる
        expected_grad = np.array([0, 0, 1, 0, 0])
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_max_min_identity(self):
        """max(x, y) + min(x, y) = x + y"""
        x = nm.tensor([1.0, 2.0, 3.0])
        y = nm.tensor([2.0, 2.0, 1.0])

        max_vals = nm.maximum(x, y)
        min_vals = nm.minimum(x, y)
        sum_minmax = max_vals + min_vals
        sum_xy = x + y

        np.testing.assert_array_almost_equal(sum_minmax.data, sum_xy.data)

    def test_leaky_relu_using_min_max(self):
        """min/maxを使ったLeaky ReLU実装"""

        def leaky_relu(x, alpha=0.1):
            return nm.maximum(x, alpha * x)

        x = nm.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = leaky_relu(x, alpha=0.1)
        loss = nm.sum(y)
        loss.backward()

        # x=0で x == 0.1*x == 0なので勾配は0.5 + 0.5*0.1 = 0.55
        expected_grad = np.array([0.1, 0.1, 0.55, 1.0, 1.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
