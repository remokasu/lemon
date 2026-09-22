import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


def _numerical_grads(a, b, R, h=1e-6):
    """L = sum((a @ b) * R) の a, b についての数値微分"""
    grads = []
    for which in (0, 1):
        arr = (a, b)[which]
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            p, m = arr.copy(), arr.copy()
            p[idx] += h
            m[idx] -= h
            fp = np.sum(((p @ b) if which == 0 else (a @ p)) * R)
            fm = np.sum(((m @ b) if which == 0 else (a @ m)) * R)
            num[idx] = (fp - fm) / (2 * h)
        grads.append(num)
    return grads


class TestBatchedMatmul:
    """N次元の matmul（最後の2軸で行列積、それより前の軸はバッチ）のテスト"""

    @pytest.mark.parametrize(
        "x_shape, y_shape",
        [
            ((3, 4), (4, 5)),
            ((2, 3, 4), (4, 5)),
            ((3, 4), (2, 4, 5)),
            ((2, 3, 4), (2, 4, 5)),
            ((2, 3, 4, 5), (2, 3, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 6)),
            ((2, 3, 4, 5), (5, 6)),
            ((4,), (4, 5)),
            ((3, 4), (4,)),
            ((4,), (2, 4, 5)),
            ((2, 3, 4), (4,)),
            ((4,), (4,)),
        ],
    )
    def test_value_and_gradient(self, x_shape, y_shape):
        rng = np.random.default_rng(0)
        a = rng.normal(size=x_shape)
        b = rng.normal(size=y_shape)
        x = nm.tensor(a.copy(), requires_grad=True)
        y = nm.tensor(b.copy(), requires_grad=True)

        z = nm.matmul(x, y)
        np.testing.assert_allclose(z._data, a @ b)

        R = rng.normal(size=z.shape)
        nm.sum(z * nm.tensor(R)).backward()
        num_a, num_b = _numerical_grads(a, b, R)
        np.testing.assert_allclose(x.grad._data, num_a, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(y.grad._data, num_b, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize(
        "x_shape, y_shape",
        [
            ((1, 3, 4, 5), (2, 1, 5, 6)),  # サイズ 1 のバッチを暗黙に広げない
            ((2, 3, 4), (3, 4, 5)),  # バッチの形が違う
            ((3, 4), (5, 6)),  # 内側の次元が合わない
        ],
    )
    def test_undefined_shapes_raise(self, x_shape, y_shape):
        x = nm.tensor(np.ones(x_shape))
        y = nm.tensor(np.ones(y_shape))
        with pytest.raises(nm.DimensionError):
            nm.matmul(x, y)
