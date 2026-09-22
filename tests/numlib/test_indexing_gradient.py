import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
import numpy as np


def _grad_of_sum(shape, index):
    """sum(x[index]) の x についての勾配を返す"""
    x = nm.tensor(np.arange(float(np.prod(shape))).reshape(shape), requires_grad=True)
    nm.sum(index(x)).backward()
    return np.asarray(x.grad._data)


class TestIndexingGradient:
    """インデックスの勾配が選択行列の転置 Pᵀ になることのテスト"""

    @pytest.mark.parametrize(
        "shape, index, expected",
        [
            ((3,), lambda x: x[0:2], [1, 1, 0]),
            ((3,), lambda x: x[[0, 2]], [1, 0, 1]),
            ((3,), lambda x: x[np.array([True, False, True])], [1, 0, 1]),
            ((3,), lambda x: x[1], [0, 1, 0]),
            # 同じ要素を複数回選ぶと、勾配は足し合わされる
            ((3,), lambda x: x[[0, 0, 2]], [2, 0, 1]),
            ((3,), lambda x: x[[1, 1, 1]], [0, 3, 0]),
            ((3,), lambda x: x[np.array([2, 2])], [0, 0, 2]),
            ((3,), lambda x: x[nm.tensor(np.array([2, 2]))], [0, 0, 2]),
            ((2, 2), lambda x: x[[0, 0]], [[2, 2], [0, 0]]),
            ((2, 2), lambda x: x[[0, 0, 1], [1, 1, 0]], [[0, 2], [1, 0]]),
            ((2, 2), lambda x: x[:, [0, 0]], [[2, 0], [2, 0]]),
        ],
        ids=[
            "slice", "int_array", "bool_mask", "scalar",
            "dup_twice", "dup_three_times", "dup_ndarray", "dup_tensor_key",
            "dup_rows", "dup_row_col_pairs", "dup_with_slice",
        ],
    )
    def test_gradient(self, shape, index, expected):
        np.testing.assert_allclose(_grad_of_sum(shape, index), expected)

    def test_weighted_duplicates_match_numerical_gradient(self):
        """重複を含むインデックスに重みをかけた場合も、数値微分と一致する"""
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(4, 3))
        idx = np.array([0, 2, 0, 3, 0])
        w = rng.normal(size=(5, 3))

        x = nm.tensor(x0.copy(), requires_grad=True)
        nm.sum(x[idx] * nm.tensor(w)).backward()

        expected = np.zeros_like(x0)
        np.add.at(expected, idx, w)
        np.testing.assert_allclose(x.grad._data, expected)

    def test_integer_scalar_key_keeps_orientation(self):
        """Integer で行を取り出しても、行ベクトルになる"""
        m = nm.matrix(np.arange(6.0).reshape(2, 3))
        row = m[nm.integer(1)]
        assert isinstance(row, nm.RowVector)
        np.testing.assert_array_equal(row._data, [[3, 4, 5]])
