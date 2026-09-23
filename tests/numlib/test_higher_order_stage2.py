"""Tests for higher-order derivatives (SPEC-0001 stage 2)

matmul / dot / get_item / where / clip / expand_dims / squeeze / var / logsumexp /
concatenate / stack / split / tile と、nm.jacobian / nm.hessian。

検証は段階①と同じで、ヘッセ行列とベクトルの積（2 階微分）を、1 階の勾配の数値微分と比べる。
"""

import numpy as np
import pytest
from lemon import numlib as nm

H = 1e-5
_rng = np.random.default_rng(1)


def _t(data, requires_grad=True):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 0:
        return nm.real(data, requires_grad=requires_grad)
    return nm.tensor(data, requires_grad=requires_grad)


def _first_grads(f, datas):
    xs = [_t(d) for d in datas]
    return [g.data for g in nm.grad(f(*xs), xs)]


def _check_hvp(f, datas, rtol=1e-5):
    """H·v を、2 階微分と「1 階の勾配の数値微分」で比べる"""
    vs = [_rng.uniform(-1, 1, size=np.shape(d)) for d in datas]
    xs = [_t(d) for d in datas]
    gs = nm.grad(f(*xs), xs, create_graph=True)
    s = nm.sum(gs[0] * _t(vs[0], requires_grad=False))
    for g, v in zip(gs[1:], vs[1:]):
        s = s + nm.sum(g * _t(v, requires_grad=False))
    hvp = nm.grad(s, xs)

    plus = _first_grads(f, [d + H * v for d, v in zip(datas, vs)])
    minus = _first_grads(f, [d - H * v for d, v in zip(datas, vs)])
    for got, p, m in zip(hvp, plus, minus):
        np.testing.assert_allclose(got.data, (p - m) / (2 * H), rtol=rtol, atol=1e-6)


_W23 = nm.tensor(_rng.uniform(-1, 1, size=(2, 3)), requires_grad=False)
_W32 = nm.tensor(_rng.uniform(-1, 1, size=(3, 2)), requires_grad=False)
_W22 = nm.tensor(_rng.uniform(-1, 1, size=(2, 2)), requires_grad=False)
_W43 = nm.tensor(_rng.uniform(-1, 1, size=(4, 3)), requires_grad=False)

# 演算ごとに「スカラーを返す非線形な関数」を作る（線形な演算は 3 乗と組み合わせる）
STAGE2_OPS = {
    "expand_dims": lambda x: nm.sum(nm.expand_dims(x, 0) ** 3),
    "squeeze": lambda x: nm.sum(nm.squeeze(nm.expand_dims(x, 1)) ** 3),
    "clip": lambda x: nm.sum(nm.clip(x, -0.5, 0.5) ** 3),
    "clip_min_only": lambda x: nm.sum(nm.clip(x, -0.5, None) ** 3),
    "tile": lambda x: nm.sum(nm.tile(x, 2) ** 3),
    "tile_2d": lambda x: nm.sum(nm.tile(x, (2, 1)) ** 3 * _W43),
    "concatenate": lambda x: nm.sum(nm.concatenate([x, x * x], axis=0) ** 3 * _W43),
    "concatenate_axis1": lambda x: nm.sum(nm.concatenate([x, x**2], axis=1) ** 3),
    "stack": lambda x: nm.sum(nm.stack([x, x * x]) ** 3),
    "split": lambda x: nm.sum(nm.split(x, 3, axis=1)[1] ** 3)
    + nm.sum(nm.split(x, 3, axis=1)[2] ** 4),
    "split_indices": lambda x: nm.sum(nm.split(x, [1], axis=0)[0] ** 3),
    "where": lambda x: nm.sum(
        nm.where(nm.tensor(np.array([[True, False, True], [False, True, False]])), x**3, x**2)
    ),
    "get_item_slice": lambda x: nm.sum(x[0:1] ** 3) + nm.sum(x[1:2] ** 4),
    "get_item_int": lambda x: nm.sum(x[1] ** 3),
    "get_item_repeated": lambda x: nm.sum(x[[0, 0, 1]] ** 3),
    "var": lambda x: nm.var(x) * nm.sum(x**2),
    "var_axis": lambda x: nm.sum(nm.var(x, axis=1) ** 3),
    "var_ddof": lambda x: nm.sum(nm.var(x, axis=0, ddof=1) ** 3),
    "var_keepdims": lambda x: nm.sum(nm.var(x, axis=1, keepdims=True) ** 3),
    "logsumexp": lambda x: nm.logsumexp(x) ** 3,
    "logsumexp_axis": lambda x: nm.sum(nm.logsumexp(x, axis=1) ** 3),
    "logsumexp_keepdims": lambda x: nm.sum(nm.logsumexp(x, axis=0, keepdims=True) ** 3),
    "get_item_mask": lambda x: nm.sum(
        x[nm.tensor(np.array([[True, False, True], [False, True, False]]))] ** 3
    ),
}


@pytest.mark.parametrize("name", sorted(STAGE2_OPS))
def test_stage2_second_order(name):
    data = _rng.uniform(-1.5, 1.5, size=(2, 3))
    _check_hvp(STAGE2_OPS[name], [data])


@pytest.mark.parametrize("name", sorted(STAGE2_OPS))
def test_stage2_first_and_graph_formulas_agree(name):
    """1 階用と高階用の式が同じ値になる"""
    data = _rng.uniform(-1.5, 1.5, size=(2, 3))
    grads = []
    for create_graph in (False, True):
        x = _t(data)
        STAGE2_OPS[name](x).backward(create_graph=create_graph)
        grads.append(x.grad)
    np.testing.assert_allclose(grads[1].data, grads[0].data, rtol=1e-12)
    assert grads[1].dtype == grads[0].dtype


def test_logsumexp_over_all_elements_is_a_scalar():
    """全要素の縮約はスカラー（nm.sum / nm.mean と同じ）"""
    d = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.3]])
    r = nm.logsumexp(nm.tensor(d))
    assert r.shape == ()
    np.testing.assert_allclose(r.data, np.log(np.exp(d).sum()))
    # 数学の型でも同じ
    v = nm.vector([0.5, 1.0, 2.0])
    assert nm.logsumexp(v).shape == ()
    # 軸を指定したときは今までどおり
    assert nm.logsumexp(nm.tensor(d), axis=1).shape == (2,)
    assert nm.logsumexp(nm.tensor(d), axis=1, keepdims=True).shape == (2, 1)
