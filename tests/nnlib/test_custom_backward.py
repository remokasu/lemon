import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
from lemon import nnlib as nl
import numpy as np


def _t(*shape):
    return nm.tensor(np.random.default_rng(0).normal(size=shape), requires_grad=True)


# backward を自作している演算: (入力を作る関数, 演算)
CUSTOM_OPS = {
    "layer_norm": (lambda: [_t(4, 5), _t(5), _t(5)],
                   lambda x, w, b: nl.layer_norm(x, (5,), w, b)),
    "rms_norm": (lambda: [_t(4, 5), _t(5)], lambda x, w: nl.rms_norm(x, w)),
    "batch_norm_1d": (lambda: [_t(6, 3), _t(3), _t(3)],
                      lambda x, g, b: nl.batch_norm_1d(x, g, b, training=True)),
    "batch_norm_2d": (lambda: [_t(3, 2, 3, 3), _t(2), _t(2)],
                      lambda x, g, b: nl.batch_norm_2d(x, g, b, training=True)),
    "conv_2d": (lambda: [_t(2, 2, 4, 4), _t(3, 2, 3, 3), _t(3)],
                lambda x, w, b: nl.conv_2d(x, w, b, padding=1)),
    "silu": (lambda: [_t(4, 5)], nl.silu),
    "glu": (lambda: [_t(4, 6)], nl.glu),
    "max_pool_2d": (lambda: [_t(1, 2, 4, 4)], lambda x: nl.MaxPool2d(2)(x)),
}


class TestGradientAccumulation:
    """自作の backward が、勾配を安全に累積できることのテスト"""

    @pytest.mark.parametrize("name", CUSTOM_OPS)
    def test_accumulate_onto_readonly_gradient(self, name):
        """
        入力の勾配が先に nm.sum の backward（read-only のブロードキャスト view）から
        入っていても、累積で落ちない
        """
        make, op = CUSTOM_OPS[name]
        inputs = make()
        loss = nm.sum(op(*inputs))
        for t in inputs:
            loss = loss + nm.sum(t)
        loss.backward()  # ValueError: output array is read-only にならない

    @pytest.mark.parametrize("name", CUSTOM_OPS)
    def test_accumulated_gradient_value(self, name):
        """sum(op(x)) + sum(x) の勾配は、sum(op(x)) の勾配 + 1"""
        make, op = CUSTOM_OPS[name]

        inputs = make()
        nm.sum(op(*inputs)).backward()
        base = [np.array(t.grad._data) for t in inputs]

        inputs = make()
        loss = nm.sum(op(*inputs))
        for t in inputs:
            loss = loss + nm.sum(t)
        loss.backward()

        for t, g in zip(inputs, base):
            np.testing.assert_allclose(t.grad._data, g + 1.0, rtol=1e-10, atol=1e-12)


class TestBatchNorm2dOutputType:
    """batch_norm_2d の出力と勾配の中身が、入れ子の NumType にならないことのテスト"""

    def test_training_output_is_array(self):
        x, g, b = _t(3, 2, 3, 3), _t(2), _t(2)
        y = nl.batch_norm_2d(x, g, b, training=True)
        assert isinstance(y._data, np.ndarray)

        nm.sum(y * nm.tensor(np.ones(y.shape))).backward()
        for t in (x, g, b):
            assert isinstance(t.grad._data, np.ndarray)


def _numerical_grad(f, arr, h=1e-6):
    num = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        p, m = arr.copy(), arr.copy()
        p[idx] += h
        m[idx] -= h
        num[idx] = (f(p) - f(m)) / (2 * h)
    return num


class TestLayerNormAffine:
    """layer_norm の weight と bias は、それぞれ独立に掛ける・足す"""

    @pytest.mark.parametrize("use_w, use_b", [(True, True), (True, False), (False, True)])
    def test_value_and_gradient(self, use_w, use_b):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 4))
        w = rng.normal(size=4) * 3
        b = rng.normal(size=4)
        R = rng.normal(size=(3, 4))

        def run(aa, grad=False):
            x = nm.tensor(aa, requires_grad=grad)
            W = nm.tensor(w) if use_w else None
            B = nm.tensor(b) if use_b else None
            return x, nl.layer_norm(x, (4,), W, B)

        x, y = run(a.copy(), grad=True)
        norm = (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
        expected = norm * (w if use_w else 1.0) + (b if use_b else 0.0)
        np.testing.assert_allclose(y._data, expected)

        nm.sum(y * nm.tensor(R)).backward()
        num = _numerical_grad(lambda aa: float(np.sum(run(aa)[1]._data * R)), a)
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)


class TestBatchNormGradient:
    """BatchNorm の勾配: 推論モードと、gamma / beta の片方だけのとき"""

    CASES = {
        "1d": (nl.batch_norm_1d, (6, 3), 3),
        "2d": (nl.batch_norm_2d, (3, 2, 3, 3), 2),
    }

    @pytest.mark.parametrize("kind", CASES)
    @pytest.mark.parametrize(
        "training, use_g, use_b, running",
        [
            (False, True, True, True),  # 推論モード: 移動平均・移動分散は定数
            (True, True, False, False),  # gamma のみ
            (True, False, True, False),  # beta のみ
        ],
        ids=["eval_running_stats", "gamma_only", "beta_only"],
    )
    def test_gradient(self, kind, training, use_g, use_b, running):
        fn, shape, C = self.CASES[kind]
        rng = np.random.default_rng(0)
        a = rng.normal(size=shape)
        g = rng.normal(size=C) * 2
        b = rng.normal(size=C)
        rm = rng.normal(size=C) if running else None
        rv = rng.random(C) + 0.5 if running else None
        R = rng.normal(size=shape)

        def run(aa, grad=False):
            x = nm.tensor(aa, requires_grad=grad)
            y = fn(
                x,
                nm.tensor(g) if use_g else None,
                nm.tensor(b) if use_b else None,
                running_mean=None if rm is None else rm.copy(),
                running_var=None if rv is None else rv.copy(),
                training=training,
            )
            return x, y

        x, y = run(a.copy(), grad=True)
        nm.sum(y * nm.tensor(R)).backward()
        num = _numerical_grad(lambda aa: float(np.sum(run(aa)[1]._data * R)), a)
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)


class TestMaxPoolPadding:
    """最大値プーリングの padding は -inf 扱い（埋めた値が最大値に選ばれない）"""

    def test_negative_input_with_padding(self):
        rng = np.random.default_rng(0)
        a = -np.abs(rng.normal(size=(1, 1, 4, 4))) - 1.0
        y = nl.max_pool_2d(nm.tensor(a), 3, 1, 1)
        assert float(y._data.max()) < 0.0

        padded = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
        expected = np.array(
            [[[[padded[0, 0, i : i + 3, j : j + 3].max() for j in range(4)] for i in range(4)]]]
        )
        np.testing.assert_allclose(y._data, expected)

    def test_gradient_with_padding(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(1, 2, 5, 5))
        R = rng.normal(size=(1, 2, 3, 3))
        x = nm.tensor(a.copy(), requires_grad=True)
        nm.sum(nl.max_pool_2d(x, 3, 2, 1) * nm.tensor(R)).backward()
        num = _numerical_grad(
            lambda aa: float(np.sum(nl.max_pool_2d(nm.tensor(aa), 3, 2, 1)._data * R)), a
        )
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)


class TestGradientReachesEarlierLayers:
    """プーリングやマスクの前にある層まで、勾配が流れることのテスト"""

    @pytest.mark.parametrize(
        "pool",
        [lambda: nl.AvgPool2d(2), lambda: nl.AdaptiveAvgPool2d((1, 1)),
         lambda: nl.GlobalAveragePooling2d(), lambda: nl.MaxPool2d(2)],
        ids=["AvgPool2d", "AdaptiveAvgPool2d", "GlobalAveragePooling2d", "MaxPool2d"],
    )
    def test_conv_before_pooling_gets_gradient(self, pool):
        conv = nl.Conv2d(1, 2, 3, padding=1)
        model = nl.Sequential(conv, pool(), nl.Flatten())
        out = model(nm.tensor(np.random.default_rng(0).normal(size=(2, 1, 4, 4))))
        nm.sum(out * out).backward()
        assert conv.weight.grad is not None

    @pytest.mark.parametrize(
        "shape, k, s, p",
        [((2, 2, 4, 4), 2, 2, 0), ((1, 2, 4, 4), 3, 1, 1), ((1, 2, 5, 5), 3, 2, 0)],
    )
    def test_avg_pool_gradient_matches_numerical(self, shape, k, s, p):
        rng = np.random.default_rng(0)
        a = rng.normal(size=shape)
        x = nm.tensor(a.copy(), requires_grad=True)
        y = nl.avg_pool_2d(x, k, s, p)
        R = rng.normal(size=y.shape)
        nm.sum(y * nm.tensor(R)).backward()
        num = _numerical_grad(
            lambda aa: float(np.sum(nl.avg_pool_2d(nm.tensor(aa), k, s, p)._data * R)), a
        )
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)

    def test_masked_attention_trains_query_and_key(self):
        """マスクを渡しても、W_q と W_k に勾配が流れる"""
        mha = nl.MultiHeadAttention(8, 2)
        x = nm.tensor(np.random.default_rng(0).normal(size=(2, 4, 8)))
        nm.sum(mha(x, x, x, mask=nl.causal_mask(4)) ** 2).backward()
        for name in ("W_q", "W_k", "W_v", "W_o"):
            assert getattr(mha, name).grad is not None, name
