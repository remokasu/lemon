import pytest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from lemon import numlib as nm
from lemon import nnlib as nl
import numpy as np


def _numerical_grad(f, arr, h=1e-6):
    num = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        p, m = arr.copy(), arr.copy()
        p[idx] += h
        m[idx] -= h
        num[idx] = (f(p) - f(m)) / (2 * h)
    return num


class TestGRUBidirectionalMultiLayer:
    """双方向・2層以上の GRU（層ごとに順方向と逆方向を処理して、つないで次の層へ）"""

    @pytest.mark.parametrize("num_layers", [2, 3])
    def test_matches_layerwise_definition(self, num_layers):
        rng = np.random.default_rng(0)
        m = nl.GRU(3, 2, num_layers=num_layers, bidirectional=True, batch_first=True)
        a = rng.normal(size=(2, 4, 3))
        with nm.autograd.off:
            out, h_n = m(nm.tensor(a))

            inputs = [nm.tensor(a[:, t, :]) for t in range(4)]
            hs = []
            for layer in range(num_layers):
                hf = nm.zeros(2, 2)
                fw = []
                for t in range(4):
                    hf = m.cells_forward[layer](inputs[t], hf)
                    fw.append(hf)
                hb = nm.zeros(2, 2)
                bw = [None] * 4
                for t in range(3, -1, -1):
                    hb = m.cells_backward[layer](inputs[t], hb)
                    bw[t] = hb
                inputs = [nm.concatenate([fw[t], bw[t]], axis=1) for t in range(4)]
                hs += [hf, hb]

        assert out.shape == (2, 4, 4)
        np.testing.assert_allclose(out._data, np.stack([v._data for v in inputs], axis=1))
        np.testing.assert_allclose(h_n._data, np.stack([h._data for h in hs]))

    def test_input_gradient(self):
        rng = np.random.default_rng(1)
        m = nl.GRU(3, 2, num_layers=2, bidirectional=True, batch_first=True)
        a = rng.normal(size=(2, 4, 3))
        x = nm.tensor(a.copy(), requires_grad=True)
        out, _ = m(x)
        R = rng.normal(size=out.shape)
        nm.sum(out * nm.tensor(R)).backward()
        num = _numerical_grad(lambda aa: float(np.sum(m(nm.tensor(aa))[0]._data * R)), a)
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)

    def test_return_last_hidden(self):
        m = nl.GRU(3, 2, num_layers=2, bidirectional=True, batch_first=True, return_sequences=False)
        assert m(nm.tensor(np.ones((2, 4, 3)))).shape == (2, 4)


class TestNonSquareConvAndPooling:
    """縦と横で別々の stride / padding / dilation"""

    @staticmethod
    def _ref_conv(x, w, stride, padding, dilation):
        N, C, H, W = x.shape
        O, _, kh, kw = w.shape
        xp = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
        ekh, ekw = dilation[0] * (kh - 1) + 1, dilation[1] * (kw - 1) + 1
        oh = (H + 2 * padding[0] - ekh) // stride[0] + 1
        ow = (W + 2 * padding[1] - ekw) // stride[1] + 1
        out = np.zeros((N, O, oh, ow))
        for i in range(oh):
            for j in range(ow):
                patch = xp[:, :, i * stride[0] : i * stride[0] + ekh : dilation[0],
                           j * stride[1] : j * stride[1] + ekw : dilation[1]]
                out[:, :, i, j] = np.einsum("nchw,ochw->no", patch, w)
        return out

    @pytest.mark.parametrize(
        "stride, padding, dilation",
        [((1, 2), (1, 1), (1, 1)), ((1, 1), (1, 0), (1, 1)), ((2, 1), (0, 1), (1, 1)),
         ((1, 2), (2, 1), (2, 1))],
    )
    def test_conv_value_and_gradient(self, stride, padding, dilation):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(1, 2, 5, 6))
        w0 = rng.normal(size=(3, 2, 3, 2))
        x = nm.tensor(x0.copy(), requires_grad=True)
        w = nm.tensor(w0.copy(), requires_grad=True)
        kw = dict(stride=stride, padding=padding, dilation=dilation)

        y = nl.conv_2d(x, w, **kw)
        np.testing.assert_allclose(y._data, self._ref_conv(x0, w0, stride, padding, dilation))

        R = rng.normal(size=y.shape)
        nm.sum(y * nm.tensor(R)).backward()
        num_x = _numerical_grad(
            lambda aa: float(np.sum(nl.conv_2d(nm.tensor(aa), nm.tensor(w0), **kw)._data * R)), x0
        )
        np.testing.assert_allclose(x.grad._data, num_x, atol=1e-6)

    @pytest.mark.parametrize("pool", [nl.max_pool_2d, nl.avg_pool_2d], ids=["max", "avg"])
    def test_pooling_gradient(self, pool):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(1, 2, 5, 6))
        f = lambda t: pool(t, (2, 3), (2, 1), (1, 0))
        x = nm.tensor(a.copy(), requires_grad=True)
        y = f(x)
        assert y.shape == (1, 2, 3, 4)
        R = rng.normal(size=y.shape)
        nm.sum(y * nm.tensor(R)).backward()
        num = _numerical_grad(lambda aa: float(np.sum(f(nm.tensor(aa))._data * R)), a)
        np.testing.assert_allclose(x.grad._data, num, atol=1e-6)


class TestTopKAccuracy:
    def test_docstring_example(self):
        m = nl.TopKAccuracy(k=2)
        y_pred = nm.tensor([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
        assert m(y_pred, nm.tensor([1, 0])) == pytest.approx(100.0)

    @pytest.mark.parametrize("k, expected", [(1, 100 / 3), (2, 100 / 3), (3, 100.0)])
    def test_values(self, k, expected):
        pred = np.array([[0.1, 0.3, 0.6], [0.5, 0.3, 0.2], [0.2, 0.7, 0.1]])
        true = np.array([0, 2, 1])
        m = nl.TopKAccuracy(k=k)
        assert m(nm.tensor(pred), nm.tensor(true)) == pytest.approx(expected)
        assert m(pred, true) == pytest.approx(expected)  # ndarray でも
