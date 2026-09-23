"""Tests for higher-order derivatives (stage 1)

backward(create_graph=True), nm.grad, and the higher-order gradient formulas of
the factory operations, pow, sum / mean / reshape / transpose / broadcast_to /
sum_to and type conversions.
"""

import numpy as np
import pytest
from lemon import numlib as nm

H = 1e-5  # 数値微分の刻み（float64）


def _t(data, dtype=np.float64, requires_grad=True):
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 0:
        return nm.real(data, kind=data.itemsize * 8, requires_grad=requires_grad)
    return nm.tensor(data, requires_grad=requires_grad)


# ------------------------------------------------------------------
# 演算の表: 名前 → (関数, 入力の生成). 入力は定義域の内側で、微分できない点を避ける
# ------------------------------------------------------------------

_rng = np.random.default_rng(0)
_SHAPE = (2, 3)


def _uniform(lo, hi):
    return lambda: _rng.uniform(lo, hi, size=_SHAPE)


def _signed(lo, hi):
    return lambda: _rng.uniform(lo, hi, size=_SHAPE) * _rng.choice([-1.0, 1.0], _SHAPE)


UNARY = {
    "neg": (nm.neg, _uniform(-2, 2)),
    "absolute": (nm.absolute, _signed(0.5, 2)),
    "sqrt": (nm.sqrt, _uniform(0.5, 2)),
    "exp": (nm.exp, _uniform(-1, 1)),
    "log": (nm.log, _uniform(0.5, 2)),
    "expm1": (nm.expm1, _uniform(-1, 1)),
    "log1p": (nm.log1p, _uniform(-0.5, 2)),
    "log2": (nm.log2, _uniform(0.5, 2)),
    "log10": (nm.log10, _uniform(0.5, 2)),
    "sin": (nm.sin, _uniform(-2, 2)),
    "cos": (nm.cos, _uniform(-2, 2)),
    "tan": (nm.tan, _uniform(-1, 1)),
    "arcsin": (nm.arcsin, _uniform(-0.8, 0.8)),
    "arccos": (nm.arccos, _uniform(-0.8, 0.8)),
    "arctan": (nm.arctan, _uniform(-2, 2)),
    "sinh": (nm.sinh, _uniform(-1, 1)),
    "cosh": (nm.cosh, _uniform(-1, 1)),
    "tanh": (nm.tanh, _uniform(-1, 1)),
    "arcsinh": (nm.arcsinh, _uniform(-2, 2)),
    "arccosh": (nm.arccosh, _uniform(1.5, 3)),
    "arctanh": (nm.arctanh, _uniform(-0.8, 0.8)),
    "square": (nm.square, _uniform(-2, 2)),
    "reciprocal": (nm.reciprocal, _signed(0.5, 2)),
    # pow（指数がリテラル）
    "pow_2": (lambda x: x**2, _uniform(-2, 2)),
    "pow_3": (lambda x: x**3, _uniform(-2, 2)),
    "pow_0.5": (lambda x: x**0.5, _uniform(0.5, 2)),
    "pow_-1": (lambda x: x**-1, _signed(0.5, 2)),
    "pow_2.5": (lambda x: x**2.5, _uniform(0.5, 2)),
    "pow_1": (lambda x: x**1, _uniform(-2, 2)),
}

BINARY = {
    "add": (nm.add, _uniform(-2, 2), _uniform(-2, 2)),
    "sub": (nm.sub, _uniform(-2, 2), _uniform(-2, 2)),
    "mul": (nm.mul, _uniform(-2, 2), _uniform(-2, 2)),
    "div": (nm.div, _uniform(-2, 2), _signed(0.5, 2)),
    "maximum": (nm.maximum, _uniform(-2, 0), _uniform(0, 2)),
    "minimum": (nm.minimum, _uniform(-2, 0), _uniform(0, 2)),
    "atan2": (nm.atan2, _signed(0.5, 2), _signed(0.5, 2)),
    "pow": (nm.pow, _uniform(0.5, 2), _uniform(0.5, 2)),
}

# 独自の _backward を持つ演算（線形なので、非線形な関数と組み合わせて調べる）
SHAPE_OPS = {
    "sum": lambda x: nm.sum(x**3),
    "sum_axis0": lambda x: nm.sum(nm.sum(x, axis=0) ** 3),
    "sum_axis1_keepdims": lambda x: nm.sum(nm.sum(x, axis=1, keepdims=True) ** 3),
    "mean": lambda x: nm.mean(x**3) * nm.mean(x),
    "mean_axis-1": lambda x: nm.sum(nm.mean(x, axis=-1) ** 3),
    "reshape": lambda x: nm.sum(nm.reshape(x, (3, 2)) ** 3 * _W32),
    "reshape_F": lambda x: nm.sum(nm.reshape(x, (3, 2), order="F") ** 3 * _W32),
    "transpose": lambda x: nm.sum(nm.transpose(x) ** 3 * _W32),
    "broadcast_to": lambda x: nm.sum(nm.broadcast_to(x, (4, 2, 3)) ** 3 * _W423),
    "sum_to": lambda x: nm.sum(nm.sum_to(x, (1, 3)) ** 3 * _W13),
    "vector_cast": lambda x: nm.sum(nm.vector(nm.reshape(x, -1)) ** 3),
    "matrix_cast": lambda x: nm.sum(nm.tensor(nm.matrix(x) * nm.matrix(x)) * _W23),
}

_W32 = nm.tensor(_rng.uniform(-1, 1, size=(3, 2)), requires_grad=False)
_W23 = nm.tensor(_rng.uniform(-1, 1, size=(2, 3)), requires_grad=False)
_W13 = nm.tensor(_rng.uniform(-1, 1, size=(1, 3)), requires_grad=False)
_W423 = nm.tensor(_rng.uniform(-1, 1, size=(4, 2, 3)), requires_grad=False)


def _first_grads(f, datas):
    """f(*xs)（スカラー）の 1 階の勾配（生の配列のリスト）"""
    xs = [_t(d) for d in datas]
    return [g.data for g in nm.grad(f(*xs), xs)]


def _check_hvp(f, datas):
    """
    ヘッセ行列とベクトルの積 H v を、2 階微分（create_graph=True）と、1 階の勾配の
    数値微分 (∇f(x + h v) - ∇f(x - h v)) / 2h で比べる
    """
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
        np.testing.assert_allclose(got.data, (p - m) / (2 * H), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_second_order(name):
    op, gen = UNARY[name]
    _check_hvp(lambda x: nm.sum(op(x)), [gen()])


@pytest.mark.parametrize("name", sorted(BINARY))
def test_binary_second_order(name):
    op, gen_x, gen_y = BINARY[name]
    _check_hvp(lambda x, y: nm.sum(op(x, y)), [gen_x(), gen_y()])


@pytest.mark.parametrize("name", sorted(set(BINARY) - {"add", "sub"}))
def test_binary_second_order_with_scalar_operand(name):
    """スカラーと配列の組み合わせ（sum_to で形を戻す経路。スカラー＋配列は定義されない）"""
    op, gen_x, gen_y = BINARY[name]
    _check_hvp(lambda x, y: nm.sum(op(x, y)), [gen_x()[0, 0], gen_y()])
    _check_hvp(lambda x, y: nm.sum(op(x, y)), [gen_x(), gen_y()[0, 0]])


@pytest.mark.parametrize("name", sorted(SHAPE_OPS))
def test_shape_ops_second_order(name):
    _check_hvp(SHAPE_OPS[name], [_uniform(0.5, 2)()])


def test_factory_ops_are_all_in_the_table():
    """_make_unary_op / _make_binary_op で作った演算は、すべて上の表で調べている"""
    factory = {
        name: obj
        for name, obj in vars(nm).items()
        if callable(obj)
        and getattr(obj, "__qualname__", "").startswith(
            ("_make_unary_op.", "_make_binary_op.")
        )
    }
    tested = [op for op, *_ in UNARY.values()] + [op for op, *_ in BINARY.values()]
    missing = sorted(
        name for name, obj in factory.items() if not any(obj is t for t in tested)
    )
    assert len(factory) >= 30
    assert not missing, f"not tested: {missing}"


# ------------------------------------------------------------------
# AC-14: 1 階用と高階用の勾配の式が同じ値（float32 / float64）
# ------------------------------------------------------------------


def _both_grads(op, datas, dtype):
    """backward(gradient=1) で、1 階用と高階用の勾配をそれぞれ求める"""
    out = []
    for create_graph in (False, True):
        xs = [_t(d, dtype=dtype) for d in datas]
        y = op(*xs)
        y.backward(gradient=nm.ones_like(y), create_graph=create_graph)
        out.append([x.grad for x in xs])
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", sorted(UNARY) + sorted(BINARY))
def test_first_and_graph_formulas_agree(name, dtype):
    if name in UNARY:
        op, gen = UNARY[name]
        datas = [gen()]
    else:
        op, gen_x, gen_y = BINARY[name]
        datas = [gen_x(), gen_y()]
    first, graph = _both_grads(op, datas, dtype)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    for a, b in zip(first, graph):
        np.testing.assert_allclose(b.data, a.data, rtol=rtol, atol=rtol)
        # 高階用の勾配は変数と同じ dtype
        assert b.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("name", sorted(SHAPE_OPS))
def test_first_and_graph_formulas_agree_shape_ops(name, dtype):
    data = _uniform(0.5, 2)()
    grads = []
    for create_graph in (False, True):
        x = _t(data, dtype=dtype)
        SHAPE_OPS[name](x).backward(create_graph=create_graph)
        grads.append(x.grad)
    rtol = 1e-4 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(grads[1].data, grads[0].data, rtol=rtol, atol=rtol)
    assert grads[1].dtype == grads[0].dtype


# ------------------------------------------------------------------
# AC-1, AC-3, AC-4, AC-5, AC-11
# ------------------------------------------------------------------


def test_pow_derivatives_up_to_4th():
    x = nm.real(2.0, requires_grad=True)
    y = x**3
    d1 = nm.grad(y, x, create_graph=True)
    d2 = nm.grad(d1, x, create_graph=True)
    d3 = nm.grad(d2, x, create_graph=True)
    d4 = nm.grad(d3, x)
    assert [float(d.data) for d in (d1, d2, d3, d4)] == [12.0, 12.0, 6.0, 0.0]
    assert all(isinstance(d, nm.Real) for d in (d1, d2, d3, d4))


def test_analytic_higher_order_of_exp_and_sin():
    x = nm.tensor([0.3, -1.2], requires_grad=True)
    g = nm.sum(nm.exp(x) * nm.sin(x))
    d1 = nm.grad(g, x, create_graph=True)
    d2 = nm.grad(nm.sum(d1), x)
    xd = x.data
    # (eˣ sin x)' = eˣ (sin x + cos x), (eˣ sin x)'' = 2 eˣ cos x
    np.testing.assert_allclose(d1.data, np.exp(xd) * (np.sin(xd) + np.cos(xd)))
    np.testing.assert_allclose(d2.data, 2 * np.exp(xd) * np.cos(xd))


def test_grad_does_not_touch_dot_grad():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    h = x * x
    y = nm.sum(h * x)
    sentinel = nm.tensor([7.0, 7.0], requires_grad=False)
    x.grad = sentinel
    g = nm.grad(y, x)
    assert x.grad is sentinel
    assert h.grad is None and y.grad is None
    np.testing.assert_allclose(g.data, 3 * x.data**2)
    # create_graph=False の戻り値は定数
    assert g.requires_grad is False
    # グラフは解放されないので、もう一度呼べる
    np.testing.assert_allclose(nm.grad(y, x).data, g.data)


def test_grad_list_of_variables():
    x = nm.real(2.0, requires_grad=True)
    w = nm.tensor([1.0, -1.0], requires_grad=True)
    y = nm.sum(w * w) * x
    gx, gw = nm.grad(y, (x, w))
    assert float(gx.data) == 2.0
    np.testing.assert_allclose(gw.data, 2 * w.data * 2.0)


def test_grad_with_respect_to_intermediate_node():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    h = x * x
    y = nm.sum(h * h)
    np.testing.assert_allclose(nm.grad(y, h).data, 2 * h.data)


def test_backward_create_graph():
    x = nm.real(2.0, requires_grad=True)
    y = x**3
    y.backward(create_graph=True)
    assert float(x.grad.data) == 12.0
    assert x.grad.requires_grad is True
    # x.grad = 3x² を逆伝播すると、x.grad に 6x = 12 が足し込まれる
    x.grad.backward()
    assert float(x.grad.data) == 24.0


def test_backward_create_graph_keeps_graph_of_x_grad():
    """retain_graph=False でも、create_graph=True なら x.grad のグラフは解放されない"""
    x = nm.real(2.0, requires_grad=True)
    y = x**3
    y.backward(retain_graph=False, create_graph=True)
    d2 = nm.grad(x.grad, x)
    assert float(d2.data) == 12.0


def test_grad_independent_returns_zeros():
    x = nm.vector([1.0, 2.0, 3.0], requires_grad=True)
    z = nm.vector([1.0, 1.0, 1.0], requires_grad=True)
    y = nm.sum(z * z)
    g = nm.grad(y, x)
    assert type(g) is nm.Vector and g.shape == x.shape
    np.testing.assert_array_equal(g.data, 0.0)
    # 定数（requires_grad=False）の y
    c = nm.sum(nm.vector([1.0, 2.0], requires_grad=False))
    g = nm.grad(c, x, create_graph=True)
    assert type(g) is nm.Vector and np.all(g.data == 0.0)
    # detach() で明示的に定数にした y も 0
    g = nm.grad(nm.sum(x * x).detach(), x)
    assert np.all(g.data == 0.0)


def test_higher_order_keeps_math_type():
    v = nm.vector([1.0, 2.0, 3.0], requires_grad=True)
    y = nm.sum(v * v * v)
    g = nm.grad(y, v, create_graph=True)
    assert type(g) is nm.Vector
    h = nm.grad(nm.sum(g), v)
    assert type(h) is nm.Vector
    np.testing.assert_allclose(h.data, 6 * v.data)

    # 転置（Vector ↔ RowVector）を通っても、勾配は変数の型
    y = nm.sum(nm.transpose(v) * nm.transpose(v))
    g = nm.grad(y, v, create_graph=True)
    assert type(g) is nm.Vector
    assert type(nm.grad(nm.sum(g * g), v)) is nm.Vector

    # 型の変換（Tensor → Vector）を通っても、Tensor の変数には Tensor の勾配
    t = nm.tensor([1.0, 2.0], requires_grad=True)
    y = nm.sum(nm.vector(t) ** 3)
    g = nm.grad(y, t, create_graph=True)
    assert type(g) is nm.Tensor and g.shape == (2,)
    assert type(nm.grad(nm.sum(g), t)) is nm.Tensor

    # backward(create_graph=True) の x.grad も同じ型
    v.grad = None
    nm.sum(v * v * v).backward(create_graph=True)
    assert type(v.grad) is nm.Vector
    m = nm.matrix([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    nm.sum(m * m).backward(create_graph=True)
    assert type(m.grad) is nm.Matrix


# ------------------------------------------------------------------
# AC-6: 対応していないものは GradientError（hint 付き）
# ------------------------------------------------------------------


def _square_op():
    def forward(x):
        return x * x, x

    def backward(x, g, needs_grad):
        return (g * 2 * x,)

    return nm.make_op(forward, backward)


def test_create_graph_unsupported_raises():
    square = _square_op()
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    y = nm.sum(square(x))
    with pytest.raises(nm.GradientError, match="make_op") as e:
        y.backward(create_graph=True)
    assert "Hint" in str(e.value) and "first-order" in str(e.value)
    # 逆伝播の前に調べるので、x.grad には何も足し込まれていない
    assert x.grad is None
    with pytest.raises(nm.GradientError, match="make_op"):
        nm.grad(y, x, create_graph=True)
    # 1 階は今までどおり
    np.testing.assert_allclose(nm.grad(y, x).data, 2 * x.data)

    # 段階②③まで対応したので、numlib の演算に未対応のものは残っていない
    A = nm.matrix([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    np.testing.assert_allclose(
        nm.grad(nm.sum(A @ A), A, create_graph=True).data,
        nm.grad(nm.sum(A @ A), A).data,
    )

    # 複素数
    z = nm.cmplx(1.0, 2.0, requires_grad=True)
    y = z * z
    with pytest.raises(nm.GradientError, match="complex") as e:
        y.backward(create_graph=True)
    assert "Hint" in str(e.value)
    with pytest.raises(nm.GradientError, match="complex"):
        nm.grad(y, z, create_graph=True)


# ------------------------------------------------------------------
# AC-12: グラフがない y はエラー（黙って 0 にしない）
# ------------------------------------------------------------------


def test_grad_without_graph_raises():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    with nm.autograd.off:
        y = nm.sum(x * x)
    with pytest.raises(nm.GradientError, match="autograd.off") as e:
        nm.grad(y, x)
    assert "detach" in str(e.value)
    # backward() も同じ hint（RuntimeError のまま捕まえられる）
    with pytest.raises(RuntimeError, match="autograd.off"):
        y.backward()
    # off の中の値を on に戻ってから使うと、定数として扱ったことになり 0
    with nm.autograd.off:
        c = x * x
    g = nm.grad(nm.sum(c * c), x)
    np.testing.assert_array_equal(g.data, 0.0)

    # 解放済みのグラフ
    y = nm.sum(x * x)
    y.backward()
    with pytest.raises(nm.GradientError, match="freed") as e:
        nm.grad(y, x)
    assert "retain_graph=True" in str(e.value)
    with pytest.raises(nm.GradientError, match="freed"):
        y.backward(create_graph=True)
    # create_graph=False の backward は今までどおり（何もしない）
    x.grad = None
    y.backward()
    assert x.grad is None

    # 解放済みのノードを使って作った新しい値も
    h = x * x
    nm.sum(h).backward()
    y = nm.sum(h * x)
    with pytest.raises(nm.GradientError, match="freed"):
        nm.grad(y, x)


def test_create_graph_under_autograd_off_raises():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    y = nm.sum(x * x)
    with nm.autograd.off:
        with pytest.raises(nm.GradientError, match="autograd.off") as e:
            nm.grad(y, x, create_graph=True)
        assert "autograd.on" in str(e.value)
        with pytest.raises(nm.GradientError, match="autograd.off"):
            y.backward(create_graph=True)
        # create_graph=False はグラフを作らないので使える
        np.testing.assert_allclose(nm.grad(y, x).data, 2 * x.data)


def test_grad_argument_errors():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(nm.GradientError, match="scalar") as e:
        nm.grad(x * x, x)
    assert "nm.sum" in str(e.value)
    c = nm.tensor([1.0, 2.0], requires_grad=False)
    with pytest.raises(nm.GradientError, match="requires_grad=True"):
        nm.grad(nm.sum(c * c), c)
    with pytest.raises(nm.GradientError, match="not differentiable"):
        nm.grad(nm.sum(x), nm.integer(3))
    with pytest.raises(nm.GradientError, match="not differentiable"):
        nm.grad(nm.sum(x), nm.boolean(True))


# ------------------------------------------------------------------
# AC-13: pow の微分は数学の定義どおり
# ------------------------------------------------------------------


def test_pow_derivative_math_rules():
    # x⁰ は定数関数 1: 微分は x = 0 を含めてどこでも 0（1〜3 階）
    x = nm.real(0.0, requires_grad=True)
    y = x**0
    d1 = nm.grad(y, x, create_graph=True)
    d2 = nm.grad(d1, x, create_graph=True)
    d3 = nm.grad(d2, x)
    assert [float(d.data) for d in (d1, d2, d3)] == [0.0, 0.0, 0.0]
    y = x**0
    y.backward()
    assert float(x.grad.data) == 0.0

    # 指数が NumType の 0 でも同じ
    t = nm.tensor([0.0, 2.0], requires_grad=True)
    e = nm.tensor([0.0, 0.0], requires_grad=False)
    g = nm.grad(nm.sum(t**e), t, create_graph=True)
    np.testing.assert_array_equal(g.data, [0.0, 0.0])
    np.testing.assert_array_equal(nm.grad(nm.sum(g), t).data, [0.0, 0.0])
    np.testing.assert_array_equal(nm.grad(nm.sum(t**e), t).data, [0.0, 0.0])

    # 指数についての微分 xʸ ln x: x < 0 は nan、x = 0 かつ y > 0 は 0、
    # x = 0 かつ y ≤ 0 は nan
    base = nm.tensor([-2.0, 0.0, 3.0, 0.0], requires_grad=False)
    for create_graph in (False, True):
        ex = nm.tensor([2.0, 2.0, 2.0, -1.0], requires_grad=True)
        with np.errstate(divide="ignore"):  # 順伝播の 0⁻¹ = inf
            g = nm.grad(nm.sum(base**ex), ex, create_graph=create_graph)
        assert np.isnan(g.data[0])
        assert g.data[1] == 0.0
        np.testing.assert_allclose(g.data[2], 9.0 * np.log(3.0))
        assert np.isnan(g.data[3])


def test_reshape_order_f_gradient():
    """order="F" の reshape の勾配は、同じ order で戻す（逆写像）"""
    x = nm.tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    w = np.arange(6.0).reshape(3, 2)
    nm.sum(nm.reshape(x, (3, 2), order="F") * nm.tensor(w, requires_grad=False)).backward()
    np.testing.assert_array_equal(x.grad.data, w.reshape(2, 3, order="F"))


@pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")
def test_higher_order_on_gpu():
    with nm.cuda.gpu:
        x = nm.tensor([0.5, 1.5], requires_grad=True)
        y = nm.sum(nm.tanh(x) * x**3)
        d1 = nm.grad(y, x, create_graph=True)
        d2 = nm.grad(nm.sum(d1), x)
        d2_gpu = d2.to_numpy()
    x = nm.tensor([0.5, 1.5], requires_grad=True)
    y = nm.sum(nm.tanh(x) * x**3)
    d1 = nm.grad(y, x, create_graph=True)
    d2 = nm.grad(nm.sum(d1), x)
    np.testing.assert_allclose(d2_gpu, d2.data, rtol=1e-10)


# ------------------------------------------------------------------
# 境界値: 空配列・n=1・3 次元・「型と形が食い違う」変数（Matrix の n×1）
# ------------------------------------------------------------------


def _make_variable(kind, data, requires_grad):
    ctor = {"vector": nm.vector, "matrix": nm.matrix, "tensor": nm.tensor}[kind]
    return ctor(data, requires_grad=requires_grad)


def _hvp_for_variable(f, x0, kind):
    """f(x)（スカラー）の HVP を数値微分と比べる。x は kind で指定した型・形の変数"""
    v = _rng.uniform(-1, 1, size=x0.shape)
    x = _make_variable(kind, x0, True)
    g = nm.grad(f(x), x, create_graph=True)
    s = nm.sum(g * _make_variable(kind, v, False))
    hvp = nm.grad(s, x)

    def first_grad(data):
        xx = _make_variable(kind, data, True)
        return nm.grad(f(xx), xx).data

    plus = first_grad(x0 + H * v)
    minus = first_grad(x0 - H * v)
    np.testing.assert_allclose(hvp.data, (plus - minus) / (2 * H), rtol=1e-5, atol=1e-6)
    return hvp


# name -> (変数の作り方, 期待する型, 形)。matrix_n1 は Vector と同じ形 (n, 1) だが型は Matrix
_BOUNDARY_SHAPES = {
    "vector_n1": ("vector", nm.Vector, (1,)),
    "vector_empty": ("vector", nm.Vector, (0,)),
    "matrix_n1": ("matrix", nm.Matrix, (3, 1)),
    "tensor_3d": ("tensor", nm.Tensor, (2, 3, 4)),
}


@pytest.mark.parametrize("name", sorted(_BOUNDARY_SHAPES))
def test_hvp_boundary_shapes(name):
    """空配列・n=1・3 次元 Tensor・Matrix(n×1) でも 2 階微分の値と型が正しい"""
    kind, expected_type, shape = _BOUNDARY_SHAPES[name]
    x0 = _rng.uniform(0.3, 1.5, size=shape)
    f = lambda x: nm.sum(nm.tanh(x) * x**3)
    x_ref = _make_variable(kind, x0, False)
    hvp = _hvp_for_variable(f, x0, kind)
    assert type(hvp) is expected_type
    assert hvp.shape == x_ref.shape


def test_higher_order_float32_vector_variable():
    """
    Vector 型・float32 の変数でも高階微分の値が正しく、dtype も float32 のまま

    nm.grad はスカラーの y を要求するため sum などで 0 次元を作ることになるが、
    0 次元の生成は既知の問題で float32 が float64 に上がる（今回は直さない対象）。
    ここでは backward(gradient=...) で非スカラーのまま逆伝播し、その問題を踏まずに
    Vector・float32 という型と dtype の組み合わせでの高階微分を確かめる。
    """
    data = np.array([0.3, -0.7, 1.1], dtype=np.float32)
    v = nm.vector(data, requires_grad=True)
    y = nm.tanh(v) * v**2  # shape (3, 1) のまま（Vector）
    y.backward(gradient=nm.ones_like(y), create_graph=True)
    g1 = v.grad
    assert type(g1) is nm.Vector and g1.dtype == np.float32
    xd = data.reshape(-1, 1).astype(np.float64)

    def fprime(x):
        return (1 - np.tanh(x) ** 2) * x**2 + 2 * x * np.tanh(x)

    np.testing.assert_allclose(
        g1.data.astype(np.float64), fprime(xd), rtol=1e-3, atol=1e-5
    )

    v.grad = None
    g1.backward(gradient=nm.ones_like(g1))
    g2 = v.grad
    assert type(g2) is nm.Vector and g2.dtype == np.float32
    h = 1e-3
    numeric_second = (fprime(xd + h) - fprime(xd - h)) / (2 * h)
    np.testing.assert_allclose(
        g2.data.astype(np.float64), numeric_second, rtol=1e-3, atol=1e-4
    )


# ------------------------------------------------------------------
# backward(create_graph=True) の累積・.grad の退避と復元
# ------------------------------------------------------------------


def test_backward_create_graph_twice_accumulates():
    """同じ y に backward(create_graph=True) を 2 回呼ぶと、葉の x.grad に 2 回ぶん足し込まれる"""
    x = nm.real(2.0, requires_grad=True)
    y = x**3
    y.backward(create_graph=True)
    assert float(x.grad.data) == 12.0
    y.backward(create_graph=True)
    assert float(x.grad.data) == 24.0
    assert x.grad.requires_grad is True


def test_grad_restores_pre_existing_grad_after_normal_completion():
    """呼ぶ前から .grad を持っていた中間ノード・変数も、nm.grad の後は元の値（オブジェクト）に戻る"""
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    h = x * x
    h_sentinel = nm.tensor([9.0, 9.0], requires_grad=False)
    h.grad = h_sentinel
    x_sentinel = nm.tensor([5.0, 5.0], requires_grad=False)
    x.grad = x_sentinel
    y = nm.sum(h * h)

    g = nm.grad(y, x)
    np.testing.assert_allclose(g.data, 4 * x.data**3)
    assert h.grad is h_sentinel
    assert x.grad is x_sentinel


def test_grad_restores_grad_on_exception():
    """逆伝播の途中で例外が起きても、finally で退避した .grad が元に戻る"""
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    h = x * x
    h.grad = nm.tensor([9.0, 9.0], requires_grad=False)
    h_sentinel = h.grad
    x.grad = nm.tensor([5.0, 5.0], requires_grad=False)
    x_sentinel = x.grad
    y = nm.sum(h * h)

    def boom(create_graph=False):
        raise RuntimeError("boom")

    h._backward = boom
    with pytest.raises(RuntimeError, match="boom"):
        nm.grad(y, x)
    assert h.grad is h_sentinel
    assert x.grad is x_sentinel


# ------------------------------------------------------------------
# nm.grad: x の重複・y が x そのもの
# ------------------------------------------------------------------


def test_grad_with_diamond_dependency():
    """x が複数の枝（ダイアモンド型の依存関係）に出てくる y でも、1 階・2 階とも正しく合算される"""
    x = nm.tensor([1.0, 2.0, -0.5], requires_grad=True)
    y = nm.sum(x * x + nm.exp(x) * x)
    g = nm.grad(y, x, create_graph=True)
    xd = x.data
    np.testing.assert_allclose(g.data, 2 * xd + np.exp(xd) * (xd + 1))
    g2 = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(g2.data, 2 + np.exp(xd) * (xd + 2))


def test_grad_duplicate_variable_in_list():
    """x がリストで重複しているとき、それぞれ独立したオブジェクトで同じ値が返る"""
    x = nm.real(3.0, requires_grad=True)
    y = x**3
    g1, g2 = nm.grad(y, [x, x])
    assert g1 is not g2
    np.testing.assert_allclose(g1.data, 27.0)
    np.testing.assert_allclose(g2.data, 27.0)


def test_grad_y_is_x_itself():
    """y が x そのもの（恒等写像）のとき、1 階微分は 1、2 階微分は 0（1 は x に依存しない定数）"""
    x = nm.real(5.0, requires_grad=True)
    g = nm.grad(x, x)
    assert float(g.data) == 1.0
    assert g.requires_grad is False

    g_graph = nm.grad(x, x, create_graph=True)
    assert float(g_graph.data) == 1.0
    g2 = nm.grad(g_graph, x)
    assert float(g2.data) == 0.0


# ------------------------------------------------------------------
# 番兵（autograd.off で作った値）: 多段合成・演算ごとの伝播
# ------------------------------------------------------------------


def test_backward_built_under_off_multi_step_composition():
    """off の中で何段も合成した値（a → b → y）でも、番兵が最後まで伝わってエラーになる"""
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    with nm.autograd.off:
        a = x * x
        b = a + x
        y = nm.sum(nm.tanh(b))
    with pytest.raises(nm.GradientError, match="autograd.off") as e:
        nm.grad(y, x)
    assert "detach" in str(e.value)


@pytest.mark.parametrize("name", ["pow", "sum", "transpose", "type_cast", "make_op"])
def test_backward_built_under_off_per_operation(name):
    """pow / sum / transpose / 型変換 / make_op それぞれが、off の中で作った値の番兵を持つ"""
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    square = _square_op()
    ops = {
        "pow": lambda t: nm.sum(t**2),
        "sum": lambda t: nm.sum(t * t),
        "transpose": lambda t: nm.sum(nm.transpose(nm.vector(t)) ** 2),
        "type_cast": lambda t: nm.sum(nm.vector(t) ** 2),
        "make_op": lambda t: nm.sum(square(t)),
    }
    with nm.autograd.off:
        y = ops[name](x)
    assert y.requires_grad is False
    with pytest.raises(nm.GradientError, match="autograd.off"):
        nm.grad(y, x)


# ------------------------------------------------------------------
# 未対応演算のエラーは逆伝播の前に検査するので、どの変数の .grad も変わらない
# ------------------------------------------------------------------


def test_unsupported_op_error_leaves_all_variable_grads_untouched():
    """対応済みの経路（a）と未対応の経路（make_op）が同じ y に混ざっていても、両方の .grad が変わらない"""
    square = _square_op()
    a = nm.tensor([1.0, 2.0], requires_grad=True)
    b = nm.tensor([3.0, 4.0], requires_grad=True)
    a.grad = nm.tensor([9.0, 9.0], requires_grad=False)
    b.grad = nm.tensor([9.0, 9.0], requires_grad=False)
    y = nm.sum(a * a) + nm.sum(square(b))
    with pytest.raises(nm.GradientError, match="make_op"):
        y.backward(create_graph=True)
    np.testing.assert_array_equal(a.grad.data, [9.0, 9.0])
    np.testing.assert_array_equal(b.grad.data, [9.0, 9.0])


# ------------------------------------------------------------------
# 3 階以上の微分が止まらないこと（解析解と比較）
# ------------------------------------------------------------------


def _nth_order_chain(f, x0, n):
    x = nm.real(x0, requires_grad=True)
    d = f(x)
    ds = []
    for i in range(n):
        d = nm.grad(d, x, create_graph=(i < n - 1))
        ds.append(float(d.data))
    return ds


def test_third_order_derivative_of_sin_matches_analytic():
    x0 = 0.6
    d1, d2, d3 = _nth_order_chain(nm.sin, x0, 3)
    np.testing.assert_allclose([d1, d2, d3], [np.cos(x0), -np.sin(x0), -np.cos(x0)])


def test_third_order_derivative_of_exp_matches_analytic():
    x0 = 0.6
    d1, d2, d3 = _nth_order_chain(nm.exp, x0, 3)
    np.testing.assert_allclose([d1, d2, d3], [np.exp(x0)] * 3)


def test_third_order_derivative_of_log_matches_analytic():
    x0 = 1.7
    d1, d2, d3 = _nth_order_chain(nm.log, x0, 3)
    np.testing.assert_allclose([d1, d2, d3], [1 / x0, -1 / x0**2, 2 / x0**3])


def test_third_order_derivative_of_tanh_matches_analytic():
    x0 = 0.4
    d1, d2, d3 = _nth_order_chain(nm.tanh, x0, 3)
    t = np.tanh(x0)
    yp = 1 - t**2
    ypp = -2 * t * yp
    yppp = -2 * yp**2 - 2 * t * ypp
    np.testing.assert_allclose([d1, d2, d3], [yp, ypp, yppp])


def test_fourth_order_derivative_does_not_stop():
    """sin は 4 階で元の関数に戻る。3 階を超えても計算が止まらないことの確認"""
    x0 = 0.6
    d1, d2, d3, d4 = _nth_order_chain(nm.sin, x0, 4)
    np.testing.assert_allclose(
        [d1, d2, d3, d4], [np.cos(x0), -np.sin(x0), -np.cos(x0), np.sin(x0)]
    )


@pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")
def test_higher_order_float32_vector_on_gpu():
    """Vector 型・float32 の変数の高階微分が、CPU と GPU で同じ値になる"""
    data = np.array([0.3, -0.7, 1.1], dtype=np.float32)

    def second_order(data):
        v = nm.vector(data, requires_grad=True)
        y = nm.tanh(v) * v**2
        y.backward(gradient=nm.ones_like(y), create_graph=True)
        g1 = v.grad
        v.grad = None
        g1.backward(gradient=nm.ones_like(g1))
        return v.grad

    with nm.cuda.gpu:
        g2_gpu = second_order(data).to_numpy()
    g2_cpu = second_order(data)
    assert g2_cpu.dtype == np.float32
    np.testing.assert_allclose(g2_gpu, g2_cpu.data, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------
# autograd.off の中で作った値は、どの演算でも番兵を持つ（黙って 0 を返さない）
# ------------------------------------------------------------------

OFF_OPS = {
    "add": lambda x, m: x + x,
    "mul": lambda x, m: x * x,
    "pow": lambda x, m: x**2,
    "sum": lambda x, m: nm.sum(x),
    "mean": lambda x, m: nm.mean(x),
    "reshape": lambda x, m: nm.reshape(x, (2, 1)),
    "transpose": lambda x, m: nm.transpose(m),
    "type_cast": lambda x, m: nm.vector(x),
    "broadcast_to": lambda x, m: nm.broadcast_to(x, (3, 2)),
    "sum_to": lambda x, m: nm.sum_to(m, (1, 2)),
    "matmul": lambda x, m: m @ m,
    "dot": lambda x, m: nm.dot(nm.vector(x), nm.vector(x)),
    "get_item": lambda x, m: x[0],
    "concatenate": lambda x, m: nm.concatenate([x, x]),
    "stack": lambda x, m: nm.stack([x, x]),
    "clip": lambda x, m: nm.clip(x, 0.0, 1.0),
    "expand_dims": lambda x, m: nm.expand_dims(x, 0),
    "squeeze": lambda x, m: nm.squeeze(nm.tensor(x.data.reshape(1, 2), requires_grad=True)),
    "var": lambda x, m: nm.var(x),
    "logsumexp": lambda x, m: nm.logsumexp(x),
    "where": lambda x, m: nm.where(nm.tensor([True, False], requires_grad=False), x, x),
    "split": lambda x, m: nm.split(x, 2)[0],
    "tile": lambda x, m: nm.tile(x, 2),
    "make_op": lambda x, m: _square_op()(x),
}


@pytest.mark.parametrize("name", sorted(OFF_OPS))
def test_value_built_under_autograd_off_is_detected(name):
    """autograd.off の中で作った値を微分しようとしたら、黙って 0 にせずエラーにする"""
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    m = nm.matrix([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    with nm.autograd.off:
        y = OFF_OPS[name](x, m)
    assert y.requires_grad is False
    with pytest.raises(nm.GradientError, match="autograd.off") as e:
        y.backward()
    assert "detach" in str(e.value)
