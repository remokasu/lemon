"""Additional coverage for device placement rules (memo/numlib_device_rule.md).

`test_device.py` で担保済みの基本（型/値/dtype の保持、`to_gpu`/`to_cpu` の恒等写像、
CUDA が無い環境でのエラー、代表的な演算での装置食い違いエラー）とは重複しない範囲を補強する:

- `to_gpu`/`to_cpu` の値のバリエーション（空配列・0 次元・高次元・float16/complex64・
  requires_grad=False・detach 済みの値）
- 勾配（装置をまたぐ計算グラフ、勾配の累積、`nm.grad` のリスト指定、高階微分）
- `autograd.off` との組み合わせ（番兵・`nm.grad` が `GradientError` になるか）
- まだテストしていない演算（`atan2`、`minimum`）の装置食い違いエラー、
  装置と形の両方が食い違う場合の優先度
- `get_array_module` を「配列そのものから判定する」に変えた影響
  （GPU を無効にしても cupy の値を演算に使えること、`nm.zeros` 等の
  「今の装置に置く」規約・`zeros_like` 等の「入力の装置のまま」規約が壊れていないこと）
- 数学の型（Vector/Matrix の演算結果）が移動で保たれること
"""

import numpy as np
import pytest
from lemon import numlib as nm

gpu_only = pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")


def _module(x):
    return type(x._data).__module__.split(".")[0]


# ------------------------------------------------------------------
# to_gpu / to_cpu の値のバリエーション
# ------------------------------------------------------------------


@gpu_only
def test_to_gpu_keeps_empty_array():
    x = nm.tensor(np.array([]))
    g = nm.to_gpu(x)
    assert _module(g) == "cupy"
    assert g.shape == (0,)
    c = nm.to_cpu(g)
    assert c.shape == (0,)


@gpu_only
def test_to_gpu_keeps_high_dimensional_tensor():
    x = nm.tensor(np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5))
    g = nm.to_gpu(x)
    assert _module(g) == "cupy"
    assert g.shape == x.shape
    np.testing.assert_array_equal(g.to_numpy(), x.data)


@gpu_only
@pytest.mark.parametrize(
    "make",
    [
        lambda: nm.tensor(np.array([1.0, 2.0], dtype=np.float16)),
        lambda: nm.tensor(np.array([1 + 2j, 3 + 4j], dtype=np.complex64)),
    ],
)
def test_to_gpu_keeps_float16_and_complex64_dtype(make):
    x = make()
    g = nm.to_gpu(x)
    assert _module(g) == "cupy"
    assert g.dtype == x.dtype
    np.testing.assert_array_equal(g.to_numpy(), x.data)


@gpu_only
@pytest.mark.parametrize(
    "make",
    [
        lambda: nm.boolean(True),
        lambda: nm.integer(200, kind=8, signed=False),
        lambda: nm.real16(1.5),
        lambda: nm.cmplx64(1 + 2j),
    ],
)
def test_to_gpu_keeps_zero_dim_scalar_kind_and_signed(make):
    """0 次元の Boolean / Integer(kind/signed) / Real16 / Complex64 も型と属性を保つ"""
    x = make()
    g = nm.to_gpu(x)
    assert _module(g) == "cupy"
    assert type(g) is type(x)
    assert g.dtype == x.dtype
    assert g.shape == ()
    for slot in ("kind", "signed"):
        if hasattr(x, slot):
            assert getattr(g, slot) == getattr(x, slot)
    c = nm.to_cpu(g)
    np.testing.assert_array_equal(c.data, x.data)


@gpu_only
def test_to_gpu_of_requires_grad_false_value_stays_non_differentiable():
    x = nm.tensor(np.array([1.0, 2.0]), requires_grad=False)
    g = nm.to_gpu(x)
    assert g.requires_grad is False
    assert _module(g) == "cupy"


@gpu_only
def test_to_gpu_of_detached_value_has_no_gradient_path():
    """detach 済みの値を移しても、元の変数への勾配は 0（数学的にそう定義される）"""
    x = nm.tensor(np.array([1.0, 2.0]), requires_grad=True)
    d = x.detach()
    g = nm.to_gpu(d)
    assert g.requires_grad is False
    y = nm.sum(g)
    result = nm.grad(y, x)
    np.testing.assert_allclose(result.data, [0.0, 0.0])


# ------------------------------------------------------------------
# 勾配: 装置をまたぐ計算グラフ
# ------------------------------------------------------------------


@gpu_only
def test_gradient_round_trip_cpu_gpu_cpu():
    """CPU -> GPU -> CPU と往復する式でも、勾配は元の（CPU の）装置に戻る"""
    x = nm.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = nm.sum(nm.to_cpu(nm.to_gpu(x) ** 2))
    y.backward()
    assert _module(x.grad) == "numpy"
    np.testing.assert_allclose(x.grad.data, 2 * x.data)


@gpu_only
def test_backward_accumulates_across_device_transfer_over_two_calls():
    """同じ変数を経由するグラフで backward() を 2 回呼ぶと勾配が累積する"""
    x = nm.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = nm.sum(nm.to_gpu(x) ** 2)
    y.backward(retain_graph=True)
    y.backward(retain_graph=True)
    np.testing.assert_allclose(x.grad.data, 2 * (2 * x.data))


@gpu_only
def test_grad_with_list_of_variables_on_different_devices():
    """nm.grad にリストで渡した変数がそれぞれ別の装置にあっても、各々元の装置で返る"""
    x_cpu = nm.tensor(np.array([1.0, 2.0]), requires_grad=True)
    with nm.cuda.gpu:
        x_gpu = nm.tensor([3.0, 4.0], requires_grad=True)
    y = nm.sum(nm.to_gpu(x_cpu) * x_gpu)
    g_cpu, g_gpu = nm.grad(y, [x_cpu, x_gpu])
    assert _module(g_cpu) == "numpy"
    assert _module(g_gpu) == "cupy"
    np.testing.assert_allclose(g_cpu.data, x_gpu.to_numpy())
    np.testing.assert_allclose(g_gpu.to_numpy(), x_cpu.data)


@gpu_only
def test_create_graph_gradient_of_gradient_through_to_cpu():
    """to_cpu 側でも高階微分できる（to_gpu 側は test_device.py に既存）"""
    with nm.cuda.gpu:
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        g = nm.grad(nm.sum(nm.to_cpu(x) ** 3), x, create_graph=True)
        assert _module(g) == "cupy"
        np.testing.assert_allclose(g.to_numpy(), 3 * x.to_numpy() ** 2)
        h = nm.grad(nm.sum(g), x)
        np.testing.assert_allclose(h.to_numpy(), 6 * x.to_numpy())


# ------------------------------------------------------------------
# autograd.off との組み合わせ
# ------------------------------------------------------------------


@gpu_only
def test_to_gpu_of_off_created_leaf_is_graphless_inside_off():
    """autograd.off の中で requires_grad な葉を to_gpu すると、番兵が付いて
    off の中で nm.grad を呼べば GradientError になる"""
    x = nm.tensor(np.array([1.0, 2.0]), requires_grad=True)  # off の外で作った葉
    with nm.cuda.gpu, nm.autograd.off:
        g = nm.to_gpu(x)
        assert g.requires_grad is False
        with pytest.raises(nm.GradientError):
            nm.grad(nm.sum(g), x)


@gpu_only
def test_to_gpu_of_leaf_moved_outside_off_is_a_normal_leaf():
    """比較対象: off の外で移した場合は普通に微分できる"""
    x = nm.tensor(np.array([1.0, 2.0]), requires_grad=True)
    g = nm.to_gpu(x)
    y = nm.sum(g**2)
    result = nm.grad(y, x)
    np.testing.assert_allclose(result.data, 2 * x.data)


# ------------------------------------------------------------------
# 装置の食い違いエラー: まだテストしていない演算
# ------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: nm.atan2(a, b),
        lambda a, b: nm.minimum(a, b),
    ],
)
def test_mixed_device_atan2_and_minimum_raise_type_mismatch_with_hint(op):
    with nm.cuda.gpu:
        a = nm.tensor(np.ones((2, 2)))  # CPU
        b = nm.tensor([[1.0, 2.0], [3.0, 4.0]])  # GPU
        with pytest.raises(nm.TypeMismatchError) as e:
            op(a, b)
        msg = str(e.value)
        assert "to_gpu" in msg or "to_cpu" in msg
        assert "CPU" in msg and "GPU" in msg


@gpu_only
def test_mixed_device_dot_error_takes_priority_over_shape_mismatch():
    """装置の食い違いと形の食い違いが両方あっても、装置のエラー（hint 付き）が出る"""
    with nm.cuda.gpu:
        a = nm.vector(np.ones(3))  # CPU, shape 3
        b = nm.vector([1.0, 2.0, 3.0, 4.0])  # GPU, shape 4 (形も違う)
        with pytest.raises(nm.TypeMismatchError) as e:
            nm.dot(a, b)
        msg = str(e.value)
        assert "CPU" in msg and "GPU" in msg


def test_type_error_from_pow_that_is_not_a_device_mismatch_is_not_swallowed():
    """pow の try/except は装置の食い違いだけを拾う。他の TypeError は素通りする"""
    x = nm.tensor(np.ones(3))
    with pytest.raises(TypeError):
        x ** {"not": "a number"}


def test_type_error_from_concatenate_that_is_not_a_device_mismatch_is_not_swallowed():
    x = nm.tensor(np.ones((2, 2)))
    with pytest.raises(TypeError):
        nm.concatenate([x, {"not": "a tensor"}])


# ------------------------------------------------------------------
# get_array_module の変更の影響
# ------------------------------------------------------------------


@gpu_only
def test_gpu_value_stays_usable_after_leaving_gpu_context():
    """GPU モードを抜けた（今の装置が CPU に戻った）あとでも、
    すでにある cupy の値は cupy のまま演算できる（配列そのものから装置を判定するため）"""
    with nm.cuda.gpu:
        g = nm.tensor([1.0, 2.0, 3.0])
    assert nm.get_array_module(g._data).__name__ == "cupy"
    r = g * 2.0
    assert _module(r) == "cupy"
    np.testing.assert_allclose(r.to_numpy(), [2.0, 4.0, 6.0])
    s = nm.sum(g)
    assert _module(s) == "cupy"


@gpu_only
@pytest.mark.parametrize(
    "make",
    [
        lambda: nm.zeros(3),
        lambda: nm.ones(3),
        lambda: nm.randn(3),
    ],
)
def test_zeros_ones_randn_follow_the_current_device(make):
    """nm.zeros/ones/randn のような「新しく配列を作る」ファクトリは今の装置に従う
    （datasets / nnlib/data がこのパターンに依存している）"""
    with nm.cuda.gpu:
        v = make()
        assert _module(v) == "cupy"
    v2 = make()
    assert _module(v2) == "numpy"


@gpu_only
def test_zeros_like_follows_the_input_device_not_the_current_device():
    """zeros_like/ones_like は「既存の配列」を受け取るので、今の装置ではなく
    入力の装置のまま置かれる（規約表の 2 行目）"""
    x_cpu = nm.tensor(np.ones(3))
    with nm.cuda.gpu:
        z = nm.zeros_like(x_cpu)
        o = nm.ones_like(x_cpu)
    assert _module(z) == "numpy"
    assert _module(o) == "numpy"


# ------------------------------------------------------------------
# 数学の型（Vector / Matrix）が移動で保たれること
# ------------------------------------------------------------------


@gpu_only
def test_dot_result_stays_real_type_after_device_move():
    v1 = nm.vector(np.array([1.0, 2.0, 3.0]))
    v2 = nm.vector(np.array([4.0, 5.0, 6.0]))
    d = nm.dot(v1, v2)
    assert type(d) is nm.Real
    g = nm.to_gpu(d)
    assert type(g) is nm.Real
    assert _module(g) == "cupy"
    np.testing.assert_allclose(g.to_numpy(), d.data)


@gpu_only
def test_matmul_result_stays_matrix_type_after_device_move():
    m1 = nm.matrix(np.ones((2, 3)))
    m2 = nm.matrix(np.ones((3, 2)))
    mm = m1 @ m2
    assert type(mm) is nm.Matrix
    g = nm.to_gpu(mm)
    assert type(g) is nm.Matrix
    assert _module(g) == "cupy"
    np.testing.assert_array_equal(g.to_numpy(), mm.data)
