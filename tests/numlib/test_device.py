"""Tests for device placement rules

規約: 新しく配列を作るときは今の装置に置く。すでにある配列は動かさない。
装置を移すときは `nm.to_gpu` / `nm.to_cpu` で明示する。
"""

import numpy as np
import pytest
from lemon import numlib as nm

gpu_only = pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")


def _module(x):
    return type(x._data).__module__.split(".")[0]


# ------------------------------------------------------------------
# 置かれる装置の規約
# ------------------------------------------------------------------


@gpu_only
def test_new_array_follows_the_current_device():
    """リテラルから新しく作る値は、今の装置に置かれる"""
    with nm.cuda.gpu:
        assert _module(nm.tensor([1.0, 2.0])) == "cupy"
        assert _module(nm.zeros(3)) == "cupy"
    assert _module(nm.tensor([1.0, 2.0])) == "numpy"


@gpu_only
def test_existing_array_is_not_moved():
    """すでにある配列は、GPU モードの中でも動かさない"""
    with nm.cuda.gpu:
        x = nm.tensor(np.ones((2, 2)))
        assert _module(x) == "numpy"
        assert _module(x * x) == "numpy"  # 結果も入力と同じ装置


# ------------------------------------------------------------------
# nm.to_gpu / nm.to_cpu
# ------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize(
    "make",
    [
        lambda: nm.tensor(np.ones((2, 3))),
        lambda: nm.vector(np.ones(3)),
        lambda: nm.rowvector(np.ones(3)),
        lambda: nm.matrix(np.ones((2, 3))),
        lambda: nm.real32(2.0),
        lambda: nm.integer(3),
    ],
)
def test_to_gpu_keeps_type_value_and_dtype(make):
    x = make()
    g = nm.to_gpu(x)
    assert _module(g) == "cupy"
    assert type(g) is type(x)
    assert g.dtype == x.dtype
    np.testing.assert_array_equal(g.to_numpy(), x.to_numpy())
    for slot in ("kind", "signed"):
        if hasattr(x, slot):
            assert getattr(g, slot) == getattr(x, slot)
    # 戻せる
    c = nm.to_cpu(g)
    assert _module(c) == "numpy"
    assert type(c) is type(x)
    np.testing.assert_array_equal(c.data, x.data)


@gpu_only
def test_to_device_returns_the_same_object_when_already_there():
    x = nm.tensor(np.ones(3))
    assert nm.to_cpu(x) is x
    g = nm.to_gpu(x)
    assert nm.to_gpu(g) is g


@gpu_only
def test_to_gpu_does_not_change_the_original():
    x = nm.tensor(np.ones(3), requires_grad=True)
    g = nm.to_gpu(x)
    assert _module(x) == "numpy"
    assert g is not x


@gpu_only
def test_to_gpu_is_a_differentiable_identity():
    """装置を移すのは恒等写像。勾配は元の装置に戻って流れる"""
    x = nm.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = nm.sum(nm.to_gpu(x) ** 2)
    y.backward()
    assert _module(x.grad) == "numpy"
    np.testing.assert_allclose(x.grad.data, 2 * x.data)


@gpu_only
def test_to_gpu_higher_order_gradient():
    x = nm.tensor(np.array([1.0, 2.0]), requires_grad=True)
    g = nm.grad(nm.sum(nm.to_gpu(x) ** 3), x, create_graph=True)
    assert _module(g) == "numpy"
    np.testing.assert_allclose(g.data, 3 * x.data**2)
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(h.data, 6 * x.data)


@gpu_only
def test_to_cpu_gradient_flows_back_to_gpu():
    with nm.cuda.gpu:
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        y = nm.sum(nm.to_cpu(x) ** 2)
        y.backward()
        assert _module(x.grad) == "cupy"
        np.testing.assert_allclose(x.grad.to_numpy(), 2 * x.to_numpy())


def test_to_cpu_on_cpu_value_is_identity():
    x = nm.tensor(np.ones(3), requires_grad=True)
    assert nm.to_cpu(x) is x


@pytest.mark.skipif(nm.cuda_available(), reason="CUDA is available")
def test_to_gpu_without_cupy_raises_with_hint():
    with pytest.raises(RuntimeError) as e:
        nm.to_gpu(nm.tensor(np.ones(3)))
    assert "cupy" in str(e.value).lower()


# ------------------------------------------------------------------
# 装置が食い違ったときのエラー
# ------------------------------------------------------------------


@gpu_only
@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a * b,
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a / b,
        lambda a, b: a**b,
        lambda a, b: nm.maximum(a, b),
        lambda a, b: a @ b,
        lambda a, b: nm.where(nm.tensor([[True, False], [False, True]]), a, b),
        lambda a, b: nm.concatenate([a, b]),
        lambda a, b: nm.stack([a, b]),
    ],
)
def test_mixed_device_raises_type_mismatch_with_hint(op):
    with nm.cuda.gpu:
        a = nm.tensor(np.ones((2, 2)))  # CPU
        b = nm.tensor([[1.0, 2.0], [3.0, 4.0]])  # GPU
        with pytest.raises(nm.TypeMismatchError) as e:
            op(a, b)
        msg = str(e.value)
        assert "to_gpu" in msg or "to_cpu" in msg
        assert "CPU" in msg and "GPU" in msg


@gpu_only
def test_mixed_device_dot_raises_type_mismatch():
    with nm.cuda.gpu:
        v_cpu = nm.vector(np.ones(3))
        v_gpu = nm.vector([1.0, 2.0, 3.0])
        with pytest.raises(nm.TypeMismatchError):
            nm.dot(v_cpu, v_gpu)


@gpu_only
def test_same_device_after_to_gpu_works():
    """hint のとおりに直せば通る"""
    with nm.cuda.gpu:
        a = nm.tensor(np.ones((2, 2)))
        b = nm.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = nm.to_gpu(a) * b
        assert _module(r) == "cupy"
        np.testing.assert_allclose(r.to_numpy(), b.to_numpy())


def test_type_error_that_is_not_a_device_mismatch_is_not_swallowed():
    """装置の食い違いでない TypeError は、そのまま出す"""
    x = nm.tensor(np.ones(3))
    with pytest.raises(TypeError):
        x * {"not": "a number"}


# ------------------------------------------------------------------
# インデックス・where の condition・off の中の連鎖（test-writer が見つけた抜け）
# ------------------------------------------------------------------


@gpu_only
def test_indexing_with_an_index_on_another_device_raises():
    """配列のインデックスは値と同じ装置のものだけ。黙って装置間コピーをしない"""
    with nm.cuda.gpu:
        idx_gpu = nm.tensor([0, 1])
        v_gpu = nm.tensor([1.0, 2.0, 3.0])
    x_cpu = nm.tensor(np.ones(5))
    idx_cpu = nm.tensor(np.array([0, 1]))

    with pytest.raises(nm.TypeMismatchError) as e:
        x_cpu[idx_gpu]
    assert "to_gpu" in str(e.value) or "to_cpu" in str(e.value)
    # 逆向き（cupy が host の配列を黙って受け取ってしまう）も止める
    with pytest.raises(nm.TypeMismatchError):
        v_gpu[idx_cpu]
    # 同じ装置なら通る
    np.testing.assert_allclose(v_gpu[nm.to_gpu(idx_cpu)].to_numpy(), [1.0, 2.0])


@gpu_only
def test_where_checks_the_device_of_the_condition():
    with nm.cuda.gpu:
        a = nm.tensor([1.0, 2.0])
        b = nm.tensor([3.0, 4.0])
    cond_cpu = nm.tensor(np.array([True, False]))
    with pytest.raises(nm.TypeMismatchError) as e:
        nm.where(cond_cpu, a, b)
    assert "to_gpu" in str(e.value) or "to_cpu" in str(e.value)


@gpu_only
def test_to_device_propagates_the_off_sentinel():
    """autograd.off の中で作った値を装置移動しても、黙って 0 にならない"""
    with nm.autograd.off, nm.cuda.gpu:
        x = nm.tensor([1.0, 2.0], requires_grad=True)
        z = nm.to_cpu(x + x)
        with pytest.raises(nm.GradientError, match="autograd.off"):
            nm.grad(nm.sum(z), x)


@gpu_only
def test_device_error_points_at_the_value_that_differs():
    """3 つ以上取る演算でも、実際に食い違っている値がエラーに出る"""
    with nm.cuda.gpu:
        cond = nm.tensor([True, False])
        a = nm.tensor([1.0, 2.0])
        a2 = nm.tensor([5.0, 6.0])
    b = nm.tensor(np.array([3.0, 4.0]))  # CPU
    for call in (
        lambda: nm.where(cond, a, b),
        lambda: nm.concatenate([a, a2, b]),
        lambda: nm.stack([a, a2, b]),
    ):
        with pytest.raises(nm.TypeMismatchError) as e:
            call()
        msg = str(e.value)
        assert "on GPU" in msg and "on CPU" in msg


@gpu_only
@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a == b,
        lambda a, b: a != b,
        lambda a, b: a < b,
        lambda a, b: a <= b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
    ],
)
def test_comparison_with_another_device_raises(op):
    """比較（x > 0 のようなマスク作り）でも hint つきのエラーにする"""
    with nm.cuda.gpu:
        a = nm.tensor([1.0, 2.0])
    b = nm.tensor(np.array([3.0, 4.0]))
    with pytest.raises(nm.TypeMismatchError) as e:
        op(a, b)
    assert "to_gpu" in str(e.value) or "to_cpu" in str(e.value)


@gpu_only
def test_comparison_on_the_same_device_still_works():
    with nm.cuda.gpu:
        a = nm.tensor([1.0, 2.0])
        mask = a > nm.tensor([0.0, 3.0])
    # 比較は生の配列を返す（NumType ではない）
    np.testing.assert_array_equal(nm.as_numpy(mask), [True, False])
