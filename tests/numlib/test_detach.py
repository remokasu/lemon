"""Tests for NumType.detach()"""

import numpy as np
import pytest
from lemon import numlib as nm


@pytest.mark.parametrize(
    "make",
    [
        lambda: nm.tensor([[1.0, 2.0], [3.0, 4.0]]),
        lambda: nm.vector([1.0, 2.0]),
        lambda: nm.rowvector([1.0, 2.0]),
        lambda: nm.matrix([[1.0, 2.0], [3.0, 4.0]]),
        lambda: nm.real(2.0),
        lambda: nm.real32(2.0),
        lambda: nm.cmplx64(1.0, 2.0),
        lambda: nm.int8(3),
        lambda: nm.uint16(3),
        lambda: nm.boolean(True),
    ],
)
def test_detach_keeps_type_value_and_dtype(make):
    x = make()
    d = x.detach()
    assert type(d) is type(x)
    assert d.dtype == x.dtype
    np.testing.assert_array_equal(d.data, x.data)
    assert d.requires_grad is False
    for slot in ("kind", "signed"):
        if hasattr(x, slot):
            assert getattr(d, slot) == getattr(x, slot)


def test_detach_stops_gradient_on_that_path_only():
    x = nm.real(3.0, requires_grad=True)
    y = x * x.detach()  # d/dx = x.detach() = 3 (the detached side contributes nothing)
    y.backward()
    assert float(x.grad) == pytest.approx(3.0)


def test_detach_result_has_no_graph():
    x = nm.vector([1.0, 2.0], requires_grad=True)
    d = (x * x).detach()
    assert d.grad is None
    assert d._prev == set()


def test_detach_shares_memory():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    d = x.detach()
    assert np.shares_memory(d.data, x.data)
    d += nm.tensor([10.0, 10.0])
    np.testing.assert_array_equal(x.data, [11.0, 12.0])


def test_detach_does_not_touch_original():
    x = nm.vector([1.0, 2.0], requires_grad=True)
    y = x * x
    y.detach()
    assert x.requires_grad is True
    assert y._prev != set()


def test_straight_through_estimator():
    x = nm.tensor([0.4, 1.6], requires_grad=True)
    y = x + (nm.tensor(np.round(x.data)) - x).detach()
    np.testing.assert_array_equal(y.data, [0.0, 2.0])
    nm.sum(y).backward()
    np.testing.assert_array_equal(x.grad.data, [1.0, 1.0])


# ==============================
# Numerical differentiation
# ==============================


def _central_diff(f, x, eps=1e-5):
    """Central difference gradient of ``f`` at ``x`` (float64 ndarray, in place)."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_pos = f(x)
        x[idx] = orig - eps
        f_neg = f(x)
        x[idx] = orig
        grad[idx] = (f_pos - f_neg) / (2 * eps)
    return grad


def test_detach_gradient_matches_numerical_diff_of_frozen_constant_branch():
    """d/dx sum(x * x.detach()) == x.detach()'s value, treated as a constant.

    Naively re-evaluating ``x * x.detach()`` at each perturbed ``x`` would
    just numerically differentiate ``x * x`` (since a fresh ``detach()``
    tracks the perturbed value too), giving ``2x`` instead of ``x`` -- that
    would not exercise the "cut this branch" semantics at all. So the
    reference function below freezes the detached branch at the value it had
    when ``x.detach()`` was called, which is what backward() is actually
    supposed to compute.
    """
    x0 = np.array([0.3, -1.7, 2.5], dtype=np.float64)
    c = x0.copy()  # value frozen at the point x.detach() is taken

    def f(v):
        return np.sum(v * c)

    numerical_grad = _central_diff(f, x0.copy())

    x = nm.tensor(x0.copy(), requires_grad=True)
    y = nm.sum(x * x.detach())
    y.backward()

    np.testing.assert_allclose(x.grad.data, numerical_grad, atol=1e-6)
    np.testing.assert_allclose(x.grad.data, c, atol=1e-10)


def test_detach_gradient_flows_through_undetached_path_matches_numerical_diff():
    """a and a.detach() are summed: gradient must flow only through the plain a."""
    x0 = np.array([2.0, -0.5], dtype=np.float64)

    def f(v):
        a = v * v
        c = x0 * x0  # frozen value of a.detach()
        return np.sum(a + c)

    numerical_grad = _central_diff(f, x0.copy())

    x = nm.tensor(x0.copy(), requires_grad=True)
    a = x * x
    y = nm.sum(a + a.detach())
    y.backward()

    np.testing.assert_allclose(x.grad.data, numerical_grad, atol=1e-6)
    np.testing.assert_allclose(x.grad.data, 2 * x0, atol=1e-10)


# ==============================
# autograd.off
# ==============================


def test_detach_under_autograd_off():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    with nm.autograd.off:
        d = x.detach()
        y = x * d
    assert d.requires_grad is False
    assert d._prev == set()
    np.testing.assert_array_equal(d.data, x.data)
    # under autograd.off no result tracks gradients at all, detached or not
    assert y.requires_grad is False
    assert y._prev == set()


# ==============================
# name / grad slots
# ==============================


def test_detach_preserves_name():
    x = nm.tensor([1.0, 2.0], name="my_tensor")
    d = x.detach()
    assert d.name == "my_tensor"


def test_detach_of_object_with_existing_grad_has_no_grad():
    x = nm.real(3.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad is not None  # sanity check: x actually has a grad now

    d = x.detach()
    assert d.grad is None
    # detaching must not clear the grad already accumulated on the original
    assert x.grad is not None


# ==============================
# retain_graph
# ==============================


def test_detach_result_does_not_affect_retain_graph_on_other_path():
    x = nm.real(2.0, requires_grad=True)
    a = x * x
    y = a + a.detach()  # only the plain `a` path should keep contributing 2x

    y.backward(retain_graph=True)
    assert float(x.grad) == pytest.approx(4.0)

    # a's graph is kept by retain_graph=True, so a second backward() adds
    # exactly one more 2x to the leaf gradient
    y.backward(retain_graph=True)
    assert float(x.grad) == pytest.approx(8.0)


# ==============================
# dtype preservation
# ==============================


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_detach_preserves_tensor_dtype(dtype):
    x = nm.tensor([1.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    d = x.detach()
    assert d.dtype == np.dtype(dtype)
    assert d.dtype == x.dtype
    np.testing.assert_array_equal(d.data, x.data)


# ==============================
# copy() after detach
# ==============================


def test_detach_then_copy_is_independent_of_original():
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    d = x.detach().copy()
    assert not np.shares_memory(d.data, x.data)
    d += nm.tensor([10.0, 10.0])
    np.testing.assert_array_equal(x.data, [1.0, 2.0])
    np.testing.assert_array_equal(d.data, [11.0, 12.0])
