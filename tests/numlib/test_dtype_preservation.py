"""Tests that operations keep the dtype of their inputs

0 次元（スカラー）に縮約する演算や添字は、入力の dtype を変えてはならない。
勾配も変数と同じ dtype でなければならない。
"""

import numpy as np
import pytest
from lemon import numlib as nm

FLOATS = [np.float16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", FLOATS)
def test_reduction_to_scalar_keeps_dtype(dtype):
    v = nm.tensor(np.array([1.0, 2.0, 3.0], dtype=dtype))
    assert nm.sum(v).dtype == dtype
    assert nm.mean(v).dtype == dtype
    assert nm.var(v).dtype == dtype
    assert nm.dot(nm.vector(v), nm.vector(v)).dtype == dtype


@pytest.mark.parametrize("dtype", FLOATS)
def test_scalar_result_kind_matches_dtype(dtype):
    """Real の kind と中身の dtype は必ず一致する（repr もこの kind で決まる）"""
    s = nm.sum(nm.tensor(np.array([1.0, 2.0], dtype=dtype)))
    assert isinstance(s, nm.Real)
    assert s.kind == np.dtype(dtype).itemsize * 8
    assert s.dtype == dtype


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.int8, np.int32, np.uint16, np.complex64]
)
def test_single_element_indexing_keeps_dtype(dtype):
    """t[0] は numpy と同じく dtype を保つ（値を取り出すだけで、型は変わらない）"""
    t = nm.tensor(np.array([1, 2, 3], dtype=dtype))
    assert t[0].dtype == dtype


def test_complex_reduction_keeps_dtype():
    c = nm.tensor(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
    assert nm.sum(c).dtype == np.complex64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_division_keeps_dtype(dtype):
    kind = np.dtype(dtype).itemsize * 8
    a = nm.real(np.array(3.0, dtype=dtype), kind=kind)
    b = nm.real(np.array(2.0, dtype=dtype), kind=kind)
    assert (a / b).dtype == dtype
    assert (b / a).dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gradient_keeps_dtype_of_variable(dtype):
    """スカラーの損失から逆伝播しても、勾配は変数と同じ dtype"""
    w = nm.tensor(np.ones((2, 3), dtype=dtype), requires_grad=True)
    x = nm.tensor(np.ones((2, 3), dtype=dtype), requires_grad=False)
    loss = nm.sum(w * w * x)
    assert loss.dtype == dtype
    loss.backward()
    assert w.grad.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_higher_order_gradient_keeps_dtype_of_variable(dtype):
    w = nm.tensor(np.ones((2, 3), dtype=dtype), requires_grad=True)
    g = nm.grad(nm.sum(w * w * w), w, create_graph=True)
    assert g.dtype == dtype
    assert nm.grad(nm.sum(g), w).dtype == dtype


def test_optimizer_state_keeps_dtype():
    """float32 で学習したとき、optimizer の内部状態も float32 のまま"""
    import lemon as lm

    model = lm.Linear(3, 2)
    for p in model.parameters():
        p.data._data = p.data._data.astype(np.float32)
    opt = lm.Adam(model.parameters(), lr=0.01)
    x = nm.tensor(np.ones((4, 3), dtype=np.float32), requires_grad=False)
    t = nm.tensor(np.zeros((4, 2), dtype=np.float32), requires_grad=False)
    opt.zero_grad()
    lm.MSELoss()(model(x), t).backward()
    for p in model.parameters():
        assert p.grad.dtype == np.float32
    opt.step()
    for m in opt.m:
        assert m._data.dtype == np.float32


@pytest.mark.parametrize("dtype", FLOATS + [np.int32, np.complex64])
def test_ones_like_zeros_like_keep_dtype(dtype):
    """ones_like / zeros_like は、配列でもスカラーでも dtype を保つ"""
    t = nm.tensor(np.ones(3, dtype=dtype))
    assert nm.ones_like(t).dtype == dtype
    assert nm.zeros_like(t).dtype == dtype
    s = nm.sum(t)
    # 整数の和は NumPy 自身が int64 に上げるので、その dtype に合わせる
    expected = np.ones(3, dtype=dtype).sum().dtype
    assert nm.ones_like(s).dtype == expected
    assert nm.zeros_like(s).dtype == expected


def test_complex_real_imag_keep_precision():
    z = nm.cmplx64(1.0, 2.0)
    assert z.real.dtype == np.float32
    assert z.imag.dtype == np.float32


# ------------------------------------------------------------------
# リテラル（Python の数）は相手の精度に合わせる（memo/numlib_literal_dtype.md 案 A）
# 種類は bool < int < float < complex の大きいほう、精度は相手に合わせる。
# 種類が上がるときだけ既定の 64 ビット
# ------------------------------------------------------------------

LITERAL_CASES = [
    (np.float32, 2, np.float32),
    (np.float32, 2.0, np.float32),
    (np.float32, 2j, np.complex64),
    (np.float16, 2.0, np.float16),
    (np.int32, 2, np.int32),
    (np.int32, 2.0, np.float64),  # 種類が上がるので 64 ビット
    (np.complex64, 2, np.complex64),
    (np.complex64, 2.0, np.complex64),
    (np.float64, 2, np.float64),
]


@pytest.mark.parametrize("dtype,literal,expected", LITERAL_CASES)
def test_literal_takes_precision_of_the_other_operand(dtype, literal, expected):
    t = nm.tensor(np.ones(3, dtype=dtype))
    assert (t * literal).dtype == expected
    assert (literal * t).dtype == expected


@pytest.mark.parametrize("dtype,literal,expected", LITERAL_CASES)
def test_literal_with_scalar_variable(dtype, literal, expected):
    """スカラーの変数でも同じ規則（整数の和は NumPy 自身が int64 に上げるので、それを基準にする）"""
    s = nm.sum(nm.tensor(np.ones(3, dtype=dtype)))
    assert (s * literal).dtype == np.result_type(s.dtype, literal)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_literal_division_keeps_precision(dtype):
    t = nm.tensor(np.ones(3, dtype=dtype))
    assert (t / 2).dtype == dtype
    assert (t / 2.0).dtype == dtype
    assert (2.0 / t).dtype == dtype


def test_literal_does_not_change_the_shape():
    """dtype は合わせるが、形は今までどおり変えない（スカラー＋配列は定義されない）"""
    t = nm.tensor(np.ones(3, dtype=np.float32))
    with pytest.raises(nm.TypeMismatchError):
        t + 1.0


@pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "make",
    [
        lambda d: nm.Real(d, kind=32),
        lambda d: nm.Real(d),
        lambda d: nm.Integer(np.asarray(2, dtype=np.int32), kind=32),
        lambda d: nm.Complex(np.asarray(2 + 0j, dtype=np.complex64), kind=64),
    ],
)
def test_scalar_keeps_the_array_module_of_its_data(make):
    """cuda.gpu の中でも、numpy の配列から作ったスカラーは numpy のまま（混ざらない）"""
    with nm.cuda.gpu:
        s = make(np.asarray(2.0, dtype=np.float32))
        assert type(s.data).__module__.startswith("numpy")
