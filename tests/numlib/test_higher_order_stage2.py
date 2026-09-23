"""Tests for higher-order derivatives (stage 2)

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
# 追加の境界値テスト用（1 次元・バッチ行列積、追加の get_item）
_W62 = nm.tensor(_rng.uniform(-1, 1, size=(6, 2)), requires_grad=False)
_W26 = nm.tensor(_rng.uniform(-1, 1, size=(2, 6)), requires_grad=False)
_B34 = nm.tensor(_rng.uniform(-1, 1, size=(3, 4)), requires_grad=False)
_B4D = nm.tensor(_rng.uniform(-1, 1, size=(1, 2, 3, 4)), requires_grad=False)

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
    "dot_1d": lambda x: nm.dot(nm.reshape(x, (-1,)), nm.reshape(x * x, (-1,))) ** 3,
    "dot_matrix": lambda x: nm.sum(nm.dot(x, _W32) ** 3),
    "matmul_tensor": lambda x: nm.sum((x @ _W32) ** 3),
    "matmul_both": lambda x: nm.sum((nm.transpose(x) @ x) ** 3),
    "matmul_batched": lambda x: nm.sum(
        (nm.reshape(x, (2, 1, 3)) @ nm.reshape(x * x, (2, 3, 1))) ** 3
    ),
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
    # --- 追加: get_item の境界値（負の添字・スライスの組み合わせ・2次元の一部・
    #     同じ要素を複数回選ぶ整数配列（2次元・両方の軸）） ---
    "get_item_negative_index": lambda x: nm.sum(x[-1] ** 3) + nm.sum(x[:, -2:] ** 2),
    "get_item_slice_combo": lambda x: nm.sum(x[1:, ::2] ** 3),
    "get_item_2d_partial": lambda x: nm.sum(x[0:1, 1:3] ** 3),
    "get_item_2d_fancy_repeated": lambda x: nm.sum(x[[0, 0, 1], [1, 2, 0]] ** 3),
    # --- 追加: 1 次元入力の matmul（dot ではなく @ 演算子）とバッチ行列積 ---
    "matmul_1d_1d": lambda x: (nm.reshape(x, (-1,)) @ nm.reshape(x * x, (-1,))) ** 3,
    "matmul_1d_by_2d": lambda x: nm.sum((nm.reshape(x, (-1,)) @ _W62) ** 3),
    "matmul_2d_by_1d": lambda x: nm.sum((_W26 @ nm.reshape(x, (-1,))) ** 3),
    "matmul_batch_broadcast": lambda x: nm.sum(
        (nm.reshape(x, (2, 1, 3)) @ _B34) ** 3
    ),  # バッチ軸が片方にしかない（Cᵢ = A Bᵢ の形）
    "matmul_multi_batch_axis": lambda x: nm.sum(
        (nm.reshape(x, (1, 2, 1, 3)) @ _B4D) ** 3
    ),  # バッチ軸が 2 つ
    # --- 追加: var / logsumexp の axis がタプルのとき ---
    "var_axis_tuple": lambda x: nm.var(x, axis=(0, 1)) * nm.sum(x**2),
    "logsumexp_axis_tuple": lambda x: nm.logsumexp(x, axis=(0, 1)) ** 3,
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


# ------------------------------------------------------------------
# 数学の型（Vector / RowVector / Matrix）の行列積
# ------------------------------------------------------------------

MATH_MATMUL = {
    "rowvec_vec": lambda v: nm.dot(v, v) ** 3,
    "mat_vec": lambda v: nm.sum((_M33 @ v) ** 3),
    "rowvec_mat": lambda v: nm.sum((nm.transpose(v) @ _M33) ** 3),
    "vec_rowvec": lambda v: nm.sum((v @ nm.transpose(v)) ** 3),
    "mat_mat": lambda v: nm.sum((nm.matrix(v @ nm.transpose(v)) @ _M33) ** 3),
}

_M33 = nm.matrix(_rng.uniform(-1, 1, size=(3, 3)), requires_grad=False)


@pytest.mark.parametrize("name", sorted(MATH_MATMUL))
def test_math_type_matmul_second_order(name):
    f = MATH_MATMUL[name]
    data = _rng.uniform(-1.0, 1.0, size=(3, 1))
    vs = _rng.uniform(-1, 1, size=(3, 1))

    def grads(d):
        v = nm.vector(d, requires_grad=True)
        return nm.grad(f(v), v).data

    v = nm.vector(data, requires_grad=True)
    g = nm.grad(f(v), v, create_graph=True)
    assert type(g) is nm.Vector
    hvp = nm.grad(nm.sum(g * nm.vector(vs, requires_grad=False)), v)
    assert type(hvp) is nm.Vector
    num = (grads(data + H * vs) - grads(data - H * vs)) / (2 * H)
    np.testing.assert_allclose(hvp.data, num, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------
# AC-7 / AC-8: nm.jacobian / nm.hessian
# ------------------------------------------------------------------


def test_jacobian_linear_map():
    """J(A v) = A"""
    a = _rng.uniform(-1, 1, size=(4, 3))
    A = nm.matrix(a, requires_grad=False)
    v = nm.vector(_rng.uniform(-1, 1, size=(3, 1)), requires_grad=True)
    J = nm.jacobian(lambda u: A @ u, v)
    assert type(J) is nm.Matrix and J.shape == (4, 3)
    np.testing.assert_allclose(J.data, a)
    assert J.requires_grad is False


def test_hessian_quadratic_form():
    """H(½ vᵀAv) = (A + Aᵀ)/2"""
    a = _rng.uniform(-1, 1, size=(3, 3))
    A = nm.matrix(a, requires_grad=False)
    v = nm.vector(_rng.uniform(-1, 1, size=(3, 1)), requires_grad=True)
    H = nm.hessian(lambda u: nm.dot(u, A @ u) * 0.5, v)
    assert type(H) is nm.Matrix and H.shape == (3, 3)
    np.testing.assert_allclose(H.data, (a + a.T) / 2, rtol=1e-10, atol=1e-12)


def test_hessian_of_a_nonlinear_function_matches_numerical_diff():
    def f(u):
        return nm.sum(nm.exp(u) * nm.sin(u)) + nm.dot(u, u) ** 2

    d = _rng.uniform(-0.5, 0.5, size=(3, 1))
    v = nm.vector(d, requires_grad=True)
    H = nm.hessian(f, v)

    def grad_at(data):
        u = nm.vector(data, requires_grad=True)
        return nm.grad(f(u), u).data

    num = np.hstack(
        [
            (grad_at(d + H_ * np.eye(3, 1, -i)) - grad_at(d - H_ * np.eye(3, 1, -i)))
            / (2 * H_)
            for i, H_ in enumerate([H] * 0 + [1e-5] * 3)
        ]
    )
    np.testing.assert_allclose(H.data, num, rtol=1e-5, atol=1e-6)


def test_jacobian_and_hessian_keep_the_graph_with_create_graph():
    v = nm.vector([0.3, -0.7], requires_grad=True)
    H = nm.hessian(lambda u: nm.sum(u**4), v, create_graph=True)
    assert type(H) is nm.Matrix and H.requires_grad is True
    # H の成分をさらに微分できる（d/dv of 12 v² は 24 v）
    g = nm.grad(nm.sum(H), v)
    np.testing.assert_allclose(g.data, 24 * v.data)


def test_jacobian_boundary_sizes():
    v1 = nm.vector([2.0], requires_grad=True)
    J = nm.jacobian(lambda u: u * u, v1)
    assert J.shape == (1, 1)
    np.testing.assert_allclose(J.data, [[4.0]])
    H = nm.hessian(lambda u: nm.sum(u**3), v1)
    assert H.shape == (1, 1)
    np.testing.assert_allclose(H.data, [[12.0]])


def test_hessian_jacobian_type_errors():
    """AC-8: RowVector / Tensor / スカラーでない出力はエラー（hint つき）"""
    row = nm.rowvector([1.0, 2.0], requires_grad=True)
    with pytest.raises(nm.TypeMismatchError) as e:
        nm.hessian(lambda u: nm.sum(u**2), row)
    assert "nm.vector" in str(e.value)

    t = nm.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(nm.TypeMismatchError) as e:
        nm.jacobian(lambda u: u * u, t)
    assert "nm.vector" in str(e.value)

    v = nm.vector([1.0, 2.0], requires_grad=True)
    # f(x) が Vector でない
    with pytest.raises(nm.TypeMismatchError) as e:
        nm.jacobian(lambda u: nm.sum(u), v)
    assert "Vector" in str(e.value)
    # hessian の f(x) がスカラーでない
    with pytest.raises(nm.GradientError) as e:
        nm.hessian(lambda u: u * u, v)
    assert "scalar" in str(e.value)
    # requires_grad=False
    const = nm.vector([1.0, 2.0], requires_grad=False)
    with pytest.raises(nm.GradientError, match="requires_grad=True"):
        nm.jacobian(lambda u: u * u, const)


# ------------------------------------------------------------------
# 段階③: random_mask / random_mask_channel
# ------------------------------------------------------------------


@pytest.mark.parametrize("training", [True, False])
def test_random_mask_second_order(training):
    """マスクは定数なので、2 階微分もマスクを掛けたものになる"""
    d = _rng.uniform(-1.0, 1.0, size=(4, 3))
    nm.seed(0)
    x = nm.tensor(d, requires_grad=True)
    y = nm.sum(nm.random_mask(x, p=0.5, training=training) ** 3)
    g = nm.grad(y, x, create_graph=True)
    h = nm.grad(nm.sum(g), x)
    # 1 階の勾配 3 m x²、2 階は 6 m x（m はマスク。0 の所は 0 のまま）
    np.testing.assert_allclose(h.data * d, 2 * g.data, rtol=1e-10)


def test_random_mask_channel_second_order():
    d = _rng.uniform(-1.0, 1.0, size=(2, 3, 2, 2))
    nm.seed(1)
    x = nm.tensor(d, requires_grad=True)
    g = nm.grad(nm.sum(nm.random_mask_channel(x, p=0.5) ** 3), x, create_graph=True)
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(h.data * d, 2 * g.data, rtol=1e-10)


def test_only_make_op_and_complex_remain_unsupported():
    """段階③まで終われば、未対応は make_op と複素数だけ"""
    def forward(a):
        return a * a, a

    def backward(a, g, needs_grad):
        return (g * 2 * a,)

    square = nm.make_op(forward, backward)
    x = nm.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(nm.GradientError, match="make_op"):
        nm.grad(nm.sum(square(x)), x, create_graph=True)

    z = nm.cmplx(1.0, 2.0, requires_grad=True)
    with pytest.raises(nm.GradientError, match="complex"):
        nm.grad(z * z, z, create_graph=True)


# ------------------------------------------------------------------
# 数学の型の保存: get_item / split / where（Vector / Matrix）の 2 階微分
#
# matmul / dot は MATH_MATMUL（上）で確認済み。concatenate / stack は
# Vector 同士でも Tensor に下がる（既存の順伝播の決まりで、数学の型として
# 結果を定義できないため）ので、ここでは扱わない。
# ------------------------------------------------------------------


def test_get_item_on_vector_keeps_vector_type_in_second_order():
    v = nm.vector([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    def f(u):
        return nm.sum(u[1:3] ** 3)

    g = nm.grad(f(v), v, create_graph=True)
    assert type(g) is nm.Vector and g.shape == v.shape
    h = nm.grad(nm.sum(g), v)
    assert type(h) is nm.Vector
    np.testing.assert_allclose(h.data.ravel(), [0.0, 12.0, 18.0, 0.0])


def test_get_item_on_matrix_keeps_matrix_type_in_second_order():
    """行の一部（RowVector 経由）と 1 列（Vector 経由）の両方を通っても型は Matrix"""
    data = _rng.uniform(-1, 1, size=(3, 3))

    def f(u):
        return nm.sum(u[0:1, :] ** 3) + nm.sum(u[:, 0] ** 4)

    m = nm.matrix(data, requires_grad=True)
    g = nm.grad(f(m), m, create_graph=True)
    assert type(g) is nm.Matrix and g.shape == m.shape
    h = nm.grad(nm.sum(g), m)
    assert type(h) is nm.Matrix

    def grad_at(d):
        u = nm.matrix(d, requires_grad=True)
        return nm.grad(f(u), u).data

    v = _rng.uniform(-1, 1, size=(3, 3))
    num = (grad_at(data + H * v) - grad_at(data - H * v)) / (2 * H)
    hvp = nm.grad(nm.sum(g * nm.matrix(v, requires_grad=False)), m)
    np.testing.assert_allclose(hvp.data, num, rtol=1e-5, atol=1e-6)


def test_split_on_vector_keeps_vector_type_in_second_order():
    v = nm.vector([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    def f(u):
        parts = nm.split(u, 2, axis=0)
        return nm.sum(parts[0] ** 3) + nm.sum(parts[1] ** 4)

    g = nm.grad(f(v), v, create_graph=True)
    assert type(g) is nm.Vector
    h = nm.grad(nm.sum(g), v)
    assert type(h) is nm.Vector
    np.testing.assert_allclose(h.data.ravel(), [6.0, 12.0, 108.0, 192.0])


def test_where_on_vector_keeps_vector_type_in_second_order():
    v = nm.vector([1.0, -2.0, 3.0, -4.0], requires_grad=True)
    cond = nm.tensor(np.array([[True], [False], [True], [False]]))

    def f(u):
        return nm.sum(nm.where(cond, u**3, u**2))

    g = nm.grad(f(v), v, create_graph=True)
    assert type(g) is nm.Vector
    h = nm.grad(nm.sum(g), v)
    assert type(h) is nm.Vector
    # True の位置は 3 階の項（6x）、False の位置は 2 階の定数（2）
    np.testing.assert_allclose(h.data.ravel(), [6 * 1.0, 2.0, 6 * 3.0, 2.0])


# ------------------------------------------------------------------
# nm.jacobian / nm.hessian: 追加の境界値・エラー系
# ------------------------------------------------------------------


def test_hessian_does_not_symmetrize_the_result():
    """
    H は対称にそろえない。

    nm.hessian(f, x) は nm.jacobian(lambda u: nm.grad(f(u), u, create_graph=True), x)
    の生の値そのもの（(H + Hᵀ)/2 のような後処理をしない）ことを、同じ式を自分で
    組み立てた場合とビット単位で一致することで確かめる。
    """

    def f(u):
        return nm.sum(nm.exp(u) * nm.sin(u) * nm.cos(u)) + nm.dot(u, u) ** 2

    d = _rng.uniform(-0.5, 0.5, size=(4, 1))

    v1 = nm.vector(d, requires_grad=True)
    H = nm.hessian(f, v1)

    v2 = nm.vector(d, requires_grad=True)
    H_raw = nm.jacobian(lambda u: nm.grad(f(u), u, create_graph=True), v2)

    np.testing.assert_array_equal(H.data, H_raw.data)


def test_jacobian_and_hessian_on_empty_vector():
    """AC-14 のエッジケース: 空配列（n=0）は Matrix (0x0) / (m x 0) / (0 x n)"""
    v = nm.vector(np.zeros((0, 1)), requires_grad=True)

    J = nm.jacobian(lambda u: u * 2, v)
    assert type(J) is nm.Matrix and J.shape == (0, 0)

    Hm = nm.hessian(lambda u: nm.sum(u), v)
    assert type(Hm) is nm.Matrix and Hm.shape == (0, 0)

    # m > 0, n = 0（f が x に触れずに定数ベクトルを返す）
    J2 = nm.jacobian(lambda u: nm.vector([1.0, 2.0], requires_grad=False), v)
    assert J2.shape == (2, 0)

    # m = 0, n > 0（f が空ベクトルを返す）
    v3 = nm.vector([1.0, 2.0, 3.0], requires_grad=True)
    J3 = nm.jacobian(lambda u: nm.vector(np.zeros((0, 1)), requires_grad=False), v3)
    assert J3.shape == (0, 3)


def test_jacobian_and_hessian_are_zero_when_f_does_not_depend_on_x():
    """y が x に依存しないときの決まり（SPEC のエッジケース）を jacobian/hessian にも適用"""
    v = nm.vector([1.0, 2.0, 3.0], requires_grad=True)

    J = nm.jacobian(lambda u: nm.vector([1.0, 2.0], requires_grad=False), v)
    assert J.shape == (2, 3)
    np.testing.assert_array_equal(J.data, np.zeros((2, 3)))

    Hm = nm.hessian(lambda u: nm.real(5.0, requires_grad=False), v)
    assert Hm.shape == (3, 3)
    np.testing.assert_array_equal(Hm.data, np.zeros((3, 3)))


def test_hessian_create_graph_up_to_fourth_order():
    """create_graph=True で 3 階・4 階微分まで（既存テストの 3 階微分をさらに 1 段延ばす）"""
    v = nm.vector([0.3, -0.7], requires_grad=True)
    # f(v) = sum(v**4): 1 階 4v³、2 階（Hessian の対角）12v²、3 階 24v、4 階 24（定数）
    Hm = nm.hessian(lambda u: nm.sum(u**4), v, create_graph=True)
    d3 = nm.grad(nm.sum(Hm), v, create_graph=True)  # 3 階: 24 v
    np.testing.assert_allclose(d3.data, 24 * v.data)
    d4 = nm.grad(nm.sum(d3), v)  # 4 階: 24（定数）
    np.testing.assert_allclose(d4.data.ravel(), [24.0, 24.0])


def test_jacobian_under_autograd_off_raises():
    v = nm.vector([1.0, 2.0], requires_grad=True)
    with nm.autograd.off:
        # create_graph=False: f(x) 自体に計算グラフが無い
        with pytest.raises(nm.GradientError, match="autograd.off") as e:
            nm.jacobian(lambda u: u * u, v)
        assert "detach" in str(e.value)
        # create_graph=True: 勾配自体のグラフが作れない
        with pytest.raises(nm.GradientError, match="autograd.off") as e:
            nm.jacobian(lambda u: u * u, v, create_graph=True)
        assert "autograd.on" in str(e.value)


def test_hessian_under_autograd_off_raises():
    """hessian は内部で常に create_graph=True の nm.grad を使うので、
    hessian(create_graph=False) を off の中で呼んでもエラーになる"""
    v = nm.vector([1.0, 2.0], requires_grad=True)
    with nm.autograd.off:
        with pytest.raises(nm.GradientError, match="autograd.off") as e:
            nm.hessian(lambda u: nm.sum(u**2), v)
        assert "autograd.on" in str(e.value)


def test_jacobian_result_graph_can_be_freed_and_then_raises():
    """jacobian(create_graph=True) が返す Matrix も、backward() で解放すれば
    以降の nm.grad は freed のエラーになる（AC-12 と同じ決まり）"""
    v = nm.vector([1.0, 2.0], requires_grad=True)
    J = nm.jacobian(lambda u: u * u, v, create_graph=True)
    nm.sum(J).backward()  # retain_graph=False（デフォルト）で解放
    with pytest.raises(nm.GradientError, match="freed") as e:
        nm.grad(nm.sum(J), v)
    assert "retain_graph=True" in str(e.value)


def test_jacobian_propagates_python_exception_without_touching_grad():
    """f が途中で例外を投げたら、その例外がそのまま伝わり x.grad も変わらない"""
    v = nm.vector([1.0, 2.0], requires_grad=True)
    v.grad = nm.vector([9.0, 9.0], requires_grad=False)

    def bad_f(u):
        _ = u * u  # x を使った計算をいったん行う
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        nm.jacobian(bad_f, v)
    np.testing.assert_array_equal(v.grad.data.ravel(), [9.0, 9.0])


def test_jacobian_propagates_unsupported_op_error_without_touching_grad():
    """f の中で make_op（未対応演算）を使うと、jacobian も GradientError になり
    x.grad は変わらない（m 行のうちどの行でも、逆伝播を始める前にまとめて調べる）"""

    def forward(a):
        return a * a, a

    def backward(a, g, needs_grad):
        return (g * 2 * a,)

    square = nm.make_op(forward, backward)
    v = nm.vector([1.0, 2.0], requires_grad=True)
    v.grad = nm.vector([9.0, 9.0], requires_grad=False)

    with pytest.raises(nm.GradientError, match="make_op"):
        nm.jacobian(lambda u: nm.vector(square(u)), v, create_graph=True)
    np.testing.assert_array_equal(v.grad.data.ravel(), [9.0, 9.0])


# ------------------------------------------------------------------
# var / logsumexp: 大きな値での数値の安定性
# ------------------------------------------------------------------


def test_logsumexp_gradient_is_stable_for_large_values():
    """大きな値（オーバーフロー域）でも softmax の勾配が有限で、解析解と一致する"""
    d = 1.0e4 + _rng.uniform(-2.0, 2.0, size=(5,))
    x = nm.tensor(d, requires_grad=True)
    y = nm.logsumexp(x)
    np.testing.assert_allclose(y.data, 1.0e4 + np.log(np.exp(d - 1.0e4).sum()))

    g = nm.grad(y, x, create_graph=True)
    assert np.all(np.isfinite(g.data))
    soft = np.exp(d - d.max())
    soft = soft / soft.sum()
    np.testing.assert_allclose(g.data, soft, rtol=1e-10)

    # 2 階微分（softmax のヤコビアン: diag(s) - s sᵀ）も有限で解析解と一致する
    direction = _rng.uniform(-1, 1, size=(5,))
    hvp = nm.grad(nm.sum(g * nm.tensor(direction, requires_grad=False)), x)
    assert np.all(np.isfinite(hvp.data))
    analytic = soft * direction - soft * np.dot(soft, direction)
    np.testing.assert_allclose(hvp.data, analytic, rtol=1e-8)


def test_logsumexp_gradient_is_stable_for_very_negative_values():
    """アンダーフロー域（すべて非常に小さい値）でも有限で、softmax と一致する"""
    d = -1.0e4 + _rng.uniform(-2.0, 2.0, size=(4,))
    x = nm.tensor(d, requires_grad=True)
    g = nm.grad(nm.logsumexp(x), x, create_graph=True)
    assert np.all(np.isfinite(g.data))
    soft = np.exp(d - d.max())
    soft = soft / soft.sum()
    np.testing.assert_allclose(g.data, soft, rtol=1e-10)
    h = nm.grad(nm.sum(g), x)
    assert np.all(np.isfinite(h.data))


# ------------------------------------------------------------------
# 段階③: random_mask / random_mask_channel の境界値
# ------------------------------------------------------------------


def test_random_mask_training_false_matches_the_plain_cube_exactly():
    """training=False は恒等写像（x を素通りする）なので、3 乗の勾配そのものになる"""
    d = _rng.uniform(-1.0, 1.0, size=(3, 2))
    x = nm.tensor(d, requires_grad=True)
    y = nm.random_mask(x, p=0.5, training=False)
    np.testing.assert_array_equal(y.data, d)  # 同じ配列（マスクをかけない）

    g = nm.grad(nm.sum(y**3), x, create_graph=True)
    np.testing.assert_allclose(g.data, 3 * d**2)
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(h.data, 6 * d, rtol=1e-10)


def test_random_mask_p_zero_is_also_identity():
    """p=0 も training に関わらず恒等写像（マスクの分岐に入らない）"""
    d = _rng.uniform(-1.0, 1.0, size=(3, 2))
    x = nm.tensor(d, requires_grad=True)
    y = nm.random_mask(x, p=0.0, training=True)
    np.testing.assert_array_equal(y.data, d)
    g = nm.grad(nm.sum(y**3), x)
    np.testing.assert_allclose(g.data, 3 * d**2)


def test_random_mask_all_kept_matches_analytic_gradient(monkeypatch):
    """マスクがすべて 1 のとき（決定的に固定）、解析解どおりの 1 階・2 階微分になる"""
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.full(shape, 0.99))
    d = _rng.uniform(-1.0, 1.0, size=(3, 2))
    x = nm.tensor(d, requires_grad=True)
    p = 0.5
    scale = 1.0 / (1.0 - p)

    y = nm.random_mask(x, p=p, training=True)
    np.testing.assert_allclose(y.data, d * scale)

    g = nm.grad(nm.sum(y**3), x, create_graph=True)
    np.testing.assert_allclose(g.data, 3 * scale**3 * d**2)
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(h.data, 6 * scale**3 * d, rtol=1e-10)


def test_random_mask_all_dropped_gives_zero_output_and_zero_gradient(monkeypatch):
    """マスクがすべて 0 のとき（決定的に固定）、出力も 1 階・2 階の勾配も 0"""
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.full(shape, 0.01))
    d = _rng.uniform(-1.0, 1.0, size=(3, 2))
    x = nm.tensor(d, requires_grad=True)

    y = nm.random_mask(x, p=0.5, training=True)
    np.testing.assert_array_equal(y.data, np.zeros_like(d))

    g = nm.grad(nm.sum(y**3), x, create_graph=True)
    np.testing.assert_array_equal(g.data, np.zeros_like(d))
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_array_equal(h.data, np.zeros_like(d))


def test_random_mask_channel_all_dropped_gives_zero_output_and_zero_gradient(
    monkeypatch,
):
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.full(shape, 0.01))
    d = _rng.uniform(-1.0, 1.0, size=(2, 3, 2, 2))
    x = nm.tensor(d, requires_grad=True)

    y = nm.random_mask_channel(x, p=0.5, training=True)
    np.testing.assert_array_equal(y.data, np.zeros_like(d))

    g = nm.grad(nm.sum(y**3), x, create_graph=True)
    np.testing.assert_array_equal(g.data, np.zeros_like(d))
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_array_equal(h.data, np.zeros_like(d))


def test_random_mask_channel_training_false_matches_the_plain_cube_exactly():
    d = _rng.uniform(-1.0, 1.0, size=(2, 3, 2, 2))
    x = nm.tensor(d, requires_grad=True)
    y = nm.random_mask_channel(x, p=0.5, training=False)
    np.testing.assert_array_equal(y.data, d)
    g = nm.grad(nm.sum(y**3), x, create_graph=True)
    np.testing.assert_allclose(g.data, 3 * d**2)
    h = nm.grad(nm.sum(g), x)
    np.testing.assert_allclose(h.data, 6 * d, rtol=1e-10)


# ------------------------------------------------------------------
# GPU: 代表的な段階②③の演算・jacobian / hessian が CPU と同じ値になる
#
# 装置は値を渡した場所に留まる（`nm.to_gpu` などで明示しない限り動かさない）ので、
# `with nm.cuda.gpu:` の中では、GPU に新しく作られるようリスト（Python の list）から
# 作る。NumPy 配列をそのまま渡すと CPU に留まってしまうことに注意
# （`nm.to_gpu` / `nm.to_cpu` の docstring のとおり）。
# ------------------------------------------------------------------


@pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")
def test_stage2_representative_ops_match_between_cpu_and_gpu():
    data = _rng.uniform(-1.5, 1.5, size=(2, 3)).tolist()
    w = _rng.uniform(-1, 1, size=(3, 2)).tolist()

    def f(u, w_):
        return (
            nm.sum((u @ w_) ** 3)
            + nm.sum(u[[0, 0, 1], [1, 2, 0]] ** 3)
            + nm.logsumexp(u) ** 2
            + nm.var(u, axis=1)[0] ** 3
        )

    def run():
        x = nm.tensor(data, requires_grad=True)
        wt = nm.tensor(w, requires_grad=False)
        g = nm.grad(f(x, wt), x, create_graph=True)
        h = nm.grad(nm.sum(g), x)
        return g, h

    g_cpu, h_cpu = run()
    with nm.cuda.gpu:
        g_gpu, h_gpu = run()
        g_gpu_np, h_gpu_np = g_gpu.to_numpy(), h_gpu.to_numpy()

    np.testing.assert_allclose(g_gpu_np, g_cpu.data, rtol=1e-8)
    np.testing.assert_allclose(h_gpu_np, h_cpu.data, rtol=1e-8)


@pytest.mark.skipif(not nm.cuda_available(), reason="CUDA is not available")
def test_jacobian_and_hessian_match_between_cpu_and_gpu():
    a = _rng.uniform(-1, 1, size=(3, 3)).tolist()
    data = _rng.uniform(-1, 1, size=(3, 1)).tolist()

    def run():
        A = nm.matrix(a, requires_grad=False)
        v = nm.vector(data, requires_grad=True)
        J = nm.jacobian(lambda u: A @ u, v)
        Hm = nm.hessian(lambda u: nm.dot(u, A @ u) * 0.5, v)
        return J, Hm

    J_cpu, H_cpu = run()
    with nm.cuda.gpu:
        J_gpu, H_gpu = run()
        J_gpu_np, H_gpu_np = J_gpu.to_numpy(), H_gpu.to_numpy()

    np.testing.assert_allclose(J_gpu_np, J_cpu.data, rtol=1e-10)
    np.testing.assert_allclose(H_gpu_np, H_cpu.data, rtol=1e-10)


def test_dot_between_a_vector_and_a_matrix_raises():
    """内積はベクトルどうしだけ。行列との組み合わせは hint 付きのエラー

    今までは順伝播だけ通り、逆伝播で生の ValueError になっていた。
    """
    v = nm.vector([1.0, 2.0, 3.0], requires_grad=True)
    A = nm.matrix(_rng.uniform(-1, 1, size=(3, 2)), requires_grad=False)
    T = nm.tensor(_rng.uniform(-1, 1, size=(3, 2)), requires_grad=False)
    for a, b in ((v, A), (A, v), (v, T), (T, v)):
        with pytest.raises(nm.TypeMismatchError) as e:
            nm.dot(a, b)
        assert "@" in str(e.value)
    # 定義される組み合わせは今までどおり
    assert nm.dot(v, v).shape == ()
    assert nm.dot(nm.reshape(v, (-1,)), nm.reshape(v, (-1,))).shape == ()
    assert nm.dot(nm.matrix(np.ones((2, 2))), nm.matrix(np.ones((2, 2)))).shape == (2, 2)
    # hint のとおりに書けば通る
    assert (nm.transpose(v) @ A).shape == (1, 2)
