import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.parameter import Parameter
from lemon.nnlib.init import uniform_
import math


class GRUCell(Module):
    """
    GRU (Gated Recurrent Unit) セル

    単一タイムステップのGRU計算を実行する。
    リセットゲートと更新ゲートの2つのゲートを持つ。

    Parameters
    ----------
    input_size : int
        入力特徴量の次元数
    hidden_size : int
        隠れ状態の次元数
    bias : bool, optional
        バイアス項を使用するか (デフォルト: True)

    Attributes
    ----------
    weight_ih : Parameter
        入力から隠れ状態への重み行列 (input_size, 3*hidden_size)
        3つのゲート（reset, update, new）の重みを連結
    weight_hh : Parameter
        隠れ状態から隠れ状態への重み行列 (hidden_size, 3*hidden_size)
    bias_ih : Parameter
        入力側のバイアス (3*hidden_size,)
    bias_hh : Parameter
        隠れ状態側のバイアス (3*hidden_size,)

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> from lemon.nnlib.layer.rnn import GRUCell
    >>>
    >>> # GRUセルの作成
    >>> cell = GRUCell(input_size=10, hidden_size=20)
    >>>
    >>> # 入力と隠れ状態
    >>> x = nm.randn(3, 10)   # (batch, input_size)
    >>> h = nm.randn(3, 20)   # (batch, hidden_size)
    >>>
    >>> # 1タイムステップの計算
    >>> h_next = cell(x, h)
    >>> h_next.shape
    (3, 20)

    Notes
    -----
    GRUの計算式:
        r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # リセットゲート
        z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # 更新ゲート
        n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # 新しい値
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # 隠れ状態更新

    LSTMよりもパラメータが少なく、計算が高速。
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # 重み行列: 3つのゲート（reset, update, new）を連結
        self.weight_ih = Parameter(nm.zeros(input_size, 3 * hidden_size))
        self.weight_hh = Parameter(nm.zeros(hidden_size, 3 * hidden_size))

        if bias:
            self.bias_ih = Parameter(nm.zeros(3 * hidden_size))
            self.bias_hh = Parameter(nm.zeros(3 * hidden_size))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self._reset_parameters()

    def _reset_parameters(self):
        """パラメータを初期化"""
        # PyTorchと同じ初期化方法: uniform(-sqrt(k), sqrt(k)), k = 1/hidden_size
        stdv = 1.0 / math.sqrt(self.hidden_size)

        uniform_(self.weight_ih, -stdv, stdv)
        uniform_(self.weight_hh, -stdv, stdv)

        if self.use_bias:
            uniform_(self.bias_ih, -stdv, stdv)
            uniform_(self.bias_hh, -stdv, stdv)

    def forward(self, x, h=None):
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力テンソル (batch, input_size)
        h : Tensor, optional
            前の隠れ状態 (batch, hidden_size)
            Noneの場合、ゼロで初期化される

        Returns
        -------
        h_next : Tensor
            次の隠れ状態 (batch, hidden_size)
        """
        if h is None:
            batch_size = x.shape[0]
            h = nm.zeros(batch_size, self.hidden_size)

        # 入力側の線形変換
        gi = x @ self.weight_ih.data
        if self.use_bias:
            gi = gi + self.bias_ih.data

        # 隠れ状態側の線形変換
        gh = h @ self.weight_hh.data
        if self.use_bias:
            gh = gh + self.bias_hh.data

        # ゲートを分割
        # gi[:, 0:h] = input reset gate
        # gi[:, h:2h] = input update gate
        # gi[:, 2h:3h] = input new gate
        i_r = gi[:, 0:self.hidden_size]
        i_z = gi[:, self.hidden_size:2*self.hidden_size]
        i_n = gi[:, 2*self.hidden_size:3*self.hidden_size]

        h_r = gh[:, 0:self.hidden_size]
        h_z = gh[:, self.hidden_size:2*self.hidden_size]
        h_n = gh[:, 2*self.hidden_size:3*self.hidden_size]

        # リセットゲートと更新ゲート
        r = 1 / (1 + nm.exp(-(i_r + h_r)))  # sigmoid(i_r + h_r)
        z = 1 / (1 + nm.exp(-(i_z + h_z)))  # sigmoid(i_z + h_z)

        # 新しい値（リセットゲートを適用）
        n = nm.tanh(i_n + r * h_n)

        # 隠れ状態を更新
        h_next = (1 - z) * n + z * h

        return h_next

    def __repr__(self):
        return f"GRUCell({self.input_size}, {self.hidden_size})"
