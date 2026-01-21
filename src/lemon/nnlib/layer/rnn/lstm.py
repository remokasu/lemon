import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.layer.rnn.lstm_cell import LSTMCell
from lemon.nnlib.layer.dropout import dropout
from lemon.nnlib.train_control import train


class LSTM(Module):
    """
    LSTM (Long Short-Term Memory) レイヤー

    長期依存関係を学習可能なRNNの一種。
    勾配消失問題を軽減するためにゲート機構を使用。

    Parameters
    ----------
    input_size : int
        入力特徴量の次元数
    hidden_size : int
        隠れ状態の次元数
    num_layers : int, optional
        積み重ねるLSTMレイヤーの数 (デフォルト: 1)
    bias : bool, optional
        バイアス項を使用するか (デフォルト: True)
    batch_first : bool, optional
        Trueの場合、入力形状は(batch, seq, feature) (デフォルト: False)
    dropout : float, optional
        レイヤー間のドロップアウト率。0の場合はドロップアウトなし (デフォルト: 0.0)
    bidirectional : bool, optional
        双方向LSTMにするか (デフォルト: False)
    return_sequences : bool, optional
        Trueの場合、全タイムステップの出力を返す。
        Falseの場合、最後のレイヤーの最終隠れ状態のみ返す (デフォルト: True)

    Attributes
    ----------
    cells_forward : list
        順方向のLSTMCellのリスト
    cells_backward : list
        逆方向のLSTMCellのリスト（bidirectional=Trueの場合のみ）

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> from lemon.nnlib.layer.rnn import LSTM
    >>>
    >>> # 単層LSTM
    >>> lstm = LSTM(input_size=10, hidden_size=20)
    >>> x = nm.randn(5, 3, 10)  # (seq_len, batch, input_size)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape
    (5, 3, 20)
    >>> h_n.shape
    (1, 3, 20)
    >>> c_n.shape
    (1, 3, 20)
    >>>
    >>> # 多層双方向LSTM
    >>> lstm = LSTM(10, 20, num_layers=2, bidirectional=True, dropout=0.3)
    >>> x = nm.randn(5, 3, 10)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape
    (5, 3, 40)  # 20 * 2 (bidirectional)
    >>> h_n.shape
    (4, 3, 20)  # 2 layers * 2 directions
    >>>
    >>> # batch_first=True
    >>> lstm = LSTM(10, 20, batch_first=True)
    >>> x = nm.randn(3, 5, 10)  # (batch, seq_len, input_size)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape
    (3, 5, 20)
    >>>
    >>> # return_sequences=False (分類タスクに便利)
    >>> lstm = LSTM(10, 20, return_sequences=False)
    >>> x = nm.randn(5, 3, 10)  # (seq_len, batch, input_size)
    >>> output = lstm(x)  # タプルではなくTensorを直接返す
    >>> output.shape
    (3, 20)  # (batch, hidden_size)

    Notes
    -----
    入力形状は(seq_len, batch, input_size)または
    batch_first=Trueの場合は(batch, seq_len, input_size)。

    return_sequences=Trueの場合:
        出力形状は(seq_len, batch, hidden_size * num_directions)。
        戻り値は(output, (h_n, c_n))のタプル。

    return_sequences=Falseの場合:
        出力形状は(batch, hidden_size * num_directions)。
        戻り値は最終隠れ状態のTensorのみ。
        分類タスクで便利。

    双方向の場合、順方向と逆方向の出力が連結される。

    隠れ状態h_nとセル状態c_nの形状は
    (num_layers * num_directions, batch, hidden_size)。

    dropoutは最後のレイヤーには適用されない。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, return_sequences=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.return_sequences = return_sequences

        # 各レイヤーのLSTMCellを作成
        self.cells_forward = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            cell = LSTMCell(layer_input_size, hidden_size, bias)
            setattr(self, f'cell_forward_{layer}', cell)
            self.cells_forward.append(cell)

        # 双方向の場合、逆方向のセルも作成
        if bidirectional:
            self.cells_backward = []
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                cell = LSTMCell(layer_input_size, hidden_size, bias)
                setattr(self, f'cell_backward_{layer}', cell)
                self.cells_backward.append(cell)

    def forward(self, x, hx=None):
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力テンソル
            - batch_first=False: (seq_len, batch, input_size)
            - batch_first=True: (batch, seq_len, input_size)
        hx : tuple of Tensor, optional
            (h_0, c_0) のタプル
            - h_0: (num_layers * num_directions, batch, hidden_size)
            - c_0: (num_layers * num_directions, batch, hidden_size)
            Noneの場合、ゼロで初期化される

        Returns
        -------
        output : Tensor
            出力テンソル
            - batch_first=False: (seq_len, batch, hidden_size * num_directions)
            - batch_first=True: (batch, seq_len, hidden_size * num_directions)
        (h_n, c_n) : tuple of Tensor
            最終隠れ状態とセル状態
            - h_n: (num_layers * num_directions, batch, hidden_size)
            - c_n: (num_layers * num_directions, batch, hidden_size)
        """
        # batch_firstの場合、(batch, seq, feature) -> (seq, batch, feature)に変換
        if self.batch_first:
            x = x.transpose((1, 0, 2))

        seq_len, batch_size, _ = x.shape

        # 隠れ状態の初期化
        if hx is None:
            h_0 = nm.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            c_0 = nm.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        else:
            h_0, c_0 = hx

        # レイヤーごとに処理（双方向の場合は順方向と逆方向を同時に処理）
        layer_input = x  # (seq_len, batch, input_size)
        h_list = []
        c_list = []

        for layer in range(self.num_layers):
            # 順方向の処理
            h_forward = h_0[layer * self.num_directions]
            c_forward = c_0[layer * self.num_directions]

            outputs_forward = []
            for t in range(seq_len):
                h_forward, c_forward = self.cells_forward[layer](
                    layer_input[t], (h_forward, c_forward)
                )
                outputs_forward.append(h_forward)

            output_forward = nm.stack(outputs_forward, axis=0)  # (seq_len, batch, hidden_size)

            # 双方向の場合、逆方向も処理
            if self.bidirectional:
                h_backward = h_0[layer * self.num_directions + 1]
                c_backward = c_0[layer * self.num_directions + 1]

                outputs_backward = []
                for t in range(seq_len - 1, -1, -1):
                    h_backward, c_backward = self.cells_backward[layer](
                        layer_input[t], (h_backward, c_backward)
                    )
                    outputs_backward.append(h_backward)

                outputs_backward.reverse()
                output_backward = nm.stack(outputs_backward, axis=0)

                # 順方向と逆方向を連結
                layer_output = nm.concatenate([output_forward, output_backward], axis=2)

                # 隠れ状態を保存
                h_list.extend([h_forward, h_backward])
                c_list.extend([c_forward, c_backward])
            else:
                layer_output = output_forward
                h_list.append(h_forward)
                c_list.append(c_forward)

            # ドロップアウト（最後のレイヤー以外）
            if layer < self.num_layers - 1 and self.dropout_p > 0:
                # 各タイムステップにドロップアウトを適用
                layer_output_list = []
                for t in range(seq_len):
                    output_t = dropout(layer_output[t], p=self.dropout_p, training=train.is_on())
                    layer_output_list.append(output_t)
                layer_output = nm.stack(layer_output_list, axis=0)

            # 次のレイヤーへの入力
            layer_input = layer_output

        # 最終的な出力と隠れ状態
        output = layer_output
        h_n = nm.stack(h_list, axis=0)
        c_n = nm.stack(c_list, axis=0)

        # batch_firstの場合、(seq, batch, feature) -> (batch, seq, feature)に戻す
        if self.batch_first:
            output = output.transpose((1, 0, 2))

        # return_sequences=Falseの場合、最後のレイヤーの最終隠れ状態のみ返す
        if not self.return_sequences:
            if self.bidirectional:
                # 双方向の場合: 最後のレイヤーの順方向と逆方向を連結
                # h_n[-2] が順方向、h_n[-1] が逆方向
                return nm.concatenate([h_n[-2], h_n[-1]], axis=1)
            else:
                return h_n[-1]  # (batch, hidden_size)

        return output, (h_n, c_n)

    def __repr__(self):
        return (f"LSTM({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"batch_first={self.batch_first}, dropout={self.dropout_p}, "
                f"bidirectional={self.bidirectional})")
