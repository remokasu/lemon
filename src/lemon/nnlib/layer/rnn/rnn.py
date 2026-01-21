import lemon.numlib as nm
from lemon.nnlib.module import Module
from lemon.nnlib.layer.rnn.rnn_cell import RNNCell
from lemon.nnlib.layer.dropout import dropout
from lemon.nnlib.train_control import train


class RNN(Module):
    """
    Vanilla RNN レイヤー

    単純なRNN（Recurrent Neural Network）。
    勾配消失問題があるため、長い系列にはLSTMやGRUを推奨。

    Parameters
    ----------
    input_size : int
        入力特徴量の次元数
    hidden_size : int
        隠れ状態の次元数
    num_layers : int, optional
        積み重ねるRNNレイヤーの数 (デフォルト: 1)
    nonlinearity : str, optional
        活性化関数。'tanh' または 'relu' (デフォルト: 'tanh')
    bias : bool, optional
        バイアス項を使用するか (デフォルト: True)
    batch_first : bool, optional
        Trueの場合、入力形状は(batch, seq, feature) (デフォルト: False)
    dropout : float, optional
        レイヤー間のドロップアウト率 (デフォルト: 0.0)
    bidirectional : bool, optional
        双方向RNNにするか (デフォルト: False)
    return_sequences : bool, optional
        Trueの場合、全タイムステップの出力を返す。
        Falseの場合、最後のレイヤーの最終隠れ状態のみ返す (デフォルト: True)

    Examples
    --------
    >>> import lemon.numlib as nm
    >>> from lemon.nnlib.layer.rnn import RNN
    >>>
    >>> # 単層RNN
    >>> rnn = RNN(input_size=10, hidden_size=20)
    >>> x = nm.randn(5, 3, 10)  # (seq_len, batch, input_size)
    >>> output, h_n = rnn(x)
    >>> output.shape
    (5, 3, 20)
    >>> h_n.shape
    (1, 3, 20)
    >>>
    >>> # ReLU活性化関数を使用
    >>> rnn = RNN(10, 20, nonlinearity='relu')
    >>>
    >>> # 多層双方向RNN
    >>> rnn = RNN(10, 20, num_layers=2, bidirectional=True)
    >>> x = nm.randn(5, 3, 10)
    >>> output, h_n = rnn(x)
    >>> output.shape
    (5, 3, 40)  # 20 * 2 (bidirectional)

    Notes
    -----
    Vanilla RNNは勾配消失問題を抱えやすい。
    長い系列（> 10-20ステップ）を扱う場合は、LSTMやGRUの使用を推奨。
    """

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False, return_sequences=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.return_sequences = return_sequences

        # 各レイヤーのRNNCellを作成
        self.cells_forward = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
            setattr(self, f'cell_forward_{layer}', cell)
            self.cells_forward.append(cell)

        # 双方向の場合、逆方向のセルも作成
        if bidirectional:
            self.cells_backward = []
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
                setattr(self, f'cell_backward_{layer}', cell)
                self.cells_backward.append(cell)

    def forward(self, x, h=None):
        """
        順伝播

        Parameters
        ----------
        x : Tensor
            入力テンソル
            - batch_first=False: (seq_len, batch, input_size)
            - batch_first=True: (batch, seq_len, input_size)
        h : Tensor, optional
            初期隠れ状態 (num_layers * num_directions, batch, hidden_size)
            Noneの場合、ゼロで初期化される

        Returns
        -------
        output : Tensor
            出力テンソル
            - batch_first=False: (seq_len, batch, hidden_size * num_directions)
            - batch_first=True: (batch, seq_len, hidden_size * num_directions)
        h_n : Tensor
            最終隠れ状態 (num_layers * num_directions, batch, hidden_size)
        """
        # batch_firstの場合、(batch, seq, feature) -> (seq, batch, feature)に変換
        if self.batch_first:
            x = x.transpose((1, 0, 2))

        seq_len, batch_size, _ = x.shape

        # 隠れ状態の初期化
        if h is None:
            h_0 = nm.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        else:
            h_0 = h

        # 順方向の処理
        h_forward = [h_0[i] for i in range(self.num_layers)]

        outputs_forward = []
        for t in range(seq_len):
            x_t = x[t]

            for layer in range(self.num_layers):
                h_forward[layer] = self.cells_forward[layer](x_t, h_forward[layer])
                x_t = h_forward[layer]

                if layer < self.num_layers - 1 and self.dropout_p > 0:
                    x_t = dropout(x_t, p=self.dropout_p, training=train.is_on())

            outputs_forward.append(x_t)

        output_forward = nm.stack(outputs_forward, axis=0)

        # 双方向の場合、逆方向も処理
        if self.bidirectional:
            h_backward = [h_0[self.num_layers + i] for i in range(self.num_layers)]

            # 逆方向用に各レイヤーの出力を保存
            layer_outputs_backward = [[] for _ in range(self.num_layers)]

            for t in range(seq_len - 1, -1, -1):
                # 第1層は元の入力から、第2層以降は前の層の出力から
                for layer in range(self.num_layers):
                    if layer == 0:
                        x_t = x[t]
                    else:
                        # 前の層の同じタイムステップの出力を使う
                        prev_forward = output_forward[t, :, :]
                        prev_backward = layer_outputs_backward[layer-1][-1]
                        x_t = nm.concatenate([prev_forward, prev_backward], axis=1)

                    h_backward[layer] = self.cells_backward[layer](x_t, h_backward[layer])
                    output_t = h_backward[layer]

                    if layer < self.num_layers - 1 and self.dropout_p > 0:
                        output_t = dropout(output_t, p=self.dropout_p, training=train.is_on())

                    layer_outputs_backward[layer].append(output_t)

            # 最終層の出力を取得（逆順なので反転）
            final_backward = layer_outputs_backward[-1]
            final_backward.reverse()
            output_backward = nm.stack(final_backward, axis=0)

            # 順方向と逆方向を連結
            output = nm.concatenate([output_forward, output_backward], axis=2)
            h_n = nm.stack(h_forward + h_backward, axis=0)
        else:
            output = output_forward
            h_n = nm.stack(h_forward, axis=0)

        # batch_firstの場合、(seq, batch, feature) -> (batch, seq, feature)に戻す
        if self.batch_first:
            output = output.transpose((1, 0, 2))

        # return_sequences=Falseの場合、最後のレイヤーの最終隠れ状態のみ返す
        if not self.return_sequences:
            if self.bidirectional:
                # 双方向の場合: 最後のレイヤーの順方向と逆方向を連結
                return nm.concatenate([h_n[-2], h_n[-1]], axis=1)
            else:
                return h_n[-1]  # (batch, hidden_size)

        return output, h_n

    def __repr__(self):
        return (f"RNN({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"nonlinearity='{self.nonlinearity}', batch_first={self.batch_first}, "
                f"dropout={self.dropout_p}, bidirectional={self.bidirectional})")
