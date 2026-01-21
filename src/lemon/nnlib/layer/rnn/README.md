# RNN Layers

RNN (Recurrent Neural Network) 系レイヤーの実装

## 概要

時系列データや系列データを処理するための再帰的ニューラルネットワークレイヤー。
LSTM、GRU、Vanilla RNN の3種類。

## 提供するレイヤー

### セルレベル (単一タイムステップ)

- **RNNCell**: Vanilla RNN セル
- **LSTMCell**: LSTM (Long Short-Term Memory) セル
- **GRUCell**: GRU (Gated Recurrent Unit) セル

### 多層ラッパー (系列処理)

- **RNN**: 多層 Vanilla RNN
- **LSTM**: 多層 LSTM
- **GRU**: 多層 GRU

## 主な機能

- 多層スタッキング (`num_layers` パラメータ)
- 双方向処理 (`bidirectional=True`)
- レイヤー間ドロップアウト
- `batch_first` モード対応
- 完全な勾配計算サポート

## 使用例

### LSTM

```python
import lemon.nnlib as nl
import lemon.numlib as nm

# 多層双方向 LSTM
lstm = nl.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    dropout=0.3
)

# 入力: (seq_len, batch, input_size)
x = nm.randn(10, 32, 128)
output, (h_n, c_n) = lstm(x)

# output: (10, 32, 512)  # 256 * 2 (bidirectional)
# h_n: (4, 32, 256)      # 2 layers * 2 directions
# c_n: (4, 32, 256)
```

### GRU

```python
# 単層 GRU
gru = nl.GRU(input_size=64, hidden_size=128, num_layers=1)

x = nm.randn(20, 16, 64)
output, h_n = gru(x)

# output: (20, 16, 128)
# h_n: (1, 16, 128)
```

### batch_first モード

```python
# batch_first=True の場合、入力は (batch, seq_len, input_size)
lstm = nl.LSTM(input_size=50, hidden_size=100, batch_first=True)

x = nm.randn(32, 10, 50)  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)

# output: (32, 10, 100)  # (batch, seq_len, hidden_size)
```

## アーキテクチャ

### LSTM

4つのゲートを持つ:
- 入力ゲート (input gate): 新しい情報をどれだけ受け入れるか
- 忘却ゲート (forget gate): 過去の情報をどれだけ忘れるか
- セルゲート (cell gate): 新しい候補値
- 出力ゲート (output gate): 出力をどれだけ制御するか

### GRU

3つのゲートを持つ (LSTM より軽量):
- リセットゲート (reset gate): 過去の情報をどれだけリセットするか
- 更新ゲート (update gate): 過去と現在の情報のバランス
- 新しい値 (new gate): 候補となる新しい隠れ状態

### Vanilla RNN

最もシンプルな RNN。長い系列では勾配消失問題が発生しやすい。

## パラメータ初期化

PyTorch と同じ初期化方法を採用:
- 重み: uniform(-sqrt(k), sqrt(k)), k = 1/hidden_size
- LSTM の忘却ゲートバイアス: 1.0 (勾配消失防止)

## 実装ファイル

- `rnn_cell.py`: RNNCell 実装
- `lstm_cell.py`: LSTMCell 実装
- `gru_cell.py`: GRUCell 実装
- `rnn.py`: 多層 RNN ラッパー
- `lstm.py`: 多層 LSTM ラッパー
- `gru.py`: 多層 GRU ラッパー

## テスト

包括的なテストスイートを提供:
- 勾配計算の検証
- 数値勾配チェック
- 多層・双方向の動作確認
- 系列学習タスク

テスト実行:
```bash
pytest tests/nnlib/test_rnn_layers.py -v
```
