#!/usr/bin/env python3
"""
RNNモデルの基本的な動作確認テスト

Note: VGG/ResNetはexample/に移動したため、ここではテストしない
"""
import sys
sys.path.insert(0, 'src')

import lemon.numlib as nm
import lemon.nnlib as nl


def test_rnn_import():
    """RNNレイヤーのインポートテスト"""
    print("Testing RNN imports...")

    # Cell
    assert hasattr(nl, 'RNNCell')
    assert hasattr(nl, 'LSTMCell')
    assert hasattr(nl, 'GRUCell')

    # Multi-layer
    assert hasattr(nl, 'RNN')
    assert hasattr(nl, 'LSTM')
    assert hasattr(nl, 'GRU')

    print("✓ RNN imports OK")


def test_lstm_basic():
    """LSTMの基本動作テスト"""
    print("Testing LSTM basic operations...")

    # 単層LSTM
    lstm = nl.LSTM(input_size=10, hidden_size=20)
    x = nm.randn(5, 3, 10)  # (seq_len, batch, input_size)
    output, (h_n, c_n) = lstm(x)

    assert output.shape == (5, 3, 20), f"Expected (5, 3, 20), got {output.shape}"
    assert h_n.shape == (1, 3, 20), f"Expected (1, 3, 20), got {h_n.shape}"
    assert c_n.shape == (1, 3, 20), f"Expected (1, 3, 20), got {c_n.shape}"

    # 多層双方向LSTM
    lstm = nl.LSTM(10, 20, num_layers=2, bidirectional=True)
    output, (h_n, c_n) = lstm(x)

    assert output.shape == (5, 3, 40), f"Expected (5, 3, 40), got {output.shape}"
    assert h_n.shape == (4, 3, 20), f"Expected (4, 3, 20), got {h_n.shape}"

    print("✓ LSTM basic operations OK")


def test_lstm_parameters():
    """LSTMパラメータの取得テスト"""
    print("Testing LSTM parameter access...")

    lstm = nl.LSTM(10, 20, num_layers=2)
    params = list(lstm.parameters())
    assert len(params) > 0, "No parameters found in LSTM"

    print("✓ LSTM parameter access OK")


if __name__ == '__main__':
    print("=" * 50)
    print("RNN Models Test Suite")
    print("=" * 50)
    print()

    try:
        test_rnn_import()
        test_lstm_basic()
        test_lstm_parameters()

        print()
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except Exception as e:
        print()
        print("=" * 50)
        print(f"Test failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        sys.exit(1)
