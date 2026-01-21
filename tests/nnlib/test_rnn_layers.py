"""
Comprehensive tests for RNN layers (RNN, LSTM, GRU)

Tests cover:
1. Basic gradient computation
2. Numerical gradient check
3. Multi-layer and bidirectional configurations
4. Simple sequence learning task
5. Hidden state management
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
from lemon import nnlib as nl
import numpy as np


def numerical_gradient(f, x, eps=1e-4):
    """
    Compute numerical gradient using finite differences

    Parameters
    ----------
    f : callable
        Function that takes no arguments and returns a scalar
    x : ndarray
        Point at which to compute gradient
    eps : float
        Finite difference step size

    Returns
    -------
    ndarray
        Numerical gradient
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + eps)
        x[idx] = old_value + eps
        fxh_pos = f()

        # f(x - eps)
        x[idx] = old_value - eps
        fxh_neg = f()

        # Numerical gradient
        grad[idx] = (fxh_pos - fxh_neg) / (2 * eps)

        # Restore
        x[idx] = old_value
        it.iternext()

    return grad


def test_lstm_gradient_computation():
    """Test 1: LSTM gradient computation"""
    print("=" * 70)
    print("Test 1: LSTM Gradient Computation")
    print("=" * 70)

    nm.autograd.enable()

    # Small LSTM for easy verification
    lstm = nl.LSTM(input_size=5, hidden_size=8, num_layers=1)
    x = nm.randn(3, 2, 5, requires_grad=True)  # (seq_len, batch, input_size)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, (h_n, c_n) = lstm(x)
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Compute loss
    loss = nm.sum(output)
    print(f"Loss: {loss._data}")

    # Backward pass
    loss.backward()

    # Check gradients
    print(f"\nAfter backward:")
    print(f"  x.grad is not None: {x.grad is not None}")

    # Check all LSTM parameters have gradients
    has_grad = True
    for name, param in lstm.named_parameters():
        if param.grad is None:
            print(f"  WARNING: {name}.grad is None")
            has_grad = False

    if has_grad:
        print(f"  All LSTM parameters have gradients ✓")

    assert x.grad is not None, "Input gradient not computed"
    assert has_grad, "Some LSTM parameters don't have gradients"

    print("\n✓ Test 1 passed\n")


def test_lstm_numerical_gradient():
    """Test 2: LSTM numerical gradient check"""
    print("=" * 70)
    print("Test 2: LSTM Numerical Gradient Check")
    print("=" * 70)

    nm.autograd.enable()

    # Very small LSTM for faster numerical gradient computation
    lstm = nl.LSTM(input_size=3, hidden_size=4, num_layers=1)
    x = nm.randn(2, 1, 3, requires_grad=True)  # (seq_len, batch, input_size)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, (h_n, c_n) = lstm(x)
    loss = nm.sum(output)
    loss.backward()

    # Check numerical gradient for input
    print("\nChecking input gradient...")
    x_data = x._data

    def loss_fn():
        x_temp = nm.tensor(x_data, requires_grad=False)
        out, _ = lstm(x_temp)
        return nm.sum(out)._data

    numerical_grad = numerical_gradient(loss_fn, x_data, eps=1e-4)
    analytical_grad = x.grad._data

    # Sample a few points to check
    max_diff = 0
    for i in range(min(5, x_data.size)):
        idx = np.unravel_index(i, x_data.shape)
        num_g = numerical_grad[idx]
        ana_g = analytical_grad[idx]
        diff = abs(num_g - ana_g)
        max_diff = max(max_diff, diff)
        print(f"  Position {idx}: numerical={num_g:.6f}, analytical={ana_g:.6f}, diff={diff:.6e}")

    print(f"\nMax gradient difference: {max_diff:.6e}")

    # Gradient check with relative error
    relative_error = max_diff / (abs(numerical_grad).max() + 1e-8)
    print(f"Relative error: {relative_error:.6e}")

    assert relative_error < 1e-2, f"Gradient check failed: relative_error={relative_error}"

    print("\n✓ Test 2 passed\n")


def test_bidirectional_lstm():
    """Test 3: Bidirectional LSTM"""
    print("=" * 70)
    print("Test 3: Bidirectional LSTM")
    print("=" * 70)

    nm.autograd.enable()

    # Bidirectional LSTM
    lstm = nl.LSTM(input_size=5, hidden_size=8, num_layers=2, bidirectional=True)
    x = nm.randn(4, 3, 5, requires_grad=True)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, (h_n, c_n) = lstm(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 3, 16)  # 8 * 2 (bidirectional)")
    assert output.shape == (4, 3, 16), f"Wrong output shape: {output.shape}"

    print(f"h_n shape: {h_n.shape}")
    print(f"Expected: (4, 3, 8)  # 2 layers * 2 directions")
    assert h_n.shape == (4, 3, 8), f"Wrong h_n shape: {h_n.shape}"

    # Backward pass
    loss = nm.sum(output)
    loss.backward()

    assert x.grad is not None, "Input gradient not computed"

    print("\n✓ Test 3 passed\n")


def test_gru_basic():
    """Test 4: GRU basic functionality"""
    print("=" * 70)
    print("Test 4: GRU Basic Functionality")
    print("=" * 70)

    nm.autograd.enable()

    # GRU
    gru = nl.GRU(input_size=6, hidden_size=10, num_layers=2)
    x = nm.randn(5, 2, 6, requires_grad=True)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, h_n = gru(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == (5, 2, 10), f"Wrong output shape: {output.shape}"

    print(f"h_n shape: {h_n.shape}")
    assert h_n.shape == (2, 2, 10), f"Wrong h_n shape: {h_n.shape}"

    # Backward pass
    loss = nm.sum(output)
    loss.backward()

    assert x.grad is not None, "Input gradient not computed"

    print("\n✓ Test 4 passed\n")


def test_rnn_basic():
    """Test 5: Vanilla RNN basic functionality"""
    print("=" * 70)
    print("Test 5: Vanilla RNN Basic Functionality")
    print("=" * 70)

    nm.autograd.enable()

    # RNN
    rnn = nl.RNN(input_size=4, hidden_size=6, num_layers=1, nonlinearity='tanh')
    x = nm.randn(3, 2, 4, requires_grad=True)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, h_n = rnn(x)

    print(f"Output shape: {output.shape}")
    assert output.shape == (3, 2, 6), f"Wrong output shape: {output.shape}"

    print(f"h_n shape: {h_n.shape}")
    assert h_n.shape == (1, 2, 6), f"Wrong h_n shape: {h_n.shape}"

    # Backward pass
    loss = nm.sum(output)
    loss.backward()

    assert x.grad is not None, "Input gradient not computed"

    print("\n✓ Test 5 passed\n")


def test_lstm_sequence_learning():
    """Test 6: LSTM simple sequence learning"""
    print("=" * 70)
    print("Test 6: LSTM Sequence Learning (Copy Task)")
    print("=" * 70)

    nm.autograd.enable()

    # Simple copy task: learn to copy input sequence
    seq_len = 5
    input_size = 3
    hidden_size = 8

    lstm = nl.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
    linear = nl.Linear(hidden_size, input_size)
    optimizer = nl.Adam(list(lstm.parameters()) + list(linear.parameters()), lr=0.01)

    # Training data: random sequences
    np.random.seed(42)
    train_sequences = []
    for _ in range(10):
        seq = nm.randn(seq_len, 1, input_size)
        train_sequences.append(seq)

    print(f"\nTraining LSTM on copy task...")
    print(f"Sequence length: {seq_len}")
    print(f"Input size: {input_size}")

    # Train for a few epochs
    initial_loss = None
    final_loss = None

    for epoch in range(20):
        epoch_loss = 0
        for seq in train_sequences:
            # Forward
            output, _ = lstm(seq)
            predictions = linear(output)
            loss = nm.sum((predictions - seq) ** 2)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss._data

        avg_loss = epoch_loss / len(train_sequences)

        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={avg_loss:.6f}")

    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Improvement:  {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

    # Check that loss decreased significantly
    assert final_loss < initial_loss * 0.5, "LSTM failed to learn (loss didn't decrease enough)"

    print("\n✓ Test 6 passed\n")


def test_batch_first_mode():
    """Test 7: batch_first mode"""
    print("=" * 70)
    print("Test 7: batch_first Mode")
    print("=" * 70)

    nm.autograd.enable()

    # LSTM with batch_first=True
    lstm = nl.LSTM(input_size=5, hidden_size=8, num_layers=1, batch_first=True)
    x = nm.randn(3, 4, 5, requires_grad=True)  # (batch, seq_len, input_size)

    print(f"\nInput shape (batch_first): {x.shape}")

    # Forward pass
    output, (h_n, c_n) = lstm(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: (3, 4, 8)  # (batch, seq_len, hidden_size)")
    assert output.shape == (3, 4, 8), f"Wrong output shape: {output.shape}"

    # Backward pass
    loss = nm.sum(output)
    loss.backward()

    assert x.grad is not None, "Input gradient not computed"

    print("\n✓ Test 7 passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RNN Layers Comprehensive Test Suite")
    print("=" * 70 + "\n")

    try:
        test_lstm_gradient_computation()
        test_lstm_numerical_gradient()
        test_bidirectional_lstm()
        test_gru_basic()
        test_rnn_basic()
        test_lstm_sequence_learning()
        test_batch_first_mode()

        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
