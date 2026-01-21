import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import lemon.nnlib as nl
import lemon.numlib as nm


def test_sequential():
    """Test Sequential container"""
    print("Testing Sequential...")

    # Test 1: Simple sequential
    class Identity(nl.Module):
        def forward(self, x):
            return x

    class AddOne(nl.Module):
        def forward(self, x):
            return x + 1

    seq = nl.Sequential(Identity(), AddOne(), AddOne())

    x = nm.tensor([1.0, 2.0])
    y = seq(x)
    assert nm.get_array_module(y._data).allclose(y._data, [3.0, 4.0]), (
        "Sequential should apply layers in order"
    )
    print("  ✅ Simple sequential")

    # Test 2: Sequential with parameters
    class LinearLayer(nl.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = nl.Parameter(nm.tensor([scale]))

        def forward(self, x):
            return x * self.scale.data

    seq_param = nl.Sequential(LinearLayer(2.0), LinearLayer(3.0))

    x = nm.tensor([1.0])
    y = seq_param(x)
    assert nm.get_array_module(y._data).allclose(y._data, [6.0]), (
        "Should multiply by 2 then 3"
    )

    # Check parameters
    param_count = sum(1 for _ in seq_param.parameters())
    assert param_count == 2, f"Should have 2 parameters, got {param_count}"
    print("  ✅ Sequential with parameters")

    # Test 3: Empty sequential
    empty_seq = nl.Sequential()
    x = nm.tensor([1.0, 2.0])
    y = empty_seq(x)
    assert nm.get_array_module(y._data).allclose(y._data, x._data), (
        "Empty sequential should return input unchanged"
    )
    print("  ✅ Empty sequential")

    # Test 4: Single layer sequential
    single_seq = nl.Sequential(AddOne())
    x = nm.tensor([5.0])
    y = single_seq(x)
    assert nm.get_array_module(y._data).allclose(y._data, [6.0]), (
        "Single layer should work"
    )
    print("  ✅ Single layer sequential")

    # Test 5: __repr__
    repr_str = repr(seq)
    assert "Sequential" in repr_str, "repr should contain Sequential"
    assert "Identity" in repr_str or "AddOne" in repr_str, (
        "repr should contain layer names"
    )
    print("  ✅ __repr__")

    # Test 6: zero_grad propagation
    nm.autograd.enable()
    seq_grad = nl.Sequential(LinearLayer(2.0), LinearLayer(3.0))

    x = nm.tensor([1.0])
    y = seq_grad(x)

    # 修正: スカラーにしてから backward
    loss = nm.sum(y)  # shape: () - スカラー
    loss.backward()

    # Check gradients exist
    for param in seq_grad.parameters():
        assert param.grad is not None, "Gradient should exist"

    # Zero grad
    seq_grad.zero_grad()
    for param in seq_grad.parameters():
        assert param.grad is None, "Gradient should be None after zero_grad"
    print("  ✅ zero_grad() propagation in Sequential")

    print("✅ All Sequential tests passed!\n")
