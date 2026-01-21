import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_sgd_optimizer():
    """Test SGD optimizer"""

    print("Testing SGD optimizer...")

    xp = nm.np

    # Test 1: Basic SGD
    nm.autograd.enable()

    # Simple model
    W = nl.Parameter(nm.tensor([[1.0, 2.0], [3.0, 4.0]]))
    b = nl.Parameter(nm.tensor([0.5, 0.5]))

    optimizer = nl.SGD([W, b], lr=0.1)

    # Forward
    x = nm.tensor([[1.0, 1.0]])
    y_true = nm.tensor([[2.0, 3.0]])
    y_pred = x @ W.data + b.data
    loss = nl.mean_squared_error(y_pred, y_true)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Save initial values
    W_before = W.data._data.copy()
    b_before = b.data._data.copy()

    # Step
    optimizer.step()

    # Check parameters were updated
    assert not xp.allclose(W.data._data, W_before), "W should be updated"
    assert not xp.allclose(b.data._data, b_before), "b should be updated"
    print("  ✅ Basic SGD")

    # Test 2: SGD with model
    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))

    optimizer = nl.SGD(model.parameters(), lr=0.01)

    x = nm.randn(3, 10)
    y_true = nm.randn(3, 2)

    # Training step
    y_pred = model(x)
    loss = nl.mean_squared_error(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("  ✅ SGD with model")

    # Test 3: SGD with momentum
    W = nl.Parameter(nm.randn(5, 3))
    optimizer = nl.SGD([W], lr=0.1, momentum=0.9)

    assert optimizer.velocity is not None, (
        "Velocity should be initialized with momentum"
    )

    # Multiple steps
    for _ in range(5):
        y = W.data @ nm.randn(3, 2)
        loss = nm.sum(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("  ✅ SGD with momentum")

    # Test 4: SGD with weight decay
    W = nl.Parameter(nm.randn(5, 3))
    optimizer = nl.SGD([W], lr=0.1, weight_decay=0.01)

    W_before = W.data._data.copy()

    y = W.data @ nm.randn(3, 2)
    loss = nm.sum(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Weight decay should cause additional shrinkage
    assert not xp.allclose(W.data._data, W_before), (
        "W should be updated with weight decay"
    )
    print("  ✅ SGD with weight decay")

    # Test 5: Multiple parameters
    params = [nl.Parameter(nm.randn(3, 3)) for _ in range(5)]
    optimizer = nl.SGD(params, lr=0.01)

    assert len(optimizer.params) == 5, "Optimizer should track all parameters"
    print("  ✅ SGD with multiple parameters")

    # Test 6: repr
    optimizer = nl.SGD([W], lr=0.01, momentum=0.9, weight_decay=0.001)
    repr_str = repr(optimizer)
    assert "SGD" in repr_str, "repr should contain 'SGD'"
    assert "0.01" in repr_str, "repr should contain lr"
    print("  ✅ SGD repr")

    print("✅ All SGD optimizer tests passed!\n")


def test_adam_optimizer():
    """Test Adam optimizer"""

    print("Testing Adam optimizer...")

    xp = nm.np

    # Test 1: Basic Adam
    nm.autograd.enable()

    W = nl.Parameter(nm.tensor([[1.0, 2.0], [3.0, 4.0]]))
    b = nl.Parameter(nm.tensor([0.5, 0.5]))

    optimizer = nl.Adam([W, b], lr=0.01)

    # Forward
    x = nm.tensor([[1.0, 1.0]])
    y_true = nm.tensor([[2.0, 3.0]])
    y_pred = x @ W.data + b.data
    loss = nl.mean_squared_error(y_pred, y_true)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Save initial values
    W_before = W.data._data.copy()
    b_before = b.data._data.copy()

    # Step
    optimizer.step()

    # Check parameters were updated
    assert not xp.allclose(W.data._data, W_before), "W should be updated"
    assert not xp.allclose(b.data._data, b_before), "b should be updated"
    assert optimizer.t == 1, "Time step should be incremented"
    print("  ✅ Basic Adam")

    # Test 2: Adam with model
    model = nl.Sequential(nl.Linear(10, 5), nl.Relu(), nl.Linear(5, 2))

    optimizer = nl.Adam(model.parameters(), lr=0.001)

    x = nm.randn(3, 10)
    y_true = nm.randn(3, 2)

    # Training step
    y_pred = model(x)
    loss = nl.mean_squared_error(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("  ✅ Adam with model")

    # Test 3: Multiple Adam steps
    W = nl.Parameter(nm.randn(5, 3))
    optimizer = nl.Adam([W], lr=0.001)

    initial_loss = None
    for i in range(10):
        y = W.data @ nm.randn(3, 2)
        loss = nm.sum(y**2)

        if i == 0:
            initial_loss = loss._data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert optimizer.t == i + 1, f"Time step should be {i + 1}"

    print("  ✅ Multiple Adam steps")

    # Test 4: Adam with weight decay
    W = nl.Parameter(nm.randn(5, 3))
    optimizer = nl.Adam([W], lr=0.001, weight_decay=0.01)

    W_before = W.data._data.copy()

    y = W.data @ nm.randn(3, 2)
    loss = nm.sum(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not xp.allclose(W.data._data, W_before), (
        "W should be updated with weight decay"
    )
    print("  ✅ Adam with weight decay")

    # Test 5: Adam with custom betas
    W = nl.Parameter(nm.randn(3, 3))
    optimizer = nl.Adam([W], lr=0.001, betas=(0.8, 0.99))

    assert optimizer.beta1 == 0.8, "beta1 should be 0.8"
    assert optimizer.beta2 == 0.99, "beta2 should be 0.99"
    print("  ✅ Adam with custom betas")

    # Test 6: repr
    optimizer = nl.Adam([W], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    repr_str = repr(optimizer)
    assert "Adam" in repr_str, "repr should contain 'Adam'"
    assert "0.001" in repr_str, "repr should contain lr"
    print("  ✅ Adam repr")

    print("✅ All Adam optimizer tests passed!\n")


def test_optimizer_integration():
    """Test optimizer integration with full training loop"""
    print("Testing optimizer integration...")

    nm.autograd.enable()
    nl.train.on()

    # Test 1: Complete training loop with SGD
    model = nl.Sequential(
        nl.Linear(10, 20), nl.Relu(), nl.Dropout(0.5), nl.Linear(20, 5)
    )

    optimizer = nl.SGD(model.parameters(), lr=0.01)

    x = nm.randn(32, 10)
    y_true = nm.tensor([i % 5 for i in range(32)])

    initial_loss = None
    for epoch in range(10):
        # Forward
        y_pred = model(x)
        loss = nl.softmax_cross_entropy(y_pred, y_true)

        if epoch == 0:
            initial_loss = loss._data

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Loss should decrease (not guaranteed but likely)
    print(f"    Initial loss: {initial_loss:.4f}, Final loss: {loss._data:.4f}")
    print("  ✅ Complete training loop with SGD")

    # Test 2: Complete training loop with Adam
    model = nl.Sequential(nl.Linear(10, 20), nl.Relu(), nl.Linear(20, 5))

    optimizer = nl.Adam(model.parameters(), lr=0.001)

    initial_loss = None
    for epoch in range(10):
        # Forward
        y_pred = model(x)
        loss = nl.softmax_cross_entropy(y_pred, y_true)

        if epoch == 0:
            initial_loss = loss._data

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"    Initial loss: {initial_loss:.4f}, Final loss: {loss._data:.4f}")
    print("  ✅ Complete training loop with Adam")

    # Test 3: Evaluation mode
    nl.train.off()

    y_pred = model(x)
    assert y_pred is not None, "Model should work in eval mode"
    print("  ✅ Evaluation mode")

    print("✅ All optimizer integration tests passed!\n")


def test_optimizer_edge_cases():
    """Test optimizer edge cases for coverage"""
    print("Testing optimizer edge cases...")

    nm.autograd.enable()

    # Test 1: Optimizer base class (for coverage)
    try:
        opt = nl.Optimizer([])
        opt.step()  # Should raise NotImplementedError
        assert False, "Should raise NotImplementedError"
    except NotImplementedError as e:
        assert "Subclasses must implement step()" in str(e)

    # Test base class __repr__
    opt = nl.Optimizer([])
    repr_str = repr(opt)
    assert "Optimizer" in repr_str, "Base class repr should work"
    print("  ✅ Optimizer base class")

    # Test 2: SGD with None gradients
    W1 = nl.Parameter(nm.randn(3, 3))
    W2 = nl.Parameter(nm.randn(3, 3))

    optimizer = nl.SGD([W1, W2], lr=0.01)

    # Only compute gradient for W1, leave W2.grad as None
    y = W1.data @ nm.randn(3, 2)
    loss = nm.sum(y)
    loss.backward()

    # W2.grad is None
    assert W2.grad is None, "W2 should have no gradient"

    # Step should skip W2
    W1_before = W1.data._data.copy()
    W2_before = W2.data._data.copy()

    optimizer.step()

    xp = nm.get_array_module(W1.data._data)
    assert not xp.allclose(W1.data._data, W1_before), "W1 should be updated"
    assert xp.allclose(W2.data._data, W2_before), (
        "W2 should not be updated (no gradient)"
    )
    print("  ✅ SGD with None gradients")

    # Test 3: Adam with None gradients
    W1 = nl.Parameter(nm.randn(3, 3))
    W2 = nl.Parameter(nm.randn(3, 3))

    optimizer = nl.Adam([W1, W2], lr=0.001)

    # Only compute gradient for W1
    y = W1.data @ nm.randn(3, 2)
    loss = nm.sum(y)
    loss.backward()

    assert W2.grad is None, "W2 should have no gradient"

    W1_before = W1.data._data.copy()
    W2_before = W2.data._data.copy()

    optimizer.step()

    assert not xp.allclose(W1.data._data, W1_before), "W1 should be updated"
    assert xp.allclose(W2.data._data, W2_before), (
        "W2 should not be updated (no gradient)"
    )
    print("  ✅ Adam with None gradients")

    # Test 4: SGD momentum with velocity initialization
    # This tests the "if self.velocity[i] is None" branch
    W = nl.Parameter(nm.randn(3, 3))
    optimizer = nl.SGD([W], lr=0.1, momentum=0.9)

    # First step initializes velocity
    y = W.data @ nm.randn(3, 2)
    loss = nm.sum(y)

    optimizer.zero_grad()
    loss.backward()

    # Before step, velocity exists but might be zeros_like
    assert optimizer.velocity is not None, "Velocity should be initialized"

    # First step
    optimizer.step()

    # Velocity should now be updated
    assert optimizer.velocity[0] is not None, "Velocity should be set after first step"
    print("  ✅ SGD momentum velocity initialization")

    # Test 5: Multiple parameters with mixed gradients
    params = [nl.Parameter(nm.randn(2, 2)) for _ in range(5)]
    optimizer = nl.SGD(params, lr=0.01)

    # Compute gradients for only some parameters
    y = params[0].data @ nm.randn(2, 1) + params[2].data @ nm.randn(2, 1)
    loss = nm.sum(y)
    loss.backward()

    # params[1], params[3], params[4] have no gradients
    assert params[0].grad is not None
    assert params[1].grad is None
    assert params[2].grad is not None
    assert params[3].grad is None
    assert params[4].grad is None

    # Step should only update params with gradients
    before = [p.data._data.copy() for p in params]
    optimizer.step()
    after = [p.data._data for p in params]

    assert not xp.allclose(before[0], after[0]), "params[0] should update"
    assert xp.allclose(before[1], after[1]), "params[1] should not update"
    assert not xp.allclose(before[2], after[2]), "params[2] should update"
    assert xp.allclose(before[3], after[3]), "params[3] should not update"
    assert xp.allclose(before[4], after[4]), "params[4] should not update"
    print("  ✅ Mixed gradients with SGD")

    # Test 6: Same for Adam
    params = [nl.Parameter(nm.randn(2, 2)) for _ in range(3)]
    optimizer = nl.Adam(params, lr=0.001)

    # Only first parameter gets gradient
    y = params[0].data @ nm.randn(2, 1)
    loss = nm.sum(y)
    loss.backward()

    before = [p.data._data.copy() for p in params]
    optimizer.step()
    after = [p.data._data for p in params]

    assert not xp.allclose(before[0], after[0]), "params[0] should update"
    assert xp.allclose(before[1], after[1]), "params[1] should not update"
    assert xp.allclose(before[2], after[2]), "params[2] should not update"
    print("  ✅ Mixed gradients with Adam")

    print("✅ All optimizer edge cases passed!\n")
