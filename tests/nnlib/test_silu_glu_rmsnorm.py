import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_silu():
    print("Testing SiLU...")
    xp = nm.np

    # Test 1: output shape
    x = nm.randn(3, 4)
    y = nl.silu(x)
    assert y.shape == (3, 4), f"Shape mismatch: {y.shape}"
    print("  ✅ output shape")

    # Test 2: silu(0) == 0
    x = nm.tensor([0.0])
    y = nl.silu(x)
    assert xp.allclose(y._data, xp.array([0.0])), "silu(0) should be 0"
    print("  ✅ silu(0) == 0")

    # Test 3: silu(x) ≈ x * sigmoid(x) numerically
    x_val = xp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=xp.float32)
    x = nm.tensor(x_val)
    y = nl.silu(x)
    sig = 1.0 / (1.0 + xp.exp(-x_val))
    expected = x_val * sig
    assert xp.allclose(y._data, expected, atol=1e-6), "silu(x) != x*sigmoid(x)"
    print("  ✅ silu(x) == x * sigmoid(x)")

    # Test 4: gradient
    nm.autograd.enable()
    x = nm.randn(2, 5, requires_grad=True)
    y = nl.silu(x)
    loss = nm.sum(y)
    loss.backward()
    assert x.grad is not None, "Gradient should be computed"
    print("  ✅ gradient")

    # Test 5: Module form
    model = nl.Silu()
    x = nm.randn(2, 8)
    y = model(x)
    assert y.shape == (2, 8)
    assert "Silu" in repr(model)
    print("  ✅ Module form and repr")

    print("✅ All SiLU tests passed!\n")


def test_glu():
    print("Testing GLU...")
    xp = nm.np

    # Test 1: output shape (last dim halved)
    x = nm.randn(2, 8)
    y = nl.glu(x)
    assert y.shape == (2, 4), f"Shape should be (2,4), got {y.shape}"
    print("  ✅ output shape (2D)")

    # Test 2: 3D input
    x = nm.randn(2, 10, 16)
    y = nl.glu(x)
    assert y.shape == (2, 10, 8), f"Shape should be (2,10,8), got {y.shape}"
    print("  ✅ output shape (3D)")

    # Test 3: explicit dim
    x = nm.randn(4, 6)
    y = nl.glu(x, dim=0)
    assert y.shape == (2, 6), f"Shape should be (2,6), got {y.shape}"
    print("  ✅ explicit dim=0")

    # Test 4: odd size raises
    try:
        nl.glu(nm.randn(2, 5))
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  ✅ odd dim raises ValueError")

    # Test 5: numerical correctness
    x_val = xp.array([[1.0, 2.0, 3.0, 4.0]], dtype=xp.float32)
    x = nm.tensor(x_val)
    y = nl.glu(x)
    a = x_val[:, :2]
    b = x_val[:, 2:]
    expected = a * (1.0 / (1.0 + xp.exp(-b)))
    assert xp.allclose(y._data, expected, atol=1e-6), "GLU values wrong"
    print("  ✅ numerical correctness")

    # Test 6: gradient
    nm.autograd.enable()
    x = nm.randn(2, 8, requires_grad=True)
    y = nl.glu(x)
    loss = nm.sum(y)
    loss.backward()
    assert x.grad is not None, "Gradient should be computed"
    assert x.grad.shape == (2, 8), "Gradient shape should match input"
    print("  ✅ gradient")

    # Test 7: Module form
    model = nl.Glu()
    x = nm.randn(3, 10)
    y = model(x)
    assert y.shape == (3, 5)
    assert "Glu" in repr(model)
    print("  ✅ Module form and repr")

    print("✅ All GLU tests passed!\n")


def test_rms_norm():
    print("Testing RMSNorm...")
    xp = nm.np

    # Test 1: output shape
    norm = nl.RMSNorm(16)
    x = nm.randn(2, 10, 16)
    y = norm(x)
    assert y.shape == (2, 10, 16), f"Shape mismatch: {y.shape}"
    print("  ✅ output shape (3D)")

    # Test 2: 2D input
    norm = nl.RMSNorm(8)
    x = nm.randn(4, 8)
    y = norm(x)
    assert y.shape == (4, 8)
    print("  ✅ output shape (2D)")

    # Test 3: RMS of output ≈ weight (when weight=1, RMS≈1)
    norm = nl.RMSNorm(32, elementwise_affine=False)
    x = nm.randn(8, 32)
    y = norm(x)
    rms = xp.sqrt(xp.mean(y._data ** 2, axis=-1))
    assert xp.allclose(rms, 1.0, atol=1e-5), f"RMS should be ≈1, got {rms}"
    print("  ✅ RMS of output ≈ 1 (no affine)")

    # Test 4: no learnable params when elementwise_affine=False
    norm_no_affine = nl.RMSNorm(8, elementwise_affine=False)
    assert len(list(norm_no_affine.parameters())) == 0
    print("  ✅ no parameters when elementwise_affine=False")

    # Test 5: gradient
    nm.autograd.enable()
    norm = nl.RMSNorm(16)
    x = nm.randn(2, 5, 16, requires_grad=True)
    y = norm(x)
    loss = nm.sum(y)
    loss.backward()
    assert x.grad is not None, "Input gradient should be computed"
    for p in norm.parameters():
        assert p.grad is not None, "Parameter gradient should be computed"
    print("  ✅ gradient (input + weight)")

    # Test 6: repr
    assert "RMSNorm" in repr(nl.RMSNorm(64))
    print("  ✅ repr")

    print("✅ All RMSNorm tests passed!\n")


def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING SiLU / GLU / RMSNorm TESTS")
    print("=" * 70 + "\n")

    try:
        test_silu()
        test_glu()
        test_rms_norm()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
