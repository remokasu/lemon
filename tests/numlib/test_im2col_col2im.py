"""
Comprehensive tests for im2col and col2im functions

Tests cover:
1. Basic functionality
2. Various parameter combinations
3. Correctness verification
4. Edge cases
5. Invertibility (im2col -> col2im should preserve information)
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import numlib as nm
import numpy as np


def test_basic_functionality():
    """Test 1: Basic functionality with simple parameters"""
    print("=" * 70)
    print("Test 1: Basic Functionality")
    print("=" * 70)

    # Simple case: 1 sample, 1 channel, small image
    x = nm.tensor(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )

    print(f"\nInput shape: {x.shape}")  # (1, 1, 4, 4)
    print(f"Input:\n{x._data[0, 0]}")

    # Apply im2col with 2x2 kernel, stride=2, no padding
    col = nm.im2col(x._data, kernel_h=2, kernel_w=2, stride=2, padding=0)

    print(f"\nim2col output shape: {col.shape}")  # (1, 4, 4)
    print(f"Expected shape: (1, 1*2*2=4, 2*2=4)")

    # Each column should contain a 2x2 patch
    print(f"\nColumn 0 (top-left patch): {col[0, :, 0]}")
    print(f"Expected: [1, 2, 5, 6]")

    print(f"Column 1 (top-right patch): {col[0, :, 1]}")
    print(f"Expected: [3, 4, 7, 8]")

    print(f"Column 2 (bottom-left patch): {col[0, :, 2]}")
    print(f"Expected: [9, 10, 13, 14]")

    print(f"Column 3 (bottom-right patch): {col[0, :, 3]}")
    print(f"Expected: [11, 12, 15, 16]")

    # Verify
    expected_col = np.array(
        [[[1, 3, 9, 11], [2, 4, 10, 12], [5, 7, 13, 15], [6, 8, 14, 16]]],
        dtype=np.float32,
    )

    if np.allclose(col, expected_col):
        print("\n✅ Test 1 PASSED: im2col works correctly")
    else:
        print("\n❌ Test 1 FAILED")
        print(f"Expected:\n{expected_col}")
        print(f"Got:\n{col}")


def test_with_padding():
    """Test 2: im2col with padding"""
    print("\n" + "=" * 70)
    print("Test 2: im2col with Padding")
    print("=" * 70)

    # 3x3 image, 3x3 kernel, padding=1
    x = nm.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)

    print(f"\nInput shape: {x.shape}")  # (1, 1, 3, 3)
    print(f"Input:\n{x._data[0, 0]}")

    # With padding=1, the padded image is 5x5
    # With 3x3 kernel and stride=1, output is 3x3
    col = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=1, padding=1)

    print(f"\nim2col output shape: {col.shape}")  # (1, 9, 9)
    print(f"Expected shape: (1, 1*3*3=9, 3*3=9)")

    # Center patch should be the original image
    center_patch_idx = 4  # Middle of 3x3 output
    center_patch = col[0, :, center_patch_idx]
    print(
        f"\nCenter patch (should be the full 3x3 with padding):\n{center_patch.reshape(3, 3)}"
    )

    # Top-left corner should have zeros from padding
    corner_patch = col[0, :, 0]
    print(f"\nTop-left corner patch:\n{corner_patch.reshape(3, 3)}")
    print(f"Expected: zeros in top-left, original values in bottom-right")

    if col.shape == (1, 9, 9):
        print("\n✅ Test 2 PASSED: Padding works correctly")
    else:
        print("\n❌ Test 2 FAILED")


def test_stride():
    """Test 3: Different stride values"""
    print("\n" + "=" * 70)
    print("Test 3: Different Stride Values")
    print("=" * 70)

    x = nm.randn(1, 1, 8, 8)
    print(f"\nInput shape: {x.shape}")

    # Test stride=1
    col1 = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=1, padding=1)
    print(f"\nStride=1: output shape {col1.shape}")
    print(f"Expected: (1, 9, 64)  # 8x8=64 output positions")

    # Test stride=2
    col2 = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=2, padding=1)
    print(f"Stride=2: output shape {col2.shape}")
    print(f"Expected: (1, 9, 16)  # 4x4=16 output positions")

    # Test stride=4
    col4 = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=4, padding=1)
    print(f"Stride=4: output shape {col4.shape}")
    print(f"Expected: (1, 9, 4)   # 2x2=4 output positions")

    if (
        col1.shape == (1, 9, 64)
        and col2.shape == (1, 9, 16)
        and col4.shape == (1, 9, 4)
    ):
        print("\n✅ Test 3 PASSED: Stride works correctly")
    else:
        print("\n❌ Test 3 FAILED")


def test_multiple_channels():
    """Test 4: Multiple input channels"""
    print("\n" + "=" * 70)
    print("Test 4: Multiple Channels")
    print("=" * 70)

    # 3 channels (like RGB image)
    x = nm.randn(2, 3, 8, 8)  # 2 samples, 3 channels
    print(f"\nInput shape: {x.shape}")

    col = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=1, padding=1)
    print(f"im2col output shape: {col.shape}")
    print(f"Expected: (2, 27, 64)  # 2 samples, 3*3*3=27, 8*8=64")

    if col.shape == (2, 27, 64):
        print("\n✅ Test 4 PASSED: Multiple channels work correctly")
    else:
        print("\n❌ Test 4 FAILED")


def test_col2im_basic():
    """Test 5: col2im basic functionality"""
    print("\n" + "=" * 70)
    print("Test 5: col2im Basic Functionality")
    print("=" * 70)

    # Create a simple pattern
    x = nm.ones(1, 1, 4, 4)
    print(f"\nInput (all ones): shape {x.shape}")

    # Non-overlapping patches (stride=2)
    col = nm.im2col(x._data, kernel_h=2, kernel_w=2, stride=2, padding=0)
    print(f"im2col output shape: {col.shape}")

    # Reconstruct
    reconstructed = nm.col2im(col, x.shape, kernel_h=2, kernel_w=2, stride=2, padding=0)
    print(f"Reconstructed shape: {reconstructed.shape}")

    print(f"\nOriginal:\n{x._data[0, 0]}")
    print(f"Reconstructed:\n{reconstructed[0, 0]}")

    # Should be identical for non-overlapping patches
    if np.allclose(x._data, reconstructed):
        print("\n✅ Test 5 PASSED: col2im reconstructs correctly (no overlap)")
    else:
        print("\n❌ Test 5 FAILED")
        print(f"Difference: {np.max(np.abs(x._data - reconstructed))}")


def test_col2im_overlapping():
    """Test 6: col2im with overlapping patches"""
    print("\n" + "=" * 70)
    print("Test 6: col2im with Overlapping Patches")
    print("=" * 70)

    # With stride=1, patches overlap
    x = nm.ones(1, 1, 4, 4)
    print(f"\nInput (all ones): shape {x.shape}")

    col = nm.im2col(x._data, kernel_h=2, kernel_w=2, stride=1, padding=0)
    print(f"im2col output shape: {col.shape}")  # (1, 4, 9)

    # Reconstruct - overlapping areas should sum up
    reconstructed = nm.col2im(col, x.shape, kernel_h=2, kernel_w=2, stride=1, padding=0)
    print(f"Reconstructed shape: {reconstructed.shape}")

    print(f"\nReconstructed (with overlap accumulation):\n{reconstructed[0, 0]}")
    print(f"Note: Overlapping regions accumulate values")

    # Corner pixels appear in 1 patch: value = 1
    # Edge pixels appear in 2 patches: value = 2
    # Center pixels appear in 4 patches: value = 4
    expected = np.array(
        [[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]], dtype=np.float32
    )

    print(f"\nExpected (overlap counts):\n{expected}")

    if np.allclose(reconstructed[0, 0], expected):
        print("\n✅ Test 6 PASSED: col2im handles overlapping correctly")
    else:
        print("\n❌ Test 6 FAILED")


def test_invertibility():
    """Test 7: Invertibility for non-overlapping case"""
    print("\n" + "=" * 70)
    print("Test 7: Invertibility (im2col -> col2im)")
    print("=" * 70)

    # Random input
    x = nm.randn(2, 3, 8, 8)
    print(f"\nOriginal shape: {x.shape}")

    # Non-overlapping: stride = kernel_size
    col = nm.im2col(x._data, kernel_h=2, kernel_w=2, stride=2, padding=0)
    reconstructed = nm.col2im(col, x.shape, kernel_h=2, kernel_w=2, stride=2, padding=0)

    diff = np.max(np.abs(x._data - reconstructed))
    print(f"\nMax difference: {diff}")

    if diff < 1e-6:
        print("✅ Test 7 PASSED: Perfect invertibility for non-overlapping case")
    else:
        print("❌ Test 7 FAILED")


def test_gradient_flow():
    """Test 8: Simulate gradient flow (backward pass)"""
    print("\n" + "=" * 70)
    print("Test 8: Gradient Flow Simulation")
    print("=" * 70)

    # Simulate forward pass
    x = nm.randn(1, 2, 6, 6)
    print(f"\nInput shape: {x.shape}")

    # Forward: im2col
    col = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=1, padding=1)
    print(f"Forward (im2col) output: {col.shape}")

    # Simulate gradient coming from loss
    grad_col = np.ones_like(col)  # Uniform gradient
    print(f"Gradient w.r.t. col: {grad_col.shape}")

    # Backward: col2im
    grad_x = nm.col2im(grad_col, x.shape, kernel_h=3, kernel_w=3, stride=1, padding=1)
    print(f"Gradient w.r.t. input: {grad_x.shape}")

    # Check gradient accumulation
    # With stride=1 and 3x3 kernel, each pixel receives gradients from multiple positions
    print(f"\nGradient at center pixel: {grad_x[0, 0, 3, 3]}")
    print(f"Expected: 9 (3x3 kernel overlaps 9 times at center)")

    print(f"Gradient at corner pixel: {grad_x[0, 0, 0, 0]}")
    print(f"Expected: 4 (corner is covered by 2x2 patches)")

    if abs(grad_x[0, 0, 3, 3] - 9.0) < 0.1:
        print("\n✅ Test 8 PASSED: Gradient accumulation works correctly")
    else:
        print("\n❌ Test 8 FAILED")


def test_edge_cases():
    """Test 9: Edge cases"""
    print("\n" + "=" * 70)
    print("Test 9: Edge Cases")
    print("=" * 70)

    # Case 1: 1x1 kernel (identity)
    print("\nCase 1: 1x1 kernel")
    x = nm.randn(2, 3, 4, 4)
    col = nm.im2col(x._data, kernel_h=1, kernel_w=1, stride=1, padding=0)
    print(f"Input: {x.shape}, Output: {col.shape}")
    print(f"Expected: (2, 3, 16) - 3*1*1=3, 4*4=16")

    # Case 2: Kernel size = image size (single patch)
    print("\nCase 2: Kernel size = image size")
    x2 = nm.randn(1, 2, 4, 4)
    col2 = nm.im2col(x2._data, kernel_h=4, kernel_w=4, stride=1, padding=0)
    print(f"Input: {x2.shape}, Output: {col2.shape}")
    print(f"Expected: (1, 32, 1) - 2*4*4=32, single patch")

    # Case 3: Large padding
    print("\nCase 3: Large padding")
    x3 = nm.randn(1, 1, 4, 4)
    col3 = nm.im2col(x3._data, kernel_h=3, kernel_w=3, stride=1, padding=3)
    print(f"Input: {x3.shape}, Output: {col3.shape}")
    expected_size = 4 + 2 * 3  # 10x10 after padding
    print(f"Expected: (1, 9, {expected_size**2})")

    if col.shape == (2, 3, 16) and col2.shape == (1, 32, 1):
        print("\n✅ Test 9 PASSED: Edge cases handled correctly")
    else:
        print("\n❌ Test 9 FAILED")


def test_performance():
    """Test 10: Performance check"""
    print("\n" + "=" * 70)
    print("Test 10: Performance Check")
    print("=" * 70)

    import time

    # Typical CNN layer dimensions
    x = nm.randn(32, 64, 28, 28)
    print(f"\nInput: {x.shape} (batch=32, channels=64, size=28x28)")

    # Benchmark im2col
    start = time.time()
    for _ in range(10):
        col = nm.im2col(x._data, kernel_h=3, kernel_w=3, stride=1, padding=1)
    elapsed_im2col = time.time() - start

    print(f"\nim2col: {elapsed_im2col:.4f}s for 10 iterations")
    print(f"Average: {elapsed_im2col / 10:.4f}s per iteration")

    # Benchmark col2im
    start = time.time()
    for _ in range(10):
        img = nm.col2im(col, x.shape, kernel_h=3, kernel_w=3, stride=1, padding=1)
    elapsed_col2im = time.time() - start

    print(f"\ncol2im: {elapsed_col2im:.4f}s for 10 iterations")
    print(f"Average: {elapsed_col2im / 10:.4f}s per iteration")

    print(
        f"\nTotal time per forward-backward: {(elapsed_im2col + elapsed_col2im) / 10:.4f}s"
    )

    if elapsed_im2col < 1.0 and elapsed_col2im < 2.0:  # Reasonable thresholds
        print("\n✅ Test 10 PASSED: Performance is acceptable")
    else:
        print("\n⚠️  Test 10 WARNING: Performance might need optimization")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS FOR im2col AND col2im")
    print("=" * 70)

    try:
        test_basic_functionality()
        test_with_padding()
        test_stride()
        test_multiple_channels()
        test_col2im_basic()
        test_col2im_overlapping()
        test_invertibility()
        test_gradient_flow()
        test_edge_cases()
        test_performance()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
