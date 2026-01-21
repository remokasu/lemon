"""
Test suite for nnlib
"""

import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lemon import nnlib as nl
from lemon import numlib as nm


def test_parameter():
    """Test Parameter class"""
    print("Testing Parameter class...")

    # Test 1: Create from list
    p1 = nl.Parameter([1.0, 2.0, 3.0])
    assert isinstance(p1.data, nm.NumType), "Parameter should wrap NumType"
    assert p1.data.requires_grad == True, (
        "Parameter should have requires_grad=True by default"
    )
    print("  ✅ Create Parameter from list")

    # Test 2: Create from NumType
    tensor = nm.tensor([4.0, 5.0, 6.0])
    p2 = nl.Parameter(tensor)
    assert p2.data.requires_grad == True, "Parameter should set requires_grad=True"
    print("  ✅ Create Parameter from NumType")

    # Test 3: requires_grad=False
    p3 = nl.Parameter([1.0, 2.0], requires_grad=False)
    assert p3.data.requires_grad == False, (
        "Parameter with requires_grad=False should work"
    )
    print("  ✅ Parameter with requires_grad=False")

    # Test 4: Shape property
    p4 = nl.Parameter(nm.randn(3, 4))
    assert p4.shape == (3, 4), f"Shape should be (3, 4), got {p4.shape}"
    print("  ✅ Shape property")

    # Test 5: Dtype property
    p5 = nl.Parameter(nm.randn(2, 2))
    assert p5.dtype is not None, "Dtype should be accessible"
    print("  ✅ Dtype property")

    # Test 6: Gradient
    nm.autograd.enable()
    p6 = nl.Parameter(nm.tensor([1.0, 2.0, 3.0]))
    y = nm.sum(p6.data**2)
    y.backward()
    assert p6.grad is not None, "Gradient should be computed"
    assert p6.grad.shape == p6.shape, "Gradient shape should match parameter shape"
    print("  ✅ Gradient computation")

    # Test 7: zero_grad
    p6.zero_grad()
    assert p6.grad is None, "Gradient should be None after zero_grad()"
    print("  ✅ zero_grad()")

    # Test 8: __repr__
    p7 = nl.Parameter([1.0, 2.0])
    repr_str = repr(p7)
    assert "Parameter" in repr_str, (
        f"__repr__ should contain 'Parameter', got {repr_str}"
    )
    print("  ✅ __repr__")

    # Test 9: Access NumType methods via __getattr__
    p8 = nl.Parameter(nm.randn(2, 3))
    assert hasattr(p8, "ndim"), "Should be able to access ndim via __getattr__"
    assert p8.ndim == 2, "ndim should be 2"
    print("  ✅ __getattr__ delegation")

    # Test 10: Parameter with different shapes
    shapes = [(5,), (3, 4), (2, 3, 4)]
    for shape in shapes:
        p = nl.Parameter(nm.randn(*shape))
        assert p.shape == shape, f"Shape should be {shape}, got {p.shape}"
    print("  ✅ Parameters with different shapes")

    # Test 11: Parameter from integer (should convert to float)
    p9 = nl.Parameter([1, 2, 3], requires_grad=True)
    assert p9.data.requires_grad == True, (
        "Integer data should be convertible with requires_grad=True"
    )
    print("  ✅ Parameter from integer data")

    # Test 12: Multiple backward passes
    nm.autograd.enable()
    p10 = nl.Parameter(nm.tensor([1.0, 2.0]))

    # First backward
    y1 = nm.sum(p10.data**2)
    y1.backward()
    grad1 = p10.grad._data.copy() if p10.grad is not None else None

    # Second backward (after zero_grad)
    p10.zero_grad()
    y2 = nm.sum(p10.data**3)
    y2.backward()
    grad2 = p10.grad._data.copy() if p10.grad is not None else None

    assert grad1 is not None and grad2 is not None, "Both gradients should be computed"
    assert not nm.get_array_module(grad1).array_equal(grad1, grad2), (
        "Gradients should be different"
    )
    print("  ✅ Multiple backward passes")

    print("✅ All Parameter tests passed!\n")
