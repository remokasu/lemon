"""
Test suite for nnlib
"""

import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import lemon.nnlib as nl
import lemon.numlib as nm


def test_module():
    """Test Module base class"""
    print("Testing Module class...")

    # Test 1: Basic Module subclass
    class SimpleModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.param1 = nl.Parameter(nm.tensor([1.0, 2.0]))
            self.param2 = nl.Parameter(nm.tensor([3.0, 4.0]))

        def forward(self, x):
            return x + self.param1.data + self.param2.data

    module = SimpleModule()
    assert len(list(module.parameters())) == 2, "Should have 2 parameters"
    print("  ✅ Basic Module subclass")

    # Test 2: forward() not implemented
    class IncompleteModule(nl.Module):
        pass

    incomplete = IncompleteModule()
    try:
        incomplete(nm.tensor([1.0]))
        assert False, "Should raise NotImplementedError"
    except NotImplementedError as e:
        assert "forward()" in str(e), "Error message should mention forward()"
    print("  ✅ NotImplementedError for missing forward()")

    # Test 3: __call__ invokes forward
    class CallableModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def forward(self, x):
            self.called = True
            return x * 2

    callable_mod = CallableModule()
    x = nm.tensor([1.0, 2.0])
    y = callable_mod(x)
    assert callable_mod.called == True, "forward() should be called via __call__"
    assert nm.get_array_module(y._data).allclose(y._data, [2.0, 4.0]), (
        "Output should be correct"
    )
    print("  ✅ __call__ invokes forward()")

    # Test 4: Nested modules
    class InnerModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.weight = nl.Parameter(nm.tensor([1.0]))

        def forward(self, x):
            return x * self.weight.data

    class OuterModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.inner = InnerModule()
            self.bias = nl.Parameter(nm.tensor([2.0]))

        def forward(self, x):
            return self.inner(x) + self.bias.data

    outer = OuterModule()
    params = list(outer.parameters())
    assert len(params) == 2, f"Should have 2 parameters (nested), got {len(params)}"
    print("  ✅ Nested modules")

    # Test 5: zero_grad
    nm.autograd.enable()
    module_grad = SimpleModule()
    x = nm.tensor([1.0, 2.0])
    y = module_grad(x)
    loss = nm.sum(y)
    loss.backward()

    # Check gradients exist
    for param in module_grad.parameters():
        assert param.grad is not None, "Gradient should exist after backward"

    # Zero grad
    module_grad.zero_grad()
    for param in module_grad.parameters():
        assert param.grad is None, "Gradient should be None after zero_grad"
    print("  ✅ zero_grad()")

    # Test 6: parameters() iterator
    class MultiParamModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nl.Parameter(nm.zeros(2))
            self.p2 = nl.Parameter(nm.zeros(3))
            self.p3 = nl.Parameter(nm.zeros(4))

        def forward(self, x):
            return x

    multi = MultiParamModule()
    param_count = sum(1 for _ in multi.parameters())
    assert param_count == 3, f"Should have 3 parameters, got {param_count}"
    print("  ✅ parameters() iterator")

    # Test 7: __repr__
    simple = SimpleModule()
    repr_str = repr(simple)
    assert "SimpleModule" in repr_str, "repr should contain class name"
    assert "param1" in repr_str, "repr should contain parameter names"
    print("  ✅ __repr__")

    # Test 8: Empty module
    empty = nl.Module()
    assert len(list(empty.parameters())) == 0, "Empty module should have no parameters"
    print("  ✅ Empty module")

    print("✅ All Module tests passed!\n")


def test_coverage_edge_cases():
    """Test edge cases for 100% coverage"""

    print("Testing coverage edge cases...")

    # Test 1: Module.__repr__ with submodules
    class SubModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.weight = nl.Parameter(nm.tensor([1.0]))

        def forward(self, x):
            return x

    class ParentModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.param = nl.Parameter(nm.tensor([2.0]))
            self.submodule = SubModule()  # ← サブモジュールを追加

        def forward(self, x):
            return self.submodule(x)

    parent = ParentModule()
    repr_str = repr(parent)

    # 277-280行をカバーするための確認
    assert "submodule" in repr_str, "Submodule name should appear in repr"
    assert "SubModule" in repr_str, "Submodule class name should appear in repr"
    assert "weight" in repr_str, "Submodule's parameter should appear in repr"

    print("  ✅ Module.__repr__ with submodules (lines 277-280 covered)")

    # Test 2: 複数のサブモジュール
    class MultiSubModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.sub1 = SubModule()
            self.sub2 = SubModule()
            self.param = nl.Parameter(nm.tensor([3.0]))

        def forward(self, x):
            return x

    multi = MultiSubModule()
    repr_str = repr(multi)
    assert "sub1" in repr_str and "sub2" in repr_str, "All submodules should appear"
    print("  ✅ Module.__repr__ with multiple submodules")

    # Test 3: ネストしたサブモジュール（改行のインデント処理を確認）
    class DeepModule(nl.Module):
        def __init__(self):
            super().__init__()
            self.level1 = ParentModule()  # ParentModule は SubModule を含む

        def forward(self, x):
            return x

    deep = DeepModule()
    repr_str = repr(deep)

    # 改行とインデントを確認
    lines = repr_str.split("\n")
    assert len(lines) > 3, "Deep nesting should create multiple lines"

    # インデントが正しく適用されているか確認
    indented_lines = [line for line in lines if line.startswith("    ")]
    assert len(indented_lines) > 0, "Nested modules should be indented"

    print("  ✅ Module.__repr__ with deep nesting (indent handling)")

    print("✅ All coverage edge cases passed!\n")
