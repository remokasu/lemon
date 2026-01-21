"""
Test cases for Module state_dict, load_state_dict, named_parameters, named_modules
"""

import pytest
import pickle
import tempfile
import os

import lemon.numlib as nm
import lemon.nnlib as nl


class SimpleModel(nl.Module):
    """Simple test model"""

    def __init__(self):
        super().__init__()
        self.fc1 = nl.Linear(10, 20)
        self.fc2 = nl.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = nl.relu(x)
        x = self.fc2(x)
        return x


class NestedModel(nl.Module):
    """Nested model for testing hierarchical structure"""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleModel()
        self.layer2 = nl.Linear(5, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test_named_parameters():
    """Test named_parameters() method"""
    model = SimpleModel()

    # Collect all parameter names
    param_names = []
    param_shapes = []
    for name, param in model.named_parameters():
        param_names.append(name)
        param_shapes.append(param.shape)

    # Check expected names
    assert "fc1.weight" in param_names
    assert "fc1.bias" in param_names
    assert "fc2.weight" in param_names
    assert "fc2.bias" in param_names

    # Check count
    assert len(param_names) == 4

    # Check shapes
    for name, shape in zip(param_names, param_shapes):
        if name == "fc1.weight":
            assert shape == (10, 20)
        elif name == "fc1.bias":
            assert shape == (20,)
        elif name == "fc2.weight":
            assert shape == (20, 5)
        elif name == "fc2.bias":
            assert shape == (5,)


def test_named_parameters_nested():
    """Test named_parameters() with nested modules"""
    model = NestedModel()

    param_names = [name for name, _ in model.named_parameters()]

    # Check nested structure
    assert "layer1.fc1.weight" in param_names
    assert "layer1.fc1.bias" in param_names
    assert "layer1.fc2.weight" in param_names
    assert "layer1.fc2.bias" in param_names
    assert "layer2.weight" in param_names
    assert "layer2.bias" in param_names

    # Check count
    assert len(param_names) == 6


def test_named_modules():
    """Test named_modules() method"""
    model = SimpleModel()

    module_names = []
    module_types = []
    for name, module in model.named_modules():
        module_names.append(name)
        module_types.append(module.__class__.__name__)

    # Check root module
    assert "" in module_names
    assert "SimpleModel" in module_types

    # Check submodules
    assert "fc1" in module_names
    assert "fc2" in module_names


def test_named_modules_nested():
    """Test named_modules() with nested structure"""
    model = NestedModel()

    module_info = [(name, module.__class__.__name__) for name, module in model.named_modules()]

    # Check all modules
    module_names = [name for name, _ in module_info]
    assert "" in module_names  # root
    assert "layer1" in module_names
    assert "layer1.fc1" in module_names
    assert "layer1.fc2" in module_names
    assert "layer2" in module_names


def test_state_dict():
    """Test state_dict() method"""
    model = SimpleModel()

    state = model.state_dict()

    # Check keys
    assert "fc1.weight" in state
    assert "fc1.bias" in state
    assert "fc2.weight" in state
    assert "fc2.bias" in state

    # Check shapes (should be NumPy arrays)
    import numpy as np

    assert isinstance(state["fc1.weight"], np.ndarray)
    assert state["fc1.weight"].shape == (10, 20)
    assert state["fc1.bias"].shape == (20,)
    assert state["fc2.weight"].shape == (20, 5)
    assert state["fc2.bias"].shape == (5,)


def test_state_dict_nested():
    """Test state_dict() with nested modules"""
    model = NestedModel()

    state = model.state_dict()

    # Check nested keys
    assert "layer1.fc1.weight" in state
    assert "layer1.fc1.bias" in state
    assert "layer1.fc2.weight" in state
    assert "layer1.fc2.bias" in state
    assert "layer2.weight" in state
    assert "layer2.bias" in state

    assert len(state) == 6


def test_load_state_dict_strict():
    """Test load_state_dict() with strict=True"""
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Get state from model1
    state = model1.state_dict()

    # Load into model2
    model2.load_state_dict(state, strict=True)

    # Check that parameters match
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        import numpy as np
        assert name1 == name2
        assert np.allclose(nm.as_numpy(param1.data), nm.as_numpy(param2.data))


def test_load_state_dict_non_strict():
    """Test load_state_dict() with strict=False"""
    model = SimpleModel()

    # Create partial state_dict (missing some keys)
    partial_state = {"fc1.weight": nm.as_numpy(nm.randn(10, 20)), "fc1.bias": nm.as_numpy(nm.zeros(20))}

    # Should not raise error with strict=False
    model.load_state_dict(partial_state, strict=False)


def test_load_state_dict_strict_missing_keys():
    """Test load_state_dict() raises error for missing keys in strict mode"""
    model = SimpleModel()

    # Create partial state_dict
    partial_state = {"fc1.weight": nm.as_numpy(nm.randn(10, 20))}

    # Should raise error with strict=True
    with pytest.raises(KeyError, match="Missing keys"):
        model.load_state_dict(partial_state, strict=True)


def test_load_state_dict_strict_unexpected_keys():
    """Test load_state_dict() raises error for unexpected keys in strict mode"""
    model = SimpleModel()

    # Create state_dict with extra keys
    state = model.state_dict()
    state["extra_param"] = nm.as_numpy(nm.randn(5, 5))

    # Should raise error with strict=True
    with pytest.raises(KeyError, match="Unexpected keys"):
        model.load_state_dict(state, strict=True)


def test_save_load_with_pickle():
    """Test saving and loading state_dict with pickle"""
    model1 = SimpleModel()

    # Forward pass to initialize
    x = nm.randn(2, 10)
    y1 = model1(x)

    # Save state_dict
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_file = f.name
        pickle.dump(model1.state_dict(), f)

    try:
        # Create new model and load state
        model2 = SimpleModel()

        with open(temp_file, "rb") as f:
            state = pickle.load(f)

        model2.load_state_dict(state)

        # Compare outputs
        y2 = model2(x)
        import numpy as np
        assert np.allclose(nm.as_numpy(y1), nm.as_numpy(y2))

    finally:
        # Cleanup
        os.unlink(temp_file)


def test_transfer_learning_scenario():
    """Test partial loading for transfer learning"""
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Get state from model1
    pretrained_state = model1.state_dict()

    # Simulate transfer learning: load only fc1 layer
    filtered_state = {k: v for k, v in pretrained_state.items() if "fc1" in k}

    # Load with strict=False
    model2.load_state_dict(filtered_state, strict=False)

    # Check fc1 parameters match
    import numpy as np
    for name, param in model2.named_parameters():
        if "fc1" in name:
            assert np.allclose(nm.as_numpy(param.data), pretrained_state[name])


def test_parameter_modification():
    """Test that state_dict reflects parameter changes"""
    model = SimpleModel()

    # Get initial state
    state1 = model.state_dict()

    # Modify a parameter
    for name, param in model.named_parameters():
        if name == "fc1.weight":
            param.data = nm.zeros_like(param.data)

    # Get new state
    state2 = model.state_dict()

    # Check that states differ
    import numpy as np

    assert not np.allclose(state1["fc1.weight"], state2["fc1.weight"])
    assert np.allclose(state2["fc1.weight"], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
