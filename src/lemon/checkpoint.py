"""
checkpoint - Training checkpoint utilities

This library provides functionality to save and load training checkpoints,
including model parameters, optimizer state, scheduler state, and training metadata.

Key Features
------------
- Save/load complete training state in a single file
- Support for model, optimizer, and scheduler states
- Pickle-based format for Python object persistence
- Resume training from exact state
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import lemon.numlib as nm


def save_checkpoint(
    filepath: str,
    model,
    optimizer,
    epoch: int,
    loss: float,
    schedulers=None,
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """
    Save complete training checkpoint

    Parameters
    ----------
    filepath : str
        Path to save checkpoint file
    model : Module
        Model to save
    optimizer : Optimizer
        Optimizer to save
    epoch : int
        Current epoch number
    loss : float
        Current loss value
    schedulers : List[Scheduler], optional
        List of learning rate schedulers to save
    metadata : dict, optional
        Additional metadata to save
    verbose : bool, optional
        Print save information (default: True)

    Examples
    --------
    >>> save_checkpoint(
    ...     'checkpoint.pkl',
    ...     model=model,
    ...     optimizer=optimizer,
    ...     epoch=10,
    ...     loss=0.5,
    ...     schedulers=[scheduler1, scheduler2]
    ... )
    """
    if verbose:
        print(f"Saving checkpoint to {filepath}...")

    # Collect model parameters
    model_state = {}
    for i, param in enumerate(model.parameters()):
        model_state[f"param_{i}"] = nm.as_numpy(param.data)

    # Collect optimizer state dynamically
    optimizer_state = {
        "class": optimizer.__class__.__name__,
    }

    for key, value in optimizer.__dict__.items():
        if key == "params" or key.startswith("_"):
            continue

        try:
            # Try to pickle simple values directly
            pickle.dumps(value)
            optimizer_state[key] = value
        except:
            # Handle lists/arrays with tensor data
            if isinstance(value, list):
                converted = []
                for item in value:
                    if item is None:
                        converted.append(None)
                    elif hasattr(item, "_data"):
                        converted.append(nm.as_numpy(item))
                    else:
                        converted.append(item)
                optimizer_state[key] = converted
            elif hasattr(value, "_data"):
                optimizer_state[key] = nm.as_numpy(value)
            # Skip non-serializable attributes

    # Collect scheduler state if provided
    schedulers_state = None
    if schedulers is not None:
        schedulers_state = []
        for sched in schedulers:
            sched_state = {
                "class": sched.__class__.__name__,
            }
            # Add scheduler-specific state
            if hasattr(sched, "__dict__"):
                for key, value in sched.__dict__.items():
                    if not key.startswith("_") and key != "optimizer":
                        try:
                            # Only save serializable attributes
                            pickle.dumps(value)
                            sched_state[key] = value
                        except:
                            pass
            schedulers_state.append(sched_state)

    # Build checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "loss": float(loss),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "schedulers_state": schedulers_state,
        "metadata": metadata or {},
    }

    # Save to file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint, f)

    if verbose:
        print(f"  Epoch: {epoch}, Loss: {loss:.6f}")
        if schedulers_state:
            sched_names = [s["class"] for s in schedulers_state]
            print(f"  Schedulers: {', '.join(sched_names)}")
        print("  Checkpoint saved")


def load_checkpoint(
    filepath: str, model, optimizer, schedulers=None, verbose: bool = True
) -> Dict[str, Any]:
    """
    Load complete training checkpoint

    Parameters
    ----------
    filepath : str
        Path to checkpoint file
    model : Module
        Model to load parameters into
    optimizer : Optimizer
        Optimizer to load state into
    schedulers : List[Scheduler], optional
        List of schedulers to load state into
    verbose : bool, optional
        Print loading information (default: True)

    Returns
    -------
    dict
        Dictionary with 'epoch', 'loss', and 'metadata'

    Examples
    --------
    >>> info = load_checkpoint('checkpoint.pkl', model, optimizer, schedulers=[sched1, sched2])
    >>> print(f"Resuming from epoch {info['epoch']}")
    """
    if verbose:
        print(f"Loading checkpoint from {filepath}...")

    # Load checkpoint file
    with open(filepath, "rb") as f:
        checkpoint = pickle.load(f)

    # Restore model parameters
    model_params = list(model.parameters())
    for i, param in enumerate(model_params):
        if f"param_{i}" in checkpoint["model_state"]:
            param_data = checkpoint["model_state"][f"param_{i}"]
            xp = nm.get_array_module(param.data._data)
            param.data._data = xp.asarray(param_data)

    # Restore optimizer state dynamically
    opt_state = checkpoint["optimizer_state"]

    for key, value in opt_state.items():
        if key == "class" or not hasattr(optimizer, key):
            continue

        attr = getattr(optimizer, key)

        # Handle list attributes (e.g., velocity, m, v)
        if isinstance(value, list) and isinstance(attr, list):
            for i, item_data in enumerate(value):
                if item_data is None:
                    # Restore None as-is
                    if i < len(attr):
                        attr[i] = None
                    continue

                if i < len(attr):
                    # Convert numpy arrays using tensor() function from numlib
                    if isinstance(item_data, nm.np.ndarray):
                        attr[i] = nm.tensor(item_data)
                    elif not isinstance(item_data, (int, float, bool, type(None))):
                        # Other array-like objects - try to convert to tensor
                        try:
                            attr[i] = nm.tensor(item_data)
                        except:
                            # If conversion fails, just assign directly
                            attr[i] = item_data
                    else:
                        # Plain scalar value
                        attr[i] = item_data
            # List attribute already updated in-place, skip setattr
            continue
        # Handle tensor attributes
        elif hasattr(attr, "_data"):
            xp = nm.get_array_module(attr._data)
            attr._data = xp.asarray(value)
        # Handle simple attributes
        else:
            setattr(optimizer, key, value)

    # Restore scheduler state if provided
    # Support both old format (scheduler_state) and new format (schedulers_state)
    schedulers_state_key = "schedulers_state" if "schedulers_state" in checkpoint else "scheduler_state"
    schedulers_state_data = checkpoint.get(schedulers_state_key)

    if schedulers_state_data is not None and schedulers is not None:
        # Convert old single scheduler format to list
        if not isinstance(schedulers_state_data, list):
            schedulers_state_data = [schedulers_state_data]

        # Restore each scheduler
        for i, (target_sched, sched_state) in enumerate(zip(schedulers, schedulers_state_data)):
            if target_sched is None:
                continue

            # First restore scheduler's attributes
            for key, value in sched_state.items():
                if key != "class" and key != "optimizer" and hasattr(target_sched, key):
                    setattr(target_sched, key, value)

            # Recompute and apply the current value to optimizer based on restored state
            # This handles the case where scheduler has modified optimizer parameters
            if hasattr(target_sched, "param_name"):
                param_name = target_sched.param_name
                # Get the value that should be set based on current last_epoch
                current_value = target_sched.get_value()
                if hasattr(optimizer, param_name):
                    setattr(optimizer, param_name, current_value)

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    metadata = checkpoint.get("metadata", {})

    if verbose:
        print(f"  Epoch: {epoch}, Loss: {loss:.6f}")
        if schedulers_state_data:
            if isinstance(schedulers_state_data, list):
                sched_names = [s["class"] for s in schedulers_state_data]
                print(f"  Schedulers: {', '.join(sched_names)}")
            else:
                print(f"  Scheduler: {schedulers_state_data['class']}")
        print("  Checkpoint loaded")

    return {"epoch": epoch, "loss": loss, "metadata": metadata}


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]
