import threading


_train_state = threading.local()
_train_state.enabled = True


def _set_train_enabled(mode: bool):
    """Set global training mode state (internal)"""
    _train_state.enabled = mode


def _get_train_enabled() -> bool:
    """Get current training mode state (internal)"""
    return getattr(_train_state, "enabled", True)


class TrainControl:
    """Single object that acts as both function and context manager"""

    def __init__(self, enable: bool, name: str):
        self.enable = enable
        self.name = name

    def __call__(self):
        """Function call: set training state globally"""
        _set_train_enabled(self.enable)

    def __enter__(self):
        """Context manager: temporarily set training state"""
        self.prev = _get_train_enabled()
        _set_train_enabled(self.enable)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore previous state even if exception occurred"""
        _set_train_enabled(self.prev)

    def __repr__(self):
        return f"train.{self.name}"


class TrainNamespace:
    """Training mode control namespace"""

    def __init__(self):
        self._on = TrainControl(True, "on")
        self._off = TrainControl(False, "off")

    @property
    def on(self):
        return self._on

    @property
    def off(self):
        return self._off

    def is_on(self) -> bool:
        """Check if training mode is enabled"""
        return _get_train_enabled()

    def is_off(self) -> bool:
        """Check if training mode is disabled"""
        return not _get_train_enabled()

    def is_enabled(self) -> bool:
        return _get_train_enabled()

    def enable(self) -> None:
        _set_train_enabled(True)

    def disable(self) -> None:
        _set_train_enabled(False)

    def set_enabled(self, mode: bool) -> None:
        _set_train_enabled(mode)


train = TrainNamespace()
