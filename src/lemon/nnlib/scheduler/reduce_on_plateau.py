class ReduceOnPlateauScheduler:
    """
    Reduce parameter when a metric has stopped improving

    Models often benefit from reducing hyperparameters (especially learning rate)
    when the validation metric stops improving. This scheduler reads a metric
    quantity and if no improvement is seen for a 'patience' number of epochs,
    the parameter is reduced.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    better : str, optional
        How to determine if metric improved:
        - '<' or 'lower' or 'minimize': Lower values are better (e.g., loss)
        - '>' or 'higher' or 'maximize': Higher values are better (e.g., accuracy)
        (default: '<')
    factor : float, optional
        Factor by which the parameter will be reduced.
        new_value = value * factor (default: 0.1)
    patience : int, optional
        Number of epochs with no improvement after which parameter will be
        reduced (default: 10)
    threshold : float, optional
        Threshold for measuring the new optimum, to only focus on
        significant changes (default: 1e-4)
    threshold_mode : str, optional
        One of 'rel' or 'abs'. In 'rel' mode, threshold is relative change;
        in 'abs' mode, threshold is absolute change (default: 'rel')
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after
        parameter has been reduced (default: 0)
    min_value : float, optional
        A lower bound on the parameter value (default: 0)
    verbose : bool, optional
        If True, prints a message to stdout for each update (default: False)

    Examples
    --------
    >>> # Learning rate reduction based on validation loss (lower is better)
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ReduceOnPlateauScheduler(
    ...     optimizer, param_name='lr', better='<', patience=5, factor=0.1
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     scheduler.step(val_loss)
    >>> # Learning rate will be reduced if val_loss doesn't improve for 5 epochs
    >>>
    >>> # Momentum reduction based on validation accuracy (higher is better)
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> scheduler = ReduceOnPlateauScheduler(
    ...     optimizer, param_name='momentum', better='>', patience=10, factor=0.9
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_acc = validate(...)
    ...     scheduler.step(val_acc)
    >>>
    >>> # Alternative syntax (more descriptive)
    >>> scheduler = ReduceOnPlateauScheduler(
    ...     optimizer, param_name='lr', better='lower', patience=5, factor=0.1
    ... )

    Notes
    -----
    Unlike other schedulers, this one requires a metric to be passed
    to the step() method. This is typically the validation loss or
    validation accuracy.
    """

    def __init__(
        self,
        optimizer,
        param_name,
        better="<",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_value=0,
        verbose=False,
    ):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0")
        self.factor = factor

        self.optimizer = optimizer
        self.param_name = param_name
        self.min_value = min_value
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.last_epoch = 0

        # Parse 'better' parameter
        self.better = better
        self._init_is_better(better, threshold, threshold_mode)
        self._reset()

        # Validate parameter exists
        base_value = optimizer.get_param(param_name)
        if base_value is None:
            raise ValueError(
                f"{optimizer.__class__.__name__} does not have parameter '{param_name}'"
            )

    def _reset(self):
        """Reset the scheduler to initial state"""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, better, threshold, threshold_mode):
        """Initialize comparison function"""
        # Map 'better' to internal mode
        better_map = {
            "<": "min",
            "lower": "min",
            "minimize": "min",
            ">": "max",
            "higher": "max",
            "maximize": "max",
        }

        if better not in better_map:
            valid_options = list(better_map.keys())
            raise ValueError(
                f"'better' must be one of {valid_options}. Got: '{better}'\n"
                f"Use '<' or 'lower' for loss-like metrics (lower is better)\n"
                f"Use '>' or 'higher' for accuracy-like metrics (higher is better)"
            )

        self.mode = better_map[better]

        if threshold_mode not in {"rel", "abs"}:
            raise ValueError('threshold_mode must be "rel" or "abs"')

        if self.mode == "min":
            self.mode_worse = float("inf")
        else:
            self.mode_worse = -float("inf")

        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def _is_better(self, current, best):
        """Check if current metric is better than best"""
        if self.mode == "min" and self.threshold_mode == "rel":
            return current < best - (best * self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return current < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return current > best + (best * self.threshold)
        else:  # mode == 'max' and threshold_mode == 'abs'
            return current > best + self.threshold

    def step(self, metric):
        """
        Perform a single schedule step

        Parameters
        ----------
        metric : float
            The metric to monitor (e.g., validation loss or accuracy)
        """
        current = float(metric)
        self.last_epoch += 1

        # Check if this is an improvement
        # Special case: if best is still at initial value (mode_worse), any real value is better
        if self.best == self.mode_worse or self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_param()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_param(self):
        """Reduce parameter value"""
        old_value = self.optimizer.get_param(self.param_name)
        new_value = max(old_value * self.factor, self.min_value)
        self.optimizer.set_param(self.param_name, new_value)

        if self.verbose:
            print(f"Reducing {self.param_name} from {old_value:.4e} to {new_value:.4e}")

    @property
    def in_cooldown(self):
        """Check if scheduler is in cooldown period"""
        return self.cooldown_counter > 0

    def state_dict(self):
        """Returns the state of the scheduler as a dict"""
        return {
            "best": self.best,
            "cooldown_counter": self.cooldown_counter,
            "num_bad_epochs": self.num_bad_epochs,
            "last_epoch": self.last_epoch,
            "param_name": self.param_name,
            "better": self.better,
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state"""
        self.best = state_dict["best"]
        self.cooldown_counter = state_dict["cooldown_counter"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
        self.last_epoch = state_dict["last_epoch"]
        self.param_name = state_dict["param_name"]
        if "better" in state_dict:
            self.better = state_dict["better"]

    def get_last_value(self):
        """Return last computed parameter value"""
        return self.optimizer.get_param(self.param_name)

    def __repr__(self):
        return (
            f"ReduceOnPlateauScheduler(param='{self.param_name}', better='{self.better}', "
            f"factor={self.factor}, patience={self.patience}, min_value={self.min_value})"
        )


class ReduceLROnPlateau(ReduceOnPlateauScheduler):
    """
    Reduce learning rate when a metric has stopped improving

    PyTorch-compatible convenience class that automatically schedules
    the 'lr' parameter. This is a specialized version of ReduceOnPlateauScheduler
    with param_name='lr' fixed and PyTorch-style 'mode' parameter.

    Models often benefit from reducing the learning rate when the validation
    metric stops improving. This scheduler reads a metric quantity and if no
    improvement is seen for a 'patience' number of epochs, the learning rate
    is reduced.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled
    mode : str, optional
        One of 'min' or 'max'. In 'min' mode, learning rate will be reduced
        when the quantity monitored has stopped decreasing; in 'max' mode it
        will be reduced when the quantity monitored has stopped increasing
        (default: 'min')
    factor : float, optional
        Factor by which the learning rate will be reduced.
        new_lr = lr * factor (default: 0.1)
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be
        reduced (default: 10)
    threshold : float, optional
        Threshold for measuring the new optimum, to only focus on
        significant changes (default: 1e-4)
    threshold_mode : str, optional
        One of 'rel' or 'abs'. In 'rel' mode, threshold is relative change;
        in 'abs' mode, threshold is absolute change (default: 'rel')
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after
        learning rate has been reduced (default: 0)
    min_lr : float, optional
        A lower bound on the learning rate (default: 0)
    verbose : bool, optional
        If True, prints a message to stdout for each update (default: False)

    Examples
    --------
    >>> # Reduce LR when validation loss plateaus
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     scheduler.step(val_loss)
    >>> # Learning rate will be reduced if val_loss doesn't improve for 5 epochs
    >>>
    >>> # Reduce LR when validation accuracy plateaus
    >>> scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1)
    >>> for epoch in range(100):
    ...     # Training step
    ...     val_acc = validate(...)
    ...     scheduler.step(val_acc)

    Notes
    -----
    Unlike other schedulers, this one requires a metric to be passed
    to the step() method. This is typically the validation loss or
    validation accuracy.
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        verbose=False,
    ):
        # Map PyTorch 'mode' to internal 'better' parameter
        mode_map = {
            "min": "<",
            "max": ">",
        }

        if mode not in mode_map:
            raise ValueError(f"mode must be 'min' or 'max'. Got: '{mode}'")

        better = mode_map[mode]

        super().__init__(
            optimizer,
            param_name="lr",
            better=better,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_value=min_lr,
            verbose=verbose,
        )
        self.mode = mode

    def __repr__(self):
        return (
            f"ReduceLROnPlateau(mode='{self.mode}', "
            f"factor={self.factor}, patience={self.patience}, min_lr={self.min_value})"
        )


class ReduceOnLossPlateau(ReduceOnPlateauScheduler):
    """
    Reduce parameter when loss plateaus (lower is better)

    Convenient version of ReduceOnPlateauScheduler with better='<' fixed.
    Use this when monitoring loss-like metrics (lower values are better).

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    patience : int, optional
        Number of epochs with no improvement after which parameter will be
        reduced (default: 10)
    factor : float, optional
        Multiplicative factor of parameter decay (default: 0.1)
    threshold : float, optional
        Threshold for measuring improvement (default: 1e-4)
    threshold_mode : str, optional
        'rel' or 'abs' (default: 'rel')
    cooldown : int, optional
        Cooldown period (default: 0)
    min_value : float, optional
        Lower bound on parameter value (default: 0)
    verbose : bool, optional
        Print message on update (default: False)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ReduceOnLossPlateau(optimizer, param_name='lr', patience=5, factor=0.5)
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     scheduler.step(val_loss)  # Pass validation loss
    """

    def __init__(
        self,
        optimizer,
        param_name,
        patience=10,
        factor=0.1,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_value=0,
        verbose=False,
    ):
        super().__init__(
            optimizer,
            param_name,
            better="<",  # Fixed: lower is better
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_value=min_value,
            verbose=verbose,
        )


class ReduceOnMetricPlateau(ReduceOnPlateauScheduler):
    """
    Reduce parameter when metric plateaus (higher is better)

    Convenient version of ReduceOnPlateauScheduler with better='>' fixed.
    Use this when monitoring accuracy-like metrics (higher values are better).

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    patience : int, optional
        Number of epochs with no improvement after which parameter will be
        reduced (default: 10)
    factor : float, optional
        Multiplicative factor of parameter decay (default: 0.1)
    threshold : float, optional
        Threshold for measuring improvement (default: 1e-4)
    threshold_mode : str, optional
        'rel' or 'abs' (default: 'rel')
    cooldown : int, optional
        Cooldown period (default: 0)
    min_value : float, optional
        Lower bound on parameter value (default: 0)
    verbose : bool, optional
        Print message on update (default: False)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ReduceOnMetricPlateau(optimizer, param_name='lr', patience=5, factor=0.5)
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_acc = validate(...)
    ...     scheduler.step(val_acc)  # Pass validation accuracy
    """

    def __init__(
        self,
        optimizer,
        param_name,
        patience=10,
        factor=0.1,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_value=0,
        verbose=False,
    ):
        super().__init__(
            optimizer,
            param_name,
            better=">",  # Fixed: higher is better
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_value=min_value,
            verbose=verbose,
        )
