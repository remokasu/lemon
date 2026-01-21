class Scheduler:
    """
    Base class for hyperparameter schedulers

    Schedulers adjust optimizer hyperparameters (lr, momentum, weight_decay, etc.)
    during training to improve convergence and final performance.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule (e.g., 'lr', 'momentum', 'weight_decay')
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> # Learning rate scheduling
    >>> scheduler = StepScheduler(optimizer, param_name='lr', step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    ...     train(...)
    ...     validate(...)
    ...     scheduler.step()
    >>>
    >>> # Momentum scheduling
    >>> scheduler = StepScheduler(optimizer, param_name='momentum', step_size=50, gamma=0.9)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, param_name, last_epoch=-1):
        self.optimizer = optimizer
        self.param_name = param_name
        self.last_epoch = last_epoch

        # Validate and store base value
        self.base_value = optimizer.get_param(param_name)
        if self.base_value is None:
            raise ValueError(
                f"{optimizer.__class__.__name__} does not have parameter '{param_name}'"
            )

    def get_value(self):
        """
        Compute parameter value for current epoch

        Returns
        -------
        float
            The parameter value for current epoch

        Notes
        -----
        Subclasses must implement this method to define the scheduling policy.
        """
        raise NotImplementedError("Subclasses must implement get_value()")

    def step(self, metric=None):
        """
        Perform a single schedule step

        Should be called after each epoch.

        Parameters
        ----------
        metric : float, optional
            Metric for plateau-based schedulers (e.g., validation loss).
            Only used by ReduceOnPlateauScheduler.

        Examples
        --------
        >>> # Standard schedulers
        >>> scheduler.step()
        >>>
        >>> # ReduceOnPlateauScheduler
        >>> val_loss = validate(...)
        >>> scheduler.step(val_loss)
        """
        self.last_epoch += 1
        value = self.get_value()
        self.optimizer.set_param(self.param_name, value)

    def get_last_value(self):
        """
        Return last computed parameter value

        Returns
        -------
        float
            The current value of the scheduled parameter
        """
        return self.optimizer.get_param(self.param_name)

    def state_dict(self):
        """
        Returns the state of the scheduler as a dict

        Returns
        -------
        dict
            Dictionary containing scheduler state
        """
        return {
            "last_epoch": self.last_epoch,
            "base_value": self.base_value,
            "param_name": self.param_name,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the scheduler state

        Parameters
        ----------
        state_dict : dict
            Scheduler state dictionary
        """
        self.last_epoch = state_dict["last_epoch"]
        self.base_value = state_dict["base_value"]
        self.param_name = state_dict["param_name"]

    def __repr__(self):
        return f"{self.__class__.__name__}(param='{self.param_name}')"
