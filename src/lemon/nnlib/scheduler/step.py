from lemon.nnlib.scheduler.scheduler import Scheduler


class StepScheduler(Scheduler):
    """
    Decays parameter by gamma every step_size epochs

    This scheduler multiplies the parameter value by gamma every step_size epochs.
    Works with any optimizer parameter: lr, momentum, weight_decay, etc.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    step_size : int
        Period of parameter decay (in epochs)
    gamma : float, optional
        Multiplicative factor of decay (default: 0.1)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> # Learning rate decay: 0.01 -> 0.001 -> 0.0001
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = StepScheduler(optimizer, param_name='lr', step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    >>> # Epoch 0-29: lr=0.01
    >>> # Epoch 30-59: lr=0.001
    >>> # Epoch 60-89: lr=0.0001
    >>> # Epoch 90-99: lr=0.00001
    >>>
    >>> # Momentum decay
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> scheduler = StepScheduler(optimizer, param_name='momentum', step_size=50, gamma=0.95)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, param_name, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, param_name, last_epoch)

    def get_value(self):
        """
        Compute parameter value for current epoch

        Returns
        -------
        float
            Parameter value for current epoch
        """
        num_steps = self.last_epoch // self.step_size
        return self.base_value * (self.gamma**num_steps)

    def __repr__(self):
        return (
            f"StepScheduler(param='{self.param_name}', "
            f"step_size={self.step_size}, gamma={self.gamma})"
        )


class StepLR(StepScheduler):
    """
    Decays learning rate by gamma every step_size epochs

    PyTorch-compatible convenience class that automatically schedules
    the 'lr' parameter. This is a specialized version of StepScheduler
    with param_name='lr' fixed.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled
    step_size : int
        Period of learning rate decay (in epochs)
    gamma : float, optional
        Multiplicative factor of decay (default: 0.1)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> # Learning rate decay: 0.01 -> 0.001 -> 0.0001
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    >>> # Epoch 0-29: lr=0.01
    >>> # Epoch 30-59: lr=0.001
    >>> # Epoch 60-89: lr=0.0001
    >>> # Epoch 90-99: lr=0.00001
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super().__init__(
            optimizer,
            param_name="lr",
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"
