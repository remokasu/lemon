from lemon.nnlib.scheduler.scheduler import Scheduler


class ExponentialScheduler(Scheduler):
    """
    Decays parameter by gamma every epoch

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    gamma : float
        Multiplicative factor of decay per epoch
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ExponentialScheduler(optimizer, param_name='lr', gamma=0.9)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, param_name, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, param_name, last_epoch)

    def get_value(self):
        return self.base_value * (self.gamma**self.last_epoch)

    def __repr__(self):
        return f"ExponentialScheduler(param='{self.param_name}', gamma={self.gamma})"


class ExponentialLR(ExponentialScheduler):
    """
    Decays learning rate by gamma every epoch

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled
    gamma : float
        Multiplicative factor of decay per epoch
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = ExponentialLR(optimizer, gamma=0.9)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        super().__init__(optimizer, param_name="lr", gamma=gamma, last_epoch=last_epoch)

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"
