from lemon.nnlib.scheduler.scheduler import Scheduler


class LinearScheduler(Scheduler):
    """
    Linearly interpolates parameter from start_factor to end_factor over total_iters epochs

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    start_factor : float
        Multiplier at epoch 0 (default: 1.0/3.0)
    end_factor : float
        Multiplier at total_iters (default: 1.0)
    total_iters : int
        Number of epochs over which to interpolate (default: 5)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = LinearScheduler(
    ...     optimizer, param_name='lr', start_factor=0.1, end_factor=1.0, total_iters=10
    ... )
    >>> for epoch in range(10):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        param_name,
        start_factor=1.0 / 3.0,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError("start_factor must be in (0, 1]")
        if end_factor > 1.0 or end_factor <= 0:
            raise ValueError("end_factor must be in (0, 1]")
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, param_name, last_epoch)

    def get_value(self):
        if self.last_epoch >= self.total_iters:
            return self.base_value * self.end_factor
        t = self.last_epoch
        factor = (
            self.start_factor
            + (self.end_factor - self.start_factor) * t / self.total_iters
        )
        return self.base_value * factor

    def __repr__(self):
        return (
            f"LinearScheduler(param='{self.param_name}', "
            f"start_factor={self.start_factor}, end_factor={self.end_factor}, "
            f"total_iters={self.total_iters})"
        )


class LinearLR(LinearScheduler):
    """
    Linearly interpolates learning rate from start_factor to end_factor over total_iters epochs

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled
    start_factor : float
        Multiplier at epoch 0 (default: 1.0/3.0)
    end_factor : float
        Multiplier at total_iters (default: 1.0)
    total_iters : int
        Number of epochs over which to interpolate (default: 5)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    >>> for epoch in range(10):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        start_factor=1.0 / 3.0,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
    ):
        super().__init__(
            optimizer,
            param_name="lr",
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
        )

    def __repr__(self):
        return (
            f"LinearLR(start_factor={self.start_factor}, "
            f"end_factor={self.end_factor}, total_iters={self.total_iters})"
        )
