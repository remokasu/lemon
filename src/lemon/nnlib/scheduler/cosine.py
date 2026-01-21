import math
from lemon.nnlib.scheduler.scheduler import Scheduler


class CosineAnnealingScheduler(Scheduler):
    """
    Set the parameter using a cosine annealing schedule

    The parameter is annealed from the initial value to eta_min using
    a cosine curve over T_max epochs.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose parameter will be scheduled
    param_name : str
        Name of parameter to schedule
    T_max : int
        Maximum number of iterations (epochs)
    eta_min : float, optional
        Minimum parameter value (default: 0)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> # Cosine learning rate annealing over 100 epochs
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = CosineAnnealingScheduler(optimizer, param_name='lr', T_max=100)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    >>> # Learning rate smoothly decreases from 0.01 to 0 following cosine curve
    >>>
    >>> # Cosine momentum annealing
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> scheduler = CosineAnnealingScheduler(
    ...     optimizer, param_name='momentum', T_max=100, eta_min=0.5
    ... )

    Notes
    -----
    This scheduler is particularly effective for training deep networks
    and is commonly used with ResNet and other modern architectures.
    The cosine curve provides a smooth decay that often works better
    than step-based schedules.
    """

    def __init__(self, optimizer, param_name, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, param_name, last_epoch)

    def get_value(self):
        """
        Compute parameter value for current epoch using cosine annealing

        Returns
        -------
        float
            Parameter value for current epoch

        Notes
        -----
        The formula is:
        value = eta_min + (base_value - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
        """
        if self.last_epoch == 0:
            return self.base_value
        elif self.last_epoch < self.T_max:
            return (
                self.eta_min
                + (self.base_value - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
            )
        else:
            return self.eta_min

    def __repr__(self):
        return (
            f"CosineAnnealingScheduler(param='{self.param_name}', "
            f"T_max={self.T_max}, eta_min={self.eta_min})"
        )


class CosineAnnealingLR(CosineAnnealingScheduler):
    """
    Set learning rate using a cosine annealing schedule

    PyTorch-compatible convenience class that automatically schedules
    the 'lr' parameter. This is a specialized version of CosineAnnealingScheduler
    with param_name='lr' fixed.

    The learning rate is annealed from the initial value to eta_min using
    a cosine curve over T_max epochs.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled
    T_max : int
        Maximum number of iterations (epochs)
    eta_min : float, optional
        Minimum learning rate (default: 0)
    last_epoch : int, optional
        The index of the last epoch (default: -1)

    Examples
    --------
    >>> # Cosine learning rate annealing over 100 epochs
    >>> optimizer = Adam(model.parameters(), lr=0.01)
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    >>> # Learning rate smoothly decreases from 0.01 to 0 following cosine curve

    Notes
    -----
    This scheduler is particularly effective for training deep networks
    and is commonly used with ResNet and other modern architectures.
    The cosine curve provides a smooth decay that often works better
    than step-based schedules.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(
            optimizer,
            param_name="lr",
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min})"
