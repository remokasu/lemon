from lemon.nnlib.optimizer.optimizer import Optimizer
import lemon.numlib as nm


class SGD(Optimizer):  # TODO : Create test cases
    """
    Stochastic Gradient Descent optimizer

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate (default: 0.01)
    momentum : float, optional
        Momentum factor (default: 0)
    weight_decay : float, optional
        Weight decay (L2 penalty) (default: 0)

    Examples
    --------
    >>> optimizer = SGD(model.parameters(), lr=0.01)
    >>>
    >>> for epoch in range(100):
    ...     y_pred = model(x)
    ...     loss = mean_squared_error(y_pred, y_true)
    ...
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    """

    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        if self.momentum > 0:
            self.velocity = [None for _ in self.params]
        else:
            self.velocity = None

    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad._data

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data._data

            # Momentum
            if self.momentum > 0:
                if self.velocity[i] is None:
                    self.velocity[i] = nm.zeros_like(param.data)

                # v = momentum * v + grad
                self.velocity[i]._data = self.momentum * self.velocity[i]._data + grad
                grad = self.velocity[i]._data

            # Update parameters
            param.data._data -= self.lr * grad

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"
