from lemon.nnlib.optimizer.optimizer import Optimizer
import lemon.numlib as nm


class Adagrad(Optimizer):  # TODO : Create test cases
    """
    Adagrad optimizer (Adaptive Gradient Algorithm)

    Adapts learning rate for each parameter based on historical gradient information.
    Good for sparse data. Historical importance for understanding adaptive methods.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate (default: 0.01)
    lr_decay : float, optional
        Learning rate decay (default: 0)
    weight_decay : float, optional
        Weight decay (L2 penalty) (default: 0)
    eps : float, optional
        Term added for numerical stability (default: 1e-10)
    initial_accumulator_value : float, optional
        Initial value for gradient accumulator (default: 0)

    Examples
    --------
    >>> optimizer = Adagrad(model.parameters(), lr=0.01)
    >>>
    >>> for epoch in range(100):
    ...     y_pred = model(x)
    ...     loss = mean_squared_error(y_pred, y_true)
    ...
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()

    Notes
    -----
    Update rule:
        G = G + grad^2  (accumulate squared gradients)
        theta = theta - (lr / (sqrt(G) + eps)) * grad

    With lr_decay:
        effective_lr = lr / (1 + t * lr_decay)

    Adagrad automatically decreases learning rate for frequently updated parameters
    and increases it for infrequent ones. However, the learning rate can become
    infinitesimally small over time due to monotonic accumulation.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        lr_decay=0,
        weight_decay=0,
        eps=1e-10,
        initial_accumulator_value=0,
    ):
        super().__init__(params)
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        self.initial_accumulator_value = initial_accumulator_value

        # Initialize gradient accumulator (sum of squared gradients)
        if initial_accumulator_value == 0:
            self.state_sum = [nm.zeros_like(p.data) for p in self.params]
        else:
            xp = nm.get_array_module(self.params[0].data._data)
            self.state_sum = [
                nm.full_like(p.data, initial_accumulator_value) for p in self.params
            ]

        self.t = 0  # Time step for lr_decay

    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        xp = nm.get_array_module(self.params[0].data._data)

        # Compute effective learning rate with decay
        if self.lr_decay != 0:
            effective_lr = self.lr / (1 + (self.t - 1) * self.lr_decay)
        else:
            effective_lr = self.lr

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad._data

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data._data

            # Accumulate squared gradients
            self.state_sum[i]._data += grad**2

            # Compute adaptive learning rate
            std = xp.sqrt(self.state_sum[i]._data + self.eps)

            # Update parameters
            param.data._data -= effective_lr * grad / std

    def __repr__(self):
        return f"Adagrad(lr={self.lr}, lr_decay={self.lr_decay}, weight_decay={self.weight_decay}, eps={self.eps})"
