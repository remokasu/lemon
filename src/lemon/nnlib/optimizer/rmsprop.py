from lemon.nnlib.optimizer.optimizer import Optimizer
import lemon.numlib as nm


class RMSprop(Optimizer):  # TODO : Create test cases
    """
    RMSprop optimizer (Root Mean Square Propagation)

    Divides learning rate by exponentially decaying average of squared gradients.
    Popular for RNNs and online learning scenarios.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate (default: 0.01)
    alpha : float, optional
        Smoothing constant (default: 0.99)
    eps : float, optional
        Term added for numerical stability (default: 1e-8)
    weight_decay : float, optional
        Weight decay (L2 penalty) (default: 0)
    momentum : float, optional
        Momentum factor (default: 0)
    centered : bool, optional
        If True, compute centered RMSprop (default: False)

    Examples
    --------
    >>> optimizer = RMSprop(model.parameters(), lr=0.01)
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
        v = alpha * v + (1 - alpha) * grad^2
        theta = theta - lr * grad / (sqrt(v) + eps)

    With momentum:
        buf = momentum * buf + grad / (sqrt(v) + eps)
        theta = theta - lr * buf

    Centered variant additionally tracks mean of gradients.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

        # Initialize squared gradient moving average
        self.v = [nm.zeros_like(p.data) for p in self.params]

        # Initialize momentum buffer if needed
        if self.momentum > 0:
            self.buf = [nm.zeros_like(p.data) for p in self.params]
        else:
            self.buf = None

        # Initialize gradient mean if centered
        if self.centered:
            self.grad_avg = [nm.zeros_like(p.data) for p in self.params]
        else:
            self.grad_avg = None

    def step(self):
        """Perform a single optimization step"""
        xp = nm.get_array_module(self.params[0].data._data)

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad._data

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data._data

            # Update squared gradient moving average
            self.v[i]._data = self.alpha * self.v[i]._data + (1 - self.alpha) * (
                grad**2
            )

            # Compute denominator
            if self.centered:
                # Update gradient mean
                self.grad_avg[i]._data = (
                    self.alpha * self.grad_avg[i]._data + (1 - self.alpha) * grad
                )
                # Centered variance: E[g^2] - E[g]^2
                denom = xp.sqrt(
                    self.v[i]._data - self.grad_avg[i]._data ** 2 + self.eps
                )
            else:
                denom = xp.sqrt(self.v[i]._data + self.eps)

            # Compute update
            if self.momentum > 0:
                # With momentum
                self.buf[i]._data = self.momentum * self.buf[i]._data + grad / denom
                param.data._data -= self.lr * self.buf[i]._data
            else:
                # Without momentum
                param.data._data -= self.lr * grad / denom

    def __repr__(self):
        return f"RMSprop(lr={self.lr}, alpha={self.alpha}, eps={self.eps}, weight_decay={self.weight_decay}, momentum={self.momentum}, centered={self.centered})"
