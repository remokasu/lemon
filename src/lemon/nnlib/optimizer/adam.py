from lemon.nnlib.optimizer.optimizer import Optimizer
import lemon.numlib as nm


class Adam(Optimizer):  # TODO : Create test cases
    """
    Adam optimizer (Adaptive Moment Estimation)

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float
        Learning rate (default: 0.001)
    betas : tuple, optional
        Coefficients for computing running averages (default: (0.9, 0.999))
    eps : float, optional
        Term added for numerical stability (default: 1e-8)
    weight_decay : float, optional
        Weight decay (L2 penalty) (default: 0)

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>>
    >>> for epoch in range(100):
    ...     y_pred = model(x)
    ...     loss = mean_squared_error(y_pred, y_true)
    ...
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [nm.zeros_like(p.data) for p in self.params]  # First moment
        self.v = [nm.zeros_like(p.data) for p in self.params]  # Second moment
        self.t = 0  # Time step

    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        xp = nm.get_array_module(self.params[0].data._data)

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad._data

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data._data

            # Update biased first moment estimate
            self.m[i]._data = self.beta1 * self.m[i]._data + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i]._data = self.beta2 * self.v[i]._data + (1 - self.beta2) * (
                grad**2
            )

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i]._data / (1 - self.beta1**self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i]._data / (1 - self.beta2**self.t)

            # Update parameters
            param.data._data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps}, weight_decay={self.weight_decay})"
