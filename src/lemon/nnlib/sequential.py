from lemon.nnlib.module import Module


class Sequential(Module):
    """
    Sequential container

    Modules will be added in the order they are passed.

    Parameters
    ----------
    *modules : Module
        Modules to be added

    Examples
    --------
    >>> model = Sequential(
    ...     Linear(784, 256),
    ...     ReLU(),
    ...     Linear(256, 10)
    ... )
    >>> y = model(x)
    """

    def __init__(self, *modules):
        super().__init__()
        self._modules = list(modules)

    @property
    def modules(self):
        """
        Get list of modules

        Returns
        -------
        list
            List of modules in the sequential container

        Examples
        --------
        >>> model = Sequential(Linear(10, 5), ReLU())
        >>> for module in model.modules:
        ...     print(module)
        Linear(in_features=10, out_features=5)
        ReLU()
        """
        return self._modules

    def forward(self, x):
        for module in self._modules:
            x = module(x)
        return x

    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def __repr__(self):
        module_strs = [f"  ({i}): {module}" for i, module in enumerate(self._modules)]
        return "Sequential(\n" + "\n".join(module_strs) + "\n)"
