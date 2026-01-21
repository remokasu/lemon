class Optimizer:
    """
    Base class for all optimizers

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    """

    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        """Zero out the gradients of all parameters"""
        for param in self.params:
            param.zero_grad()

    def step(self):
        """Perform a single optimization step (parameter update)"""
        raise NotImplementedError("Subclasses must implement step()")

    def get_param(self, name, default=None):
        """
        Get optimizer hyperparameter value

        Parameters
        ----------
        name : str
            Name of the hyperparameter (e.g., 'lr', 'momentum', 'weight_decay')
        default : any, optional
            Default value to return if parameter doesn't exist

        Returns
        -------
        any
            The parameter value, or default if not found

        Examples
        --------
        >>> optimizer = Adam(model.parameters(), lr=0.01)
        >>> optimizer.get_param('lr')
        0.01
        >>> optimizer.get_param('momentum', default=0)
        0
        """
        return getattr(self, name, default)

    def set_param(self, name, value):
        """
        Set optimizer hyperparameter value

        Parameters
        ----------
        name : str
            Name of the hyperparameter (e.g., 'lr', 'momentum', 'weight_decay')
        value : any
            New value for the parameter

        Raises
        ------
        ValueError
            If the optimizer doesn't have the specified parameter

        Examples
        --------
        >>> optimizer = Adam(model.parameters(), lr=0.01)
        >>> optimizer.set_param('lr', 0.001)
        >>> optimizer.get_param('lr')
        0.001
        """
        if not hasattr(self, name):
            raise ValueError(
                f"{self.__class__.__name__} does not have parameter '{name}'"
            )
        setattr(self, name, value)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
