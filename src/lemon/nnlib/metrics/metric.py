"""
Base class for all metrics
"""


class Metric:
    """
    Base class for all metrics

    Subclasses must implement __call__ and name methods.
    """

    def __call__(self, y_pred, y_true):
        """
        Calculate metric value

        Parameters
        ----------
        y_pred : Tensor
            Predicted values
        y_true : Tensor
            True labels

        Returns
        -------
        float
            Metric value
        """
        raise NotImplementedError

    def name(self):
        """
        Get metric name

        Returns
        -------
        str
            Metric name
        """
        raise NotImplementedError

    def format(self, value):
        """
        Format metric value for display

        This method controls how the metric value is displayed in training logs.
        Subclasses can override this to customize the display format.

        Parameters
        ----------
        value : float
            Metric value to format

        Returns
        -------
        str
            Formatted string for display

        Examples
        --------
        >>> metric = Accuracy()
        >>> metric.format(0.8523)
        'accuracy=85.23%'
        """
        # Default: 4 decimal places
        return f"{self.name()}={value:.4f}"
