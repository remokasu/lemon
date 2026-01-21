"""
Sampler base class

Base class for custom samplers. Currently a placeholder for future extensions.
"""


class Sampler:
    """
    Base class for all samplers

    Samplers are used to control the order in which DataLoader iterates over
    dataset indices. This is a placeholder for future extensions.

    Examples
    --------
    >>> class RandomSampler(Sampler):
    ...     def __init__(self, dataset):
    ...         self.dataset = dataset
    ...
    ...     def __iter__(self):
    ...         # Return random indices
    ...         pass
    ...
    ...     def __len__(self):
    ...         return len(self.dataset)
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
