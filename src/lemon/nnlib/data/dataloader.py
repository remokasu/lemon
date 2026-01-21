"""
DataLoader

Provides an iterable over a dataset with batching and shuffling support.
"""

import lemon.numlib as nm


class DataLoader:
    """
    Data loader

    Combines a dataset and provides an iterable over the dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset from which to load the data
    batch_size : int, optional
        How many samples per batch to load (default: 1)
    shuffle : bool, optional
        Set to True to have the data reshuffled at every epoch (default: False)
    drop_last : bool, optional
        Set to True to drop the last incomplete batch (default: False)

    Examples
    --------
    >>> dataset = TensorDataset(X, y)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for X_batch, y_batch in loader:
    ...     # training code
    ...     pass
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)

        xp = nm.get_array_module(nm.zeros(1)._data)  # Get numpy or cupy

        if self.shuffle:
            indices = xp.random.permutation(n)
        else:
            indices = xp.arange(n)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)

            # Skip last batch if drop_last=True and it's incomplete
            if self.drop_last and end - start < self.batch_size:
                continue

            batch_indices = indices[start:end]

            # Get batch
            batch = [self.dataset[int(i)] for i in batch_indices]

            # Transpose: [(x1, y1), (x2, y2)] -> ([x1, x2], [y1, y2])
            if len(batch[0]) == 2:  # (X, y) pairs
                X_list = [item[0] for item in batch]
                y_list = [item[1] for item in batch]

                # Stack into batches
                if hasattr(X_list[0], "_data"):
                    # NumType objects
                    X_batch = nm.stack([x for x in X_list])
                    if hasattr(y_list[0], "_data"):
                        y_batch = nm.stack([y for y in y_list])
                    else:
                        # y_batch = xp.array(
                        #     y_list
                        # )
                        y_batch = nm.tensor(xp.array(y_list))
                else:
                    # Raw arrays - CONVERT TO TENSOR
                    X_batch = nm.tensor(xp.array(X_list))
                    y_batch = nm.tensor(xp.array(y_list))

                yield X_batch, y_batch
            else:
                # General case: multiple outputs
                batch_data = list(zip(*batch))
                result = []
                for data in batch_data:
                    if hasattr(data[0], "_data"):
                        result.append(nm.stack(list(data)))
                    else:
                        # Raw arrays - CONVERT TO TENSOR
                        result.append(nm.tensor(xp.array(data)))
                yield tuple(result)
