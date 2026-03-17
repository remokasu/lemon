import lemon.numlib as nm


def causal_mask(seq_len):
    """
    Generate a causal (autoregressive) mask for decoder self-attention.

    Prevents position i from attending to positions > i.
    Used in GPT-like models.

    Parameters
    ----------
    seq_len : int
        Sequence length

    Returns
    -------
    Tensor  shape (seq_len, seq_len)
        1 where attention is allowed, 0 where it is masked

    Examples
    --------
    >>> mask = causal_mask(4)
    >>> # [[1, 0, 0, 0],
    >>> #  [1, 1, 0, 0],
    >>> #  [1, 1, 1, 0],
    >>> #  [1, 1, 1, 1]]
    """
    xp = nm.np
    mask = xp.tril(xp.ones((seq_len, seq_len), dtype=xp.float32))
    return nm.tensor(mask)


def padding_mask(lengths, max_len=None):
    """
    Generate a padding mask from sequence lengths.

    Marks valid positions as 1 and padding positions as 0.

    Parameters
    ----------
    lengths : list of int or Tensor
        Actual length of each sequence in the batch
    max_len : int, optional
        Maximum sequence length. If None, uses max(lengths).

    Returns
    -------
    Tensor  shape (batch, max_len)
        1 for valid positions, 0 for padding

    Examples
    --------
    >>> mask = padding_mask([3, 5, 2], max_len=5)
    >>> # [[1, 1, 1, 0, 0],
    >>> #  [1, 1, 1, 1, 1],
    >>> #  [1, 1, 0, 0, 0]]
    """
    xp = nm.np
    if hasattr(lengths, "_data"):
        lengths = lengths._data.tolist()
    if max_len is None:
        max_len = max(lengths)
    batch = len(lengths)
    mask = xp.zeros((batch, max_len), dtype=xp.float32)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
    return nm.tensor(mask)
