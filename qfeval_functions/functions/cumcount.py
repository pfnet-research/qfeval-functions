import torch

from .stable_sort import stable_sort


def cumcount(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Number each occurrence of unique values along a dimension.

    This function assigns a cumulative count to each occurrence of unique
    values along the specified dimension. For each unique value, the first
    occurrence is numbered 0, the second occurrence is numbered 1, and so on.
    This is similar to the behavior of ``pandas.GroupBy.cumcount()``.

    Args:
        x (Tensor):
            The input tensor containing values to be counted.
        dim (int, optional):
            The dimension along which to perform cumulative counting.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, where each element
            contains the cumulative count (0-indexed) of that value's
            occurrence along the specified dimension.

    Example:

        >>> x = torch.tensor([1, 2, 1, 3, 2, 1, 3])
        >>> QF.cumcount(x)
        tensor([0, 0, 1, 0, 1, 2, 1])

        >>> x = torch.tensor([[1, 2, 1, 2],
        ...                   [3, 3, 4, 3]])
        >>> QF.cumcount(x, dim=1)
        tensor([[0, 0, 1, 1],
                [0, 1, 0, 2]])

        >>> x = torch.tensor([[1, 2, 3],
        ...                   [1, 2, 3],
        ...                   [1, 2, 3]])
        >>> QF.cumcount(x, dim=0)
        tensor([[0, 0, 0],
                [1, 1, 1],
                [2, 2, 2]])
    """

    # 1. Flatten the input tensor.
    dim = len(x.shape) + dim if dim < 0 else dim
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape(x.shape[0], -1)

    # 2. Computes the index of a group for each sorted element.
    v, idx = stable_sort(x, dim=0)
    a = torch.arange(x.shape[0], device=x.device)[:, None]
    b = torch.where(
        torch.eq(v, v.roll(1, 0)),
        torch.zeros_like(a),
        a,
    )
    g_idx = a - b.cummax(0).values

    # 3. Distribute the indexes to the original locations.
    g_idx = idx.scatter(0, idx, g_idx)

    # 4. Restore the shape.
    return g_idx.reshape(shape).transpose(0, dim)  # type: ignore[no-any-return]
