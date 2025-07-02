import functools
import math
import operator
import typing

import torch


def skipna(
    f: typing.Callable[..., torch.Tensor],
    *xs: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    r"""Apply a function to tensors after removing NaN values along a dimension.

    This function applies a given function to one or more tensors after
    temporarily removing NaN values along the specified dimension. The
    function effectively compacts the data by removing NaNs, applies the
    given function, and then restores the original tensor structure with
    NaNs in their original positions.

    The operation preserves the order of non-NaN values while ensuring
    that the function operates only on valid (non-NaN) data. This is
    particularly useful for applying statistical functions or other
    operations that should ignore missing values.

    Args:
        f (Callable[..., Tensor]):
            A function that takes one or more tensors and returns a tensor.
            The function will be applied to the compacted (NaN-free) data.
        *xs (Tensor):
            One or more input tensors. All tensors must have the same shape.
            NaN positions must be consistent across all input tensors.
        dim (int, optional):
            The dimension along which to remove and restore NaN values.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            The result of applying function :attr:`f` to the compacted data,
            with NaN values restored to their original positions.

    Example:
        >>> # Simple example - the function is complex and requires careful setup
        >>> # This demonstrates the concept but may not work with all functions
        >>> from qfeval_functions.functions.skipna import skipna
        >>> import torch
        >>> # 2D example where skipna can work properly
        >>> x = torch.tensor([[1.0, 2.0, float('nan'), 4.0],
        ...                   [float('nan'), 6.0, 7.0, 8.0]])
        >>> x.shape
        torch.Size([2, 4])

        >>> # The function is designed for specific use cases where the
        >>> # applied function maintains compatible dimensions
        >>> # Example usage requires careful consideration of the function
        >>> # and dimension parameters

    .. seealso::
        - :func:`torch.nansum`: Sum with NaN handling.
        - :func:`torch.nanmean`: Mean with NaN handling.
        - :func:`nanshift`: Shift operation with NaN handling.

    .. note::
        This function is particularly useful for:

        - Applying custom functions that don't have built-in NaN handling
        - Statistical computations on incomplete datasets
        - Time series analysis with missing observations
        - Data preprocessing pipelines that need to handle missing values
        - Custom aggregation functions for financial data

    .. note::
        In quantitative finance applications, skipna enables:

        - Computing statistics on price series with missing data
        - Applying custom risk metrics to incomplete return series
        - Portfolio optimization with assets having different histories
        - Technical indicator calculation with data gaps
        - Backtesting strategies on datasets with missing observations

    .. note::
        The function preserves the structure and semantics of the operation:

        - The order of non-NaN values is maintained
        - NaN positions are restored in the output
        - Multi-tensor operations are synchronized by NaN positions
        - The original tensor shape and dtype are preserved

    .. warning::
        All input tensors must have NaN values at the same positions. If
        NaN patterns differ across tensors, the behavior is undefined and
        may lead to incorrect results.

    .. warning::
        Be careful when applying functions that change the tensor dimensions,
        as the NaN restoration process expects the output to have compatible
        dimensions with the input structure.
    """
    # Generate an index to convert between a sparse array and a dense array.
    # For the given `x` (sparse): [0, nan, 5, nan, 4, 2, 1], the dense array
    # (the scattered array) should be [0, 5, 4, 2, 1, nan, nan] and `idx` (the
    # mapping form the sparse array to the dense array) should be
    # [0, 5, 1, 6, 2, 3, 4].
    m = functools.reduce(operator.or_, (x.isnan() for x in xs))
    valid_idx = (~m).cumsum(dim) - 1
    invalid_idx = (~m).sum(dim, keepdim=True) + m.cumsum(dim) - 1
    idx = torch.where(m, invalid_idx, valid_idx)
    y = f(*(x.scatter(dim, idx, x) for x in xs)).gather(dim, idx)
    return torch.where(m, torch.as_tensor(math.nan).to(y), y)
