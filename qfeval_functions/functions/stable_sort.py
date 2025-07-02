import math
import typing

import torch


class StableSortResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def _unsafe_stable_sort(x: torch.Tensor, dim: int) -> StableSortResult:
    r"""Sorts the given tensor without NaN values preserving the order of
    equivalent elements.
    """

    # 1. First sort values using an unstable algorithm.
    # NOTE: v should be the same as a stable algorithm computes.
    v, idx = x.sort(dim=dim)

    # 2. Computes the rank of v.
    v_rank = torch.ne(v, v.roll(1, dim)).cumsum(dim=dim)

    # 3. Sort idx within a group having the same v's rank.
    idx = v_rank * idx.shape[dim] + idx
    idx = idx.sort(dim=dim).values % idx.shape[dim]

    # 4. Return the result of a stable sorting.
    return StableSortResult(v, idx)


def stable_sort(x: torch.Tensor, dim: int = -1) -> StableSortResult:
    r"""Perform stable sorting that preserves the order of equivalent elements.

    This function sorts the input tensor along the specified dimension while
    maintaining the relative order of elements that compare equal. Unlike
    PyTorch's built-in ``torch.sort``, this implementation guarantees stable
    sorting behavior, which is crucial for maintaining consistency in
    applications where the original order of equal elements matters.

    The function handles NaN values gracefully by placing them at the end
    of the sorted sequence while preserving their relative order. This makes
    it suitable for financial data analysis where missing values are common.

    Stable sorting ensures that when multiple elements have the same value,
    they appear in the same relative order as in the original tensor. This
    property is essential for consistent ranking and index-based operations.

    Args:
        x (Tensor):
            Input tensor to be sorted. Can contain NaN values which will be
            handled appropriately.
        dim (int, optional):
            Dimension along which to sort. Default is -1 (last dimension).

    Returns:
        StableSortResult:
            A named tuple containing:
            
            - ``values`` (Tensor): The sorted values with the same shape as input.
            - ``indices`` (Tensor): The indices that would sort the original tensor.

    Example:
        >>> # Basic stable sorting
        >>> x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        >>> result = stable_sort(x)
        >>> result.values
        tensor([1., 1., 3., 4., 5.])
        >>> result.indices
        tensor([1, 3, 0, 2, 4])

        >>> # Verify stability: original order preserved for equal elements
        >>> x = torch.tensor([[2.0, 1.0, 1.0, 3.0]])
        >>> result = stable_sort(x, dim=1)
        >>> result.values
        tensor([[1., 1., 2., 3.]])
        >>> result.indices  # First 1.0 comes before second 1.0
        tensor([[1, 2, 0, 3]])

        >>> # 2D sorting along different dimensions
        >>> x = torch.tensor([[3.0, 1.0, 4.0],
        ...                   [1.0, 5.0, 2.0]])
        >>> result = stable_sort(x, dim=1)
        >>> result.values
        tensor([[1., 3., 4.],
                [1., 2., 5.]])
        >>> result.indices
        tensor([[1, 0, 2],
                [0, 2, 1]])

        >>> # Sorting along rows (dim=0)
        >>> result = stable_sort(x, dim=0)
        >>> result.values
        tensor([[1., 1., 2.],
                [3., 5., 4.]])

        >>> # Handling NaN values
        >>> x = torch.tensor([3.0, float('nan'), 1.0, float('nan'), 2.0])
        >>> result = stable_sort(x)
        >>> result.values
        tensor([1., 2., 3., nan, nan])
        >>> result.indices
        tensor([2, 4, 0, 1, 3])

        >>> # Multi-dimensional with NaN
        >>> x = torch.tensor([[float('nan'), 2.0, 1.0],
        ...                   [3.0, float('nan'), 4.0]])
        >>> result = stable_sort(x, dim=1)
        >>> result.values
        tensor([[1., 2., nan],
                [3., 4., nan]])

        >>> # Financial data ranking with ties
        >>> returns = torch.tensor([0.12, 0.08, 0.12, 0.15, 0.08])
        >>> result = stable_sort(returns, dim=0)
        >>> result.values  # Stable order for equal returns
        tensor([0.0800, 0.0800, 0.1200, 0.1200, 0.1500])
        >>> result.indices  # Original indices preserved for ties
        tensor([1, 4, 0, 2, 3])

        >>> # Batch processing of portfolios
        >>> portfolio_values = torch.tensor([[100.0, 150.0, 100.0, 200.0],
        ...                                  [80.0, 120.0, 80.0, 160.0]])
        >>> result = stable_sort(portfolio_values, dim=1)
        >>> result.values
        tensor([[100., 100., 150., 200.],
                [ 80.,  80., 120., 160.]])

    .. seealso::
        - :func:`torch.sort`: Standard sorting (not guaranteed to be stable).
        - :func:`torch.argsort`: Return indices that would sort a tensor.
        - :func:`soft_topk`: Differentiable top-k selection.

    .. note::
        Stable sorting is particularly important in quantitative finance for:
        
        - Consistent ranking of assets with equal performance metrics
        - Maintaining portfolio construction order for equal-weight strategies
        - Preserving temporal order in time series with identical values
        - Ensuring reproducible results in backtesting and simulations
        - Risk management with consistent exposure ordering
        - Regulatory reporting requiring deterministic ordering

    .. note::
        Applications in financial data analysis include:
        
        - Ranking assets by returns while preserving temporal order for ties
        - Portfolio optimization with consistent asset selection
        - Risk factor analysis with stable factor loadings
        - Performance attribution with consistent sector ordering
        - ESG scoring with stable ranking for equal scores
        - Credit rating analysis with consistent ordering within ratings

    .. note::
        The stable sorting algorithm has O(n log n) time complexity and uses
        a two-phase approach: first sorting with an unstable algorithm, then
        re-ordering within groups of equal elements to maintain stability.

    .. note::
        For NaN handling, the algorithm treats NaN values as the largest
        possible values and places them at the end of the sorted sequence.
        The relative order of NaN values is preserved, maintaining stability
        even in the presence of missing data.

    .. warning::
        Large tensors with many duplicate values may require significant
        memory for the intermediate sorting steps. Consider using chunk-based
        processing for very large datasets if memory is a concern.

    References:
        - Stable sorting algorithms: https://en.wikipedia.org/wiki/Sorting_algorithm#Stability
        - Financial data ranking: https://en.wikipedia.org/wiki/Ranking#Finance
    """

    # If x has no NaN values, it is okay to apply _unsafe_stable_sort.
    isnan = x.isnan()
    if not isnan.any():
        return _unsafe_stable_sort(x, dim)

    # If x has NaN values, computes the stable result using two results.
    safe_x = x.nan_to_num(math.inf, math.inf, -math.inf)
    result = _unsafe_stable_sort(safe_x, dim)
    # Mapping NaN to 0.0, +inf to -1.0 and other values to -2.0.
    # NOTE: Multiplying -inf to the result should recover NaN and +inf.
    nan_result = _unsafe_stable_sort(
        torch.where(
            safe_x.isposinf(),
            x.nan_to_num(0.0, -1.0),
            torch.full_like(x, -2.0),
        ),
        dim,
    )
    return StableSortResult(
        torch.where(
            result.values.isposinf(),
            nan_result.values * -math.inf,
            result.values,
        ),
        torch.where(
            result.values.isposinf(),
            nan_result.indices,
            result.indices,
        ),
    )
