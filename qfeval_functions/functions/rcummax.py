import typing

import torch


class Result(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def rcummax(x: torch.Tensor, dim: int) -> Result:
    r"""Compute reverse cumulative maximum along a specified dimension.

    This function computes the cumulative maximum in reverse order along the
    specified dimension. For each position, it returns the maximum value from
    that position to the end of the tensor along the given dimension, along
    with the indices of those maximum values.

    Unlike :func:`torch.cummax` which computes cumulative maximum from the
    beginning to each position, this function computes from each position
    to the end, effectively computing ``cummax`` on the reversed tensor and
    then reversing the result.

    Args:
        x (Tensor):
            The input tensor.
        dim (int):
            The dimension along which to compute the reverse cumulative
            maximum.

    Returns:
        Result:
            A named tuple containing:

            - **values** (*Tensor*): The reverse cumulative maximum values
              with the same shape as the input tensor.
            - **indices** (*Tensor*): The indices of the maximum values
              with the same shape as the input tensor.

    Example:
        >>> # 1D example
        >>> x = torch.tensor([1, 5, 3, 9, 2, 7])
        >>> result = QF.rcummax(x, dim=0)
        >>> result.values
        tensor([9, 9, 9, 9, 7, 7])
        >>> result.indices
        tensor([3, 3, 3, 3, 5, 5])

        >>> # 2D example
        >>> x = torch.tensor([[1, 4, 2],
        ...                   [3, 1, 5]])
        >>> result = QF.rcummax(x, dim=1)
        >>> result.values
        tensor([[4, 4, 2],
                [5, 5, 5]])
        >>> result.indices
        tensor([[1, 1, 2],
                [2, 2, 2]])

        >>> # 3D example with different dimensions
        >>> x = torch.tensor([[[1, 2], [3, 4]],
        ...                   [[5, 6], [7, 8]]])
        >>> result_dim0 = QF.rcummax(x, dim=0)
        >>> result_dim0.values.shape
        torch.Size([2, 2, 2])

        >>> # Compare with regular cummax
        >>> regular = torch.cummax(x, dim=1)
        >>> reverse = QF.rcummax(x, dim=1)
        >>> # They should be different for most cases

    .. seealso::
        - :func:`torch.cummax`: Cumulative maximum from beginning to each position.
        - :func:`rcumsum`: Reverse cumulative sum.

    .. note::
        This function is useful in financial analysis for:

        - Computing trailing maximum values for stop-loss calculations
        - Maximum drawdown analysis from any point forward
        - Peak detection and resistance level identification
        - Portfolio value ceiling tracking for performance analysis
        - Risk management scenarios requiring future maximum exposure

    .. note::
        The reverse cumulative maximum is particularly valuable for:

        - Calculating the maximum value that will be seen in the future
        - Implementing lookback straddles and barrier options
        - Analyzing maximum adverse excursion in trading strategies
        - Computing rolling maximum values in reverse time order

    .. note::
        The indices returned correspond to the original positions in the
        input tensor, not the positions in the flipped tensor used internally.
    """
    result = torch.cummax(torch.flip(x, [dim]), dim)
    return Result(
        values=torch.flip(result.values, [dim]),
        indices=x.shape[dim] - 1 - torch.flip(result.indices, [dim]),
    )
