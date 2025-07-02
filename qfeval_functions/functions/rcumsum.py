import torch


def rcumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Compute reverse cumulative sum along a specified dimension.

    This function computes the cumulative sum in reverse order along the
    specified dimension. For each position, it returns the sum of all values
    from that position to the end of the tensor along the given dimension.

    Unlike :func:`torch.cumsum` which computes cumulative sum from the
    beginning to each position, this function computes from each position
    to the end, effectively computing ``cumsum`` on the reversed tensor and
    then reversing the result.

    Args:
        x (Tensor):
            The input tensor.
        dim (int):
            The dimension along which to compute the reverse cumulative sum.

    Returns:
        Tensor:
            The reverse cumulative sum with the same shape as the input tensor.

    Example:
        >>> # 1D example
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> QF.rcumsum(x, dim=0)
        tensor([15, 14, 12,  9,  5])

        >>> # 2D example
        >>> x = torch.tensor([[1, 2, 3],
        ...                   [4, 5, 6]])
        >>> QF.rcumsum(x, dim=1)
        tensor([[ 6,  5,  3],
                [15, 11,  6]])

        >>> # 3D example with different dimensions
        >>> x = torch.tensor([[[1, 2], [3, 4]],
        ...                   [[5, 6], [7, 8]]])
        >>> QF.rcumsum(x, dim=0).shape
        torch.Size([2, 2, 2])
        >>> QF.rcumsum(x, dim=1).shape
        torch.Size([2, 2, 2])
        >>> QF.rcumsum(x, dim=2).shape
        torch.Size([2, 2, 2])

        >>> # Compare with regular cumsum
        >>> regular = torch.cumsum(x, dim=1)
        >>> reverse = QF.rcumsum(x, dim=1)
        >>> # The last elements should be the same
        >>> regular[:, -1, :] == reverse[:, 0, :]
        tensor([[True, True],
                [True, True]])

        >>> # Verify property: rcumsum + cumsum - original = total sum
        >>> x_1d = torch.tensor([1, 2, 3, 4])
        >>> forward = torch.cumsum(x_1d, dim=0)
        >>> backward = QF.rcumsum(x_1d, dim=0)
        >>> (forward + backward - x_1d).unique()
        tensor([10])

    See Also:
        :func:`torch.cumsum`: Cumulative sum from beginning to each position.
        :func:`rcummax`: Reverse cumulative maximum.

    .. note::
        This function is essential in financial analysis for:

        - Computing remaining cash flows from any point forward
        - Calculating future value accumulations in investment scenarios
        - Reverse discounting for present value calculations
        - Tail sum computations for risk assessment
        - Portfolio rebalancing with forward-looking constraints

    .. note::
        The reverse cumulative sum is particularly useful for:

        - Calculating total remaining exposure or commitments
        - Computing final settlement amounts in derivative contracts
        - Analyzing cumulative returns from exit points
        - Budget allocation with remaining period constraints
        - Stress testing with forward-looking scenarios

    .. note::
        For time series data, this function provides the sum of all future
        values from each time point, which is valuable for forecasting
        and planning applications.
    """
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])
