import torch


def ffill(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Forward fill missing values along the specified dimension.

    This function propagates the last valid (non-NaN) values forward along
    the specified dimension. For each NaN value, it is replaced with the
    most recent valid value that appears before it along the dimension. If no
    valid value exists before a NaN, it remains NaN.

    The forward fill operation is commonly used in time series analysis to
    handle missing data by carrying forward the last observed value.

    Args:
        x (Tensor):
            The input tensor containing values to be filled. May contain NaN
            values that will be replaced.
        dim (int, optional):
            The dimension along which to perform forward fill.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, with NaN values replaced
            by the nearest preceding valid values where possible.

    Example:

        >>> # Simple 1D forward fill
        >>> x = torch.tensor([1.0, nan, nan, 4.0, 5.0])
        >>> QF.ffill(x)
        tensor([1., 1., 1., 4., 5.])

        >>> # 2D example with dim=1
        >>> x = torch.tensor([[nan, 2.0, nan, 4.0],
        ...                   [5.0, nan, nan, 8.0]])
        >>> QF.ffill(x, dim=1)
        tensor([[nan, 2., 2., 4.],
                [5., 5., 5., 8.]])

        >>> # Forward fill along columns (dim=0)
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [nan, 2.0, nan],
        ...                   [4.0, nan, nan]])
        >>> QF.ffill(x, dim=0)
        tensor([[1., nan, 3.],
                [1., 2., 3.],
                [4., 2., 3.]])

    .. note::
        This implementation uses an efficient algorithm based on cumulative
        maximum of boolean masks to track the last valid value indices.
    """
    if x.shape[dim] == 0:
        return x
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    indices = torch.cummax(~torch.isnan(x), dim=0).indices
    x = x[indices, torch.arange(x.shape[1], device=x.device)[None]]
    return x.reshape(shape).transpose(0, dim)
