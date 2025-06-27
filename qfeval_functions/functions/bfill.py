import torch

from .ffill import ffill


def bfill(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Backward fill missing values along the specified dimension.

    This function propagates the last valid (non-NaN) values backward along
    the specified dimension. For each NaN value, it is replaced with the
    nearest valid value that appears after it along the dimension. If no valid
    value exists after a NaN, it remains NaN.

    Args:
        x (Tensor):
            The input tensor containing values to be filled.
        dim (int, optional):
            The dimension along which to perform backward fill.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, with NaN values replaced
            by the nearest subsequent valid values where possible.

    Example:

        >>> x = torch.tensor([1.0, nan, nan, 4.0, 5.0])
        >>> QF.bfill(x)
        tensor([1., 4., 4., 4., 5.])

        >>> x = torch.tensor([[nan, 2.0, nan, 4.0],
        ...                   [5.0, nan, nan, nan]])
        >>> QF.bfill(x, dim=1)
        tensor([[2., 2., 4., 4.],
                [5., nan, nan, nan]])
    """
    return torch.flip(ffill(torch.flip(x, [dim]), dim), [dim])
