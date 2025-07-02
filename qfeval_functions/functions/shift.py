import math
import typing

import torch


@typing.overload
def shift(x: torch.Tensor, shifts: int, dims: int) -> torch.Tensor:
    pass


@typing.overload
def shift(
    x: torch.Tensor,
    shifts: typing.Tuple[int, ...],
    dims: typing.Tuple[int, ...],
) -> torch.Tensor:
    pass


def shift(
    x: torch.Tensor,
    shifts: typing.Union[int, typing.Tuple[int, ...]],
    dims: typing.Union[int, typing.Tuple[int, ...]],
) -> torch.Tensor:
    r"""Shift tensor elements along specified dimensions with NaN padding.

    This function shifts the elements of the input tensor along the specified
    dimensions by the given number of positions. The shifted positions are
    filled with NaN values, making this operation different from a circular
    shift. Positive shifts move elements forward (to higher indices), while
    negative shifts move elements backward (to lower indices).

    The function supports both single and multi-dimensional shifts, allowing
    for complex data transformations commonly used in time series analysis
    and sliding window operations.

    Args:
        x (Tensor):
            The input tensor to be shifted.
        shifts (int or tuple of ints):
            The number of positions to shift along each dimension. Positive
            values shift forward, negative values shift backward. If a single
            int is provided with multiple dimensions, the same shift is
            applied to all specified dimensions.
        dims (int or tuple of ints):
            The dimension(s) along which to shift. Must have the same length
            as :attr:`shifts` if :attr:`shifts` is a tuple.

    Returns:
        Tensor:
            The shifted tensor with the same shape as the input. Positions
            that are shifted in from outside the tensor boundaries are
            filled with NaN values.

    Raises:
        RuntimeError:
            If the length of :attr:`shifts` and :attr:`dims` don't match
            when both are tuples.

    Example:
        >>> # 1D shift example
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> shift(x, shifts=2, dims=0)
        tensor([nan, nan, 1., 2., 3.])

        >>> # Negative shift
        >>> shift(x, shifts=-1, dims=0)
        tensor([2., 3., 4., 5., nan])

        >>> # 2D shift along rows
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0],
        ...                   [7.0, 8.0, 9.0]])
        >>> shift(x, shifts=1, dims=0)
        tensor([[nan, nan, nan],
                [1.,  2.,  3.],
                [4.,  5.,  6.]])

        >>> # 2D shift along columns
        >>> shift(x, shifts=-1, dims=1)
        tensor([[2., 3., nan],
                [5., 6., nan],
                [8., 9., nan]])

        >>> # Multi-dimensional shift
        >>> shift(x, shifts=(1, -1), dims=(0, 1))
        tensor([[nan, nan, nan],
                [2.,  3., nan],
                [5.,  6., nan]])

        >>> # Batch processing
        >>> batch = torch.randn(2, 3, 4)
        >>> shifted = shift(batch, shifts=1, dims=2)
        >>> shifted.shape
        torch.Size([2, 3, 4])

    .. seealso::
        - :func:`torch.roll`: Circular shift without NaN padding.
        - :func:`nanshift`: Shift operation that handles NaN values specially.

    .. note::
        The shift operation preserves tensor shape and data type. Large shift
        values are automatically clamped to prevent out-of-bounds access.
    """

    # 1. Force dims/shifts to be tuples.
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(shifts, int):
        shifts = (shifts,) * len(dims)
    if len(shifts) != len(dims):
        raise RuntimeError(
            f"Inconsistent number of dimensions: shifts={shifts} dims={dims}"
        )
    # Prevent shifts from causing an out-of-index error.
    shifts = tuple(
        max(min(s, x.shape[dims[i]]), -x.shape[dims[i]])
        for i, s in enumerate(shifts)
    )

    # 2. Build a mask to fill NaNs.
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    for shift, dim in zip(shifts, dims):
        s = slice(shift, None) if shift < 0 else slice(0, shift)
        key = tuple(s if i == dim else slice(None) for i in range(len(x.shape)))
        mask[key] = True

    # 3. Apply torch.roll and fill rolled values with NaNs using the mask.
    x = x.roll(shifts, dims)
    return torch.where(mask, torch.as_tensor(math.nan).to(x), x)
