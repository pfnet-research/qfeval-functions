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
    r"""Shifts array elements along specified dimensions, filling vacated
    positions with NaN.

    This function behaves like :func:`torch.roll` except that elements
    shifted beyond the boundary do not wrap around; instead, the vacated
    positions are filled with NaN.  This matches the behavior of
    :meth:`pandas.DataFrame.shift` and is useful for creating lagged (or
    leading) time series.  A positive shift moves elements toward larger
    indices, and a negative shift moves them toward smaller indices.

    Args:
        x (Tensor):
            The input tensor.  It should have a floating-point dtype
            because vacated positions are filled with NaN.
        shifts (int or tuple of ints):
            The number of places by which the elements are shifted.  If it
            is an int, the same shift is applied to all dimensions in
            ``dims``.  Shifts whose magnitudes exceed the dimension size are
            clamped to the dimension size (i.e., the result becomes all
            NaN).
        dims (int or tuple of ints):
            The dimension or dimensions along which to shift.  If both
            ``shifts`` and ``dims`` are tuples, they must have the same length.

    Returns:
        Tensor:
            A tensor of the same shape as the input, with elements shifted
            and vacated positions filled with NaN.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> QF.shift(x, 1, 0)
        tensor([nan, 1., 2., 3.])

        >>> QF.shift(x, -1, 0)
        tensor([2., 3., 4., nan])

        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> QF.shift(x, (1, 1), (0, 1))
        tensor([[nan, nan],
                [nan, 1.]])

    .. seealso::
        - :func:`nanshift`: Shift function that skips NaN values.
        - :func:`group_shift`: Shift operation within groups.
        - ``torch.roll``: Circular shift where elements wrap around.
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
