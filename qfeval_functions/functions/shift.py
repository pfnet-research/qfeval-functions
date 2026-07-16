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
            The input tensor.  It must have a floating-point dtype
            because vacated positions are filled with NaN.
        shifts (int or tuple of ints):
            The number of places by which the elements are shifted.  If it
            is an int, the same shift is applied to all dimensions in
            ``dims``.  Shifts whose magnitudes exceed the dimension size are
            clamped to the dimension size (i.e., the result becomes all
            NaN).
        dims (int or tuple of ints):
            The dimension or dimensions along which to shift.  Negative
            values are counted from the last dimension.  If both ``shifts``
            and ``dims`` are tuples, they must have the same length.  If the
            same dimension is given multiple times, its shifts are summed,
            matching the behavior of :func:`torch.roll`.

    Returns:
        Tensor:
            A tensor of the same shape as the input, with elements shifted
            and vacated positions filled with NaN.

    Raises:
        TypeError: If ``x`` does not have a floating-point dtype.
        RuntimeError: If ``shifts`` and ``dims`` have different lengths.
        IndexError: If a dimension in ``dims`` is out of range.

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

    if not x.is_floating_point():
        raise TypeError(
            f"shift requires a floating-point tensor because vacated "
            f"positions are filled with NaN, but got dtype: {x.dtype}."
        )

    # 1. Force dims/shifts to be tuples.
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(shifts, int):
        shifts = (shifts,) * len(dims)
    if len(shifts) != len(dims):
        raise RuntimeError(
            f"Inconsistent number of dimensions: shifts={shifts} dims={dims}"
        )

    # 2. Normalize negative dimensions and sum shifts of duplicated
    # dimensions so that torch.roll and the mask below agree on the
    # effective shift of each dimension.
    merged_shifts: typing.Dict[int, int] = {}
    for shift, dim in zip(shifts, dims):
        if not -x.ndim <= dim < x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {dim})"
            )
        dim = dim % x.ndim
        merged_shifts[dim] = merged_shifts.get(dim, 0) + shift
    dims = tuple(merged_shifts.keys())
    # Prevent shifts from causing an out-of-index error.
    shifts = tuple(
        max(min(s, x.shape[dim]), -x.shape[dim])
        for dim, s in merged_shifts.items()
    )

    # 3. Build a mask to fill NaNs.
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    for shift, dim in zip(shifts, dims):
        s = slice(shift, None) if shift < 0 else slice(0, shift)
        key = tuple(s if i == dim else slice(None) for i in range(len(x.shape)))
        mask[key] = True

    # 4. Apply torch.roll and fill rolled values with NaNs using the mask.
    x = x.roll(shifts, dims)
    return torch.where(mask, torch.as_tensor(math.nan).to(x), x)
