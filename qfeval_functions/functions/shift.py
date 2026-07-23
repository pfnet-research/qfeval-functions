import math
import typing

import torch


def _fill_value(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the value used to fill vacated positions for ``x``'s dtype.

    Floating-point and complex tensors are filled with NaN (``nan+0j`` for
    complex), integer tensors with ``0``, and boolean tensors with
    ``False``.
    """
    if x.is_floating_point() or x.is_complex():
        return torch.as_tensor(math.nan).to(x)
    return torch.zeros((), dtype=x.dtype, device=x.device)


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
    positions with a dtype-appropriate value.

    This function behaves like :func:`torch.roll` except that elements
    shifted beyond the boundary do not wrap around; instead, the vacated
    positions are filled with a dtype-appropriate value.  This matches the
    behavior of :meth:`pandas.DataFrame.shift` and is useful for creating
    lagged (or leading) time series.  A positive shift moves elements toward
    larger indices, and a negative shift moves them toward smaller indices.

    Args:
        x (Tensor):
            The input tensor.  Vacated positions are filled with NaN for
            floating-point (and complex) tensors, ``0`` for integer tensors,
            and ``False`` for boolean tensors.
        shifts (int or tuple of ints):
            The number of places by which the elements are shifted.  If it
            is an int, the same shift is applied to all dimensions in
            ``dims``.  Shifts whose magnitudes exceed the dimension size are
            clamped to the dimension size (i.e., the result becomes entirely
            the fill value).
        dims (int or tuple of ints):
            The dimension or dimensions along which to shift.  Negative
            values are counted from the last dimension.  If both ``shifts``
            and ``dims`` are tuples, they must have the same length.  If the
            same dimension is given multiple times, its shifts are summed,
            matching the behavior of :func:`torch.roll`.

    Returns:
        Tensor:
            A tensor of the same shape as the input, with elements shifted
            and vacated positions filled with a dtype-appropriate value (NaN
            for floating-point and complex, ``0`` for integer, ``False`` for
            boolean).

    Raises:
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

        >>> # Integer tensors are filled with 0.
        >>> QF.shift(torch.tensor([1, 2, 3, 4]), 1, 0)
        tensor([0, 1, 2, 3])

        >>> # Boolean tensors are filled with False.
        >>> QF.shift(torch.tensor([True, True, True]), 1, 0)
        tensor([False,  True,  True])

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

    # 4. Apply torch.roll and fill rolled values using the mask.
    x = x.roll(shifts, dims)
    return torch.where(mask, _fill_value(x), x)
