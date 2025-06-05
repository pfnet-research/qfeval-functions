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
    r"""Shifts array elements along specified dimensions."""

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
