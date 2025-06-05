import torch

from .shift import shift as _shift


def nanshift(
    x: torch.Tensor,
    shift: int = 1,
    dim: int = -1,
) -> torch.Tensor:
    r"""Shifts array elements along specified dimensions, ignoring NaNs."""

    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape((shape[0], -1))

    # 2. Build a mapping to gather/scatter valid values from/to a sparse tensor.
    priority = torch.arange(shape[0], device=x.device)[:, None]
    priority = priority + x.isnan().int() * shape[0]
    index = torch.argsort(priority, dim=0)
    reversed_index = torch.argsort(index, dim=0)

    # 3. Gather valid values, apply shift, and scatter them.
    y = x[index, torch.arange(x.shape[1])[None, :]]
    y = torch.where(y.isnan(), torch.as_tensor(y).to(y), _shift(y, shift, 0))
    x = y[reversed_index, torch.arange(y.shape[1])[None, :]]

    # 4. Restore the shape and the order of dimensions.
    return x.reshape(shape).transpose(0, dim)
