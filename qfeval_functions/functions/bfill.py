import torch

from .ffill import ffill


def bfill(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Propagates last valid values backward."""
    return torch.flip(ffill(torch.flip(x, [dim]), dim), [dim])
