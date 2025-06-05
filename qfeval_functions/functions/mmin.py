import torch

from .mmax import mmax


def mmin(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Returns the moving min of the given tensor."""
    return -mmax(-x, span, dim)
