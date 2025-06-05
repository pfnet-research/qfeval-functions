import torch

from .mvar import mvar


def mstd(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Returns the moving standard deviation of the given tensor."""
    result: torch.Tensor = mvar(x, span=span, dim=dim, ddof=ddof) ** 0.5
    return result
