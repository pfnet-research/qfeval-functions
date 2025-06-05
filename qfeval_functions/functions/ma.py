import torch

from .msum import msum


def ma(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Returns the moving average of the given tensor."""
    return msum(x, span, dim) / span
