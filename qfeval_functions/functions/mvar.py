import torch

from .msum import msum


def mvar(
    x: torch.Tensor, span: int, dim: int = -1, ddof: int = 1
) -> torch.Tensor:
    r"""Returns the moving variance of the given tensor."""
    numerator = msum(x**2, span, dim) - msum(x, span, dim) ** 2 / span
    result: torch.Tensor = numerator / (span - ddof)
    return result
