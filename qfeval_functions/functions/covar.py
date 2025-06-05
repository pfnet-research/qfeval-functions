import torch

from .mulmean import mulmean


def covar(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    ddof: int = 1,
) -> torch.Tensor:
    r"""Calculates the covariance between the given tensors along the specified
    dimension.

    When performing broadcast between the given tensors, the number of elements
    may increase and the space complexity may increase, but this function still
    performs the calculation with the original space complexity.  For instance,
    when operating on tensors with shapes (N, 1, D) and (1, M, D), the space
    complexity remains at O(ND + MD) instead of O(NMD).

    CAVEAT: Differently from np.cov, this calculates a covariance for each
    batch index instead of producing a covariance matrix.
    """
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    return mulmean(x, y, dim=dim, keepdim=keepdim, _ddof=ddof)
