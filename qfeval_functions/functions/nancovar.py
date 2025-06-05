import torch

from .nanmean import nanmean
from .nanmulmean import nanmulmean
from .nanones import nanones


def nancovar(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    ddof: int = 1,
) -> torch.Tensor:
    """Calculates the covariance between the given tensors.

    CAVEAT: Differently from np.cov, this calculates a covariance for each
    batch index instead of producing a covariance matrix.
    CAVEAT: The calculation result would be low-precision (approximately
    half-precision) especially in case of many NaNs due to prioritizing memory
    efficiency.  In order to improve the precision is necessary to write CUDA
    code via PyTorch JIT (torch.utils.cpp_extension.load).
    """
    # Improve the precision by subtracting their averages first.
    x = x - nanmean(x, dim=dim, keepdim=True)
    y = y - nanmean(y, dim=dim, keepdim=True)
    mx = nanmulmean(x, nanones(y), dim=dim, keepdim=keepdim, _ddof=ddof)
    my = nanmulmean(nanones(x), y, dim=dim, keepdim=keepdim, _ddof=ddof)
    mxy = nanmulmean(x, y, dim=dim, keepdim=keepdim, _ddof=ddof)
    # NOTE: E((X - E[X])(Y - E[Y])) = E(XY) - E(X)E(Y)
    return (mxy - mx * my).to(x)
