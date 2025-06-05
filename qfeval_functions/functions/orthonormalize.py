import torch

from .einsum import einsum


def orthonormalize(a: torch.Tensor) -> torch.Tensor:
    r"""Orthonormalizes the given vectors and returns the corresponding
    orthonormal vectors.  Ignoring numerical errorrs, this function returns
    the same results as the Gram-Schmidt process.

    If the given vectors is orthonormal, this function must return the
    identical vectors (CAVEAT: It may have a little numerical errors).

    Shape:
        - a: :math:`(*, N, M)` where `*` means any number of additional
          dimensions, `N` means the number of vectors, and `M` means the
          number of dimensions.
    """
    assert a.shape[-2] <= a.shape[-1], (
        "The dimension of vectors must be larger than the number of "
        f"vectors, but: {a.shape}"
    )
    # 1. Squash the batch shape.
    shape = a.shape
    a = a.reshape(-1, shape[-2], shape[-1])

    # 2. Calculate orthonormal vectors.
    q, r = torch.linalg.qr(a.transpose(-1, -2))
    a = (q * einsum("bii->bi", r)[:, None, :].sign()).transpose(-1, -2)

    # 3. Restore the batch shape.
    return a.reshape(*shape)
