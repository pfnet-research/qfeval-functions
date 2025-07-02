import torch

from .einsum import einsum


def orthonormalize(a: torch.Tensor) -> torch.Tensor:
    r"""Compute orthonormal vectors using the Gram-Schmidt process.

    This function takes a set of vectors and returns orthonormal vectors that
    span the same subspace. The orthonormalization is performed using QR
    decomposition, which is numerically stable and equivalent to the
    Gram-Schmidt process.

    The resulting vectors are orthogonal (perpendicular to each other) and
    normalized (unit length). If the input vectors are already orthonormal,
    the output will be identical (up to numerical precision).

    .. note::
        The number of vectors must not exceed the dimension of the vector space
        (i.e., :attr:`a.shape[-2] <= a.shape[-1]`).

    Args:
        a (Tensor):
            Input vectors of shape ``(..., N, M)`` where ``N`` is the number
            of vectors and ``M`` is the dimension of each vector. The batch
            dimensions ``...`` are preserved.

    Returns:
        Tensor:
            Orthonormal vectors of the same shape as the input. The vectors
            span the same subspace as the input vectors.

    Raises:
        AssertionError:
            If the number of vectors exceeds the vector dimension
            (``a.shape[-2] > a.shape[-1]``).

    Example:
        >>> # 2D vectors in 3D space
        >>> vectors = torch.tensor([[[1.0, 1.0, 0.0],
        ...                          [1.0, 0.0, 1.0]]])
        >>> orthonormal = QF.orthonormalize(vectors)
        >>> orthonormal.shape
        torch.Size([1, 2, 3])

        >>> # Verify orthogonality: dot product should be zero
        >>> torch.sum(orthonormal[0, 0] * orthonormal[0, 1])
        tensor(0.)

        >>> # Verify normalization: each vector should have unit length
        >>> torch.norm(orthonormal, dim=-1)
        tensor([[1., 1.]])

        >>> # Already orthonormal vectors remain unchanged
        >>> identity = torch.eye(3).unsqueeze(0)
        >>> result = QF.orthonormalize(identity)
        >>> torch.allclose(result, identity)
        True
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
