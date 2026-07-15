import torch

from .einsum import einsum


def orthonormalize(a: torch.Tensor) -> torch.Tensor:
    r"""Orthonormalize the given vectors and return the corresponding
    orthonormal vectors.

    Ignoring numerical errors, this function returns the same results as the
    Gram-Schmidt process: the :math:`i`-th output vector is the :math:`i`-th
    input vector orthogonalized with respect to all preceding vectors and
    then normalized to unit length.  Internally, it is implemented with QR
    decomposition for numerical stability and efficiency.

    If the given vectors are already orthonormal, this function must return
    the identical vectors (CAVEAT: it may have a little numerical errors).

    Args:
        a (Tensor):
            The input tensor containing sets of linearly independent
            vectors.  Each of the trailing :math:`N` rows is a vector of
            :math:`M` elements, and :math:`N` must not exceed :math:`M`.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            corresponding orthonormal vectors.

    Shape:
        - a: :math:`(*, N, M)` where ``*`` means any number of additional
          dimensions, ``N`` means the number of vectors, and ``M`` means the
          number of dimensions.
        - return: :math:`(*, N, M)`, the same shape as the input.

    Example:

        >>> a = torch.tensor([[3.0, 4.0],
        ...                   [0.0, 5.0]])
        >>> QF.orthonormalize(a)
        tensor([[ 0.6000,  0.8000],
                [-0.8000,  0.6000]])

    .. seealso::
        - :func:`orthogonalize`: Orthogonalize vectors against another set
          of vectors.
        - :func:`project`: Project a tensor with a projection matrix.
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
