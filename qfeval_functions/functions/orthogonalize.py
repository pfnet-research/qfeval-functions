import torch


def orthogonalize(
    x: torch.Tensor, y: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    r"""Orthogonalize ``x`` with respect to ``y`` along the specified dimension.

    This function subtracts from ``x`` its orthogonal projection onto ``y``,
    so that the result is orthogonal to ``y`` along ``dim`` (i.e., their dot
    product along ``dim`` is zero):

    .. math::
        x - \frac{\langle x, y \rangle}{\langle y, y \rangle} y

    This is the elementary step of the Gram-Schmidt process.  In quantitative
    finance, it is useful for neutralizing a signal against another factor
    (e.g., removing market exposure from an alpha signal).

    Args:
        x (Tensor):
            The tensor to be orthogonalized.
        y (Tensor):
            The tensor with respect to which ``x`` will be orthogonalized.
            Its shape must be broadcastable to the shape of ``x``.
        dim (int, optional):
            The dimension along which the orthogonalization will be
            performed.  Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as ``x`` that is orthogonal to ``y``
            along the specified dimension.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> y = torch.tensor([1.0, 1.0, 1.0])
        >>> QF.orthogonalize(x, y)
        tensor([-1.,  0.,  1.])

        >>> # Each row is orthogonalized independently along dim=1.
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> QF.orthogonalize(x, y, dim=1)
        tensor([[0., 2.],
                [3., 0.]])

    .. note::
        If :attr:`y` is a zero vector along the dimension, the projection
        involves division by zero and the result is NaN.

    .. seealso::
        - :func:`orthonormalize`: Orthonormalize a set of vectors.
        - :func:`project`: Project a tensor with a projection matrix.
    """
    # Calculate the dot product of x and y along the specified dimension.
    dot_product = (x * y).sum(dim=dim, keepdim=True)

    # Compute the projection of x onto y.
    projection = dot_product * y / y.square().sum(dim=dim, keepdim=True)

    # Subtract the projection from x to obtain the orthogonalized tensor.
    return x - projection
