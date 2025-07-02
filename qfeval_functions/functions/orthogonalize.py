import torch


def orthogonalize(
    x: torch.Tensor, y: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    r"""Orthogonalize tensor x with respect to tensor y.

    This function performs the Gram-Schmidt orthogonalization process,
    removing the component of :attr:`x` that is parallel to :attr:`y` along
    the specified dimension. The result is a tensor that is orthogonal to
    :attr:`y` while preserving the orthogonal component of :attr:`x`.

    The orthogonalization is computed as:

    .. math::
        \text{orthogonalized}(x) = x - \frac{\langle x, y \rangle}{\langle y, y \rangle} y

    where :math:`\langle \cdot, \cdot \rangle` denotes the dot product along
    the specified dimension.

    Args:
        x (Tensor):
            The tensor to be orthogonalized.
        y (Tensor):
            The reference tensor with respect to which :attr:`x` will be
            orthogonalized. Must have the same shape as :attr:`x`.
        dim (int, optional):
            The dimension along which the orthogonalization is performed.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            The orthogonalized tensor with the same shape as :attr:`x`.

    Example:
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> y = torch.tensor([[1.0, 0.0, 0.0],
        ...                   [1.0, 0.0, 0.0]])
        >>> QF.orthogonalize(x, y, dim=1)
        tensor([[0., 2., 3.],
                [0., 5., 6.]])

        >>> # Verify orthogonality: dot product should be zero
        >>> result = QF.orthogonalize(x, y, dim=1)
        >>> torch.sum(result * y, dim=1)
        tensor([0., 0.])
    """
    # Calculate the dot product of x and y along the specified dimension.
    dot_product = (x * y).sum(dim=dim, keepdim=True)

    # Compute the projection of x onto y.
    projection = dot_product * y / y.square().sum(dim=dim, keepdim=True)

    # Subtract the projection from x to obtain the orthogonalized tensor.
    return x - projection
