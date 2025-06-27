import typing

import torch

from .mulsum import mulsum


def mulmean(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Compute the mean of element-wise product in a memory-efficient way.

    This function calculates the mean of the element-wise product of two
    tensors ``(x * y).mean(dim)`` without creating the intermediate product
    tensor in memory. This is particularly useful when working with large
    tensors where memory efficiency is critical, or when broadcasting between
    tensors would result in a very large intermediate tensor.

    The function is mathematically equivalent to ``(x * y).mean(dim)`` but
    uses a more memory-efficient implementation that avoids materializing
    the full product tensor.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the mean. If not specified
            (default is empty tuple), the mean is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.
        _ddof (int, optional):
            Delta degrees of freedom for internal calculations.
            The divisor used is ``N - _ddof``, where ``N`` is the number of
            elements. Default is 0. This is an internal parameter.

    Returns:
        Tensor:
            The mean of the element-wise product. The shape depends on the
            input dimensions, :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple element-wise product mean
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = torch.tensor([2.0, 3.0, 4.0, 5.0])
        >>> QF.mulmean(x, y)
        tensor(10.)

        >>> # Equivalent to (x * y).mean()
        >>> torch.allclose(QF.mulmean(x, y), (x * y).mean())
        True

        >>> # 2D tensors with specific dimension
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> QF.mulmean(x, y, dim=1)
        tensor([2.0000, 5.5000])

        >>> # Broadcasting example
        >>> x = torch.randn(1000, 1)
        >>> y = torch.randn(1, 1000)
        >>> # Memory-efficient: doesn't create 1000x1000 intermediate tensor
        >>> result = QF.mulmean(x, y)

        >>> # With keepdim
        >>> x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        >>> y = torch.tensor([[[2.0, 1.0]], [[1.0, 2.0]]])
        >>> QF.mulmean(x, y, dim=(1, 2), keepdim=True)
        tensor([[[2.0000]],
        <BLANKLINE>
                [[5.5000]]])

    .. seealso::
        :func:`mulsum`: The underlying function for memory-efficient
        multiplication and summation.
        :func:`covar`: Uses this function for covariance calculations.
    """
    return mulsum(x, y, dim=dim, keepdim=keepdim, mean=True, _ddof=_ddof)
