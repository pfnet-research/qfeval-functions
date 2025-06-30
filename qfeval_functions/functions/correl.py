import typing

import torch

from .mulmean import mulmean


def correl(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute Pearson correlation coefficient between two tensors.

    This function calculates the Pearson correlation coefficient between
    tensors :attr:`x` and :attr:`y` along the specified dimension(s). The
    correlation coefficient measures the linear relationship between two
    variables and ranges from -1 (perfect negative correlation) to 1
    (perfect positive correlation), with 0 indicating no linear correlation.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be the same shape as :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the correlation. If not
            specified (default is empty tuple), computes element-wise
            correlation and sums the result.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim` retained or not.
            Default is False.

    Returns:
        Tensor:
            The Pearson correlation coefficient(s). The shape depends on the
            input dimensions and the :attr:`keepdim` parameter.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> QF.correl(x, y, dim=0)
        tensor(1.)

        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> y = torch.tensor([[2.0, 4.0, 5.0],
        ...                   [8.0, 10.0, 12.0]])
        >>> QF.correl(x, y, dim=1)
        tensor([0.9820, 1.0000])

        >>> QF.correl(x, y, dim=1, keepdim=True)
        tensor([[0.9820],
                [1.0000]])
    """
    ax = x - x.mean(dim=dim, keepdim=True)
    ay = y - y.mean(dim=dim, keepdim=True)
    axy = mulmean(ax, ay, dim=dim, keepdim=True)
    ax2 = (ax**2).mean(dim=dim, keepdim=True)
    ay2 = (ay**2).mean(dim=dim, keepdim=True)
    result: torch.Tensor = axy / ax2.sqrt() / ay2.sqrt()
    result = result.sum(dim=dim, keepdim=keepdim)
    return result
