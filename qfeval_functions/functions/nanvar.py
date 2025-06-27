import math
import typing

import torch

from .nanmean import nanmean
from .nansum import nansum


def nanvar(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    unbiased: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the variance of a tensor, ignoring NaN values.

    This function calculates the variance of tensor elements along specified
    dimensions while excluding NaN values. The variance can be computed using
    either the unbiased estimator (dividing by N-1) or the biased estimator
    (dividing by N), where N is the number of non-NaN elements.

    Args:
        x (Tensor):
            The input tensor.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to compute the variance. If not
            specified (default is an empty tuple), the variance is computed
            over all elements. Can be a single dimension or multiple dimensions.
        unbiased (bool, optional):
            If ``True`` (default), uses Bessel's correction and divides by
            ``N-1`` where ``N`` is the number of non-NaN elements. If ``False``,
            divides by ``N``.
        keepdim (bool, optional):
            If ``True``, the output tensor has the same number of dimensions
            as the input, with the reduced dimensions having size 1.
            If ``False`` (default), the reduced dimensions are removed.

    Returns:
        Tensor:
            The variance of non-NaN elements. The shape depends on the
            :attr:`dim` and :attr:`keepdim` parameters.

    Example:
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanvar(x)
        tensor(3.3333)

        >>> # With biased estimator
        >>> QF.nanvar(x, unbiased=False)
        tensor(2.5000)

        >>> # 2D tensor with dimension specification
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nanvar(x, dim=1)
        tensor([0.5000, 2.0000])

        >>> # Keep dimensions
        >>> QF.nanvar(x, dim=1, keepdim=True)
        tensor([[0.5000],
                [2.0000]])
    """
    n = (~x.isnan()).to(x).sum(dim=dim, keepdim=True)
    if unbiased:
        n = n - 1
        # Prevent gradients from being divided by zeros.
        x = torch.where(n <= 0, torch.as_tensor(math.nan).to(x), x)
    x2 = (x - nanmean(x, dim=dim, keepdim=True)) ** 2
    r = nansum(x2, dim=dim, keepdim=True) / n
    return r.sum(dim=dim, keepdim=keepdim)
