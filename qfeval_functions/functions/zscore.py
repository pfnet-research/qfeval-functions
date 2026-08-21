import typing

import torch

from .nanmean import nanmean
from .nanvar import nanvar


def zscore(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = -1,
    unbiased: bool = True,
) -> torch.Tensor:
    r"""Compute the z-score along specified dimensions, ignoring NaN
    values.

    This function standardizes tensor elements by subtracting the
    NaN-aware mean and dividing by the NaN-aware standard deviation along
    the specified dimension(s).  NaN values are excluded from the
    statistics and stay NaN in the output.  The result is compatible with
    :func:`scipy.stats.zscore` with ``nan_policy="omit"``, where
    :attr:`unbiased` corresponds to its ``ddof`` parameter (``True`` to
    ``ddof=1`` and ``False`` to ``ddof=0``).

    The z-score is computed as:

    .. math::
        \text{ZSCORE}[i] = \frac{x[i] - \mu}{\sigma}

    where :math:`\mu` and :math:`\sigma` are the mean and the standard
    deviation of the valid (non-NaN) elements along the specified
    dimension(s).

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to standardize elements.  If
            None, elements are standardized over all dimensions.
            Default is -1 (the last dimension).
        unbiased (bool, optional):
            If ``True`` (default), the standard deviation is computed
            with Bessel's correction, dividing the sum of squared
            deviations by ``N - 1``, where ``N`` is the number of valid
            elements.  If ``False``, divides by ``N``.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing z-scores.
            Positions holding NaN in the input are NaN in the output.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.zscore(x)
        tensor([-1.2649, -0.6325,  0.0000,  0.6325,  1.2649])

        >>> # Biased (population) standard deviation
        >>> QF.zscore(x, unbiased=False)
        tensor([-1.4142, -0.7071,  0.0000,  0.7071,  1.4142])

        >>> # NaN values are excluded from the statistics.
        >>> QF.zscore(torch.tensor([1.0, nan, 3.0]))
        tensor([-0.7071,     nan,  0.7071])

        >>> # Each row is standardized independently.
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [2.0, 4.0, 6.0]])
        >>> QF.zscore(x, dim=1)
        tensor([[-1.,  0.,  1.],
                [-1.,  0.,  1.]])

        >>> # Standardization over all elements
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> QF.zscore(x, dim=None, unbiased=False)
        tensor([[-1.3416, -0.4472],
                [ 0.4472,  1.3416]])

        >>> # A constant slice yields NaN.
        >>> QF.zscore(torch.full((3,), 2.0))
        tensor([nan, nan, nan])

    .. note::
        A constant slice has zero standard deviation, so its z-scores
        are NaN (:math:`0 / 0`).  A slice with a single valid element
        also yields NaN: the variance itself is NaN if :attr:`unbiased`
        is true, and the zero standard deviation leads to :math:`0 / 0`
        otherwise.  A slice with no valid elements stays NaN.  A slice
        containing infinite values yields NaN because the mean and the
        standard deviation become infinite or undefined.

    .. note::
        Wherever a slice has at least two distinct valid values, the
        output has (approximately) zero mean and unit standard deviation
        over the valid elements along the specified dimension(s).

    .. seealso::
        - :func:`mzscore`: Moving (sliding window) version of this
          function.
        - :func:`rank`: Cross-sectional percentile rank, another way to
          normalize values along a dimension.
        - :func:`winsorize`: Cross-sectional clipping of extreme values.
        - :func:`nanmean`: NaN-aware mean function.
        - :func:`nanvar`: NaN-aware variance function.
    """
    mean = nanmean(x, dim=dim, keepdim=True)
    std = nanvar(x, dim=dim, unbiased=unbiased, keepdim=True) ** 0.5
    return (x - mean) / std
