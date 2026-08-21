import typing

import torch

from .nanquantile import nanquantile


def winsorize(
    x: torch.Tensor,
    lower: float = 0.05,
    upper: float = 0.95,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = -1,
) -> torch.Tensor:
    r"""Clamp tensor values to quantile bounds, ignoring NaN values.

    This function limits the influence of outliers by clipping the
    values of :attr:`x` to the range
    ``[nanquantile(x, lower), nanquantile(x, upper)]``, where both
    bounds are computed along :attr:`dim` while ignoring NaN values.
    Values below the lower quantile are replaced with the lower
    quantile, and values above the upper quantile are replaced with the
    upper quantile.  NaN values are excluded from the quantile
    computation and remain NaN in the result.  This is equivalent to
    the pandas idiom ``s.clip(s.quantile(lower), s.quantile(upper))``
    applied to each slice along :attr:`dim`.

    Args:
        x (Tensor):
            The input tensor containing values.
        lower (float, optional):
            The quantile used as the lower clipping bound.  Must
            satisfy ``0 <= lower <= upper <= 1``.  Default is 0.05.
        upper (float, optional):
            The quantile used as the upper clipping bound.  Must
            satisfy ``0 <= lower <= upper <= 1``.  Default is 0.95.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the quantile
            bounds.  If None, the bounds are computed over all
            elements.  Default is -1, so each cross section along the
            last dimension is winsorized independently.

    Returns:
        Tensor:
            A tensor of the same shape as the input with values clipped
            to the quantile bounds.  NaN values remain NaN.

    Raises:
        ValueError: If ``0 <= lower <= upper <= 1`` is violated.

    Example:

        >>> # Clip outliers with the default 5%/95% quantile bounds
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
        >>> QF.winsorize(x)
        tensor([ 1.2000,  2.0000,  3.0000,  4.0000, 80.8000])

        >>> # Tighter bounds clip more aggressively
        >>> QF.winsorize(x, lower=0.25, upper=0.75)
        tensor([2., 2., 3., 4., 4.])

        >>> # NaN values are ignored and preserved
        >>> x = torch.tensor([nan, 1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.winsorize(x, lower=0.25, upper=0.75)
        tensor([nan, 2., 2., 3., 4., 4.])

        >>> # Each row is winsorized independently by default (dim=-1)
        >>> x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 10.0],
        ...                   [5.0, 6.0, 7.0, 8.0, 9.0]])
        >>> QF.winsorize(x, lower=0.25, upper=0.75)
        tensor([[1., 1., 2., 3., 3.],
                [6., 6., 7., 8., 8.]])

        >>> # Global winsorization over all elements
        >>> QF.winsorize(x, lower=0.25, upper=0.75, dim=None)
        tensor([[2.2500, 2.2500, 2.2500, 3.0000, 7.7500],
                [5.0000, 6.0000, 7.0000, 7.7500, 7.7500]])

    .. note::
        The clipping bounds are linearly interpolated quantiles, so
        clipped values are generally not elements of the input.  This
        differs from ``scipy.stats.mstats.winsorize``, which replaces
        the smallest and largest values with the nearest remaining
        order statistics (actual input values).

    .. seealso::
        - :func:`nanquantile`: NaN-aware quantile used to compute the
          clipping bounds.
        - :func:`fillna`: Replace NaN and infinity values.
        - :func:`zscore`: Standardize values along a dimension.
        - :func:`rank`: Rank-based normalization robust to outliers.
    """
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(
            "winsorize() requires 0 <= lower <= upper <= 1, but got "
            f"lower={lower} and upper={upper}."
        )
    lower_bound = nanquantile(x, lower, dim=dim, keepdim=True)
    upper_bound = nanquantile(x, upper, dim=dim, keepdim=True)
    return torch.clamp(x, lower_bound, upper_bound)
