import math

import torch


def rank(x: torch.Tensor, dim: int = -1, pct: bool = True) -> torch.Tensor:
    r"""Compute the cross-sectional rank of elements along a dimension.

    This function ranks all valid (non-NaN) elements along the specified
    dimension in ascending order, resolving ties by the average of their
    1-based ranks.  NaN values keep their positions as NaN in the output
    and are excluded from both the ranking and the percentile denominator.
    The result is compatible with
    ``pandas.DataFrame.rank(method="average", na_option="keep", pct=pct)``
    applied along the same axis.

    The rank of a valid element is computed as:

    .. math::
        \text{RANK}[i] = \#\{j : x[j] < x[i]\}
        + \frac{\#\{j : x[j] = x[i]\} + 1}{2}

    where :math:`j` ranges over the valid elements along the selected
    dimension.  If :attr:`pct` is true, the rank is further divided by
    the number of valid elements :math:`N_{\text{valid}}`.

    Args:
        x (Tensor):
            The input tensor containing values.  Must be a floating point
            tensor.  Infinite values are valid and are ranked as the
            smallest (:math:`-\infty`) or the largest (:math:`+\infty`)
            values.
        dim (int, optional):
            The dimension along which to rank elements.
            Default is -1 (the last dimension).
        pct (bool, optional):
            If ``True``, return the percentile rank, i.e., the average
            rank divided by the number of valid elements, which lies in
            :math:`(0, 1]`.  If ``False``, return the 1-based average
            rank, which lies in :math:`[1, N_{\text{valid}}]`.
            Default is ``True``.

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the rank
            values.  Positions holding NaN in the input are NaN in the
            output.

    Raises:
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Percentile rank (default)
        >>> x = torch.tensor([3.0, 1.0, 2.0])
        >>> QF.rank(x)
        tensor([1.0000, 0.3333, 0.6667])

        >>> # 1-based average rank
        >>> QF.rank(x, pct=False)
        tensor([3., 1., 2.])

        >>> # Ties are resolved by the average rank.
        >>> QF.rank(torch.tensor([1.0, 1.0, 2.0]), pct=False)
        tensor([1.5000, 1.5000, 3.0000])

        >>> # NaN values keep their positions and do not affect the
        >>> # percentile denominator.
        >>> QF.rank(torch.tensor([2.0, nan, 1.0]))
        tensor([1.0000,    nan, 0.5000])

        >>> # Infinite values are ranked as the smallest/largest values.
        >>> QF.rank(torch.tensor([1.0, inf, -inf, nan]), pct=False)
        tensor([2., 3., 1., nan])

        >>> # 2D tensor with ranks along each row
        >>> x = torch.tensor([[1.0, 3.0, 2.0],
        ...                   [3.0, 1.0, 2.0]])
        >>> QF.rank(x, dim=1, pct=False)
        tensor([[1., 3., 2.],
                [3., 1., 2.]])

    .. note::
        If all elements along the dimension are NaN, the result is NaN
        for all of them.  A slice with a single valid element yields the
        rank ``1.0`` (also ``1.0`` as the percentile rank).

    .. note::
        The implementation sorts the input and then locates each element
        by binary search, so it takes :math:`O(N \log N)` time for each
        slice of length :math:`N`.

    .. seealso::
        - :func:`mrank`: Moving (sliding window) version of this
          function.
        - :func:`zscore`: Cross-sectional standardization, another way
          to normalize values along a dimension.
        - :func:`winsorize`: Cross-sectional clipping of extreme values.
    """
    if not x.is_floating_point():
        raise TypeError(
            f"rank only supports floating point tensors, but got {x.dtype}."
        )

    # 1. Move the target dimension to the last dimension because
    # torch.searchsorted operates on the last dimension.
    xt = x.transpose(dim, -1)

    # 2. Sort values so that each element can be located by binary search.
    # NOTE: torch.sort places NaN values at the end, so replacing them
    # with +inf keeps the binary search within the valid region.
    s = torch.sort(xt, dim=-1).values
    sf = s.nan_to_num(math.inf, math.inf, -math.inf)
    nv = (~xt.isnan()).sum(dim=-1, keepdim=True)

    # 3. For each element v, count the values less than v (left) and the
    # values less than or equal to v (right).  The latter must be clamped
    # by the number of valid values so that the +inf values converted
    # from NaN do not count as ties of a real +inf.
    left = torch.searchsorted(sf.contiguous(), xt.contiguous(), right=False)
    right = torch.searchsorted(sf.contiguous(), xt.contiguous(), right=True)
    right = torch.minimum(right, nv)

    # 4. The group tied with v occupies the ranks `left + 1, ..., right`,
    # so its average rank is `(left + right + 1) / 2`.
    r = (left + right + 1).to(xt.dtype) / 2
    if pct:
        r = r / nv.to(xt.dtype)

    # 5. Restore NaN values and the original dimension order.
    r = torch.where(xt.isnan(), torch.as_tensor(math.nan).to(xt), r)
    return r.transpose(dim, -1)
