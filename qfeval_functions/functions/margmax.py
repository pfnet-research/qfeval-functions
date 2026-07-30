import math

import torch

from .apply_for_axis import apply_for_axis


def _mextremum_distance(
    x: torch.Tensor, span: int, largest: bool
) -> torch.Tensor:
    """Returns the number of periods elapsed since the extremum of each
    trailing window of the given tensor ``x``, whose shape is ``(B, N)``,
    along the 2nd dimension.

    If ``largest`` is true, the extremum is the window maximum; otherwise,
    it is the window minimum.  Ties are resolved to the most recent
    occurrence, i.e., the smallest distance.
    """

    # 1. A series shorter than the window has no valid output position.
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)

    # 2. Materialize all windows as `(B, N - span + 1, span)` and mark the
    # elements attaining the window extremum.
    w = x.unfold(1, span, 1)
    if largest:
        m = w.amax(dim=-1, keepdim=True)
    else:
        m = w.amin(dim=-1, keepdim=True)
    tie = w == m

    # 3. Take the largest marked window index, i.e., the most recent
    # occurrence of the extremum.  For a window containing NaN, `m` is NaN,
    # so `tie` is all false and the index falls back to 0, but such
    # windows are overwritten with NaN below anyway.
    idx = torch.arange(span, device=x.device)
    most_recent = (tie * idx).amax(dim=-1)
    dist = (span - 1 - most_recent).to(x.dtype)

    # 4. A window containing NaN has no well-defined extremum.
    dist = dist.masked_fill(torch.isnan(w).any(dim=-1), math.nan)

    # 5. Prepend NaNs for the first `span - 1` positions.
    nans = x.new_full((x.shape[0], span - 1), math.nan)
    return torch.cat((nans, dist), dim=1)


def margmax(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the number of periods since the moving maximum.

    This function calculates, for each position, the number of periods
    elapsed since the maximum value inside the trailing window of size
    :attr:`span` along the specified dimension: ``0`` means the latest
    element is the window maximum, and ``span - 1`` means the oldest
    element in the window is.  If the maximum appears multiple times in a
    window, the most recent occurrence is used, i.e., the smallest
    distance is returned.  The output tensor has the same shape as the
    input tensor.  For positions where the sliding window cannot fully
    cover preceding elements (i.e., the first ``span - 1`` elements along
    the selected dimension), the result is ``nan``.

    The number of periods since the maximum is computed as:

    .. math::
        \text{MARGMAX}[i] = i - \max\{j : x[j] = M[i],\;
        i - \text{span} < j \le i\},
        \quad M[i] = \max_{k=i-\text{span}+1}^{i} x[k]

    This function is the building block of the Aroon indicator; for a
    window of size :attr:`span`, the Aroon up indicator is
    ``(span - margmax(x, span)) / span * 100``.

    Args:
        x (Tensor):
            The input tensor containing values.  Must be a floating point
            tensor.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the number of periods
            since the moving maximum. Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape and dtype as the input, containing
            the number of periods since the window maximum as floating
            point values.  The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Raises:
        ValueError: If ``span`` is not positive.
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Number of periods since the window maximum
        >>> x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> QF.margmax(x, span=3)
        tensor([nan, nan, 1., 0., 1.])

        >>> # Ties resolve to the most recent occurrence.
        >>> x = torch.tensor([2.0, 1.0, 2.0, 2.0])
        >>> QF.margmax(x, span=3)
        tensor([nan, nan, 0., 0.])

        >>> # 2D tensor with the number of periods computed along rows
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [4.0, 3.0, 2.0, 1.0]])
        >>> QF.margmax(x, span=2, dim=1)
        tensor([[nan, 0., 0., 0.],
                [nan, 1., 1., 1.]])

    .. note::
        If a window contains any NaN value, the result for that window is
        NaN.  A ``+inf`` value is a legitimate maximum, so the result
        measures the distance to the ``+inf`` element; a ``-inf`` value
        does not affect the result unless it is the window maximum.

    .. note::
        The implementation compares every window element with the window
        maximum, so it takes ``O(N * span)`` time and memory, unlike the
        ``O(N)`` moving aggregations such as :func:`mmax`.

    .. seealso::
        - :func:`margmin`: Number of periods since the moving minimum.
        - :func:`mmax`: Moving maximum function.
        - :func:`mrank`: Moving rank of the latest element.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    if not x.is_floating_point():
        raise TypeError(
            f"margmax only supports floating point tensors, but got "
            f"{x.dtype}."
        )
    return apply_for_axis(
        lambda x: _mextremum_distance(x, span, largest=True), x, dim
    )
