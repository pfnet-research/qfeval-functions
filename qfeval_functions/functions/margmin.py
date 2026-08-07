import torch

from .apply_for_axis import apply_for_axis
from .margmax import _mextremum_distance


def margmin(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the number of periods since the moving minimum.

    This function calculates, for each position, the number of periods
    elapsed since the minimum value inside the trailing window of size
    :attr:`span` along the specified dimension: ``0`` means the latest
    element is the window minimum, and ``span - 1`` means the oldest
    element in the window is.  If the minimum appears multiple times in a
    window, the most recent occurrence is used, i.e., the smallest
    distance is returned.  The output tensor has the same shape as the
    input tensor.  For positions where the sliding window cannot fully
    cover preceding elements (i.e., the first ``span - 1`` elements along
    the selected dimension), the result is ``nan``.

    The number of periods since the minimum is computed as:

    .. math::
        \text{MARGMIN}[i] = i - \max\{j : x[j] = m[i],\;
        i - \text{span} < j \le i\},
        \quad m[i] = \min_{k=i-\text{span}+1}^{i} x[k]

    This function is the building block of the Aroon indicator; for a
    window of size :attr:`span`, the Aroon down indicator is
    ``(span - margmin(x, span)) / span * 100``.

    Args:
        x (Tensor):
            The input tensor containing values.  Must be a floating point
            tensor.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the number of periods
            since the moving minimum. Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape and dtype as the input, containing
            the number of periods since the window minimum as floating
            point values.  The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Raises:
        ValueError: If ``span`` is not positive.
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Number of periods since the window minimum
        >>> x = torch.tensor([3.0, 1.0, 2.0, 0.0, 4.0])
        >>> QF.margmin(x, span=3)
        tensor([nan, nan, 1., 0., 1.])

        >>> # Ties resolve to the most recent occurrence.
        >>> x = torch.tensor([1.0, 2.0, 1.0, 1.0])
        >>> QF.margmin(x, span=3)
        tensor([nan, nan, 0., 0.])

        >>> # 2D tensor with the number of periods computed along rows
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [4.0, 3.0, 2.0, 1.0]])
        >>> QF.margmin(x, span=2, dim=1)
        tensor([[nan, 1., 1., 1.],
                [nan, 0., 0., 0.]])

    .. note::
        If a window contains any NaN value, the result for that window is
        NaN.  A ``-inf`` value is a legitimate minimum, so the result
        measures the distance to the ``-inf`` element; a ``+inf`` value
        does not affect the result unless it is the window minimum.

    .. note::
        ``margmin(x, span, dim)`` is equivalent to
        ``margmax(-x, span, dim)``.  Both share the same implementation,
        using a vectorized reduction for narrow windows and batched
        predecessor queries for larger windows. The latter avoids
        ``O(N * span)`` window materialization and takes
        ``O(N * log(N))`` time with linear workspace.

    .. seealso::
        - :func:`margmax`: Number of periods since the moving maximum.
        - :func:`mmin`: Moving minimum function.
        - :func:`mrank`: Moving rank of the latest element.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if span <= 0:
        raise ValueError(f"span must be a positive integer, but got {span}.")
    if not x.is_floating_point():
        raise TypeError(
            f"margmin only supports floating point tensors, but got "
            f"{x.dtype}."
        )
    return apply_for_axis(
        lambda x: _mextremum_distance(x, span, largest=False), x, dim
    )
