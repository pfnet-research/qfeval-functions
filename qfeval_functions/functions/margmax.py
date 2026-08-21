import math

import torch

from ._moving_order import _window_has_nan
from .apply_for_axis import apply_for_axis
from .mmax import mmax


def _prepend_incomplete(
    x: torch.Tensor, completed: torch.Tensor, span: int
) -> torch.Tensor:
    nans = x.new_full((x.shape[0], span - 1), math.nan)
    return torch.cat((nans, completed), dim=1)


def _mextremum_distance_compare(
    x: torch.Tensor, span: int, largest: bool
) -> torch.Tensor:
    """Compute extremum distances by reducing materialized windows."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)

    # Reversing each window makes the first extremum the most recent one,
    # so the reduction result is already the elapsed-period count.
    windows = x.unfold(1, span, 1).flip(-1)
    if largest:
        distance = windows.argmax(dim=-1)
    else:
        distance = windows.argmin(dim=-1)
    completed = distance.to(x.dtype)
    completed = completed.masked_fill(_window_has_nan(x, span), math.nan)
    return _prepend_incomplete(x, completed, span)


def _mextremum_distance_predecessor(
    x: torch.Tensor, span: int, largest: bool
) -> torch.Tensor:
    """Compute distances with moving extrema and offline predecessors."""
    if x.shape[1] < span:
        return torch.full_like(x, math.nan)
    if x.shape[0] == 0:
        return x.clone()

    batch_size, length = x.shape
    prefix_length = span - 1
    ordering_values = x if largest else -x
    moving = mmax(ordering_values, span, dim=1)

    # Interleave each complete-window extremum query immediately after its
    # input event. A stable value sort then groups equal values while
    # retaining time order, reducing every query to a predecessor lookup.
    tail = torch.stack(
        (ordering_values[:, prefix_length:], moving[:, prefix_length:]),
        dim=-1,
    ).flatten(1)
    combined = torch.cat((ordering_values[:, :prefix_length], tail), dim=1)

    completed_times = torch.arange(
        prefix_length, length, dtype=torch.long, device=x.device
    )
    tail_event_times = torch.stack(
        (completed_times, torch.full_like(completed_times, -1)),
        dim=-1,
    ).flatten()
    event_times = (
        torch.cat(
            (
                torch.arange(prefix_length, dtype=torch.long, device=x.device),
                tail_event_times,
            )
        )
        .view(1, -1)
        .expand(batch_size, -1)
    )

    ordering = combined.argsort(dim=1, stable=True)
    sorted_event_times = event_times.gather(1, ordering)
    sorted_positions = (
        torch.arange(combined.shape[1], dtype=torch.long, device=x.device)
        .view(1, -1)
        .expand(batch_size, -1)
    )
    previous_event_position = (
        torch.where(sorted_event_times >= 0, sorted_positions, -1)
        .cummax(dim=1)
        .values
    )
    latest_time_sorted = sorted_event_times.gather(
        1, previous_event_position.clamp_min(0)
    )
    latest_time = torch.empty_like(latest_time_sorted)
    latest_time.scatter_(1, ordering, latest_time_sorted)
    latest_completed = latest_time[:, span::2]

    current_time = completed_times.to(x.dtype)
    completed = current_time - latest_completed.to(x.dtype)
    completed = completed.masked_fill(_window_has_nan(x, span), math.nan)
    return _prepend_incomplete(x, completed, span)


def _mextremum_distance(
    x: torch.Tensor, span: int, largest: bool
) -> torch.Tensor:
    """Return periods since each trailing-window extremum.

    Ties resolve to the most recent occurrence. The implementation is
    selected from measured small- and large-window algorithms.
    """
    window_count = max(0, x.shape[1] - span + 1)
    enough_parallel_work = x.shape[0] * span >= 1_024
    if (
        span < 128
        or (span < 512 and not enough_parallel_work)
        or window_count <= 8
    ):
        return _mextremum_distance_compare(x, span, largest)
    return _mextremum_distance_predecessor(x, span, largest)


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
        Narrow windows use a vectorized reduction. Larger windows combine
        a linear moving maximum with batched predecessor queries, avoiding
        ``O(N * span)`` window materialization and taking
        ``O(N * log(N))`` time with linear workspace.

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
