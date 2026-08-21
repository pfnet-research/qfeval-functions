import torch

from .shift import shift


def roc(x: torch.Tensor, span: int = 1, dim: int = -1) -> torch.Tensor:
    r"""Compute the rate of change (ROC) along the specified dimension.

    ROC is a momentum indicator measuring the relative change between the
    current value and the value ``span`` positions earlier:

    .. math::
        \text{ROC}[i] = \frac{x[i]}{x[i - \text{span}]} - 1

    This is equivalent to ``pandas.DataFrame.pct_change(span)``.  The first
    ``span`` elements along the dimension are NaN because they have no
    preceding value (:func:`shift` fills vacated positions with NaN).  A
    negative ``span`` looks forward instead, matching ``pct_change`` with
    negative periods, and ``span=0`` yields all zeros for finite nonzero
    inputs.

    Division follows IEEE 754 semantics: a zero previous value yields
    ``inf`` or ``-inf``, and ``0 / 0`` yields NaN.  A NaN input affects
    exactly the output positions where it appears as the numerator or the
    denominator.

    Args:
        x (Tensor):
            The input tensor containing values.  Must be a floating point
            tensor.
        span (int, optional):
            The number of positions to look back (negative values look
            forward).  Default is 1.
        dim (int, optional):
            The dimension along which to compute the rate of change.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the rate
            of change values.  The first ``span`` elements along the
            dimension (the last ``-span`` elements for a negative
            ``span``) are NaN.

    Raises:
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> x = torch.tensor([100.0, 110.0, 121.0])
        >>> QF.roc(x)
        tensor([   nan, 0.1000, 0.1000])

        >>> # A negative span compares against future values.
        >>> QF.roc(x, span=-1)
        tensor([-0.0909, -0.0909,     nan])

        >>> # A zero previous value yields inf, and 0 / 0 yields NaN.
        >>> QF.roc(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        tensor([nan, inf, -1., nan])

    .. seealso::
        - :func:`shift`: Shift function used to align previous values.
        - :func:`nandiff`: Difference function that skips NaN values.
        - :func:`ema`: Exponential moving average function.
    """
    # NOTE: bool is a subclass of int, so it must be rejected explicitly.
    if isinstance(span, bool) or not isinstance(span, int):
        raise TypeError(f"span must be an integer, but got {span!r}.")
    if not x.is_floating_point():
        raise TypeError(
            f"roc only supports floating point tensors, but got {x.dtype}."
        )
    return x / shift(x, span, dim) - 1
