import torch

from .mquantile import mquantile


def mmedian(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) median of a tensor.

    This function calculates the median of elements within a sliding
    window of size :attr:`span` along the specified dimension. The output
    tensor has the same shape as the input tensor. For positions where the
    sliding window cannot fully cover preceding elements (i.e., the first
    ``span - 1`` elements along the selected dimension), the result is
    ``nan``.

    Letting :math:`s_0 \le s_1 \le \dots \le s_{\text{span}-1}` denote the
    sorted values of a window, the moving median is:

    .. math::
        \text{MMEDIAN}[i] =
        \begin{cases}
            s_{(\text{span}-1)/2}
            & \text{if } \text{span} \text{ is odd} \\
            \left(s_{\text{span}/2-1} + s_{\text{span}/2}\right) / 2
            & \text{if } \text{span} \text{ is even}
        \end{cases}

    This is equivalent to ``QF.mquantile(x, span, 0.5, dim)`` and
    compatible with ``pandas.DataFrame.rolling(span).median()``: for an
    even window size, the two central values are averaged (linear
    interpolation).

    Args:
        x (Tensor):
            The input tensor containing values. Must be a floating point
            tensor.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving median.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            median values. The first ``span - 1`` elements along the
            specified dimension are ``nan``.

    Raises:
        TypeError: If ``span`` is not an integer (``bool`` is rejected).
        ValueError: If ``span`` is not positive.
        TypeError: If ``x`` is not a floating point tensor.

    Example:

        >>> # Moving median with an odd window size
        >>> x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> QF.mmedian(x, span=3)
        tensor([nan, nan, 2., 3., 4.])

        >>> # An even window size averages the two central values
        >>> QF.mmedian(x, span=2)
        tensor([   nan, 2.0000, 2.5000, 3.5000, 4.5000])

        >>> # Moving median along the first dimension
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [5.0, 3.0],
        ...                   [3.0, 7.0]])
        >>> QF.mmedian(x, span=3, dim=0)
        tensor([[nan, nan],
                [nan, nan],
                [3., 3.]])

        >>> # A window containing NaN yields NaN
        >>> QF.mmedian(torch.tensor([1.0, nan, 3.0, 4.0, 5.0]), span=2)
        tensor([   nan,    nan,    nan, 3.5000, 4.5000])

    .. note::
        If a window contains any NaN value, the moving median for that
        window is NaN. Unlike ``pandas.DataFrame.rolling``, there is no
        ``min_periods``-style option to skip NaN values.

    .. note::
        This function delegates to :func:`mquantile` with its automatic
        algorithm selection: narrow windows use vectorized window
        operations, while larger-window workloads use wavelet-matrix
        range selection.

    .. seealso::
        - :func:`mquantile`: The underlying moving quantile function.
        - :func:`nanmedian`: Median over a whole dimension, skipping NaNs.
        - :func:`ma`: Moving average function.
    """
    return mquantile(x, span, 0.5, dim=dim)
