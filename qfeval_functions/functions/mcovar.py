import math

import numpy as np
import torch

from .rcumsum import rcumsum


def _mcovar(
    x: torch.Tensor, y: torch.Tensor, span: int, ddof: int
) -> torch.Tensor:
    """Returns the moving covariance of the given tensors ``x`` and ``y``,
    whose shapes are ``(B, N)``, along the 2nd dimension.

    This follows the chunked cumulative-statistics algorithm described in
    https://imoz.jp/scraps/202607_mvar.en.html (see ``_mvar``), generalized
    from second moments to comoments.  Each part (a chunk prefix or suffix)
    keeps ``n`` (count), per-series local reference values ``rx``/``ry``,
    sums of local deviations ``Sx``/``Sy`` (with ``yx = x - rx`` and
    ``yy = y - ry``), relative means ``mx_bar = Sx / n`` and
    ``my_bar = Sy / n``, and the comoment
    ``C`` (``Sum((x - mu_x) * (y - mu_y))``).
    """

    w = span

    # 1. Reshape the target dimension into length-`w` chunks with prepended
    # NaNs (the same chunk layout as `msum`).
    pad_len = w * 2 - x.shape[1] % w
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    y = torch.nn.functional.pad(y, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // w, w))
    y = y.reshape((y.shape[0], y.shape[1] // w, w))

    # 2. Compute the per-part statistics `(n, Sx, Sy, mx_bar, my_bar, C)` for
    # every chunk prefix (part B) and chunk suffix (part A).  Each series is
    # locally centered on its own reference value (the chunk's first element
    # for a prefix, its last element for a suffix), so that all cumulative
    # sums are accumulated over small local deviations.  This avoids
    # catastrophic cancellation, and lets NaN/inf contaminate exactly the
    # parts that contain them.
    #
    # Prefixes (part B): counts n = 1..w, references = first chunk elements.
    n_p = torch.arange(1, w + 1, dtype=x.dtype, device=x.device)
    rx_p = x[:, :, :1]
    ry_p = y[:, :, :1]
    yx_p = x - rx_p
    yy_p = y - ry_p
    Sx_p = yx_p.cumsum(dim=2)
    Sy_p = yy_p.cumsum(dim=2)
    mx_bar_p = Sx_p / n_p
    my_bar_p = Sy_p / n_p
    C_p = (yx_p * yy_p).cumsum(dim=2) - Sx_p * my_bar_p
    #
    # Suffixes (part A): counts n = w..1, references = last chunk elements.
    n_s = torch.arange(w, 0, -1, dtype=x.dtype, device=x.device)
    rx_s = x[:, :, -1:]
    ry_s = y[:, :, -1:]
    yx_s = x - rx_s
    yy_s = y - ry_s
    Sx_s = rcumsum(yx_s, dim=2)
    Sy_s = rcumsum(yy_s, dim=2)
    mx_bar_s = Sx_s / n_s
    my_bar_s = Sy_s / n_s
    C_s = rcumsum(yx_s * yy_s, dim=2) - Sx_s * my_bar_s

    # 3. Each window is a suffix of the previous chunk (part A) followed by a
    # prefix of the current chunk (part B); an aligned window is just a whole
    # chunk prefix.  Merge the two parts with Chan et al.'s parallel formula
    # for comoments `C = C_A + C_B + n_A * n_B / (n_A + n_B) * delta_x *
    # delta_y`, where the mean gap per series is `delta = mu_B - mu_A`.
    # Since the means are stored relative to their references, `delta =
    # (r_B - r_A) + (m_bar_B - m_bar_A)` never forms a large offset.  Here
    # `n_A + n_B = w`, so the weight is `n_A * n_B / w`.
    delta_x = (rx_p[:, 1:] - rx_s[:, :-1]) + (
        mx_bar_p[:, 1:, :-1] - mx_bar_s[:, :-1, 1:]
    )
    delta_y = (ry_p[:, 1:] - ry_s[:, :-1]) + (
        my_bar_p[:, 1:, :-1] - my_bar_s[:, :-1, 1:]
    )
    weight = n_s[1:] * n_p[:-1] / w
    C = C_s[:, :-1, 1:] + C_p[:, 1:, :-1] + delta_x * delta_y * weight
    C = torch.cat((C, C_p[:, 1:, -1:]), dim=2)

    # 4. Cov = C / (w - ddof).
    return C.flatten(start_dim=1)[:, pad_len - w :] / max(0, w - ddof)


def mcovar(
    x: torch.Tensor,
    y: torch.Tensor,
    span: int,
    dim: int = -1,
    ddof: int = 1,
) -> torch.Tensor:
    r"""Compute the moving (sliding window) covariance of two tensors.

    This function calculates the covariance between elements of :attr:`x`
    and :attr:`y` within a sliding window of size :attr:`span` along the
    specified dimension.  The input tensors are broadcast to a common shape,
    and the output tensor has that shape.  For positions where the sliding
    window cannot fully cover preceding elements (i.e., the first
    ``span - 1`` elements along the selected dimension), the result is
    ``nan``.  This is compatible with
    ``pandas.DataFrame.rolling(span).cov(other, ddof=ddof)``.

    The moving covariance is computed using the formula:

    .. math::
        \text{MCOVAR}[i] = \frac{1}{\text{span} - \text{ddof}}
        \sum_{j=i-\text{span}+1}^{i}
        \left(x[j] - \mu_x[i]\right)\left(y[j] - \mu_y[i]\right)

    where :math:`\mu_x[i]` and :math:`\mu_y[i]` are the means of ``x`` and
    ``y`` over the same window.

    The moving covariance is computed in ``O(N)`` time independent of
    :attr:`span`, using cumulative statistics of :attr:`span`-sized chunks
    merged by Chan's parallel algorithm.  All sums are taken over deviations
    from nearby reference values, so the result is numerically stable even
    when the inputs have large offsets relative to their covariance (e.g.,
    values around ``1e6`` with a covariance of ``1e-6``).

    See `A Numerically Stable and Fast Implementation of Moving Averages and
    Variances <https://imoz.jp/scraps/202607_mvar.en.html>`_ for a detailed
    description of the chunked cumulative-statistics algorithm.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving covariance.
            Default is -1 (the last dimension).
        ddof (int, optional):
            Delta degrees of freedom. The divisor used in the calculation is
            ``span - ddof``. Use 0 for population covariance. Must be less
            than ``span``; otherwise the result is ``inf`` or ``nan``.
            Default is 1 (sample covariance).

    Returns:
        Tensor:
            A tensor of the broadcast shape of the inputs, containing the
            moving covariance values. The first ``span - 1`` elements along
            the specified dimension are ``nan``.

    Example:

        >>> # Simple moving covariance with window size 3
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> QF.mcovar(x, y, span=3)
        tensor([nan, nan, 2., 2., 2.])

        >>> # Negatively related series
        >>> y = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        >>> QF.mcovar(x, y, span=3)
        tensor([nan, nan, -2., -2., -2.])

        >>> # Population covariance (ddof=0)
        >>> QF.mcovar(x, y, span=3, ddof=0)
        tensor([    nan,     nan, -1.3333, -1.3333, -1.3333])

        >>> # Covariance of a series with itself is its variance
        >>> QF.mcovar(x, x, span=2)
        tensor([   nan, 0.5000, 0.5000, 0.5000, 0.5000])

        >>> # Broadcasting: one series against each row of a matrix
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = torch.tensor([[2.0, 4.0, 6.0, 8.0],
        ...                   [4.0, 3.0, 2.0, 1.0]])
        >>> QF.mcovar(x, y, span=2)
        tensor([[    nan,  1.0000,  1.0000,  1.0000],
                [    nan, -0.5000, -0.5000, -0.5000]])

    .. note::
        If a window contains any NaN value in either input, the moving
        covariance for that window is NaN. Unlike
        ``pandas.DataFrame.rolling``, there is no ``min_periods``-style
        option to skip NaN values.

    .. seealso::
        - :func:`covar`: Covariance over an entire dimension.
        - :func:`mcorrel`: Moving Pearson correlation function.
        - :func:`mvar`: Moving variance function (this with ``y = x``).
        - :func:`nancovar`: NaN-aware covariance function.
    """

    x, y = torch.broadcast_tensors(x, y)

    # Replicate `apply_for_axis`'s transformations for the two input tensors.
    # 1. Move the target dimension to the top to make data manipulation
    # easier.
    x = x.transpose(0, dim)
    y = y.transpose(0, dim)
    shape = x.shape
    # NOTE: This does not use -1 intentionally because it fails if the given
    # tensor has one or more 0-length axes.
    x = x.reshape((shape[0], int(np.prod(shape[1:]))))
    y = y.reshape((shape[0], int(np.prod(shape[1:]))))

    # 2. Apply the core function.
    result = _mcovar(x.t(), y.t(), span, ddof).t()

    # 3. Restore the shape and the order of dimensions.
    return result.reshape(result.shape[:1] + shape[1:]).transpose(0, dim)
