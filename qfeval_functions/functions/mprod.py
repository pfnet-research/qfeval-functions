import math

import torch

from .apply_for_axis import apply_for_axis


def _mprod(x: torch.Tensor, span: int) -> torch.Tensor:
    """Returns the moving product of the given tensor ``x``, whose shape is
    ``(B, N)``, along the 2nd dimension."""

    # 1. Reshape the target dimension into `(*, span)` with prepending NaNs.
    pad_len = span * 2 - x.shape[1] % span
    x = torch.nn.functional.pad(x, (pad_len, 0), value=math.nan)
    x = x.reshape((x.shape[0], x.shape[1] // span, span))

    # 2. Calculate `prod(x[:, i:i+span], dim=1)` by splitting it into
    # `prod(x[:, i:s], dim=1)*prod(x[:, s:i+span], dim=1)` where `s` is a
    # multiple of `span`.  They can be calculated by a forward and a reverse
    # cumulative product bounded to each chunk.
    a = x.cumprod(dim=2)
    b = x.flip(2).cumprod(dim=2).flip(2)
    x = torch.cat((a[:, 1:, :-1] * b[:, :-1, 1:], a[:, 1:, -1:]), dim=2)
    return x.flatten(start_dim=1)[:, pad_len - span :]


def mprod(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) product of a tensor.

    This function calculates the product of elements within a sliding window
    of size :attr:`span` along the specified dimension. The output tensor has
    the same shape as the input tensor. For positions where the sliding
    window cannot fully cover preceding elements (i.e., the first ``span - 1``
    elements along the selected dimension), the result is ``nan``.

    The moving product is computed using the formula:

    .. math::
        \text{MPROD}[i] = \prod_{j=i-\text{span}+1}^{i} x[j]

    The moving product is computed in ``O(N)`` time independent of
    :attr:`span`, by decomposing each window into a chunk suffix and a chunk
    prefix obtained from cumulative products bounded to :attr:`span` terms.
    See `A Numerically Stable and Fast Implementation of Moving Averages and
    Variances <https://imoz.jp/scraps/202607_mvar.en.html>`_ for a detailed
    description of the chunked algorithm.  Because no division is involved,
    a window containing ``0`` yields exactly ``0.0`` without contaminating
    neighboring windows, which is a key advantage over division-based
    running products.

    Args:
        x (Tensor):
            The input tensor.
        span (int):
            The size of the sliding window.
        dim (int, optional):
            The dimension along which to compute the moving product.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            products.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> QF.mprod(x, span=3)
        tensor([nan, nan,  6., 24., 60.])

        >>> # A window containing zero is exactly zero; other windows are
        >>> # not affected.
        >>> x = torch.tensor([1.0, 2.0, 0.0, 4.0, 5.0])
        >>> QF.mprod(x, span=2)
        tensor([nan,  2.,  0.,  0., 20.])

        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                   [5.0, 6.0, 7.0, 8.0]])
        >>> QF.mprod(x, span=2, dim=1)
        tensor([[nan,  2.,  6., 12.],
                [nan, 30., 42., 56.]])

    .. note::
        If a window contains any NaN value, the moving product for that
        window is NaN. Infinities follow IEEE 754 semantics; in particular,
        a window containing both ``0`` and ``inf`` results in NaN. For large
        spans, products of many terms may overflow or underflow; for
        strictly positive data in that regime, consider computing
        ``QF.msum(x.log(), span, dim).exp()`` instead.

    .. seealso::
        - :func:`msum`: Moving sum function (the additive counterpart).
        - :func:`nancumprod`: NaN-aware cumulative product function.
        - :func:`ma`: Moving average function.
    """
    return apply_for_axis(lambda x: _mprod(x, span), x, dim)
