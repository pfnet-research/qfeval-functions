import typing

import torch

from .nanvar import nanvar


def nanstd(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    unbiased: bool = True,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the standard deviation of a tensor, ignoring NaN values.

    This function calculates the standard deviation of tensor elements
    along the specified dimension(s) while excluding NaN values, as the
    square root of :func:`nanvar`.  The standard deviation can be computed
    using either the unbiased estimator (dividing by ``N-1``) or the
    biased estimator (dividing by ``N``), where ``N`` is the number of
    non-NaN elements.

    The NaN-aware standard deviation is computed as:

    .. math::
        \text{nanstd}(X) = \sqrt{\frac{1}{N_{\text{valid}} - \delta}
            \sum_{i \text{ valid}} (X_i - \bar{X})^2}

    where :math:`\bar{X}` is the mean of valid (non-NaN) values,
    :math:`N_{\text{valid}}` is the number of valid values, and
    :math:`\delta` is 1 if :attr:`unbiased` is ``True`` and 0 otherwise.

    Args:
        x (Tensor):
            The input tensor.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the standard
            deviation. If None (default), the standard deviation is
            computed over all elements. Can be a single dimension or
            multiple dimensions.
        unbiased (bool, optional):
            If ``True`` (default), uses Bessel's correction and divides by
            ``N-1`` where ``N`` is the number of non-NaN elements. If
            ``False``, divides by ``N``.
        keepdim (bool, optional):
            If ``True``, the output tensor has the same number of
            dimensions as the input, with the reduced dimensions having
            size 1. If ``False`` (default), the reduced dimensions are
            removed.

    Returns:
        Tensor:
            The standard deviation of non-NaN elements. The shape depends
            on the :attr:`dim` and :attr:`keepdim` parameters.

    Example:

        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanstd(x)
        tensor(1.8257)

        >>> # With biased estimator
        >>> QF.nanstd(x, unbiased=False)
        tensor(1.5811)

        >>> # 2D tensor with dimension specification
        >>> x = torch.tensor([[1.0, 2.0, nan],
        ...                   [4.0, nan, 6.0]])
        >>> QF.nanstd(x, dim=1)
        tensor([0.7071, 1.4142])

        >>> # Keep dimensions
        >>> QF.nanstd(x, dim=1, keepdim=True)
        tensor([[0.7071],
                [1.4142]])

    .. note::
        With :attr:`unbiased` set to ``True``, slices with fewer than two
        valid values yield NaN because Bessel's correction divides by
        ``N-1``. With :attr:`unbiased` set to ``False``, a slice with a
        single valid value yields 0, and only all-NaN slices yield NaN.

    .. seealso::
        - :func:`nanvar`: NaN-aware variance function.
        - :func:`nanmean`: NaN-aware mean function.
        - :func:`mstd`: Moving standard deviation function.
        - ``torch.std``: Standard deviation function (NaN propagates).
    """
    result: torch.Tensor = (
        nanvar(x, dim=dim, unbiased=unbiased, keepdim=keepdim) ** 0.5
    )
    return result
