import torch

from .nanshift import nanshift


def nandiff(x: torch.Tensor, shift: int = 1, dim: int = -1) -> torch.Tensor:
    r"""Compute differences between valid values, skipping NaN values.

    This function subtracts the ``shift``-th previous valid (non-NaN)
    value from each valid value along the specified dimension, leaving
    NaN values in place.  Writing the indices of the valid elements along
    ``dim`` as :math:`v_0, v_1, \dots`, the result is:

    .. math::
        \text{nandiff}(x)[v_k] = x[v_k] - x[v_{k - \text{shift}}]

    and NaN elsewhere.  The first ``shift`` valid positions become NaN
    because they have no ``shift``-th previous valid value.  A negative
    ``shift`` computes differences against the following valid values,
    and ``shift=0`` yields zeros at valid positions.

    This is useful for computing differences (e.g., price changes) of
    time series with missing values, where a plain difference such as
    ``x.diff()`` would produce a NaN pair around every gap.  It is
    equivalent to ``x - QF.nanshift(x, shift, dim)`` and, on NaN-free
    floating-point input, to ``x - QF.shift(x, shift, dim)``.  In pandas
    terms, it matches ``s.dropna().diff(shift)`` realigned to the
    original index.

    Args:
        x (Tensor):
            The input tensor.  NaN values are skipped and remain NaN in
            the result.  Integer tensors have no NaN values to skip, so
            this behaves like ``x - QF.shift(x, shift, dim)`` except that
            vacated positions are filled with ``0`` by :func:`nanshift`,
            i.e., the first ``shift`` elements are returned as is.
        shift (int, optional):
            Number of valid positions to look back. Positive values
            compute differences against previous valid values, negative
            values against following valid values. Default is 1.
        dim (int, optional):
            The dimension along which to compute differences.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the
            differences between valid values.  NaN input positions and
            valid positions without a ``shift``-th previous (or
            following) valid value are NaN.

    Raises:
        RuntimeError: If ``x`` is a boolean tensor, because PyTorch does
            not support subtraction of boolean tensors.

    Example:

        >>> x = torch.tensor([1.0, nan, 3.0, 4.0, nan, 6.0])
        >>> QF.nandiff(x)
        tensor([nan, nan, 2., 1., nan, 2.])

        >>> # Negative shift: difference against the next valid value.
        >>> QF.nandiff(x, shift=-1)
        tensor([-2., nan, -1., -2., nan, nan])

        >>> # 2D example with dim=0
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [nan, 5.0, 6.0]])
        >>> QF.nandiff(x, shift=1, dim=0)
        tensor([[nan, nan, nan],
                [nan, nan, 3.]])

    .. note::
        Infinite values are treated as valid values and follow IEEE 754
        arithmetic: the difference between an infinite and a finite
        value is ±inf, and the difference between two infinities of the
        same sign is NaN.

    .. seealso::
        - :func:`nanshift`: Shift function that skips NaN values.
        - :func:`shift`: Standard shift function without NaN handling.
        - :func:`ffill`: Forward fill missing values.
    """
    return x - nanshift(x, shift, dim)
