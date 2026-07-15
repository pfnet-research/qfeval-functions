import typing

import torch


# NOTE: This uses the same default value as R's signif function uses.  The
# argument name follows numpy.round and DataFrame.round.
def signif(x: torch.Tensor, decimals: int = 6) -> torch.Tensor:
    r"""Rounds the numbers of the given tensor to the specified number of
    significant digits.

    Unlike :func:`torch.round`, which rounds to a fixed number of decimal
    places, this function keeps the given number of significant digits, so
    the rounding precision adapts to the magnitude of each element (e.g.,
    rounding to 3 significant digits maps 12345 to 12300 and 0.012345 to
    0.0123).  This is the same behavior as R's ``signif`` function, and the
    default number of digits follows it.  Zeros and non-finite values (NaN
    and infinities) are returned unchanged.

    Args:
        x (Tensor):
            The input tensor.
        decimals (int, optional):
            The number of significant digits to keep.  Default is 6, the
            same default as R's ``signif`` function.

    Returns:
        Tensor:
            A tensor of the same shape as the input, with each element
            rounded to the specified number of significant digits.

    Example:

        >>> x = torch.tensor([3.14159265, 2.71828183])
        >>> QF.signif(x, 3)
        tensor([3.1400, 2.7200])

        >>> # Zeros are kept as is, and the magnitude adapts per element.
        >>> x = torch.tensor([0.0, 100.0])
        >>> QF.signif(x, 3)
        tensor([  0., 100.])

    .. seealso::
        - ``torch.round``: Rounds to a number of decimal places instead of
          significant digits.
    """
    e = 10 ** (decimals - x.abs().log10().ceil())
    e = torch.where(e.isfinite() & e.ne(0.0), e, torch.tensor(1).to(e))
    return typing.cast(torch.Tensor, (x * e).round() / e)
