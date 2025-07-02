import typing

import torch


# NOTE: This uses the same default value as R's signif function uses.  The
# argument name follows numpy.round and DataFrame.round.
def signif(x: torch.Tensor, decimals: int = 6) -> torch.Tensor:
    r"""Round tensor values to a specified number of significant digits.

    This function rounds the elements of the input tensor to the specified
    number of significant digits, similar to R's ``signif()`` function.
    Significant digits are counted from the first non-zero digit, providing
    a more meaningful representation for values across different orders of
    magnitude compared to decimal place rounding.

    The rounding is performed using the formula:

    .. math::
        \text{signif}(x, n) = \text{round}\left(x \times 10^{n - \lceil \log_{10}|x| \rceil}\right) \times 10^{\lceil \log_{10}|x| \rceil - n}

    where :math:`n` is the number of significant digits.

    Args:
        x (Tensor):
            The input tensor containing values to be rounded.
        decimals (int, optional):
            The number of significant digits to preserve. Must be positive.
            Default is 6, following R's ``signif()`` function convention.

    Returns:
        Tensor:
            A tensor with the same shape as the input, containing values
            rounded to the specified number of significant digits.

    Example:
        >>> # Basic significant digit rounding
        >>> x = torch.tensor([1.23456, 123.456, 0.0012345])
        >>> signif(x, decimals=3)
        tensor([    1.2300,   123.0000,     0.0012])

        >>> # Different precision levels
        >>> values = torch.tensor([3.14159, 2.71828, 1.41421])
        >>> signif(values, decimals=2)
        tensor([3.1000, 2.7000, 1.4000])
        >>> signif(values, decimals=4)
        tensor([3.1420, 2.7180, 1.4140])

        >>> # Large and small numbers
        >>> mixed = torch.tensor([1234567.89, 0.000123456, 98.7654])
        >>> signif(mixed, decimals=3)
        tensor([1230000.0000,     0.0001,    98.8000])

        >>> # Multi-dimensional arrays
        >>> matrix = torch.tensor([[12.345, 67.890],
        ...                        [0.012345, 98765.4]])
        >>> signif(matrix, decimals=2)
        tensor([[   12.0000,    68.0000],
                [    0.0120, 98999.9922]])

        >>> # Edge cases with zeros and very small numbers
        >>> edge_cases = torch.tensor([0.0, 1e-10, 1e10])
        >>> signif(edge_cases, decimals=3)
        tensor([    0.0000,     0.0000, 10000000000.0000])

        >>> # Financial data example
        >>> prices = torch.tensor([123.456, 12.3456, 1.23456, 0.123456])
        >>> signif(prices, decimals=4)
        tensor([123.5000,  12.3500,   1.2350,   0.1235])

    .. seealso::
        - :func:`torch.round`: Round to nearest integer.
        - :func:`torch.ceil`: Round up to nearest integer.
        - :func:`torch.floor`: Round down to nearest integer.

    .. note::
        This function is particularly useful in quantitative finance for:

        - Standardizing numerical precision across different assets
        - Reporting financial results with appropriate precision
        - Data preprocessing for machine learning models
        - Risk management calculations requiring consistent precision
        - Regulatory reporting with specified precision requirements

    .. note::
        In scientific computing and financial applications, significant digit
        rounding is often preferred over decimal place rounding because:

        - It maintains relative precision across different scales
        - It's more appropriate for percentage-based calculations
        - It provides consistent precision for values spanning multiple orders of magnitude
        - It aligns with significant figure conventions in scientific notation

    .. note::
        The function handles special cases gracefully:

        - Zero values remain unchanged
        - Infinite values are preserved
        - NaN values are preserved
        - Very large or very small finite values are rounded appropriately

    .. warning::
        Be aware that floating-point precision limitations may affect the
        exact representation of rounded values, especially for very large
        or very small numbers.
    """
    e = 10 ** (decimals - x.abs().log10().ceil())
    e = torch.where(e.isfinite() & e.ne(0.0), e, torch.tensor(1).to(e))
    return typing.cast(torch.Tensor, (x * e).round() / e)
