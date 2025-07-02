import typing

import torch


def rms(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the root mean square (RMS) along specified dimensions.

    The root mean square is calculated as the square root of the arithmetic
    mean of the squares of the values. It provides a measure of the magnitude
    of the values, particularly useful for measuring variability or the
    effective value of oscillating quantities.

    The RMS is computed as:

    .. math::
        \text{RMS} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}

    where :math:`n` is the number of elements along the specified dimension(s).

    Args:
        x (Tensor):
            The input tensor.
        dim (int or tuple of ints, optional):
            The dimension or dimensions along which to compute the RMS.
            If empty tuple (default), computes RMS over all elements.
        keepdim (bool, optional):
            Whether the output tensor retains the specified dimensions.
            Default is False.

    Returns:
        Tensor:
            The RMS values. If :attr:`keepdim` is True, the output tensor
            has the same number of dimensions as the input with the reduced
            dimensions having size 1.

    Example:
        >>> # Simple 1D example
        >>> x = torch.tensor([3.0, 4.0, 0.0, -3.0, -4.0])
        >>> QF.rms(x)
        tensor(3.1623)

        >>> # 2D example with different dimensions
        >>> x = torch.tensor([[1.0, -2.0, 3.0],
        ...                   [4.0, -5.0, 6.0]])
        >>> QF.rms(x, dim=1)  # RMS along rows
        tensor([2.1602, 5.0662])
        >>> QF.rms(x, dim=0)  # RMS along columns
        tensor([2.9155, 3.8079, 4.7434])

        >>> # Multi-dimensional reduction
        >>> x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
        ...                   [[5.0, 6.0], [7.0, 8.0]]])
        >>> QF.rms(x, dim=(1, 2))  # RMS over last two dimensions
        tensor([2.7386, 6.5955])

        >>> # Keep dimensions
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> QF.rms(x, dim=1, keepdim=True)
        tensor([[2.1602],
                [5.0662]])

        >>> # Compare with standard deviation and mean
        >>> data = torch.randn(1000)
        >>> rms_val = QF.rms(data)
        >>> std_val = data.std()
        >>> mean_val = data.mean()
        >>> # For zero-mean data: RMS ≈ std
        >>> # For general data: RMS² = mean² + std²

    See Also:
        :func:`torch.norm`: L2 norm (RMS times sqrt(n)).
        :func:`torch.std`: Standard deviation.
        :func:`torch.mean`: Arithmetic mean.

    .. note::
        The RMS is particularly important in quantitative finance for:

        - Volatility measurement as a scale-invariant risk metric
        - Portfolio risk assessment through return magnitude analysis
        - Signal processing in high-frequency trading systems
        - Error measurement in model validation and backtesting
        - Normalizing financial time series for comparison
        - Computing effective exposure in multi-asset portfolios

    .. note::
        In financial applications, RMS is often used to measure:

        - Effective volatility of returns (especially for zero-mean returns)
        - Magnitude of portfolio weights regardless of direction
        - Scale of price movements for risk scaling
        - Error magnitude in forecasting models
        - Effective size of positions in risk management

    .. note::
        For zero-mean data, RMS equals the standard deviation. For non-zero
        mean data, RMS will be larger than the standard deviation since
        RMS² = mean² + variance.

    .. warning::
        Unlike standard deviation, RMS is not translation-invariant. Adding
        a constant to all values will change the RMS, making it sensitive
        to the level of the data as well as its variability.
    """
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()
