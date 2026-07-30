import math
import typing

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _make_ohlc(
    batch: int,
    length: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build synthetic OHLC data satisfying ``low <= close <= high``,
    where each series is a random walk along the last dimension."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, length)
    base = torch.randn(shape, generator=generator, dtype=dtype).cumsum(dim=1)
    high = base + torch.rand(shape, generator=generator, dtype=dtype)
    low = base - torch.rand(shape, generator=generator, dtype=dtype)
    weight = torch.rand(shape, generator=generator, dtype=dtype)
    close = low + weight * (high - low)
    return high, low, close


def _naive_true_range(
    high: torch.Tensor, low: torch.Tensor, close: torch.Tensor
) -> torch.Tensor:
    """Compute the true range of 1D OHLC series with an explicit loop."""
    tr = torch.full_like(high, math.nan)
    for i in range(1, high.shape[0]):
        tr[i] = torch.max(
            torch.stack(
                (
                    high[i] - low[i],
                    (high[i] - close[i - 1]).abs(),
                    (low[i] - close[i - 1]).abs(),
                )
            )
        )
    return tr


def _naive_atr(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    span: int,
    use_sma: bool,
) -> torch.Tensor:
    """Compute the ATR of 1D OHLC series with explicit loops implementing
    the true range plus Wilder's recursion (or its SMA variant)."""
    tr = _naive_true_range(high, low, close)
    n = tr.shape[0]
    result = torch.full_like(tr, math.nan)
    if use_sma:
        for i in range(span - 1, n):
            result[i] = tr[i - span + 1 : i + 1].mean()
        return result
    if n <= span:
        return result
    result[span] = tr[1 : span + 1].mean()
    for i in range(span + 1, n):
        result[i] = (result[i - 1] * (span - 1) + tr[i]) / span
    return result


@pytest.mark.parametrize("use_sma", [False, True])
@pytest.mark.parametrize("span", [1, 2, 5, 14])
def test_atr_matches_naive_reference(span: int, use_sma: bool) -> None:
    """ATR matches a naive loop implementation of the true range plus
    Wilder's recursion (or the SMA variant) on random OHLC data."""
    high, low, close = _make_ohlc(3, 60, seed=1)
    result = QF.atr(high, low, close, span=span, use_sma=use_sma)
    for i in range(high.shape[0]):
        expected = _naive_atr(high[i], low[i], close[i], span, use_sma)
        np.testing.assert_allclose(
            result[i].numpy(), expected.numpy(), rtol=1e-10, equal_nan=True
        )


def test_atr_span_one_equals_true_range() -> None:
    """With ``span=1``, the Wilder seed is TR[1] and the recursion has
    alpha 1, so the ATR equals the true range (one warm-up NaN)."""
    high, low, close = _make_ohlc(1, 20, seed=2)
    result = QF.atr(high[0], low[0], close[0], span=1)
    expected = _naive_true_range(high[0], low[0], close[0])
    np.testing.assert_allclose(
        result.numpy(), expected.numpy(), rtol=1e-12, equal_nan=True
    )


def test_atr_hand_computed_tiny_case() -> None:
    """Verify a small case against explicitly hand-computed numbers."""
    high = torch.tensor([10.0, 12.0, 11.0, 14.0])
    low = torch.tensor([9.0, 10.0, 9.0, 11.0])
    close = torch.tensor([9.5, 11.0, 10.0, 13.0])
    # TR[1] = max(12 - 10, |12 - 9.5|, |10 - 9.5|) = 2.5
    # TR[2] = max(11 - 9, |11 - 11|, |9 - 11|) = 2.0
    # TR[3] = max(14 - 11, |14 - 10|, |11 - 10|) = 4.0
    # Wilder: ATR[2] = (2.5 + 2.0) / 2 = 2.25
    #         ATR[3] = (2.25 * (2 - 1) + 4.0) / 2 = 3.125
    result = QF.atr(high, low, close, span=2)
    np.testing.assert_allclose(
        result.numpy(),
        np.array([math.nan, math.nan, 2.25, 3.125]),
        equal_nan=True,
    )
    # SMA: ATR[2] = (2.5 + 2.0) / 2 = 2.25
    #      ATR[3] = (2.0 + 4.0) / 2 = 3.0
    result_sma = QF.atr(high, low, close, span=2, use_sma=True)
    np.testing.assert_allclose(
        result_sma.numpy(),
        np.array([math.nan, math.nan, 2.25, 3.0]),
        equal_nan=True,
    )


def test_atr_true_range_gap_up() -> None:
    """When a bar gaps above the previous close (its low exceeds it), the
    true range is ``high - previous close``."""
    high = torch.tensor([10.5, 13.0])
    low = torch.tensor([9.5, 12.0])
    close = torch.tensor([10.0, 12.5])
    result = QF.atr(high, low, close, span=1)
    assert result[1].item() == pytest.approx(3.0)  # 13.0 - 10.0


def test_atr_true_range_gap_down() -> None:
    """When a bar gaps below the previous close (its high is under it),
    the true range is ``previous close - low``."""
    high = torch.tensor([10.5, 8.0])
    low = torch.tensor([9.5, 7.0])
    close = torch.tensor([10.0, 7.5])
    result = QF.atr(high, low, close, span=1)
    assert result[1].item() == pytest.approx(3.0)  # 10.0 - 7.0


def test_atr_true_range_inside_bar() -> None:
    """When the previous close lies inside the bar's range, the true
    range is simply ``high - low``."""
    high = torch.tensor([10.5, 11.0])
    low = torch.tensor([9.5, 9.5])
    close = torch.tensor([10.0, 10.5])
    result = QF.atr(high, low, close, span=1)
    assert result[1].item() == pytest.approx(1.5)  # 11.0 - 9.5


@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_nonnegative_for_valid_ohlc(use_sma: bool) -> None:
    """The ATR is non-negative for inputs satisfying the OHLC
    invariants."""
    high, low, close = _make_ohlc(4, 50, seed=3)
    result = QF.atr(high, low, close, span=5, use_sma=use_sma)
    valid = result[~torch.isnan(result)]
    assert valid.numel() > 0
    assert bool((valid >= 0).all())


@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_constant_prices_yield_zero(use_sma: bool) -> None:
    """Constant prices have a zero true range, so the ATR is zero after
    the warm-up period."""
    c = torch.full((20,), 7.5)
    result = QF.atr(c, c, c, span=5, use_sma=use_sma)
    assert torch.isnan(result[:5]).all()
    assert bool((result[5:] == 0).all())


def test_atr_sma_matches_pandas_rolling_mean() -> None:
    """With ``use_sma=True``, the ATR equals the pandas rolling mean of
    the true range (whose first element is NaN)."""
    span = 7
    high, low, close = _make_ohlc(1, 40, seed=4)
    result = QF.atr(high[0], low[0], close[0], span=span, use_sma=True)
    hi = pd.Series(high[0].numpy())
    lo = pd.Series(low[0].numpy())
    cl = pd.Series(close[0].numpy())
    tr = pd.concat(
        ((hi - lo), (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()),
        axis=1,
    ).max(axis=1, skipna=False)
    expected = tr.rolling(span).mean()
    np.testing.assert_allclose(result.numpy(), expected.to_numpy(), rtol=1e-10)


def test_atr_nan_bar_poisons_wilder_smoothing() -> None:
    """A NaN bar makes two adjacent true ranges NaN, and Wilder's
    recursion keeps all subsequent outputs NaN."""
    high, low, close = _make_ohlc(1, 30, seed=5)
    hi, lo, cl = high[0].clone(), low[0].clone(), close[0].clone()
    hi[15] = lo[15] = cl[15] = math.nan
    result = QF.atr(hi, lo, cl, span=5)
    assert torch.isnan(result[:5]).all()
    assert torch.isfinite(result[5:15]).all()
    assert torch.isnan(result[15:]).all()


def test_atr_nan_bar_recovers_with_sma() -> None:
    """With ``use_sma=True``, only the outputs whose windows contain the
    NaN true ranges are NaN, so the output recovers afterwards."""
    high, low, close = _make_ohlc(1, 30, seed=5)
    hi, lo, cl = high[0].clone(), low[0].clone(), close[0].clone()
    hi[15] = lo[15] = cl[15] = math.nan
    result = QF.atr(hi, lo, cl, span=5, use_sma=True)
    # TR[15] and TR[16] are NaN, affecting the windows ending at 15-20.
    assert torch.isnan(result[:5]).all()
    assert torch.isfinite(result[5:15]).all()
    assert torch.isnan(result[15:21]).all()
    assert torch.isfinite(result[21:]).all()


def test_atr_nan_close_only_poisons_next_true_range() -> None:
    """A NaN close leaves its own bar's true range valid and makes only
    the next bar's true range NaN."""
    high, low, close = _make_ohlc(1, 30, seed=6)
    hi, lo, cl = high[0], low[0], close[0].clone()
    cl[15] = math.nan
    result = QF.atr(hi, lo, cl, span=5)
    assert torch.isfinite(result[5:16]).all()
    assert torch.isnan(result[16:]).all()
    result_sma = QF.atr(hi, lo, cl, span=5, use_sma=True)
    assert torch.isfinite(result_sma[5:16]).all()
    assert torch.isnan(result_sma[16:21]).all()
    assert torch.isfinite(result_sma[21:]).all()


@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_short_series_returns_all_nan(use_sma: bool) -> None:
    """A series with ``span`` or fewer elements yields an all-NaN output
    of the input shape, because one ATR value needs ``span + 1`` bars."""
    span = 5
    for length in [0, 1, 3, 5]:
        high, low, close = _make_ohlc(1, length, seed=7)
        result = QF.atr(high[0], low[0], close[0], span=span, use_sma=use_sma)
        assert result.shape == (length,)
        assert torch.isnan(result).all()

    # `span + 1` bars are exactly enough for one ATR value.
    high, low, close = _make_ohlc(1, span + 1, seed=7)
    result = QF.atr(high[0], low[0], close[0], span=span, use_sma=use_sma)
    assert torch.isnan(result[:span]).all()
    assert torch.isfinite(result[span])


def test_atr_short_series_multi_dimensional() -> None:
    """Short series along the target dimension keep the input shape even
    for multi-dimensional inputs and negative dimensions."""
    high, low, close = _make_ohlc(4, 3, seed=8)
    for dim in [1, -1]:
        result = QF.atr(high, low, close, span=5, dim=dim)
        assert result.shape == high.shape
        assert torch.isnan(result).all()


def test_atr_broadcasting() -> None:
    """The inputs are broadcast against each other before computing the
    true range."""
    high, low, close = _make_ohlc(3, 25, seed=9)
    result = QF.atr(high[0], low[0], close, span=4)
    assert result.shape == close.shape
    for i in range(close.shape[0]):
        expected = QF.atr(high[0], low[0], close[i], span=4)
        np.testing.assert_allclose(
            result[i].numpy(), expected.numpy(), equal_nan=True
        )


@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_dim_argument(use_sma: bool) -> None:
    """Positive and negative ``dim`` values select the same dimension."""
    high, low, close = _make_ohlc(4, 30, seed=10)
    base = QF.atr(high, low, close, span=5, use_sma=use_sma)
    np.testing.assert_allclose(
        QF.atr(high, low, close, span=5, use_sma=use_sma, dim=1).numpy(),
        base.numpy(),
        equal_nan=True,
    )
    result_dim0 = QF.atr(
        high.t(), low.t(), close.t(), span=5, use_sma=use_sma, dim=0
    )
    np.testing.assert_allclose(
        result_dim0.t().numpy(), base.numpy(), equal_nan=True
    )


@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_3d_tensors(use_sma: bool) -> None:
    """3D inputs are processed independently along the target dimension,
    including a middle dimension."""
    high, low, close = _make_ohlc(6, 30, seed=11)
    base = QF.atr(high, low, close, span=5, use_sma=use_sma)
    h3 = high.reshape(2, 3, 30)
    l3 = low.reshape(2, 3, 30)
    c3 = close.reshape(2, 3, 30)
    for dim in [2, -1]:
        result = QF.atr(h3, l3, c3, span=5, use_sma=use_sma, dim=dim)
        assert result.shape == h3.shape
        np.testing.assert_allclose(
            result.reshape(6, 30).numpy(), base.numpy(), equal_nan=True
        )
    result_dim1 = QF.atr(
        h3.transpose(1, 2),
        l3.transpose(1, 2),
        c3.transpose(1, 2),
        span=5,
        use_sma=use_sma,
        dim=1,
    )
    np.testing.assert_allclose(
        result_dim1.transpose(1, 2).reshape(6, 30).numpy(),
        base.numpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_sma", [False, True])
def test_atr_dtype_and_device_preservation(
    dtype: torch.dtype, use_sma: bool
) -> None:
    """The output (including the NaN warm-up region) keeps the input
    dtype and device."""
    high, low, close = _make_ohlc(2, 20, seed=12, dtype=dtype)
    result = QF.atr(high, low, close, span=3, use_sma=use_sma)
    assert_basic_properties(result, close)

    # Short series (early return path) also keeps dtype/device.
    result_short = QF.atr(
        high[:, :3], low[:, :3], close[:, :3], span=5, use_sma=use_sma
    )
    assert_basic_properties(result_short, close[:, :3])


def test_atr_invalid_span_raises_value_error() -> None:
    """``span`` must be a positive integer."""
    high, low, close = _make_ohlc(1, 10, seed=13)
    for span in [0, -1]:
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.atr(high[0], low[0], close[0], span=span)


def test_atr_non_integer_span_raises_type_error() -> None:
    """Non-integer ``span`` values are rejected; ``bool`` is a subclass
    of ``int`` and must not be silently accepted as 1."""
    high, low, close = _make_ohlc(1, 10, seed=13)
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.atr(high[0], low[0], close[0], span=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.atr(high[0], low[0], close[0], span=math.nan)  # type: ignore[arg-type] # NOQA
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.atr(high[0], low[0], close[0], span=True)


def test_atr_non_floating_point_input_raises_type_error() -> None:
    """Non-floating-point inputs are rejected for each argument."""
    valid = torch.ones(10)
    invalid = torch.ones(10, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating point"):
        QF.atr(invalid, valid, valid)
    with pytest.raises(TypeError, match="floating point"):
        QF.atr(valid, invalid, valid)
    with pytest.raises(TypeError, match="floating point"):
        QF.atr(valid, valid, invalid)
