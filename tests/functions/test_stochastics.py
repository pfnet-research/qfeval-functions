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


def _pandas_stochastics(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    k_span: int,
    d_span: int,
    sd_span: int,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the stochastic oscillator of 1D series with pandas."""
    lowest = pd.Series(low.numpy()).rolling(k_span).min()
    highest = pd.Series(high.numpy()).rolling(k_span).max()
    k = (pd.Series(close.numpy()) - lowest) / (highest - lowest) * 100
    d = k.rolling(d_span).mean()
    slow_d = d.rolling(sd_span).mean()
    return k.to_numpy(), d.to_numpy(), slow_d.to_numpy()


@pytest.mark.parametrize("spans", [(14, 3, 3), (5, 3, 2)])
def test_stochastics_matches_pandas_reference(
    spans: typing.Tuple[int, int, int]
) -> None:
    """The oscillator matches the pandas rolling implementation on random
    OHLC data (which has no flat windows), along both dims 1 and 0."""
    k_span, d_span, sd_span = spans
    high, low, close = _make_ohlc(3, 60, seed=1)
    k, d, slow_d = QF.stochastics(
        high, low, close, k_span, d_span, sd_span, dim=1
    )
    for i in range(high.shape[0]):
        expected = _pandas_stochastics(
            high[i], low[i], close[i], k_span, d_span, sd_span
        )
        for actual, want in zip((k, d, slow_d), expected):
            np.testing.assert_allclose(
                actual[i].numpy(),
                want,
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
    # The same series along dim=0 yield the same values.
    transposed = QF.stochastics(
        high.t(), low.t(), close.t(), k_span, d_span, sd_span, dim=0
    )
    for actual, actual_t in zip((k, d, slow_d), transposed):
        np.testing.assert_allclose(
            actual_t.t().numpy(), actual.numpy(), equal_nan=True
        )


def test_stochastics_bounds_for_valid_ohlc() -> None:
    """All three outputs lie in [0, 100] for inputs satisfying the OHLC
    invariants."""
    high, low, close = _make_ohlc(4, 50, seed=2)
    outputs = QF.stochastics(high, low, close, 5, 3, 3, dim=1)
    for out in outputs:
        valid = out[~torch.isnan(out)]
        assert valid.numel() > 0
        assert bool((valid >= 0).all())
        assert bool((valid <= 100).all())


def test_stochastics_close_at_extremes() -> None:
    """%K is 100 when the close equals the highest high and 0 when it
    equals the lowest low."""
    high = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    low = high - 1.0
    k, _, _ = QF.stochastics(high, low, high, k_span=3, d_span=2, sd_span=2)
    assert torch.isnan(k[:2]).all()
    assert bool((k[2:] == 100).all())

    low2 = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    high2 = low2 + 1.0
    k2, _, _ = QF.stochastics(high2, low2, low2, k_span=3, d_span=2, sd_span=2)
    assert torch.isnan(k2[:2]).all()
    assert bool((k2[2:] == 0).all())


def test_stochastics_flat_window_returns_neutral_50() -> None:
    """A window with no price variation (0/0) yields the neutral value
    50, and %D and Slow %D become 50 once their windows are all 50.

    NOTE: This intentionally differs from TA-Lib, which returns 0 for
    such windows, and from the raw pandas formula, which yields NaN.
    """
    c = torch.full((10,), 5.0)
    k, d, slow_d = QF.stochastics(c, c, c, k_span=3, d_span=2, sd_span=2)
    assert torch.isnan(k[:2]).all()
    assert bool((k[2:] == 50).all())
    assert torch.isnan(d[:3]).all()
    assert bool((d[3:] == 50).all())
    assert torch.isnan(slow_d[:4]).all()
    assert bool((slow_d[4:] == 50).all())
    # The raw pandas formula yields NaN (0 / 0) where ours is 50.
    pandas_k, _, _ = _pandas_stochastics(c, c, c, 3, 2, 2)
    assert np.isnan(pandas_k[2:]).all()


def test_stochastics_partially_flat_series() -> None:
    """Only the windows with no price variation get the neutral value;
    other windows use the raw formula."""
    prices = torch.tensor([1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 3.0])
    k, _, _ = QF.stochastics(
        prices, prices, prices, k_span=3, d_span=2, sd_span=2
    )
    expected = np.array([math.nan, math.nan, 100.0, 100.0, 50.0, 50.0, 0.0])
    np.testing.assert_allclose(k.numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("spans", [(14, 3, 3), (5, 3, 2), (2, 4, 3), (1, 2, 2)])
def test_stochastics_nan_warmup_counts(
    spans: typing.Tuple[int, int, int]
) -> None:
    """The NaN warm-up lengths of %K, %D, and Slow %D are exactly
    ``k_span - 1``, ``k_span + d_span - 2``, and
    ``k_span + d_span + sd_span - 3``."""
    k_span, d_span, sd_span = spans
    high, low, close = _make_ohlc(1, 40, seed=3)
    k, d, slow_d = QF.stochastics(
        high[0], low[0], close[0], k_span, d_span, sd_span
    )
    for out, warmup in (
        (k, k_span - 1),
        (d, k_span + d_span - 2),
        (slow_d, k_span + d_span + sd_span - 3),
    ):
        assert torch.isnan(out[:warmup]).all()
        assert torch.isfinite(out[warmup:]).all()


def test_stochastics_k_span_one() -> None:
    """With ``k_span=1``, %K is the position of the close within the
    current bar's range and has no warm-up NaNs."""
    high, low, close = _make_ohlc(1, 20, seed=4)
    k, d, _ = QF.stochastics(
        high[0], low[0], close[0], k_span=1, d_span=3, sd_span=3
    )
    expected = (close[0] - low[0]) / (high[0] - low[0]) * 100
    assert torch.isfinite(k).all()
    np.testing.assert_allclose(k.numpy(), expected.numpy(), rtol=1e-12)
    np.testing.assert_allclose(d.numpy(), QF.ma(k, 3).numpy(), equal_nan=True)


def test_stochastics_nan_input_propagates() -> None:
    """A NaN high makes %K NaN exactly while the NaN stays in the
    trailing window, and the NaN region grows for %D and Slow %D."""
    high, low, close = _make_ohlc(1, 30, seed=5)
    hi, lo, cl = high[0].clone(), low[0], close[0]
    hi[10] = math.nan
    k, d, slow_d = QF.stochastics(hi, lo, cl, k_span=3, d_span=2, sd_span=2)
    # %K windows containing the NaN high: positions 10-12.
    assert torch.isfinite(k[2:10]).all()
    assert torch.isnan(k[10:13]).all()
    assert torch.isfinite(k[13:]).all()
    # %D windows containing a NaN %K: positions 10-13.
    assert torch.isfinite(d[3:10]).all()
    assert torch.isnan(d[10:14]).all()
    assert torch.isfinite(d[14:]).all()
    # Slow %D windows containing a NaN %D: positions 10-14.
    assert torch.isfinite(slow_d[4:10]).all()
    assert torch.isnan(slow_d[10:15]).all()
    assert torch.isfinite(slow_d[15:]).all()


def test_stochastics_infinite_range_stays_nan() -> None:
    """A window whose highest high and lowest low are both infinite has a
    NaN range (inf - inf), which stays NaN instead of becoming 50."""
    high = torch.tensor([1.0, math.inf, 2.0])
    low = torch.tensor([1.0, math.inf, 1.0])
    close = torch.tensor([1.0, 1.0, 1.75])
    k, _, _ = QF.stochastics(high, low, close, k_span=1, d_span=1, sd_span=1)
    # A zero-range window still yields the neutral value 50.
    assert k[0].item() == 50.0
    # An infinite range (inf - inf) stays NaN.
    assert math.isnan(k[1].item())
    assert k[2].item() == pytest.approx(75.0)


def test_stochastics_dim_argument_and_3d() -> None:
    """3D inputs are processed independently along the target dimension,
    selected by positive or negative ``dim`` values."""
    high, low, close = _make_ohlc(6, 30, seed=6)
    base = QF.stochastics(high, low, close, 5, 3, 2, dim=1)
    base_neg = QF.stochastics(high, low, close, 5, 3, 2, dim=-1)
    for out, out_neg in zip(base, base_neg):
        np.testing.assert_allclose(out_neg.numpy(), out.numpy(), equal_nan=True)
    h3 = high.reshape(2, 3, 30)
    l3 = low.reshape(2, 3, 30)
    c3 = close.reshape(2, 3, 30)
    for dim in [2, -1]:
        outputs = QF.stochastics(h3, l3, c3, 5, 3, 2, dim=dim)
        for out, expected in zip(outputs, base):
            assert out.shape == h3.shape
            np.testing.assert_allclose(
                out.reshape(6, 30).numpy(),
                expected.numpy(),
                equal_nan=True,
            )
    permuted = QF.stochastics(
        h3.permute(2, 0, 1),
        l3.permute(2, 0, 1),
        c3.permute(2, 0, 1),
        5,
        3,
        2,
        dim=0,
    )
    for out, expected in zip(permuted, base):
        np.testing.assert_allclose(
            out.permute(1, 2, 0).reshape(6, 30).numpy(),
            expected.numpy(),
            equal_nan=True,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stochastics_dtype_and_device_preservation(
    dtype: torch.dtype,
) -> None:
    """All outputs (including the NaN warm-up regions) keep the input
    dtype and device."""
    high, low, close = _make_ohlc(2, 20, seed=7, dtype=dtype)
    outputs = QF.stochastics(high, low, close, 5, 3, 3, dim=1)
    for out in outputs:
        assert_basic_properties(out, close)


def test_stochastics_broadcasting() -> None:
    """The inputs are broadcast against each other before computing the
    oscillator."""
    high, low, close = _make_ohlc(3, 25, seed=8)
    outputs = QF.stochastics(high[0], low[0], close, 4, 2, 2, dim=-1)
    for out in outputs:
        assert out.shape == close.shape
    for i in range(close.shape[0]):
        expected = QF.stochastics(high[0], low[0], close[i], 4, 2, 2)
        for out, want in zip(outputs, expected):
            np.testing.assert_allclose(
                out[i].numpy(), want.numpy(), equal_nan=True
            )


def test_stochastics_invalid_span_raises_value_error() -> None:
    """``k_span``, ``d_span``, and ``sd_span`` must be positive
    integers."""
    high, low, close = _make_ohlc(1, 10, seed=9)
    hi, lo, cl = high[0], low[0], close[0]
    with pytest.raises(ValueError, match="k_span must be a positive"):
        QF.stochastics(hi, lo, cl, k_span=0)
    with pytest.raises(ValueError, match="d_span must be a positive"):
        QF.stochastics(hi, lo, cl, d_span=-1)
    with pytest.raises(ValueError, match="sd_span must be a positive"):
        QF.stochastics(hi, lo, cl, sd_span=0)


def test_stochastics_non_integer_span_raises_type_error() -> None:
    """Non-integer span values are rejected; ``bool`` is a subclass of
    ``int`` and must not be silently accepted as 1."""
    high, low, close = _make_ohlc(1, 10, seed=9)
    hi, lo, cl = high[0], low[0], close[0]
    with pytest.raises(TypeError, match="k_span must be an integer"):
        QF.stochastics(hi, lo, cl, k_span=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="k_span must be an integer"):
        QF.stochastics(hi, lo, cl, k_span=True)
    with pytest.raises(TypeError, match="d_span must be an integer"):
        QF.stochastics(hi, lo, cl, d_span=2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sd_span must be an integer"):
        QF.stochastics(hi, lo, cl, sd_span=math.nan)  # type: ignore[arg-type]


def test_stochastics_non_floating_point_input_raises_type_error() -> None:
    """Non-floating-point inputs are rejected for each argument."""
    valid = torch.ones(10)
    invalid = torch.ones(10, dtype=torch.int32)
    with pytest.raises(TypeError, match="floating point"):
        QF.stochastics(invalid, valid, valid)
    with pytest.raises(TypeError, match="floating point"):
        QF.stochastics(valid, invalid, valid)
    with pytest.raises(TypeError, match="floating point"):
        QF.stochastics(valid, valid, invalid)
