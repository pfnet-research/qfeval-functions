import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_wma(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    weights = torch.arange(1, span + 1, dtype=x.dtype)
    for i in range(span - 1, x.shape[0]):
        result[i] = (x[i - span + 1 : i + 1] * weights).sum() / weights.sum()
    return result


def _pandas_wma(df: pd.DataFrame, span: int) -> np.ndarray:
    """Pandas rolling weighted-dot reference implementation."""
    weights = np.arange(1, span + 1)
    return (
        df.rolling(span)
        .apply(lambda v: np.dot(v, weights) / weights.sum(), raw=True)
        .to_numpy()
    )


def test_wma_pandas_reference() -> None:
    """Test weighted moving average against pandas rolling apply."""
    torch.manual_seed(0)
    a = torch.randn(60, 4, dtype=torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (1, 2, 3, 5, 8):
        np.testing.assert_allclose(
            QF.wma(a, span, dim=0).numpy(),
            _pandas_wma(df, span),
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            QF.wma(a, span, dim=1).numpy(),
            _pandas_wma(df.T, span).T,
            rtol=1e-10,
            atol=1e-12,
        )


def test_wma_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    The implementation splits the data into span-sized chunks, so this
    exercises all relative positions of windows and chunk boundaries,
    including data shorter than, equal to, and longer than the window.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 26):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            result = QF.wma(x, span, dim=0)
            expected = _reference_wma(x, span)
            np.testing.assert_allclose(
                result.numpy(), expected.numpy(), rtol=1e-10, atol=1e-12
            )


def test_wma_known_values() -> None:
    """Test weighted moving average with hand-computed values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.wma(x, 3, dim=0)
    # WMA[2] = (1*1 + 2*2 + 3*3) / 6 = 14 / 6
    # WMA[3] = (2*1 + 3*2 + 4*3) / 6 = 20 / 6
    # WMA[4] = (3*1 + 4*2 + 5*3) / 6 = 26 / 6
    expected = torch.tensor(
        [math.nan, math.nan, 14.0 / 6.0, 20.0 / 6.0, 26.0 / 6.0]
    )
    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), rtol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_wma_constant_input() -> None:
    """Test that a constant input yields the same constant."""
    x = torch.full((10,), 7.0)
    result = QF.wma(x, 4, dim=0)
    np.testing.assert_allclose(result[3:].numpy(), np.full(7, 7.0))
    assert torch.isnan(result[:3]).all()


def test_wma_leads_ma_for_increasing_data() -> None:
    """Test that WMA exceeds MA for increasing data (it weights recency)."""
    x = torch.arange(1.0, 21.0)
    wma_result = QF.wma(x, 5, dim=0)
    ma_result = QF.ma(x, 5, dim=0)
    mask = torch.isfinite(ma_result)
    assert mask.any()
    assert (wma_result[mask] > ma_result[mask]).all()


def test_wma_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it.

    The chunked implementation keeps all partial sums local to a chunk, so
    this sweeps a NaN through every position to verify that no window
    outside the NaN's reach is affected.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.wma(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.wma(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_wma_infinity_in_window() -> None:
    """Test that infinity makes exactly the containing windows non-finite.

    Per IEEE 754 evaluation of the weighted sum, such windows are ``inf``
    or NaN; windows not containing the infinity are unaffected.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.wma(x, span, dim=0)
        for value in (math.inf, -math.inf):
            for position in range(n):
                xp = x.clone()
                xp[position] = value
                result = QF.wma(xp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert not torch.isfinite(result[i])
                    else:
                        assert result[i] == clean[i]


def test_wma_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with a large offset relative to the variance.

    The result is compared with a float64 computation on the same (already
    quantized) input, so the tolerance covers only the error of the
    algorithm itself.
    """
    torch.manual_seed(0)
    x = (torch.randn(1000, dtype=torch.float64) + 1e6).to(torch.float32)

    result = QF.wma(x, 20, dim=0)
    expected = QF.wma(x.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    relative_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / expected[mask].abs()
    assert relative_error.max().item() < 1e-4


def test_wma_numerical_stability_drift() -> None:
    """Test float32 accuracy on a drifting series (random walk).

    Unlike a constant offset, a drift cannot be fixed by subtracting a
    global constant, so this checks that the computation is locally
    centered.
    """
    torch.manual_seed(1)
    steps = torch.randn(10000, dtype=torch.float64) * 0.01
    x = (steps.cumsum(dim=0) + 1000).to(torch.float32)

    result = QF.wma(x, 50, dim=0)
    expected = QF.wma(x.to(torch.float64), 50, dim=0)

    mask = torch.isfinite(expected)
    relative_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / expected[mask].abs()
    assert relative_error.max().item() < 1e-4


def test_wma_span_one_identity() -> None:
    """Test that span=1 returns the input values unchanged."""
    x = torch.tensor([3.0, -1.5, 0.25, 7.0])
    result = QF.wma(x, 1, dim=0)
    np.testing.assert_allclose(result.numpy(), x.numpy())


def test_wma_window_larger_than_data() -> None:
    """Test weighted moving average when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.wma(x, 5, dim=0)
    assert torch.isnan(result).all()


def test_wma_single_element() -> None:
    """Test weighted moving average with a single-element input."""
    x = torch.tensor([42.0])
    result = QF.wma(x, 1, dim=0)
    assert result.shape == x.shape
    assert result.item() == 42.0


def test_wma_2d_and_3d_dims() -> None:
    """Test weighted moving average on 2D/3D tensors per series."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 15, dtype=torch.float64)
    result = QF.wma(x, 4, dim=2)
    assert result.shape == x.shape
    for i in range(3):
        for j in range(4):
            expected = _reference_wma(x[i, j], 4)
            np.testing.assert_allclose(
                result[i, j].numpy(), expected.numpy(), rtol=1e-10, atol=1e-12
            )
    result0 = QF.wma(x, 2, dim=0)
    for j in range(4):
        for t in range(15):
            expected = _reference_wma(x[:, j, t], 2)
            np.testing.assert_allclose(
                result0[:, j, t].numpy(),
                expected.numpy(),
                rtol=1e-10,
                atol=1e-12,
            )


def test_wma_negative_dim() -> None:
    """Test weighted moving average with negative dimension indexing."""
    torch.manual_seed(0)
    x = torch.randn(5, 12)
    np.testing.assert_allclose(
        QF.wma(x, 3, dim=-1).numpy(), QF.wma(x, 3, dim=1).numpy()
    )
    x3 = torch.randn(2, 6, 4)
    np.testing.assert_allclose(
        QF.wma(x3, 2, dim=-2).numpy(), QF.wma(x3, 2, dim=1).numpy()
    )


def test_wma_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.wma(x, 3, dim=dim)
            assert_basic_properties(result, x)
