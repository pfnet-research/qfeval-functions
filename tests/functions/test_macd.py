import typing

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _pandas_macd(
    x: "np.typing.NDArray[np.float64]",
    fast_span: int,
    slow_span: int,
    signal_span: int,
) -> typing.Tuple[
    "np.typing.NDArray[np.float64]",
    "np.typing.NDArray[np.float64]",
    "np.typing.NDArray[np.float64]",
]:
    """Reference MACD based on pandas ewm(span=n, adjust=True)."""
    s = pd.Series(x)
    macd_line = (
        s.ewm(span=fast_span, adjust=True).mean()
        - s.ewm(span=slow_span, adjust=True).mean()
    )
    signal_line = macd_line.ewm(span=signal_span, adjust=True).mean()
    histogram = macd_line - signal_line
    return (
        macd_line.to_numpy(),
        signal_line.to_numpy(),
        histogram.to_numpy(),
    )


def _assert_macd_matches_pandas(
    x: torch.Tensor,
    combos: typing.List[typing.Tuple[int, int, int]],
) -> None:
    for fast_span, slow_span, signal_span in combos:
        macd_line, signal_line, histogram = QF.macd(
            x, fast_span, slow_span, signal_span
        )
        expected = _pandas_macd(x.numpy(), fast_span, slow_span, signal_span)
        np.testing.assert_allclose(
            macd_line.numpy(), expected[0], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            signal_line.numpy(), expected[1], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            histogram.numpy(), expected[2], rtol=1e-10, atol=1e-12
        )


def test_macd_pandas_reference() -> None:
    """MACD matches pandas ewm-based references for several span combos,
    including the default (12, 26, 9).

    NOTE: :func:`QF.ema` truncates weights below its internal ``1e-8``
    decay cutoff, so exact agreement with pandas requires the series to
    fit in the exactly-computed window: spans >= 7 cover 128 elements,
    while spans >= 2 cover 32 elements.
    """
    torch.manual_seed(42)
    x = torch.cumsum(torch.randn(100, dtype=torch.float64), dim=0) + 100.0
    _assert_macd_matches_pandas(x, [(12, 26, 9), (8, 17, 9), (7, 30, 10)])
    _assert_macd_matches_pandas(x[:32], [(2, 3, 2), (5, 20, 5), (3, 7, 4)])


def test_macd_tuple_structure() -> None:
    """MACD returns three tensors of the input shape, and the histogram is
    exactly the MACD line minus the signal line."""
    torch.manual_seed(0)
    x = torch.randn(3, 50) + 10.0
    result = QF.macd(x)
    assert isinstance(result, tuple)
    assert len(result) == 3
    macd_line, signal_line, histogram = result
    for tensor in result:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == x.shape
    torch.testing.assert_close(histogram, macd_line - signal_line)


def test_macd_constant_series() -> None:
    """A constant series yields (approximately) zero for all three lines."""
    x = torch.full((50,), 5.0, dtype=torch.float64)
    macd_line, signal_line, histogram = QF.macd(x)
    zeros = torch.zeros_like(x)
    torch.testing.assert_close(macd_line, zeros, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(signal_line, zeros, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(histogram, zeros, atol=1e-12, rtol=0.0)


def test_macd_trending_series() -> None:
    """For a steadily increasing series, the fast EMA exceeds the slow EMA,
    so the MACD line eventually becomes positive."""
    x = torch.arange(1.0, 101.0, dtype=torch.float64)
    macd_line, signal_line, histogram = QF.macd(x)
    assert macd_line[0].item() == 0.0
    assert (macd_line[10:] > 0).all()
    assert macd_line[-1].item() > 0


def test_macd_nan_contamination() -> None:
    """A NaN input contaminates all outputs at and after its position."""
    torch.manual_seed(1)
    x = torch.randn(60, dtype=torch.float64) + 100.0
    k = 25
    x[k] = torch.nan
    macd_line, signal_line, histogram = QF.macd(x)
    for tensor in (macd_line, signal_line, histogram):
        assert torch.isfinite(tensor[:k]).all()
        assert torch.isnan(tensor[k:]).all()


def test_macd_multi_dimensional() -> None:
    """2D (dim=0/1), 3D, and negative dimensions match per-slice results."""
    torch.manual_seed(2)
    x = torch.randn(4, 40, dtype=torch.float64) + 50.0

    macd_dim1, signal_dim1, hist_dim1 = QF.macd(x, 3, 7, 5, dim=1)
    macd_dim0, signal_dim0, hist_dim0 = QF.macd(x.t(), 3, 7, 5, dim=0)
    torch.testing.assert_close(macd_dim0, macd_dim1.t())
    torch.testing.assert_close(signal_dim0, signal_dim1.t())
    torch.testing.assert_close(hist_dim0, hist_dim1.t())

    for row in range(x.shape[0]):
        row_macd, row_signal, row_hist = QF.macd(x[row], 3, 7, 5)
        torch.testing.assert_close(macd_dim1[row], row_macd)
        torch.testing.assert_close(signal_dim1[row], row_signal)
        torch.testing.assert_close(hist_dim1[row], row_hist)

    macd_neg, signal_neg, hist_neg = QF.macd(x, 3, 7, 5, dim=-1)
    torch.testing.assert_close(macd_neg, macd_dim1)
    torch.testing.assert_close(signal_neg, signal_dim1)
    torch.testing.assert_close(hist_neg, hist_dim1)

    x3d = torch.randn(2, 3, 30, dtype=torch.float64) + 50.0
    macd_3d, signal_3d, hist_3d = QF.macd(x3d, 3, 7, 5, dim=2)
    assert macd_3d.shape == x3d.shape
    for i in range(x3d.shape[0]):
        for j in range(x3d.shape[1]):
            expected_macd, _, _ = QF.macd(x3d[i, j], 3, 7, 5)
            torch.testing.assert_close(macd_3d[i, j], expected_macd)
    assert signal_3d.shape == x3d.shape
    assert hist_3d.shape == x3d.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_macd_dtype_preservation(dtype: torch.dtype) -> None:
    """All three outputs preserve the input dtype and device."""
    torch.manual_seed(3)
    x = (torch.randn(30) + 10.0).to(dtype)
    for tensor in QF.macd(x):
        assert_basic_properties(tensor, x)


def test_macd_invalid_span_ordering_raises_value_error() -> None:
    """``fast_span`` must be strictly less than ``slow_span``."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="less than slow_span"):
        QF.macd(x, fast_span=26, slow_span=12)
    with pytest.raises(ValueError, match="less than slow_span"):
        QF.macd(x, fast_span=12, slow_span=12)


def test_macd_non_positive_span_raises_value_error() -> None:
    """Zero or negative spans are rejected."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="fast_span must be a positive"):
        QF.macd(x, fast_span=0)
    with pytest.raises(ValueError, match="slow_span must be a positive"):
        QF.macd(x, slow_span=-1)
    with pytest.raises(ValueError, match="signal_span must be a positive"):
        QF.macd(x, signal_span=0)


def test_macd_non_integer_span_raises_type_error() -> None:
    """Non-integer spans are rejected; ``bool`` is a subclass of ``int``
    and must not be silently accepted."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="fast_span must be an integer"):
        QF.macd(x, fast_span=12.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="slow_span must be an integer"):
        QF.macd(x, slow_span=26.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="signal_span must be an integer"):
        QF.macd(x, signal_span=True)
    with pytest.raises(TypeError, match="fast_span must be an integer"):
        QF.macd(x, fast_span=True)
