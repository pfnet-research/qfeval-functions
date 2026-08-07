import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mrank(x: torch.Tensor, span: int, pct: bool) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1].numpy()
        if np.isnan(window).any():
            continue
        rank = float(scipy.stats.rankdata(window)[-1])
        result[i] = rank / span if pct else rank
    return result


def test_mrank_basic_functionality() -> None:
    """Test basic moving rank functionality against pandas."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        for pct in (True, False):
            np.testing.assert_allclose(
                QF.mrank(a, span, dim=0, pct=pct).numpy(),
                df.rolling(span).rank(method="average", pct=pct).to_numpy(),
                1e-6,
                1e-6,
            )


def test_mrank_pandas_comparison_with_ties() -> None:
    """Test the average-tie behavior against pandas on tie-heavy data."""
    torch.manual_seed(0)
    a = torch.randint(0, 4, (60, 5)).to(torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        for pct in (True, False):
            np.testing.assert_allclose(
                QF.mrank(a, span, dim=0, pct=pct).numpy(),
                df.rolling(span).rank(method="average", pct=pct).to_numpy(),
                1e-6,
                1e-6,
            )


def test_mrank_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    This exercises data shorter than, equal to, and longer than the
    window, on both mostly distinct and tie-heavy data.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 25):
            for x in (
                torch.randn(n, dtype=torch.float64),
                torch.randint(0, 3, (n,)).to(torch.float64),
            ):
                for pct in (True, False):
                    result = QF.mrank(x, span, dim=0, pct=pct)
                    expected = _reference_mrank(x, span, pct)
                    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_mrank_large_window_wavelet_matches_comparison() -> None:
    """The automatic large-window path preserves ties and NaN semantics."""
    from qfeval_functions.functions.mrank import _mrank_compare
    from qfeval_functions.functions.mrank import _mrank_wavelet

    torch.manual_seed(23)
    x = torch.randn(3, 640, dtype=torch.float64)
    x[0, 311] = math.nan
    x[1] = torch.arange(640, dtype=x.dtype).remainder(17)
    x[2, 200] = math.inf
    x[2, 400] = -math.inf
    for pct in (True, False):
        expected = _mrank_compare(x, 513, pct)
        torch.testing.assert_close(
            _mrank_wavelet(x, 513, pct), expected, equal_nan=True
        )
        torch.testing.assert_close(
            QF.mrank(x, 513, dim=1, pct=pct), expected, equal_nan=True
        )


def test_mrank_known_values() -> None:
    """Test moving rank with simple known data."""
    x = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0])

    result = QF.mrank(x, 3, dim=0)
    # Window [1,2,3]: 3 is the largest of 3 -> 3/3.
    # Window [2,3,2]: ranks of the tied 2s are (1+2)/2 -> 1.5/3.
    # Window [3,2,1]: 1 is the smallest of 3 -> 1/3.
    expected = torch.tensor([math.nan, math.nan, 1.0, 0.5, 1.0 / 3.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)

    result_raw = QF.mrank(x, 3, dim=0, pct=False)
    expected_raw = torch.tensor([math.nan, math.nan, 3.0, 1.5, 1.0])
    np.testing.assert_allclose(result_raw.numpy(), expected_raw.numpy())


def test_mrank_known_values_all_tied() -> None:
    """Test moving rank where all window elements are tied."""
    x = torch.full((5,), 2.0)

    # The average rank of `span` tied elements is (span + 1) / 2.
    result = QF.mrank(x, 2, dim=0)
    expected = torch.tensor([math.nan, 0.75, 0.75, 0.75, 0.75])
    np.testing.assert_allclose(result.numpy(), expected.numpy())

    result_raw = QF.mrank(x, 3, dim=0, pct=False)
    expected_raw = torch.tensor([math.nan, math.nan, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result_raw.numpy(), expected_raw.numpy())


def test_mrank_monotonic_data() -> None:
    """Test that monotone data yields the extreme ranks after warm-up."""
    x = torch.arange(10, dtype=torch.float64)
    for span in (2, 3, 5):
        # Increasing data: the latest element is always the largest.
        result = QF.mrank(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(result[span - 1 :].numpy(), 1.0)

        # Decreasing data: the latest element is always the smallest.
        result = QF.mrank(-x, span, dim=0)
        np.testing.assert_allclose(result[span - 1 :].numpy(), 1.0 / span)


def test_mrank_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.mrank(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mrank(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mrank_with_infinity() -> None:
    """Test that infinities rank as the extreme values."""
    x = torch.tensor([1.0, math.inf, 2.0, -math.inf, 3.0, math.inf])

    result = QF.mrank(x, 3, dim=0)
    # [1,inf,2] -> 2 ranks 2nd; [inf,2,-inf] -> -inf ranks 1st;
    # [2,-inf,3] -> 3 ranks 3rd; [-inf,3,inf] -> inf ranks 3rd.
    expected = torch.tensor(
        [math.nan, math.nan, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        result.numpy(), _reference_mrank(x, 3, pct=True).numpy(), rtol=1e-6
    )

    # Tied infinities are averaged like any other tie.
    result = QF.mrank(torch.tensor([math.inf, math.inf]), 2, dim=0)
    np.testing.assert_allclose(result.numpy(), [math.nan, 0.75])


def test_mrank_span_one() -> None:
    """Test that span=1 yields 1.0 for both pct and non-pct modes."""
    x = torch.tensor([3.0, 1.0, 2.0])
    np.testing.assert_allclose(
        QF.mrank(x, 1, dim=0).numpy(), np.ones(3), rtol=0
    )
    np.testing.assert_allclose(
        QF.mrank(x, 1, dim=0, pct=False).numpy(), np.ones(3), rtol=0
    )


def test_mrank_window_larger_than_data() -> None:
    """Test moving rank when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mrank(x, 5, dim=0)
    assert result.shape == x.shape
    assert torch.isnan(result).all()


def test_mrank_single_element() -> None:
    """Test moving rank with a single-element input."""
    x = torch.tensor([42.0])
    np.testing.assert_allclose(QF.mrank(x, 1, dim=0).numpy(), [1.0])
    assert torch.isnan(QF.mrank(x, 2, dim=0)).all()


def test_mrank_multi_dimensional() -> None:
    """Test moving rank with 2D/3D tensors against per-fiber 1-D calls."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 12)

    result = QF.mrank(x, 4, dim=2)
    assert result.shape == x.shape
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(
                result[i, j],
                QF.mrank(x[i, j], 4, dim=0),
                equal_nan=True,
            )

    result = QF.mrank(x, 3, dim=1)
    for i in range(3):
        for k in range(12):
            torch.testing.assert_close(
                result[i, :, k],
                QF.mrank(x[i, :, k], 3, dim=0),
                equal_nan=True,
            )


def test_mrank_negative_dimension() -> None:
    """Test moving rank with negative dimension indexing."""
    torch.manual_seed(0)
    x = torch.randn(5, 10)
    torch.testing.assert_close(
        QF.mrank(x, 3, dim=-1), QF.mrank(x, 3, dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.mrank(x, 3, dim=-2), QF.mrank(x, 3, dim=0), equal_nan=True
    )


def test_mrank_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mrank(x, 3, dim=dim)
            assert_basic_properties(result, x)


def test_mrank_invalid_span_type() -> None:
    """Test that a non-integer span raises TypeError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mrank(x, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mrank(x, True)


def test_mrank_invalid_span_value() -> None:
    """Test that a non-positive span raises ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for span in (0, -1):
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.mrank(x, span)


def test_mrank_non_floating_input() -> None:
    """Test that a non-floating input tensor raises TypeError."""
    with pytest.raises(TypeError, match="floating point"):
        QF.mrank(torch.tensor([1, 2, 3]), 2)
