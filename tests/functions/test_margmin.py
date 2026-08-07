import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_margmin(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors.

    Ties are resolved to the most recent occurrence of the minimum.
    """
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1]
        if bool(torch.isnan(window).any()):
            continue
        best = 0
        for j in range(span):
            if window[j] <= window[best]:
                best = j
        result[i] = span - 1 - best
    return result


def _pandas_margmin(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Pandas rolling reference: ``np.argmin`` on the reversed window
    returns the offset of the most recent minimum from the latest
    element, which is exactly the number of periods since the minimum."""
    rolling = df.rolling(span)
    return rolling.apply(lambda w: float(np.argmin(w[::-1])), raw=True)


def test_margmin_all_length_span_alignments() -> None:
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
                result = QF.margmin(x, span, dim=0)
                expected = _reference_margmin(x, span)
                np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmin_large_window_predecessor_matches_reduction() -> None:
    """The large-window predecessor path preserves ties and special values."""
    from qfeval_functions.functions.margmax import _mextremum_distance_compare
    from qfeval_functions.functions.margmax import (
        _mextremum_distance_predecessor,
    )

    torch.manual_seed(41)
    x = torch.randn(4, 400, dtype=torch.float64)
    x[0, 211] = math.nan
    x[1] = torch.arange(400, dtype=x.dtype).remainder(13)
    x[2, 120] = -math.inf
    expected = _mextremum_distance_compare(x, 256, largest=False)
    torch.testing.assert_close(
        _mextremum_distance_predecessor(x, 256, largest=False),
        expected,
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.margmin(x, 256, dim=1), expected, equal_nan=True
    )


def test_margmin_pandas_cross_check() -> None:
    """Cross-check against a pandas rolling.apply implementation."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.margmin(a, span, dim=0).numpy(),
            _pandas_margmin(df, span).to_numpy(),
            1e-6,
            1e-6,
        )


def test_margmin_pandas_cross_check_with_ties() -> None:
    """Cross-check the most-recent-tie rule against pandas on tie-heavy
    data (``np.argmin`` picks the first minimum of the reversed window,
    i.e., the most recent one of the original window)."""
    torch.manual_seed(0)
    a = torch.randint(0, 4, (60, 5)).to(torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.margmin(a, span, dim=0).numpy(),
            _pandas_margmin(df, span).to_numpy(),
            1e-6,
            1e-6,
        )


def test_margmin_known_values() -> None:
    """Test the number of periods since the minimum with known data."""
    x = torch.tensor([3.0, 1.0, 2.0, 0.0, 4.0])
    result = QF.margmin(x, 3, dim=0)
    # [3,1,2] -> 1 one period ago; [1,2,0] -> 0 now; [2,0,4] -> 0 one ago.
    expected = torch.tensor([math.nan, math.nan, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmin_monotonic_data() -> None:
    """Test monotone data: increasing -> span - 1, decreasing -> 0."""
    x = torch.arange(10, dtype=torch.float64)
    for span in (1, 2, 3, 5):
        result = QF.margmin(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(result[span - 1 :].numpy(), float(span - 1))

        result = QF.margmin(-x, span, dim=0)
        np.testing.assert_allclose(result[span - 1 :].numpy(), 0.0)


def test_margmin_constant_data() -> None:
    """Test that constant data (all ties) yields 0 everywhere."""
    x = torch.full((8,), 5.0)
    for span in (1, 3, 5):
        result = QF.margmin(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(result[span - 1 :].numpy(), 0.0)


def test_margmin_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.margmin(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.margmin(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_margmin_with_negative_infinity() -> None:
    """Test that -inf is a legitimate minimum."""
    x = torch.tensor([1.0, -math.inf, 2.0, 3.0, 0.0])
    result = QF.margmin(x, 3, dim=0)
    # The -inf element stays the minimum while it is inside the window.
    expected = torch.tensor([math.nan, math.nan, 1.0, 2.0, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmin_with_positive_infinity() -> None:
    """Test that +inf does not affect the result unless it is the min."""
    x = torch.tensor([1.0, math.inf, 2.0, 3.0])
    result = QF.margmin(x, 3, dim=0)
    # [1,inf,2] -> min 1 two periods ago; [inf,2,3] -> min 2 one ago.
    expected = torch.tensor([math.nan, math.nan, 2.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())

    # A window of only +inf values still has a (tied) minimum.
    x = torch.full((3,), math.inf)
    result = QF.margmin(x, 2, dim=0)
    np.testing.assert_allclose(result.numpy(), [math.nan, 0.0, 0.0])


def test_margmin_equals_margmax_of_negated_input() -> None:
    """Test the duality margmin(x) == margmax(-x) on random data."""
    torch.manual_seed(0)
    x = torch.randn(4, 30, dtype=torch.float64)
    x[0, 5] = math.nan
    x[1, 7] = math.inf
    x[2, 9] = -math.inf
    for span in (1, 3, 7):
        torch.testing.assert_close(
            QF.margmin(x, span, dim=1),
            QF.margmax(-x, span, dim=1),
            equal_nan=True,
        )


def test_margmin_span_one() -> None:
    """Test that span=1 yields 0 everywhere."""
    x = torch.tensor([3.0, 1.0, 2.0])
    np.testing.assert_allclose(
        QF.margmin(x, 1, dim=0).numpy(), np.zeros(3), rtol=0
    )


def test_margmin_window_larger_than_data() -> None:
    """Test margmin when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.margmin(x, 5, dim=0)
    assert result.shape == x.shape
    assert torch.isnan(result).all()


def test_margmin_single_element() -> None:
    """Test margmin with a single-element input."""
    x = torch.tensor([42.0])
    np.testing.assert_allclose(QF.margmin(x, 1, dim=0).numpy(), [0.0])
    assert torch.isnan(QF.margmin(x, 2, dim=0)).all()


def test_margmin_multi_dimensional() -> None:
    """Test margmin with 2D/3D tensors against per-fiber 1-D calls."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 12)

    result = QF.margmin(x, 4, dim=2)
    assert result.shape == x.shape
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(
                result[i, j],
                QF.margmin(x[i, j], 4, dim=0),
                equal_nan=True,
            )

    result = QF.margmin(x, 3, dim=1)
    for i in range(3):
        for k in range(12):
            torch.testing.assert_close(
                result[i, :, k],
                QF.margmin(x[i, :, k], 3, dim=0),
                equal_nan=True,
            )


def test_margmin_negative_dimension() -> None:
    """Test margmin with negative dimension indexing."""
    torch.manual_seed(0)
    x = torch.randn(5, 10)
    torch.testing.assert_close(
        QF.margmin(x, 3, dim=-1), QF.margmin(x, 3, dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.margmin(x, 3, dim=-2), QF.margmin(x, 3, dim=0), equal_nan=True
    )


def test_margmin_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.margmin(x, 3, dim=dim)
            assert_basic_properties(result, x)


def test_margmin_invalid_span_type() -> None:
    """Test that a non-integer span raises TypeError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.margmin(x, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.margmin(x, True)


def test_margmin_invalid_span_value() -> None:
    """Test that a non-positive span raises ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for span in (0, -1):
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.margmin(x, span)


def test_margmin_non_floating_input() -> None:
    """Test that a non-floating input tensor raises TypeError."""
    with pytest.raises(TypeError, match="floating point"):
        QF.margmin(torch.tensor([1, 2, 3]), 2)
