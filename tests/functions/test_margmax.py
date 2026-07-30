import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_margmax(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors.

    Ties are resolved to the most recent occurrence of the maximum.
    """
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1]
        if bool(torch.isnan(window).any()):
            continue
        best = 0
        for j in range(span):
            if window[j] >= window[best]:
                best = j
        result[i] = span - 1 - best
    return result


def _pandas_margmax(df: pd.DataFrame, span: int) -> pd.DataFrame:
    """Pandas rolling reference: ``np.argmax`` on the reversed window
    returns the offset of the most recent maximum from the latest
    element, which is exactly the number of periods since the maximum."""
    rolling = df.rolling(span)
    return rolling.apply(lambda w: float(np.argmax(w[::-1])), raw=True)


def test_margmax_all_length_span_alignments() -> None:
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
                result = QF.margmax(x, span, dim=0)
                expected = _reference_margmax(x, span)
                np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmax_pandas_cross_check() -> None:
    """Cross-check against a pandas rolling.apply implementation."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.margmax(a, span, dim=0).numpy(),
            _pandas_margmax(df, span).to_numpy(),
            1e-6,
            1e-6,
        )


def test_margmax_pandas_cross_check_with_ties() -> None:
    """Cross-check the most-recent-tie rule against pandas on tie-heavy
    data (``np.argmax`` picks the first maximum of the reversed window,
    i.e., the most recent one of the original window)."""
    torch.manual_seed(0)
    a = torch.randint(0, 4, (60, 5)).to(torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.margmax(a, span, dim=0).numpy(),
            _pandas_margmax(df, span).to_numpy(),
            1e-6,
            1e-6,
        )


def test_margmax_known_values() -> None:
    """Test the number of periods since the maximum with known data."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    result = QF.margmax(x, 3, dim=0)
    # [1,3,2] -> 3 one period ago; [3,2,5] -> 5 now; [2,5,4] -> 5 one ago.
    expected = torch.tensor([math.nan, math.nan, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmax_monotonic_data() -> None:
    """Test monotone data: increasing -> 0, decreasing -> span - 1."""
    x = torch.arange(10, dtype=torch.float64)
    for span in (1, 2, 3, 5):
        result = QF.margmax(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(result[span - 1 :].numpy(), 0.0)

        result = QF.margmax(-x, span, dim=0)
        np.testing.assert_allclose(result[span - 1 :].numpy(), float(span - 1))


def test_margmax_constant_data() -> None:
    """Test that constant data (all ties) yields 0 everywhere."""
    x = torch.full((8,), 5.0)
    for span in (1, 3, 5):
        result = QF.margmax(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(result[span - 1 :].numpy(), 0.0)


def test_margmax_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.margmax(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.margmax(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_margmax_with_positive_infinity() -> None:
    """Test that +inf is a legitimate maximum."""
    x = torch.tensor([1.0, math.inf, 2.0, 3.0, 4.0])
    result = QF.margmax(x, 3, dim=0)
    # The +inf element stays the maximum while it is inside the window.
    expected = torch.tensor([math.nan, math.nan, 1.0, 2.0, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_margmax_with_negative_infinity() -> None:
    """Test that -inf does not affect the result unless it is the max."""
    x = torch.tensor([1.0, -math.inf, 2.0, 3.0])
    result = QF.margmax(x, 3, dim=0)
    expected = torch.tensor([math.nan, math.nan, 0.0, 0.0])
    np.testing.assert_allclose(result.numpy(), expected.numpy())

    # A window of only -inf values still has a (tied) maximum.
    x = torch.full((3,), -math.inf)
    result = QF.margmax(x, 2, dim=0)
    np.testing.assert_allclose(result.numpy(), [math.nan, 0.0, 0.0])


def test_margmax_span_one() -> None:
    """Test that span=1 yields 0 everywhere."""
    x = torch.tensor([3.0, 1.0, 2.0])
    np.testing.assert_allclose(
        QF.margmax(x, 1, dim=0).numpy(), np.zeros(3), rtol=0
    )


def test_margmax_window_larger_than_data() -> None:
    """Test margmax when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.margmax(x, 5, dim=0)
    assert result.shape == x.shape
    assert torch.isnan(result).all()


def test_margmax_single_element() -> None:
    """Test margmax with a single-element input."""
    x = torch.tensor([42.0])
    np.testing.assert_allclose(QF.margmax(x, 1, dim=0).numpy(), [0.0])
    assert torch.isnan(QF.margmax(x, 2, dim=0)).all()


def test_margmax_multi_dimensional() -> None:
    """Test margmax with 2D/3D tensors against per-fiber 1-D calls."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 12)

    result = QF.margmax(x, 4, dim=2)
    assert result.shape == x.shape
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(
                result[i, j],
                QF.margmax(x[i, j], 4, dim=0),
                equal_nan=True,
            )

    result = QF.margmax(x, 3, dim=1)
    for i in range(3):
        for k in range(12):
            torch.testing.assert_close(
                result[i, :, k],
                QF.margmax(x[i, :, k], 3, dim=0),
                equal_nan=True,
            )


def test_margmax_negative_dimension() -> None:
    """Test margmax with negative dimension indexing."""
    torch.manual_seed(0)
    x = torch.randn(5, 10)
    torch.testing.assert_close(
        QF.margmax(x, 3, dim=-1), QF.margmax(x, 3, dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.margmax(x, 3, dim=-2), QF.margmax(x, 3, dim=0), equal_nan=True
    )


def test_margmax_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.margmax(x, 3, dim=dim)
            assert_basic_properties(result, x)


def test_margmax_invalid_span_type() -> None:
    """Test that a non-integer span raises TypeError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.margmax(x, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.margmax(x, True)


def test_margmax_invalid_span_value() -> None:
    """Test that a non-positive span raises ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for span in (0, -1):
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.margmax(x, span)


def test_margmax_non_floating_input() -> None:
    """Test that a non-floating input tensor raises TypeError."""
    with pytest.raises(TypeError, match="floating point"):
        QF.margmax(torch.tensor([1, 2, 3]), 2)
