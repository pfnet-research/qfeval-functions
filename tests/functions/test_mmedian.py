import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mmedian(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1].numpy()
        if not np.isnan(window).any():
            result[i] = float(np.median(window))
    return result


def test_mmedian_basic_functionality() -> None:
    """Test basic moving median functionality against pandas."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.mmedian(a, span, dim=0).numpy(),
            df.rolling(span).median().to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_mmedian_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    This exercises all relative positions of windows and data boundaries,
    including data shorter than, equal to, and longer than the window,
    against a naive per-window NumPy reference.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 25):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            result = QF.mmedian(x, span, dim=0)
            expected = _reference_mmedian(x, span)
            np.testing.assert_allclose(
                result.numpy(),
                expected.numpy(),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )


def test_mmedian_known_values_odd_window() -> None:
    """Test hand-computed medians for an odd window size."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    # Sorted windows: [1, 2, 3], [2, 3, 5], [2, 4, 5].
    torch.testing.assert_close(
        QF.mmedian(x, 3, dim=0),
        torch.tensor([math.nan, math.nan, 2.0, 3.0, 4.0]),
        equal_nan=True,
    )


def test_mmedian_known_values_even_window() -> None:
    """Test hand-computed medians for even window sizes (the mean of the
    two central values)."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    torch.testing.assert_close(
        QF.mmedian(x, 2, dim=0),
        torch.tensor([math.nan, 2.0, 2.5, 3.5, 4.5]),
        equal_nan=True,
    )
    x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    # Sorted windows: [1, 2, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32].
    torch.testing.assert_close(
        QF.mmedian(x, 4, dim=0),
        torch.tensor([math.nan, math.nan, math.nan, 3.0, 6.0, 12.0]),
        equal_nan=True,
    )


def test_mmedian_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.mmedian(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mmedian(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mmedian_inf_handling() -> None:
    """Test that infinities act as extreme order statistics."""
    x = torch.tensor([1.0, math.inf, 2.0, 3.0, 4.0])
    # Sorted windows: [1, 2, inf], [2, 3, inf], [2, 3, 4].
    torch.testing.assert_close(
        QF.mmedian(x, 3, dim=0),
        torch.tensor([math.nan, math.nan, 2.0, 3.0, 3.0]),
        equal_nan=True,
    )
    # An even window whose two central values include +inf yields +inf;
    # the interpolation must not turn it into NaN.
    x = torch.tensor([1.0, math.inf, math.inf, 5.0])
    torch.testing.assert_close(
        QF.mmedian(x, 2, dim=0),
        torch.tensor([math.nan, math.inf, math.inf, math.inf]),
        equal_nan=True,
    )
    # Opposite infinities in the two central positions yield NaN.
    result = QF.mmedian(torch.tensor([-math.inf, math.inf]), 2, dim=0)
    assert torch.isnan(result[1])


def test_mmedian_span_one_returns_input() -> None:
    """Test that span=1 returns the input unchanged."""
    torch.manual_seed(0)
    x = torch.randn(10)
    torch.testing.assert_close(QF.mmedian(x, 1, dim=0), x)


def test_mmedian_window_larger_than_data() -> None:
    """Test moving median when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mmedian(x, 5, dim=0)
    assert result.shape == x.shape
    assert torch.isnan(result).all()


def test_mmedian_single_element() -> None:
    """Test moving median with single-element input."""
    x = torch.tensor([42.0])
    torch.testing.assert_close(QF.mmedian(x, 1, dim=0), x)
    assert torch.isnan(QF.mmedian(x, 2, dim=0)).all()


def test_mmedian_2d_dims() -> None:
    """Test that dim=0 and dim=1 match per-column/per-row 1-D results."""
    torch.manual_seed(0)
    x = torch.randn(6, 7, dtype=torch.float64)
    result0 = QF.mmedian(x, 3, dim=0)
    result1 = QF.mmedian(x, 3, dim=1)
    for j in range(x.shape[1]):
        torch.testing.assert_close(
            result0[:, j], _reference_mmedian(x[:, j], 3), equal_nan=True
        )
    for i in range(x.shape[0]):
        torch.testing.assert_close(
            result1[i], _reference_mmedian(x[i], 3), equal_nan=True
        )


def test_mmedian_3d_and_negative_dim() -> None:
    """Test 3D tensors and negative dimension equivalence."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 20)
    result = QF.mmedian(x, 5, dim=2)
    assert result.shape == x.shape
    torch.testing.assert_close(result, QF.mmedian(x, 5, dim=-1), equal_nan=True)
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(
                result[i, j],
                QF.mmedian(x[i, j], 5, dim=0),
                equal_nan=True,
            )
    torch.testing.assert_close(
        QF.mmedian(x, 5, dim=1), QF.mmedian(x, 5, dim=-2), equal_nan=True
    )


def test_mmedian_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mmedian(x, 3, dim=dim)
            assert_basic_properties(result, x)


def test_mmedian_invalid_span_raises_value_error() -> None:
    """``span`` must be a positive integer."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for span in (0, -1):
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.mmedian(x, span)


def test_mmedian_non_integer_span_raises_type_error() -> None:
    """Non-integer ``span`` values are rejected; ``bool`` is a subclass of
    ``int`` and must not be silently accepted as 1."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mmedian(x, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mmedian(x, True)


def test_mmedian_non_floating_point_input_raises_type_error() -> None:
    """Non-floating-point inputs are rejected."""
    for dtype in (torch.int32, torch.int64, torch.bool):
        x = torch.ones(10, dtype=dtype)
        with pytest.raises(TypeError, match="floating point"):
            QF.mmedian(x, 3)


def test_mmedian_equals_mquantile_half() -> None:
    """Test that mmedian is exactly mquantile with q=0.5, including NaN
    and infinite inputs."""
    torch.manual_seed(0)
    x = torch.randn(4, 30)
    x[1, 5] = math.nan
    x[2, 10] = math.inf
    x[3, 20] = -math.inf
    for span in (1, 2, 3, 6):
        torch.testing.assert_close(
            QF.mmedian(x, span, dim=1),
            QF.mquantile(x, span, 0.5, dim=1),
            equal_nan=True,
        )
