import math

import numpy as np
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_nanrms_vs_numpy_random_data() -> None:
    """Compare nanrms with a manual numpy reference on data with NaNs."""
    torch.manual_seed(1)
    x = torch.randn(6, 7, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.3] = math.nan
    # Ensure every row and column has at least one valid value to avoid
    # numpy warnings about empty slices.
    x[:, 0] = 1.5
    x[0, :] = -0.5
    for dim in (None, 0, 1, (0, 1)):
        for keepdim in (False, True):
            actual = QF.nanrms(x, dim=dim, keepdim=keepdim)
            expected = np.sqrt(
                np.nanmean(x.numpy() ** 2, axis=dim, keepdims=keepdim)
            )
            np.testing.assert_allclose(actual.numpy(), expected)


def test_nanrms_matches_rms_without_nan() -> None:
    """Verify nanrms equals rms on NaN-free data."""
    torch.manual_seed(2)
    x = torch.randn(4, 5, dtype=torch.float64)
    for dim in (None, 0, 1, (0, 1)):
        torch.testing.assert_close(QF.nanrms(x, dim=dim), QF.rms(x, dim=dim))
    torch.testing.assert_close(
        QF.nanrms(x, dim=1, keepdim=True), QF.rms(x, dim=1, keepdim=True)
    )


def test_nanrms_basic_functionality() -> None:
    """Test RMS with a manually calculated example containing NaN."""
    x = torch.tensor([3.0, math.nan, -4.0], dtype=torch.float64)
    expected = math.sqrt((9.0 + 16.0) / 2.0)
    torch.testing.assert_close(
        QF.nanrms(x), torch.tensor(expected, dtype=torch.float64)
    )


def test_nanrms_all_nan() -> None:
    """Test that all-NaN slices return NaN."""
    x = torch.tensor([[3.0, -4.0], [math.nan, math.nan]], dtype=torch.float64)
    result = QF.nanrms(x, dim=1)
    torch.testing.assert_close(
        result[0], torch.tensor(math.sqrt(12.5), dtype=torch.float64)
    )
    assert torch.isnan(result[1])
    # Full reduction over an all-NaN tensor.
    all_nan = torch.full((2, 2), math.nan, dtype=torch.float64)
    assert torch.isnan(QF.nanrms(all_nan))


def test_nanrms_with_infinity() -> None:
    """Test that infinite values yield an infinite RMS."""
    x_pos = torch.tensor([1.0, math.inf, math.nan], dtype=torch.float64)
    assert QF.nanrms(x_pos) == math.inf
    # Squaring makes negative infinity positive.
    x_neg = torch.tensor([1.0, -math.inf, math.nan], dtype=torch.float64)
    assert QF.nanrms(x_neg) == math.inf


def test_nanrms_sign_invariance() -> None:
    """Test that nanrms is invariant to the sign of the input."""
    torch.manual_seed(3)
    x = torch.randn(5, 6, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.2] = math.nan
    torch.testing.assert_close(
        QF.nanrms(x, dim=1), QF.nanrms(-x, dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.nanrms(x, dim=1), QF.nanrms(x.abs(), dim=1), equal_nan=True
    )


def test_nanrms_zeros() -> None:
    """Test that zero values yield a zero RMS."""
    x = torch.tensor([0.0, math.nan, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        QF.nanrms(x), torch.tensor(0.0, dtype=torch.float64)
    )


def test_nanrms_tuple_dims() -> None:
    """Test nanrms with multiple reduction dimensions on a 3D tensor."""
    torch.manual_seed(4)
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.2] = math.nan
    result = QF.nanrms(x, dim=(1, 2))
    assert result.shape == (3,)
    # Should be equivalent to flattening those dimensions.
    torch.testing.assert_close(
        result, QF.nanrms(x.reshape(3, -1), dim=1), equal_nan=True
    )
    # Negative dimensions behave the same.
    torch.testing.assert_close(
        result, QF.nanrms(x, dim=(-1, -2)), equal_nan=True
    )


def test_nanrms_keepdim_shapes() -> None:
    """Test output shapes with and without keepdim."""
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    assert QF.nanrms(x, dim=1).shape == (2, 4)
    assert QF.nanrms(x, dim=1, keepdim=True).shape == (2, 1, 4)
    assert QF.nanrms(x, dim=(1, 2), keepdim=True).shape == (2, 1, 1)
    assert QF.nanrms(x).shape == ()
    torch.testing.assert_close(
        QF.nanrms(x, dim=-1), QF.nanrms(x, dim=2), equal_nan=True
    )


def test_nanrms_dtype_preservation() -> None:
    """Test that nanrms preserves dtype and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[3.0, math.nan, -4.0]], dtype=dtype)
        result = QF.nanrms(x, dim=1, keepdim=True)
        assert_basic_properties(result, x, expected_shape=torch.Size([1, 1]))
        torch.testing.assert_close(
            result, torch.tensor([[math.sqrt(12.5)]], dtype=dtype)
        )
