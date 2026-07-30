import math
import warnings

import numpy as np
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_nanstd_vs_numpy_random_data() -> None:
    """Compare nanstd with numpy.nanstd on random data with NaN values."""
    torch.manual_seed(1)
    x = torch.randn(7, 9, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.3] = math.nan
    # NOTE: Suppress warnings of all-NaN or single-value slices:
    # RuntimeWarning: Degrees of freedom <= 0 for slice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for unbiased, ddof in ((True, 1), (False, 0)):
            for dim in (None, 0, 1):
                for keepdim in (False, True):
                    actual = QF.nanstd(
                        x, dim=dim, unbiased=unbiased, keepdim=keepdim
                    )
                    expected = np.nanstd(
                        x.numpy(), axis=dim, ddof=ddof, keepdims=keepdim
                    )
                    np.testing.assert_allclose(
                        actual.numpy(), expected, equal_nan=True
                    )


def test_nanstd_consistency_with_nanvar() -> None:
    """Verify that nanstd squared equals nanvar."""
    torch.manual_seed(2)
    x = torch.randn(5, 8, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.2] = math.nan
    for unbiased in (True, False):
        std = QF.nanstd(x, dim=1, unbiased=unbiased)
        var = QF.nanvar(x, dim=1, unbiased=unbiased)
        torch.testing.assert_close(std**2, var, equal_nan=True)


def test_nanstd_no_nan_matches_torch_std() -> None:
    """Verify nanstd matches torch.std on NaN-free data."""
    torch.manual_seed(3)
    x = torch.randn(4, 6, dtype=torch.float64)
    for unbiased in (True, False):
        torch.testing.assert_close(
            QF.nanstd(x, dim=1, unbiased=unbiased),
            torch.std(x, dim=1, unbiased=unbiased),
        )
    torch.testing.assert_close(QF.nanstd(x), torch.std(x))


def test_nanstd_basic_functionality() -> None:
    """Test standard deviation with a manually calculated example."""
    x = torch.tensor([[1.0, math.nan, 3.0, 5.0]], dtype=torch.float64)
    # Valid values: [1, 3, 5] -> mean=3, squared deviations sum = 8.
    torch.testing.assert_close(
        QF.nanstd(x, dim=1, unbiased=True),
        torch.tensor([2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        QF.nanstd(x, dim=1, unbiased=False),
        torch.tensor([(8.0 / 3.0) ** 0.5], dtype=torch.float64),
    )


def test_nanstd_single_valid_value() -> None:
    """Test slices with exactly one valid value."""
    x = torch.tensor(
        [[5.0, math.nan, math.nan], [1.0, 2.0, 3.0]], dtype=torch.float64
    )
    # Unbiased: division by n-1=0 yields NaN.
    result_unbiased = QF.nanstd(x, dim=1, unbiased=True)
    assert torch.isnan(result_unbiased[0])
    assert torch.isfinite(result_unbiased[1])
    # Biased: a single value has zero deviation.
    result_biased = QF.nanstd(x, dim=1, unbiased=False)
    torch.testing.assert_close(
        result_biased[0], torch.tensor(0.0, dtype=torch.float64)
    )


def test_nanstd_all_nan() -> None:
    """Test that all-NaN slices return NaN."""
    x = torch.tensor(
        [[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0]],
        dtype=torch.float64,
    )
    for unbiased in (True, False):
        result = QF.nanstd(x, dim=1, unbiased=unbiased)
        assert torch.isnan(result[0])
        assert torch.isfinite(result[1])
    # Full reduction over an all-NaN tensor.
    all_nan = torch.full((2, 2), math.nan, dtype=torch.float64)
    assert torch.isnan(QF.nanstd(all_nan))


def test_nanstd_tuple_dims() -> None:
    """Test nanstd with multiple reduction dimensions on a 3D tensor."""
    torch.manual_seed(4)
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.2] = math.nan
    result = QF.nanstd(x, dim=(1, 2))
    assert result.shape == (3,)
    # Should be equivalent to flattening those dimensions.
    result_flat = QF.nanstd(x.reshape(3, -1), dim=1)
    torch.testing.assert_close(result, result_flat, equal_nan=True)
    # keepdim keeps the reduced dimensions as size 1.
    result_keepdim = QF.nanstd(x, dim=(1, 2), keepdim=True)
    assert result_keepdim.shape == (3, 1, 1)
    torch.testing.assert_close(
        result_keepdim.reshape(3), result, equal_nan=True
    )


def test_nanstd_negative_dim() -> None:
    """Test nanstd with negative dimension indices."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]], dtype=torch.float64
    )
    torch.testing.assert_close(
        QF.nanstd(x, dim=-1), QF.nanstd(x, dim=1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.nanstd(x, dim=-2), QF.nanstd(x, dim=0), equal_nan=True
    )


def test_nanstd_keepdim_shapes() -> None:
    """Test output shapes with and without keepdim."""
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    assert QF.nanstd(x, dim=1).shape == (2, 4)
    assert QF.nanstd(x, dim=1, keepdim=True).shape == (2, 1, 4)
    assert QF.nanstd(x).shape == ()
    assert QF.nanstd(x, keepdim=True).shape == (1, 1, 1)
    torch.testing.assert_close(
        QF.nanstd(x, dim=2, keepdim=True).squeeze(2), QF.nanstd(x, dim=2)
    )


def test_nanstd_dtype_preservation() -> None:
    """Test that nanstd preserves dtype and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[1.0, math.nan, 3.0]], dtype=dtype)
        result = QF.nanstd(x, dim=1)
        assert result.dtype == dtype
        assert result.device == x.device
        assert result.shape == (1,)


def test_nanstd_basic_properties() -> None:
    """Test basic properties with the shared helper."""
    x = torch.tensor(
        [[1.0, math.nan, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
    )
    result = QF.nanstd(x, dim=1, keepdim=True)
    assert_basic_properties(result, x, expected_shape=torch.Size([2, 1]))


def test_nanstd_with_infinity() -> None:
    """Test that infinite values yield a non-finite standard deviation."""
    x = torch.tensor([[1.0, math.inf, 2.0]], dtype=torch.float64)
    for unbiased in (True, False):
        result = QF.nanstd(x, dim=1, unbiased=unbiased)
        assert not torch.isfinite(result[0])
