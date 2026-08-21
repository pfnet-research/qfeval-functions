import math
import warnings

import numpy as np
import scipy.stats
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_zscore_scipy_comparison_random_with_nans() -> None:
    """Test zscore against scipy on random data with random NaN values."""
    torch.manual_seed(7)
    x = torch.randn(15, 40, dtype=torch.float64)
    x[torch.rand(15, 40) < 0.2] = math.nan
    for unbiased, ddof in ((True, 1), (False, 0)):
        expected = scipy.stats.zscore(
            x.numpy(), axis=1, ddof=ddof, nan_policy="omit"
        )
        np.testing.assert_allclose(
            QF.zscore(x, dim=1, unbiased=unbiased).numpy(),
            expected,
            atol=1e-10,
            equal_nan=True,
        )
        expected_t = scipy.stats.zscore(
            x.numpy(), axis=0, ddof=ddof, nan_policy="omit"
        )
        np.testing.assert_allclose(
            QF.zscore(x, dim=0, unbiased=unbiased).numpy(),
            expected_t,
            atol=1e-10,
            equal_nan=True,
        )


def test_zscore_standardization_properties() -> None:
    """Test that outputs have zero mean and unit variance along the dim."""
    torch.manual_seed(1)
    x = torch.randn(8, 30, dtype=torch.float64)
    x[torch.rand(8, 30) < 0.2] = math.nan
    for unbiased in (True, False):
        z = QF.zscore(x, dim=1, unbiased=unbiased)
        # The NaN mask must be preserved exactly.
        assert torch.equal(z.isnan(), x.isnan())
        np.testing.assert_allclose(
            QF.nanmean(z, dim=1).numpy(), np.zeros(8), atol=1e-12
        )
        np.testing.assert_allclose(
            QF.nanvar(z, dim=1, unbiased=unbiased).numpy(),
            np.ones(8),
            atol=1e-12,
        )


def test_zscore_degenerate_slices() -> None:
    """Test constant, single-valid, and all-NaN slices."""
    x = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [math.nan, 5.0, math.nan],
            [math.nan, math.nan, math.nan],
            [1.0, 2.0, math.nan],
        ],
        dtype=torch.float64,
    )
    for unbiased in (True, False):
        z = QF.zscore(x, dim=1, unbiased=unbiased)
        # A constant slice yields 0 / 0 = NaN.
        assert z[0].isnan().all()
        # A single valid element yields NaN: the unbiased variance is NaN,
        # and the biased variance is 0, leading to 0 / 0 = NaN.
        assert z[1].isnan().all()
        # An all-NaN slice stays NaN.
        assert z[2].isnan().all()
        # This behavior matches scipy.stats.zscore with nan_policy="omit".
        # NOTE: Suppress a warning caused by the constant slice:
        # RuntimeWarning: Precision loss occurred in moment calculation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = scipy.stats.zscore(
                x.numpy(), axis=1, ddof=1 if unbiased else 0, nan_policy="omit"
            )
        np.testing.assert_allclose(
            z.numpy(), expected, atol=1e-12, equal_nan=True
        )
    # A slice with two valid elements is standardized normally.
    z = QF.zscore(x, dim=1, unbiased=True)
    expected_last = torch.tensor(
        [-math.sqrt(0.5), math.sqrt(0.5), math.nan], dtype=torch.float64
    )
    torch.testing.assert_close(z[3], expected_last, equal_nan=True)


def test_zscore_dim_none_and_tuple_dims() -> None:
    """Test dim=None, tuple dims, and negative dims on a 3D tensor."""
    torch.manual_seed(3)
    x = torch.randn(4, 5, 6, dtype=torch.float64)
    x[torch.rand(4, 5, 6) < 0.1] = math.nan
    # Global standardization is equivalent to standardizing the flattened
    # tensor.
    z_none = QF.zscore(x, dim=None, unbiased=False)
    assert z_none.shape == x.shape
    z_flat = QF.zscore(x.reshape(-1), dim=0, unbiased=False)
    torch.testing.assert_close(z_none, z_flat.reshape(x.shape), equal_nan=True)
    # Tuple dims are equivalent to flattening those dimensions.
    z_tuple = QF.zscore(x, dim=(1, 2))
    assert z_tuple.shape == x.shape
    z_flat2 = QF.zscore(x.reshape(4, -1), dim=1)
    torch.testing.assert_close(
        z_tuple, z_flat2.reshape(x.shape), equal_nan=True
    )
    # Negative dimensions are equivalent to their positive counterparts.
    torch.testing.assert_close(
        QF.zscore(x, dim=-1), QF.zscore(x, dim=2), equal_nan=True
    )
    torch.testing.assert_close(
        QF.zscore(x, dim=(-2, -1)), QF.zscore(x, dim=(1, 2)), equal_nan=True
    )
    # The output shape always matches the input shape.
    for dim in (0, 1, 2, -1):
        assert QF.zscore(x, dim=dim).shape == x.shape


def test_zscore_affine_invariance() -> None:
    """Test invariance under positive affine transformations."""
    torch.manual_seed(5)
    x = torch.randn(6, 25, dtype=torch.float64)
    x[torch.rand(6, 25) < 0.15] = math.nan
    z = QF.zscore(x, dim=1)
    z_pos = QF.zscore(x * 2.5 + 3.0, dim=1)
    torch.testing.assert_close(z_pos, z, rtol=1e-10, atol=1e-10, equal_nan=True)
    # A negative scale flips the sign of the z-scores.
    z_neg = QF.zscore(x * -1.5 + 1.0, dim=1)
    torch.testing.assert_close(
        z_neg, -z, rtol=1e-10, atol=1e-10, equal_nan=True
    )


def test_zscore_dtype_and_basic_properties() -> None:
    """Test dtype/device/shape preservation for float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(3, 10, dtype=dtype)
        z = QF.zscore(x, dim=1)
        assert_basic_properties(z, x)
        z_all = QF.zscore(x, dim=None, unbiased=False)
        assert_basic_properties(z_all, x)


def test_zscore_with_infinity() -> None:
    """Test that slices containing infinities yield NaN values."""
    x = torch.tensor(
        [
            [1.0, math.inf, 2.0],
            [1.0, -math.inf, 2.0],
            [1.0, math.inf, -math.inf],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.float64,
    )
    for unbiased in (True, False):
        z = QF.zscore(x, dim=1, unbiased=unbiased)
        # An infinite value makes the mean and the standard deviation
        # infinite or undefined, so the whole slice becomes NaN.
        assert z[:3].isnan().all()
        # A finite slice is standardized normally.
        assert z[3].isfinite().all()
