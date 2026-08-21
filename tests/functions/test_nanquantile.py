import math
import warnings
from typing import Tuple

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties

QS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
INTERPOLATIONS = ("linear", "lower", "higher", "nearest", "midpoint")


def _random_with_nans(
    shape: Tuple[int, ...], seed: int, nan_ratio: float = 0.3
) -> torch.Tensor:
    """Return a float64 tensor with randomly placed NaN values."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    x[torch.rand(shape, generator=generator) < nan_ratio] = math.nan
    return x


def test_nanquantile_vs_numpy_2d() -> None:
    """Compare with numpy.nanquantile for scalar dims on a 2D tensor."""
    x = _random_with_nans((13, 17), seed=1)
    for q in QS:
        for dim in (None, 0, 1, -1):
            for keepdim in (False, True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    expected = np.nanquantile(
                        x.numpy(), q, axis=dim, keepdims=keepdim
                    )
                actual = QF.nanquantile(x, q, dim=dim, keepdim=keepdim)
                assert actual.shape == np.asarray(expected).shape
                np.testing.assert_allclose(
                    actual.numpy(), expected, equal_nan=True
                )


def test_nanquantile_interpolation_modes_vs_numpy() -> None:
    """All five interpolation modes must match numpy's methods."""
    x = _random_with_nans((11, 19), seed=2)
    for q in QS:
        for interpolation in INTERPOLATIONS:
            expected = np.nanquantile(
                x.numpy(), q, axis=1, method=interpolation
            )
            actual = QF.nanquantile(x, q, dim=1, interpolation=interpolation)
            np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True)


def test_nanquantile_vs_numpy_tuple_dims() -> None:
    """Compare with numpy.nanquantile for tuple dims on a 3D tensor."""
    x = _random_with_nans((5, 6, 7), seed=3)
    for q in (0.0, 0.25, 0.5, 0.9, 1.0):
        for dim in ((0, 2), (1, 2), (0, 1, 2), (-1, 0), (1,)):
            for keepdim in (False, True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    expected = np.nanquantile(
                        x.numpy(), q, axis=dim, keepdims=keepdim
                    )
                actual = QF.nanquantile(x, q, dim=dim, keepdim=keepdim)
                assert actual.shape == np.asarray(expected).shape
                np.testing.assert_allclose(
                    actual.numpy(), expected, equal_nan=True
                )


def test_nanquantile_vs_numpy_4d() -> None:
    """Compare with numpy.nanquantile for tuple dims on a 4D tensor."""
    x = _random_with_nans((2, 3, 4, 5), seed=4)
    for dim in ((1, 3), (0, 2), (-1, -2)):
        for keepdim in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = np.nanquantile(
                    x.numpy(), 0.75, axis=dim, keepdims=keepdim
                )
            actual = QF.nanquantile(x, 0.75, dim=dim, keepdim=keepdim)
            assert actual.shape == np.asarray(expected).shape
            np.testing.assert_allclose(actual.numpy(), expected, equal_nan=True)


def test_nanquantile_no_nan_matches_torch_quantile() -> None:
    """Without NaN values, the result must match torch.quantile."""
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(7, 9, generator=generator, dtype=torch.float64)
    for q in QS:
        torch.testing.assert_close(QF.nanquantile(x, q), torch.quantile(x, q))
        torch.testing.assert_close(
            QF.nanquantile(x, q, dim=0), torch.quantile(x, q, dim=0)
        )
        torch.testing.assert_close(
            QF.nanquantile(x, q, dim=1, keepdim=True),
            torch.quantile(x, q, dim=1, keepdim=True),
        )


def test_nanquantile_all_nan_slice() -> None:
    """A slice consisting only of NaN values must yield NaN."""
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [math.nan, math.nan, math.nan]],
        dtype=torch.float64,
    )
    result = QF.nanquantile(x, 0.5, dim=1)
    assert result[0].item() == 2.0
    assert math.isnan(result[1].item())

    # Global reduction over an all-NaN tensor.
    all_nan = torch.tensor([math.nan, math.nan])
    assert math.isnan(QF.nanquantile(all_nan, 0.5).item())

    # Tuple-dim reduction with an all-NaN slice.
    x3 = torch.full((2, 3, 4), math.nan, dtype=torch.float64)
    x3[0] = 1.0
    result3 = QF.nanquantile(x3, 0.25, dim=(1, 2))
    assert result3[0].item() == 1.0
    assert math.isnan(result3[1].item())


def test_nanquantile_invalid_q_raises_value_error() -> None:
    """Out-of-range quantiles must raise ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        QF.nanquantile(x, -0.1)
    with pytest.raises(ValueError):
        QF.nanquantile(x, 1.1)
    with pytest.raises(ValueError):
        QF.nanquantile(x, math.nan, dim=0)


def test_nanquantile_invalid_interpolation_raises() -> None:
    """An unsupported interpolation mode must raise (via torch)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        QF.nanquantile(x, 0.5, interpolation="bogus")


def test_nanquantile_single_element() -> None:
    """A single-element tensor must return its value for any q."""
    x = torch.tensor([42.0])
    for q in (0.0, 0.5, 1.0):
        torch.testing.assert_close(QF.nanquantile(x, q), torch.tensor(42.0))
    result = QF.nanquantile(x, 0.5, dim=0, keepdim=True)
    assert result.shape == (1,)
    torch.testing.assert_close(result, torch.tensor([42.0]))


def test_nanquantile_with_infinity() -> None:
    """Quantiles with infinite endpoints must match numpy."""
    x = torch.tensor(
        [1.0, math.inf, 2.0, -math.inf, math.nan], dtype=torch.float64
    )
    assert QF.nanquantile(x, 0.25).item() == -math.inf
    assert QF.nanquantile(x, 0.5).item() == 1.5
    assert QF.nanquantile(x, 0.75).item() == math.inf
    # Index-selecting interpolations return the infinite endpoints.
    assert QF.nanquantile(x, 0.0, interpolation="lower").item() == -math.inf
    assert QF.nanquantile(x, 1.0, interpolation="higher").item() == math.inf
    # Linear interpolation at an infinite endpoint yields NaN
    # (inf - inf); torch and numpy agree on this behavior.
    for q in QS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = np.nanquantile(x.numpy(), q)
        np.testing.assert_allclose(
            QF.nanquantile(x, q).numpy(), expected, equal_nan=True
        )


def test_nanquantile_keepdim_shapes() -> None:
    """Check output shapes for every form of the dim argument."""
    x = torch.randn(2, 3, 4)
    assert QF.nanquantile(x, 0.5).shape == ()
    assert QF.nanquantile(x, 0.5, keepdim=True).shape == (1, 1, 1)
    assert QF.nanquantile(x, 0.5, dim=0).shape == (3, 4)
    assert QF.nanquantile(x, 0.5, dim=1).shape == (2, 4)
    assert QF.nanquantile(x, 0.5, dim=1, keepdim=True).shape == (2, 1, 4)
    assert QF.nanquantile(x, 0.5, dim=-1).shape == (2, 3)
    assert QF.nanquantile(x, 0.5, dim=-1, keepdim=True).shape == (2, 3, 1)
    assert QF.nanquantile(x, 0.5, dim=(0, 2)).shape == (3,)
    assert QF.nanquantile(x, 0.5, dim=(0, 2), keepdim=True).shape == (1, 3, 1)
    assert QF.nanquantile(x, 0.5, dim=(-1, -3)).shape == (3,)
    assert QF.nanquantile(x, 0.5, dim=(-1, -3), keepdim=True).shape == (
        1,
        3,
        1,
    )
    assert QF.nanquantile(x, 0.5, dim=(0, 1, 2)).shape == ()
    assert QF.nanquantile(x, 0.5, dim=(0, 1, 2), keepdim=True).shape == (
        1,
        1,
        1,
    )


def test_nanquantile_negative_dims_in_tuple() -> None:
    """Negative and unsorted tuple dims must match their positive form."""
    x = _random_with_nans((4, 5, 6), seed=6)
    torch.testing.assert_close(
        QF.nanquantile(x, 0.3, dim=(-1, -3)),
        QF.nanquantile(x, 0.3, dim=(0, 2)),
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.nanquantile(x, 0.3, dim=(2, 0), keepdim=True),
        QF.nanquantile(x, 0.3, dim=(0, 2), keepdim=True),
        equal_nan=True,
    )


def test_nanquantile_dtype_preservation() -> None:
    """The result must preserve float32/float64 dtypes and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(3, 5, dtype=dtype)
        x[0, 0] = math.nan
        result = QF.nanquantile(x, 0.25, dim=1)
        assert_basic_properties(result, x, expected_shape=torch.Size([3]))
        assert QF.nanquantile(x, 0.25).dtype == dtype
        assert QF.nanquantile(x, 0.25, dim=(0, 1)).dtype == dtype
