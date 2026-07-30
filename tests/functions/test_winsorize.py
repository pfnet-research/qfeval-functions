import math
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _random_with_nans(
    shape: Tuple[int, ...], seed: int, nan_ratio: float = 0.3
) -> torch.Tensor:
    """Return a float64 tensor with randomly placed NaN values."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    x[torch.rand(shape, generator=generator) < nan_ratio] = math.nan
    return x


def test_winsorize_matches_pandas_clip_rowwise() -> None:
    """Rowwise winsorization matches the pandas clip/quantile idiom."""
    for nan_ratio in (0.0, 0.2):
        x = _random_with_nans((12, 30), seed=11, nan_ratio=nan_ratio)
        result = QF.winsorize(x, lower=0.1, upper=0.9)
        for i in range(x.shape[0]):
            s = pd.Series(x[i].numpy())
            expected = s.clip(s.quantile(0.1), s.quantile(0.9))
            np.testing.assert_allclose(
                result[i].numpy(), expected.to_numpy(), equal_nan=True
            )


def test_winsorize_matches_pandas_clip_columnwise() -> None:
    """Columnwise winsorization matches the pandas clip/quantile idiom."""
    for nan_ratio in (0.0, 0.2):
        x = _random_with_nans((25, 6), seed=12, nan_ratio=nan_ratio)
        result = QF.winsorize(x, dim=0)
        for j in range(x.shape[1]):
            s = pd.Series(x[:, j].numpy())
            expected = s.clip(s.quantile(0.05), s.quantile(0.95))
            np.testing.assert_allclose(
                result[:, j].numpy(), expected.to_numpy(), equal_nan=True
            )


def test_winsorize_preserves_nan_and_ignores_them_for_bounds() -> None:
    """NaN values stay NaN and do not affect the quantile bounds."""
    x = torch.tensor([[math.nan, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)
    result = QF.winsorize(x, lower=0.25, upper=0.75)
    assert math.isnan(result[0, 0].item())
    # The bounds must be those of the valid values only.
    torch.testing.assert_close(
        result[:, 1:], QF.winsorize(x[:, 1:], lower=0.25, upper=0.75)
    )
    expected = torch.tensor(
        [[math.nan, 2.0, 2.0, 3.0, 4.0, 4.0]], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_winsorize_full_range_is_identity() -> None:
    """lower=0 and upper=1 must leave finite data unchanged."""
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    result = QF.winsorize(x, lower=0.0, upper=1.0)
    assert torch.equal(result, x)


def test_winsorize_equal_bounds_collapse_to_quantile() -> None:
    """lower == upper maps every valid value to that quantile."""
    x = torch.tensor([[1.0, math.nan, 4.0, 2.0, 8.0]], dtype=torch.float64)
    result = QF.winsorize(x, lower=0.5, upper=0.5)
    assert QF.nanquantile(x, 0.5, dim=-1).item() == 3.0
    expected = torch.tensor(
        [[3.0, math.nan, 3.0, 3.0, 3.0]], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected, equal_nan=True)


def test_winsorize_invalid_bounds_raise_value_error() -> None:
    """Bound ordering violations must raise ValueError."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        QF.winsorize(x, lower=0.6, upper=0.4)
    with pytest.raises(ValueError):
        QF.winsorize(x, lower=-0.1, upper=0.9)
    with pytest.raises(ValueError):
        QF.winsorize(x, lower=0.1, upper=1.5)


def test_winsorize_bounds_and_interior_values() -> None:
    """Clipped extremes hit the bounds; interior values are untouched."""
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(4, 101, generator=generator, dtype=torch.float64)
    lower, upper = 0.1, 0.9
    result = QF.winsorize(x, lower=lower, upper=upper)
    lo = QF.nanquantile(x, lower, dim=-1, keepdim=True)
    hi = QF.nanquantile(x, upper, dim=-1, keepdim=True)
    # Clipping must actually have occurred in every row.
    assert bool((x < lo).any(dim=-1).all())
    assert bool((x > hi).any(dim=-1).all())
    # The extremes of the result equal the quantile bounds exactly.
    assert torch.equal(result.amin(dim=-1, keepdim=True), lo)
    assert torch.equal(result.amax(dim=-1, keepdim=True), hi)
    assert bool((result >= lo).all())
    assert bool((result <= hi).all())
    # Values inside the bounds are copied without modification.
    inside = (x >= lo) & (x <= hi)
    assert torch.equal(result[inside], x[inside])


def test_winsorize_dim_variants() -> None:
    """Check the default dim, explicit dims, and global winsorization."""
    x = _random_with_nans((6, 7), seed=15, nan_ratio=0.2)
    # The default dimension is the last one.
    torch.testing.assert_close(
        QF.winsorize(x), QF.winsorize(x, dim=-1), equal_nan=True
    )
    torch.testing.assert_close(
        QF.winsorize(x, dim=-1), QF.winsorize(x, dim=1), equal_nan=True
    )
    # Rowwise and columnwise winsorization differ in general.
    rowwise = QF.winsorize(x, dim=1)
    colwise = QF.winsorize(x, dim=0)
    assert not torch.allclose(rowwise.nan_to_num(), colwise.nan_to_num())
    # Global winsorization clamps by the global quantiles.
    result = QF.winsorize(x, lower=0.25, upper=0.75, dim=None)
    lo = QF.nanquantile(x, 0.25)
    hi = QF.nanquantile(x, 0.75)
    torch.testing.assert_close(result, x.clamp(lo, hi), equal_nan=True)


def test_winsorize_tuple_dims_3d() -> None:
    """Tuple dims winsorize jointly over the specified dimensions."""
    x = _random_with_nans((3, 4, 5), seed=16, nan_ratio=0.15)
    result = QF.winsorize(x, lower=0.2, upper=0.8, dim=(1, 2))
    assert result.shape == x.shape
    # Equivalent to flattening the last two dimensions.
    flat = QF.winsorize(x.reshape(3, -1), lower=0.2, upper=0.8, dim=-1)
    torch.testing.assert_close(result, flat.reshape(x.shape), equal_nan=True)
    # Consistent with clamping by the tuple-dim quantiles.
    lo = QF.nanquantile(x, 0.2, dim=(1, 2), keepdim=True)
    hi = QF.nanquantile(x, 0.8, dim=(1, 2), keepdim=True)
    torch.testing.assert_close(result, x.clamp(lo, hi), equal_nan=True)


def test_winsorize_dtype_preservation() -> None:
    """The result must preserve float32/float64 dtypes and shape."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(4, 6, dtype=dtype)
        x[0, 0] = math.nan
        result = QF.winsorize(x)
        assert_basic_properties(result, x)
        assert QF.winsorize(x, dim=None).dtype == dtype
        assert QF.winsorize(x, dim=(0, 1)).dtype == dtype


def test_winsorize_with_infinity() -> None:
    """Infinite values are clipped to the finite quantile bounds."""
    x = torch.tensor([-math.inf, 1.0, 2.0, 3.0, math.inf], dtype=torch.float64)
    result = QF.winsorize(x, lower=0.25, upper=0.75)
    expected = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(result, expected)
    # NaN values are still preserved alongside infinities.
    x = torch.tensor(
        [math.nan, -math.inf, 1.0, 2.0, 3.0, math.inf], dtype=torch.float64
    )
    result = QF.winsorize(x, lower=0.25, upper=0.75)
    expected = torch.tensor(
        [math.nan, 1.0, 1.0, 2.0, 3.0, 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected, equal_nan=True)
