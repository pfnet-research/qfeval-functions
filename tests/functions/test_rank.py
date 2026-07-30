import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_rank_pandas_comparison_random_with_nans() -> None:
    """Test rank against pandas on random data with random NaN values."""
    torch.manual_seed(42)
    x = torch.randn(20, 50, dtype=torch.float64)
    x[torch.rand(20, 50) < 0.2] = math.nan
    df = pd.DataFrame(x.numpy())
    for pct in (True, False):
        expected = df.rank(axis=1, method="average", pct=pct).to_numpy()
        np.testing.assert_allclose(
            QF.rank(x, dim=1, pct=pct).numpy(), expected, equal_nan=True
        )
        np.testing.assert_allclose(
            QF.rank(x, dim=-1, pct=pct).numpy(), expected, equal_nan=True
        )
        expected_t = df.rank(axis=0, method="average", pct=pct).to_numpy()
        np.testing.assert_allclose(
            QF.rank(x, dim=0, pct=pct).numpy(), expected_t, equal_nan=True
        )


def test_rank_pandas_comparison_heavy_ties() -> None:
    """Test rank on heavily tied data against pandas."""
    torch.manual_seed(0)
    x = torch.randint(0, 5, (10, 30)).to(torch.float64)
    x[torch.rand(10, 30) < 0.15] = math.nan
    df = pd.DataFrame(x.numpy())
    for pct in (True, False):
        expected = df.rank(axis=1, method="average", pct=pct).to_numpy()
        np.testing.assert_allclose(
            QF.rank(x, dim=1, pct=pct).numpy(), expected, equal_nan=True
        )


def test_rank_pandas_comparison_infinity() -> None:
    """Test rank on slices with infinities and NaNs against pandas."""
    x = torch.tensor(
        [
            [1.0, math.inf, 2.0, -math.inf],
            [math.inf, math.inf, 1.0, 2.0],
            [math.inf, math.nan, 1.0, math.inf],
            [-math.inf, math.nan, math.nan, 1.0],
            [math.inf, math.inf, math.inf, math.inf],
            [-math.inf, math.nan, math.inf, -math.inf],
        ],
        dtype=torch.float64,
    )
    df = pd.DataFrame(x.numpy())
    for pct in (True, False):
        expected = df.rank(axis=1, method="average", pct=pct).to_numpy()
        np.testing.assert_allclose(
            QF.rank(x, dim=1, pct=pct).numpy(), expected, equal_nan=True
        )


def test_rank_pandas_comparison_1d_series() -> None:
    """Test rank on a 1D tensor against a pandas Series."""
    x = torch.tensor([5.0, math.nan, 3.0, 3.0, 1.0], dtype=torch.float64)
    s = pd.Series(x.numpy())
    for pct in (True, False):
        expected = s.rank(method="average", pct=pct).to_numpy()
        for dim in (0, -1):
            np.testing.assert_allclose(
                QF.rank(x, dim=dim, pct=pct).numpy(),
                expected,
                equal_nan=True,
            )


def test_rank_known_values() -> None:
    """Test rank against manually computed values."""
    x = torch.tensor([3.0, 1.0, 2.0])
    torch.testing.assert_close(
        QF.rank(x, pct=False), torch.tensor([3.0, 1.0, 2.0])
    )
    torch.testing.assert_close(
        QF.rank(x, pct=True), torch.tensor([1.0, 1.0 / 3.0, 2.0 / 3.0])
    )
    x = torch.tensor([1.0, 1.0, 2.0])
    torch.testing.assert_close(
        QF.rank(x, pct=False), torch.tensor([1.5, 1.5, 3.0])
    )
    torch.testing.assert_close(
        QF.rank(x, pct=True), torch.tensor([0.5, 0.5, 1.0])
    )


def test_rank_all_nan_slice() -> None:
    """Test that an all-NaN slice results in all NaN values."""
    x = torch.tensor(
        [[math.nan, math.nan, math.nan], [1.0, 2.0, 3.0]],
        dtype=torch.float64,
    )
    for pct in (True, False):
        result = QF.rank(x, dim=1, pct=pct)
        assert result[0].isnan().all()
        assert not result[1].isnan().any()


def test_rank_single_valid_element() -> None:
    """Test that a single valid element gets rank 1.0 (pct rank 1.0)."""
    x = torch.tensor([[math.nan, 3.0, math.nan]], dtype=torch.float64)
    expected = torch.tensor([[math.nan, 1.0, math.nan]], dtype=torch.float64)
    for pct in (True, False):
        torch.testing.assert_close(
            QF.rank(x, dim=1, pct=pct), expected, equal_nan=True
        )
    for pct in (True, False):
        torch.testing.assert_close(
            QF.rank(torch.tensor([5.0]), pct=pct), torch.tensor([1.0])
        )


def test_rank_value_range_and_rank_sum() -> None:
    """Test range properties and the rank-sum identity."""
    torch.manual_seed(17)
    x = torch.randn(10, 20, dtype=torch.float64)
    # Without NaNs, ranks of distinct values are a permutation of 1..n.
    pct_rank = QF.rank(x, dim=1, pct=True)
    assert ((pct_rank > 0) & (pct_rank <= 1)).all()
    raw_rank = QF.rank(x, dim=1, pct=False)
    assert ((raw_rank >= 1) & (raw_rank <= 20)).all()
    expected_sum = torch.full((10,), 20 * 21 / 2, dtype=torch.float64)
    torch.testing.assert_close(raw_rank.sum(dim=1), expected_sum)
    # With NaNs, the percentile denominator is the number of valid values.
    x[torch.rand(10, 20) < 0.3] = math.nan
    valid = ~x.isnan()
    nv = valid.sum(dim=1).to(torch.float64)
    raw_rank = QF.rank(x, dim=1, pct=False)
    torch.testing.assert_close(raw_rank.nansum(dim=1), nv * (nv + 1) / 2)
    pct_rank = QF.rank(x, dim=1, pct=True)
    assert ((pct_rank[valid] > 0) & (pct_rank[valid] <= 1)).all()
    torch.testing.assert_close(pct_rank, raw_rank / nv[:, None], equal_nan=True)


def test_rank_3d_and_negative_dims() -> None:
    """Test rank on a 3D tensor along every (negative) dimension."""
    torch.manual_seed(11)
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    x[torch.rand(3, 4, 5) < 0.2] = math.nan
    for dim in (0, 1, 2, -1, -2, -3):
        for pct in (True, False):
            result = QF.rank(x, dim=dim, pct=pct)
            assert result.shape == x.shape
            xt = x.transpose(dim, -1).contiguous()
            flat = xt.reshape(-1, xt.shape[-1]).numpy()
            expected_flat = pd.DataFrame(flat).rank(
                axis=1, method="average", pct=pct
            )
            expected = (
                torch.from_numpy(expected_flat.to_numpy())
                .reshape(xt.shape)
                .transpose(dim, -1)
            )
            np.testing.assert_allclose(
                result.numpy(), expected.numpy(), equal_nan=True
            )


def test_rank_non_contiguous_input() -> None:
    """Test that non-contiguous inputs give the same result."""
    torch.manual_seed(13)
    base = torch.randn(6, 8, dtype=torch.float64)
    base[torch.rand(6, 8) < 0.2] = math.nan
    x = base.t()
    assert not x.is_contiguous()
    for pct in (True, False):
        for dim in (0, 1):
            torch.testing.assert_close(
                QF.rank(x, dim=dim, pct=pct),
                QF.rank(x.contiguous(), dim=dim, pct=pct),
                equal_nan=True,
            )
    expected = pd.DataFrame(x.numpy()).rank(axis=0, method="average", pct=True)
    np.testing.assert_allclose(
        QF.rank(x, dim=0, pct=True).numpy(),
        expected.to_numpy(),
        equal_nan=True,
    )


def test_rank_dtype_and_basic_properties() -> None:
    """Test dtype/device/shape preservation for float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[3.0, 1.0, math.nan, 2.0]], dtype=dtype)
        result = QF.rank(x, dim=1, pct=False)
        assert_basic_properties(result, x)
        expected = torch.tensor([[3.0, 1.0, math.nan, 2.0]], dtype=dtype)
        torch.testing.assert_close(result, expected, equal_nan=True)
        result_pct = QF.rank(x, dim=1, pct=True)
        assert_basic_properties(result_pct, x)


def test_rank_type_error_for_non_float_tensors() -> None:
    """Test that non-floating-point tensors are rejected."""
    with pytest.raises(TypeError):
        QF.rank(torch.tensor([3, 1, 2]))
    with pytest.raises(TypeError):
        QF.rank(torch.tensor([True, False, True]))
