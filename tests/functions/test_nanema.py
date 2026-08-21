import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import assert_basic_properties


def _naive_nanema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Naive O(n^2) reference over valid weights for a 1D tensor."""
    n = x.shape[0]
    result = torch.full((n,), math.nan, dtype=x.dtype)
    for i in range(n):
        if math.isnan(x[i].item()):
            continue
        num = 0.0
        den = 0.0
        for j in range(i + 1):
            if math.isnan(x[j].item()):
                continue
            weight = (1 - alpha) ** (i - j)
            num += x[j].item() * weight
            den += weight
        result[i] = num / den
    return result


def test_nanema_equals_ema_on_nan_free_data() -> None:
    """Test that nanema is identical to ema when the input has no NaNs."""
    torch.manual_seed(0)
    x1 = torch.randn(100)
    x2 = torch.randn(8, 40)
    for alpha in (0.1, 0.5, 0.9):
        torch.testing.assert_close(
            QF.nanema(x1, alpha), QF.ema(x1, alpha), rtol=0, atol=0
        )
        for dim in (0, 1, -1):
            torch.testing.assert_close(
                QF.nanema(x2, alpha, dim=dim),
                QF.ema(x2, alpha, dim=dim),
                rtol=0,
                atol=0,
            )


def test_nanema_matches_pandas_ewm_at_valid_positions() -> None:
    """Test nanema against pandas ewm(adjust=True, ignore_na=False).

    The outputs are compared only at valid positions because pandas
    carries the previous mean forward at NaN positions while nanema
    returns NaN there.
    """
    patterns = [
        [math.nan, math.nan, 1.0, 2.0, 3.0, 4.0],  # leading NaNs
        [1.0, 2.0, math.nan, 4.0, 5.0],  # interior single NaN
        [1.0, math.nan, math.nan, math.nan, 5.0, 6.0],  # interior run
        [1.0, 2.0, 3.0, math.nan, math.nan],  # trailing NaNs
    ]
    for values in patterns:
        for alpha in (0.1, 0.5, 0.9):
            x = torch.tensor(values, dtype=torch.float64)
            result = QF.nanema(x, alpha).numpy()
            expected = (
                pd.Series(values)
                .ewm(alpha=alpha, adjust=True, ignore_na=False)
                .mean()
                .to_numpy()
            )
            valid = ~np.isnan(np.array(values))
            np.testing.assert_allclose(
                result[valid], expected[valid], rtol=1e-6, atol=1e-6
            )


def test_nanema_matches_pandas_ewm_2d_random() -> None:
    """Test nanema against pandas ewm on random 2D data with NaNs."""
    torch.manual_seed(1)
    x = torch.randn(50, 5, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.3] = math.nan
    df = pd.DataFrame(x.numpy())
    valid = ~np.isnan(x.numpy())
    for alpha in (0.1, 0.5, 0.9):
        result = QF.nanema(x, alpha, dim=0).numpy()
        expected = (
            df.ewm(alpha=alpha, adjust=True, ignore_na=False).mean().to_numpy()
        )
        np.testing.assert_allclose(
            result[valid], expected[valid], rtol=1e-6, atol=1e-6
        )


def test_nanema_matches_naive_reference() -> None:
    """Test nanema against a naive double-loop reference."""
    torch.manual_seed(2)
    for _ in range(20):
        n = int(torch.randint(1, 31, ()).item())
        x = torch.randn(n, dtype=torch.float64)
        x[torch.rand(n) < 0.35] = math.nan
        # For these alphas and n <= 30, the decay cutoff of the
        # underlying ema kernel (weights below 1e-8 are dropped) never
        # kicks in, so results are exact up to rounding errors.
        for alpha in (0.1, 0.5):
            torch.testing.assert_close(
                QF.nanema(x, alpha),
                _naive_nanema(x, alpha),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        # For alpha=0.9, weights beyond distance 8 fall below the 1e-8
        # cutoff, so the deviation can reach roughly 1e-7.
        torch.testing.assert_close(
            QF.nanema(x, 0.9),
            _naive_nanema(x, 0.9),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )


def test_nanema_leading_nans_and_single_valid() -> None:
    """Test leading NaNs, all-NaN inputs, and single valid values."""
    # Leading NaNs stay NaN and the first valid value is returned as is.
    x = torch.tensor([math.nan, math.nan, 2.0, 4.0])
    result = QF.nanema(x, alpha=0.3)
    assert torch.isnan(result[:2]).all()
    assert result[2].item() == 2.0

    # All-NaN input yields all-NaN output.
    result = QF.nanema(torch.full((5,), math.nan), alpha=0.5)
    assert torch.isnan(result).all()

    # A single valid value is returned as is.
    x = torch.tensor([math.nan, 5.0, math.nan])
    torch.testing.assert_close(
        QF.nanema(x, alpha=0.5), x, rtol=0, atol=0, equal_nan=True
    )

    # Single-element tensors.
    assert QF.nanema(torch.tensor([7.0]), alpha=0.5).item() == 7.0
    assert math.isnan(QF.nanema(torch.tensor([math.nan]), alpha=0.5).item())


def test_nanema_empty_tensor() -> None:
    """Test that empty tensors pass through with the same shape."""
    x = torch.empty(0)
    assert QF.nanema(x, alpha=0.5).shape == x.shape


def test_nanema_output_nan_mask() -> None:
    """Test that output NaNs are exactly the input NaNs plus positions
    before the first valid value."""
    torch.manual_seed(3)
    x = torch.randn(6, 25, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.4] = math.nan
    x[:3, 0] = math.nan  # Make some rows start with NaN.
    result = QF.nanema(x, alpha=0.2, dim=1)
    valid = ~x.isnan()
    seen_valid = torch.cummax(valid, dim=1).values
    expected_nan = x.isnan() | ~seen_valid
    assert torch.equal(result.isnan(), expected_nan)

    # When every series starts with a valid value, the output NaN mask
    # equals the input NaN mask.
    x[:, 0] = 1.0
    result = QF.nanema(x, alpha=0.2, dim=1)
    assert torch.equal(result.isnan(), x.isnan())


def test_nanema_inf_propagation() -> None:
    """Test that ±inf inputs propagate into subsequent outputs."""
    x = torch.tensor([1.0, math.inf, 2.0, math.nan, 3.0])
    result = QF.nanema(x, alpha=0.5)
    assert result[0].item() == 1.0
    assert torch.isposinf(result[1])
    assert torch.isposinf(result[2])
    assert torch.isnan(result[3])  # NaN input position stays NaN.
    assert torch.isposinf(result[4])

    # Mixing +inf and -inf makes subsequent sums NaN.
    x = torch.tensor([math.inf, -math.inf, 1.0])
    result = QF.nanema(x, alpha=0.5)
    assert torch.isposinf(result[0])
    assert torch.isnan(result[1])
    assert torch.isnan(result[2])


def test_nanema_multi_dim_and_negative_dim() -> None:
    """Test nanema on 3D tensors and with negative dimensions."""
    torch.manual_seed(4)
    x = torch.randn(3, 4, 6)
    x[torch.rand(x.shape) < 0.3] = math.nan
    for dim in range(3):
        assert QF.nanema(x, 0.4, dim=dim).shape == x.shape

    # Negative dims equal their positive counterparts.
    torch.testing.assert_close(
        QF.nanema(x, 0.4, dim=-1), QF.nanema(x, 0.4, dim=2), equal_nan=True
    )
    torch.testing.assert_close(
        QF.nanema(x, 0.4, dim=-2), QF.nanema(x, 0.4, dim=1), equal_nan=True
    )

    # 3D computation matches slice-wise 1D computation.
    result = QF.nanema(x, 0.4, dim=0)
    torch.testing.assert_close(
        result[:, 1, 2], QF.nanema(x[:, 1, 2], 0.4), equal_nan=True
    )


def test_nanema_dtype_and_basic_properties() -> None:
    """Test dtype/device/shape preservation for float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=dtype)
        result = QF.nanema(x, alpha=0.3)
        assert_basic_properties(result, x)
