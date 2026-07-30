import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF
from tests.functions.test_utils import assert_basic_properties


def _naive_nandiff(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Naive reference for a 1D tensor via indices of valid elements."""
    result = torch.full_like(x, math.nan)
    valid_indices = [
        i for i in range(x.shape[0]) if not math.isnan(x[i].item())
    ]
    for k, i in enumerate(valid_indices):
        if 0 <= k - shift < len(valid_indices):
            result[i] = x[i] - x[valid_indices[k - shift]]
    return result


def test_nandiff_matches_naive_reference() -> None:
    """Test nandiff against a naive reference over valid indices."""
    torch.manual_seed(0)
    for _ in range(20):
        n = int(torch.randint(1, 26, ()).item())
        x = torch.randn(n, dtype=torch.float64)
        x[torch.rand(n) < 0.35] = math.nan
        for shift in (1, 2, 3, -1, -2, 0):
            torch.testing.assert_close(
                QF.nandiff(x, shift),
                _naive_nandiff(x, shift),
                equal_nan=True,
            )


def test_nandiff_equals_shift_difference_on_nan_free_data() -> None:
    """Test that nandiff equals x - QF.shift(x) when there are no NaNs."""
    torch.manual_seed(1)
    x = torch.randn(30)
    for shift in (1, 2, -1, 0):
        torch.testing.assert_close(
            QF.nandiff(x, shift),
            x - QF.shift(x, shift, -1),
            equal_nan=True,
        )
    x2 = torch.randn(4, 20)
    for dim in (0, 1):
        torch.testing.assert_close(
            QF.nandiff(x2, 2, dim=dim),
            x2 - QF.shift(x2, 2, dim),
            equal_nan=True,
        )


def test_nandiff_matches_pandas_diff_of_dropna() -> None:
    """Test nandiff against pandas dropna().diff() realigned to the
    original index."""
    torch.manual_seed(2)
    for _ in range(10):
        x = torch.randn(20, dtype=torch.float64)
        x[torch.rand(20) < 0.35] = math.nan
        s = pd.Series(x.numpy())
        for shift in (1, 2, 3, -1, -2, 0):
            expected = s.dropna().diff(shift).reindex(s.index).to_numpy()
            np.testing.assert_allclose(
                QF.nandiff(x, shift).numpy(), expected, equal_nan=True
            )


def test_nandiff_edge_cases() -> None:
    """Test leading valid positions, all-NaN, single valid, and shift=0."""
    # The first `shift` valid positions become NaN.
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    torch.testing.assert_close(
        QF.nandiff(x, 2),
        torch.tensor([math.nan, math.nan, math.nan, 3.0]),
        equal_nan=True,
    )

    # All-NaN input yields all-NaN output.
    assert torch.isnan(QF.nandiff(torch.full((4,), math.nan))).all()

    # A single valid value has no predecessor or successor.
    x = torch.tensor([math.nan, 5.0, math.nan])
    for shift in (1, -1):
        torch.testing.assert_close(
            QF.nandiff(x, shift),
            torch.full((3,), math.nan),
            equal_nan=True,
        )

    # shift=0 yields zeros at valid positions.
    torch.testing.assert_close(
        QF.nandiff(x, 0),
        torch.tensor([math.nan, 0.0, math.nan]),
        equal_nan=True,
    )

    # Single-element tensors.
    assert math.isnan(QF.nandiff(torch.tensor([3.0]), 1).item())
    assert QF.nandiff(torch.tensor([3.0]), 0).item() == 0.0


def test_nandiff_multi_dim_and_negative_dim() -> None:
    """Test nandiff on 2D/3D tensors and with negative dimensions."""
    torch.manual_seed(3)
    x = torch.randn(4, 5, 6)
    x[torch.rand(x.shape) < 0.3] = math.nan
    for dim in range(3):
        assert QF.nandiff(x, 1, dim=dim).shape == x.shape

    # Negative dims equal their positive counterparts.
    torch.testing.assert_close(
        QF.nandiff(x, 2, dim=-1), QF.nandiff(x, 2, dim=2), equal_nan=True
    )
    torch.testing.assert_close(
        QF.nandiff(x, 1, dim=-3), QF.nandiff(x, 1, dim=0), equal_nan=True
    )

    # 2D computations match row/column-wise 1D computations.
    x2 = x[0]
    result_dim0 = QF.nandiff(x2, 1, dim=0)
    result_dim1 = QF.nandiff(x2, 1, dim=1)
    for i in range(x2.shape[1]):
        torch.testing.assert_close(
            result_dim0[:, i], QF.nandiff(x2[:, i], 1), equal_nan=True
        )
    for i in range(x2.shape[0]):
        torch.testing.assert_close(
            result_dim1[i], QF.nandiff(x2[i], 1), equal_nan=True
        )


def test_nandiff_dtype_and_basic_properties() -> None:
    """Test dtype/device/shape preservation for float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=dtype)
        result = QF.nandiff(x)
        assert_basic_properties(result, x)


def test_nandiff_integer_dtype() -> None:
    """Test that integer tensors behave like x - shift(x) with 0 fill."""
    # Integer tensors have no NaNs to skip; nanshift fills vacated
    # positions with 0, so the first `shift` elements are returned as is.
    x = torch.tensor([5, 7, 10])
    result = QF.nandiff(x, 1)
    assert result.dtype == x.dtype
    torch.testing.assert_close(result, torch.tensor([5, 2, 3]))
    torch.testing.assert_close(result, x - QF.shift(x, 1, -1))


def test_nandiff_with_infinity() -> None:
    """Test IEEE 754 arithmetic with infinite values."""
    x = torch.tensor([1.0, math.inf, 2.0])
    result = QF.nandiff(x, 1)
    assert math.isnan(result[0].item())
    assert torch.isposinf(result[1])  # inf - 1 = inf
    assert torch.isneginf(result[2])  # 2 - inf = -inf

    # inf - inf yields NaN when infinities are shifted onto each other.
    x = torch.tensor([math.inf, math.nan, math.inf])
    result = QF.nandiff(x, 1)
    assert torch.isnan(result).all()
