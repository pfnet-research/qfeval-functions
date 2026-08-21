import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_max_drawdown_known_path() -> None:
    """A known price path yields its most negative drawdown, -0.5."""
    x = torch.tensor([100.0, 120.0, 90.0, 130.0, 65.0])
    result = QF.max_drawdown(x)
    assert result.shape == torch.Size([])
    assert result.item() == pytest.approx(-0.5)


def test_max_drawdown_monotone_increasing() -> None:
    """A monotonically non-decreasing series has zero maximum drawdown."""
    x = torch.tensor([1.0, 2.0, 2.0, 3.0, 5.0])
    assert QF.max_drawdown(x).item() == 0.0


def test_max_drawdown_numpy_reference() -> None:
    """Maximum drawdown matches the minimum of the numpy drawdown series
    on random positive walks."""
    torch.manual_seed(42)
    x = torch.exp(torch.cumsum(torch.randn(4, 50, dtype=torch.float64), 1))
    x_np = x.numpy()
    expected = (x_np / np.maximum.accumulate(x_np, axis=1) - 1).min(axis=1)
    result = QF.max_drawdown(x, dim=1)
    np.testing.assert_allclose(result.numpy(), expected)
    assert (result <= 0).all()


def test_max_drawdown_keepdim() -> None:
    """``keepdim`` retains the reduced dimension with size 1."""
    torch.manual_seed(0)
    x = torch.exp(torch.cumsum(torch.randn(3, 20, dtype=torch.float64), 1))
    result = QF.max_drawdown(x, dim=1)
    result_keepdim = QF.max_drawdown(x, dim=1, keepdim=True)
    assert result.shape == (3,)
    assert result_keepdim.shape == (3, 1)
    torch.testing.assert_close(result_keepdim.squeeze(1), result)


def test_max_drawdown_nan_slice() -> None:
    """Any NaN along the reduced dimension makes the result NaN, while
    NaN-free slices are unaffected."""
    torch.manual_seed(1)
    x = torch.exp(torch.cumsum(torch.randn(2, 20, dtype=torch.float64), 1))
    x[0, 10] = math.nan
    result = QF.max_drawdown(x, dim=1)
    assert torch.isnan(result[0])
    assert torch.isfinite(result[1])


def test_max_drawdown_2d_per_row() -> None:
    """Each row of a 2D input is reduced independently."""
    x = torch.tensor(
        [
            [100.0, 120.0, 90.0, 130.0, 65.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [100.0, 50.0, 100.0, 25.0, 100.0],
        ]
    )
    result = QF.max_drawdown(x, dim=1)
    expected = torch.tensor([-0.5, 0.0, -0.75])
    torch.testing.assert_close(result, expected)


def test_max_drawdown_multi_dimensional() -> None:
    """3D and negative dimensions match per-slice results."""
    torch.manual_seed(2)
    x = torch.exp(torch.cumsum(torch.randn(2, 3, 15, dtype=torch.float64), 2))

    result_3d = QF.max_drawdown(x, dim=2)
    assert result_3d.shape == (2, 3)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            torch.testing.assert_close(
                result_3d[i, j], QF.max_drawdown(x[i, j])
            )

    torch.testing.assert_close(QF.max_drawdown(x, dim=-1), result_3d)
    torch.testing.assert_close(
        QF.max_drawdown(x, dim=0),
        QF.drawdown(x, dim=0).amin(dim=0),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_max_drawdown_dtype_preservation(dtype: torch.dtype) -> None:
    """The output preserves dtype and device with a reduced shape."""
    x = torch.tensor([[100.0, 120.0, 90.0], [10.0, 20.0, 30.0]], dtype=dtype)
    result = QF.max_drawdown(x, dim=1)
    assert_basic_properties(result, x, expected_shape=torch.Size([2]))
    result_keepdim = QF.max_drawdown(x, dim=1, keepdim=True)
    assert_basic_properties(
        result_keepdim, x, expected_shape=torch.Size([2, 1])
    )
