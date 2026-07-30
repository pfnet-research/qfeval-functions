import math

import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_drawdown_known_path() -> None:
    """A known price path yields the expected drawdown series, starting
    at 0 for the first element."""
    x = torch.tensor([100.0, 120.0, 90.0, 130.0, 65.0])
    result = QF.drawdown(x)
    expected = torch.tensor([0.0, 0.0, -0.25, 0.0, -0.5])
    torch.testing.assert_close(result, expected)
    assert result[0].item() == 0.0


def test_drawdown_numpy_reference() -> None:
    """Drawdown matches x / np.maximum.accumulate(x) - 1 on random
    positive walks along both dimensions."""
    torch.manual_seed(42)
    x = torch.exp(torch.cumsum(torch.randn(4, 50, dtype=torch.float64), 1))
    x_np = x.numpy()

    result_dim1 = QF.drawdown(x, dim=1)
    expected_dim1 = x_np / np.maximum.accumulate(x_np, axis=1) - 1
    np.testing.assert_allclose(result_dim1.numpy(), expected_dim1)

    result_dim0 = QF.drawdown(x, dim=0)
    expected_dim0 = x_np / np.maximum.accumulate(x_np, axis=0) - 1
    np.testing.assert_allclose(result_dim0.numpy(), expected_dim0)


def test_drawdown_monotone_increasing() -> None:
    """A monotonically increasing series has zero drawdown everywhere."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    torch.testing.assert_close(QF.drawdown(x), torch.zeros_like(x))


def test_drawdown_non_positive_values() -> None:
    """All drawdown values of a positive price series are non-positive
    (up to floating point noise), and a new high resets it to 0."""
    torch.manual_seed(0)
    x = torch.exp(torch.cumsum(torch.randn(100, dtype=torch.float64), 0))
    result = QF.drawdown(x)
    assert (result <= 1e-15).all()

    x = torch.tensor([100.0, 80.0, 120.0, 110.0, 130.0])
    result = QF.drawdown(x)
    expected = torch.tensor([0.0, -0.2, 0.0, -1.0 / 12.0, 0.0])
    torch.testing.assert_close(result, expected)


def test_drawdown_nan_stickiness() -> None:
    """A NaN at position k makes all outputs at or after k NaN, while the
    outputs before k are unaffected."""
    torch.manual_seed(1)
    x = torch.exp(torch.cumsum(torch.randn(30, dtype=torch.float64), 0))
    expected_prefix = QF.drawdown(x[:10])
    k = 10
    x[k] = math.nan
    result = QF.drawdown(x)
    torch.testing.assert_close(result[:k], expected_prefix)
    assert torch.isfinite(result[:k]).all()
    assert torch.isnan(result[k:]).all()


def test_drawdown_multi_dimensional() -> None:
    """2D (dim=0/1), 3D, and negative dimensions match per-slice results."""
    torch.manual_seed(2)
    x = torch.exp(torch.cumsum(torch.randn(4, 20, dtype=torch.float64), 1))

    result_dim1 = QF.drawdown(x, dim=1)
    for row in range(x.shape[0]):
        torch.testing.assert_close(result_dim1[row], QF.drawdown(x[row]))

    result_dim0 = QF.drawdown(x.t(), dim=0)
    torch.testing.assert_close(result_dim0, result_dim1.t())

    result_neg = QF.drawdown(x, dim=-1)
    torch.testing.assert_close(result_neg, result_dim1)

    x3d = torch.exp(torch.cumsum(torch.randn(2, 3, 10, dtype=torch.float64), 2))
    result_3d = QF.drawdown(x3d, dim=2)
    assert result_3d.shape == x3d.shape
    for i in range(x3d.shape[0]):
        for j in range(x3d.shape[1]):
            torch.testing.assert_close(result_3d[i, j], QF.drawdown(x3d[i, j]))
    torch.testing.assert_close(QF.drawdown(x3d, dim=-1), result_3d)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_drawdown_dtype_preservation(dtype: torch.dtype) -> None:
    """The output preserves the input dtype, device, and shape."""
    x = torch.tensor([100.0, 120.0, 90.0, 130.0, 65.0], dtype=dtype)
    assert_basic_properties(QF.drawdown(x), x)
