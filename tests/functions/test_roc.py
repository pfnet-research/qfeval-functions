import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_roc_known_values() -> None:
    """A 10% increase per step yields a ROC of 0.1."""
    x = torch.tensor([100.0, 110.0, 121.0])
    result = QF.roc(x, span=1)
    expected = torch.tensor([math.nan, 0.1, 0.1])
    torch.testing.assert_close(result, expected, equal_nan=True)


@pytest.mark.parametrize("span", [1, 2, 5, -1, -3])
def test_roc_pandas_pct_change(span: int) -> None:
    """ROC matches pandas pct_change for positive and negative spans."""
    torch.manual_seed(42)
    x = torch.cumsum(torch.randn(50, dtype=torch.float64), dim=0) + 100.0
    result = QF.roc(x, span=span)
    expected = (
        pd.Series(x.numpy()).pct_change(span, fill_method=None).to_numpy()
    )
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)


def test_roc_zero_division() -> None:
    """A zero previous value yields +/-inf, and 0 / 0 yields NaN."""
    x = torch.tensor([0.0, 1.0, 0.0, 0.0, -2.0])
    result = QF.roc(x, span=1)
    assert torch.isnan(result[0])  # leading position
    assert result[1].item() == math.inf  # 1 / 0
    assert result[2].item() == -1.0  # 0 / 1 - 1
    assert torch.isnan(result[3])  # 0 / 0
    assert result[4].item() == -math.inf  # -2 / 0


def test_roc_nan_positions() -> None:
    """A NaN input affects exactly the outputs where it appears as the
    numerator or the denominator."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    result = QF.roc(x, span=1)
    assert torch.isnan(result[0])  # leading position
    assert result[1].item() == pytest.approx(1.0)  # 2 / 1 - 1
    assert torch.isnan(result[2])  # NaN numerator
    assert torch.isnan(result[3])  # NaN denominator
    assert result[4].item() == pytest.approx(0.25)  # 5 / 4 - 1


def test_roc_leading_nan_prefix() -> None:
    """The first ``span`` outputs are NaN and the rest are finite."""
    torch.manual_seed(0)
    x = torch.rand(20) + 1.0
    for span in [1, 3, 7]:
        result = QF.roc(x, span=span)
        assert torch.isnan(result[:span]).all()
        assert torch.isfinite(result[span:]).all()


def test_roc_span_larger_than_length() -> None:
    """A span larger than the dimension size yields all NaN."""
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isnan(QF.roc(x, span=5)).all()
    assert torch.isnan(QF.roc(x, span=-5)).all()


def test_roc_zero_span() -> None:
    """``span=0`` yields all zeros for finite nonzero inputs."""
    x = torch.tensor([1.0, -2.0, 3.5])
    torch.testing.assert_close(QF.roc(x, span=0), torch.zeros_like(x))


def test_roc_multi_dimensional() -> None:
    """2D (dim=0/1), 3D, and negative dimensions match per-slice results."""
    torch.manual_seed(1)
    x = torch.rand(4, 10, dtype=torch.float64) + 1.0

    result_dim1 = QF.roc(x, span=2, dim=1)
    for row in range(x.shape[0]):
        torch.testing.assert_close(
            result_dim1[row], QF.roc(x[row], span=2), equal_nan=True
        )

    result_dim0 = QF.roc(x.t(), span=2, dim=0)
    torch.testing.assert_close(result_dim0, result_dim1.t(), equal_nan=True)

    result_neg = QF.roc(x, span=2, dim=-1)
    torch.testing.assert_close(result_neg, result_dim1, equal_nan=True)

    x3d = torch.rand(2, 3, 8, dtype=torch.float64) + 1.0
    result_3d = QF.roc(x3d, span=2, dim=2)
    assert result_3d.shape == x3d.shape
    for i in range(x3d.shape[0]):
        for j in range(x3d.shape[1]):
            torch.testing.assert_close(
                result_3d[i, j], QF.roc(x3d[i, j], span=2), equal_nan=True
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_roc_dtype_preservation(dtype: torch.dtype) -> None:
    """The output preserves the input dtype and device."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    assert_basic_properties(QF.roc(x, span=1), x)


def test_roc_non_integer_span_raises_type_error() -> None:
    """Non-integer spans are rejected; ``bool`` is a subclass of ``int``
    and must not be silently accepted as 1."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.roc(x, span=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.roc(x, span=True)


def test_roc_non_floating_point_input_raises_type_error() -> None:
    """Non-floating-point inputs are rejected."""
    for dtype in [torch.int32, torch.int64, torch.bool]:
        x = torch.ones(5, dtype=dtype)
        with pytest.raises(TypeError, match="floating point"):
            QF.roc(x)
