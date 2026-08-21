import math
import warnings

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_mzscore_basic_functionality() -> None:
    """Test basic moving z-score functionality against pandas."""
    a = QF.randn(100, 6, dtype=torch.float64)
    df = pd.DataFrame(a.numpy())
    for w in (2, 5, 10):
        for ddof in (0, 1):
            expected = (
                (df - df.rolling(w).mean()) / df.rolling(w).std(ddof=ddof)
            ).to_numpy()
            np.testing.assert_allclose(
                QF.mzscore(a, w, dim=0, ddof=ddof).numpy(),
                expected,
                rtol=1e-6,
                atol=1e-6,
            )


def test_mzscore_simple_case() -> None:
    """Test moving z-score with simple known data."""
    x = torch.tensor([1.0, 2.0, 4.0, 0.0])

    result = QF.mzscore(x, 2, dim=0)

    # For each window of two elements, the latest element deviates from the
    # window mean by half the difference, and the sample standard deviation
    # is the difference divided by sqrt(2), so the z-score is +-1/sqrt(2).
    expected = torch.tensor(
        [math.nan, math.sqrt(0.5), math.sqrt(0.5), -math.sqrt(0.5)]
    )

    finite_mask = torch.isfinite(result)
    np.testing.assert_allclose(
        result[finite_mask].numpy(), expected[finite_mask].numpy()
    )
    assert torch.isnan(result[0])


def test_mzscore_linear_ramp() -> None:
    """Test the moving z-score of a linear ramp against its closed form.

    For a linear ramp, the latest element of each window deviates from the
    window mean by (span - 1) / 2, and the sample variance of the window is
    span * (span + 1) / 12.
    """
    x = torch.arange(12, dtype=torch.float64)
    for span in (2, 3, 4, 5):
        result = QF.mzscore(x, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        expected = ((span - 1) / 2) / math.sqrt(span * (span + 1) / 12)
        np.testing.assert_allclose(
            result[span - 1 :].numpy(),
            np.full(12 - span + 1, expected),
            rtol=1e-12,
        )


def test_mzscore_constant_window_is_undefined() -> None:
    """Test that constant windows yield NaN."""
    x = torch.tensor([1.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    result = QF.mzscore(x, 3, dim=0)
    np.testing.assert_allclose(result[2].item(), 2 / 3 / math.sqrt(4 / 3))
    assert torch.isnan(result[3])

    # A fully constant series is NaN everywhere.
    x2 = torch.full((10,), 5.0)
    assert torch.isnan(QF.mzscore(x2, 3, dim=0)).all()


def test_mzscore_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.mzscore(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mzscore(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mzscore_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity makes exactly the windows containing it
    non-finite."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mzscore(x, span, dim=0)
        for value in (math.inf, -math.inf):
            for position in range(n):
                xp = x.clone()
                xp[position] = value
                result = QF.mzscore(xp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert not torch.isfinite(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mzscore_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with a large offset relative to the variance.

    The result is compared with a float64 computation on the same (already
    quantized) input, so the tolerance covers only the error of the
    algorithm itself.  Since the z-score is dimensionless and of order one,
    the absolute error is checked.  Note that the achievable precision is
    bounded by the floating-point resolution of the offset itself (the
    numerator ``x - ma(x)`` cancels values of the offset's magnitude), so
    the offset is chosen such that ``offset * eps`` stays well below the
    tolerance.
    """
    torch.manual_seed(0)
    x = (torch.randn(1000, dtype=torch.float64) + 1e3).to(torch.float32)

    result = QF.mzscore(x, 20, dim=0)
    expected = QF.mzscore(x.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    absolute_error = (result.to(torch.float64) - expected)[mask].abs()
    assert absolute_error.max().item() < 1e-3


def test_mzscore_span_one_is_undefined() -> None:
    """Test that span=1 yields NaN: each window is a single element, so
    its standard deviation is zero (ddof=0) or undefined (ddof=1)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for ddof in (0, 1):
        assert torch.isnan(QF.mzscore(x, 1, dim=0, ddof=ddof)).all()


def test_mzscore_window_larger_than_data() -> None:
    """Test moving z-score when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isnan(QF.mzscore(x, 5, dim=0)).all()


def test_mzscore_ddof_greater_or_equal_to_span() -> None:
    """Test that ddof >= span raises no warnings and yields no meaningful
    values (the standard deviation becomes infinite, so the z-score
    degenerates to zero, or NaN for constant windows)."""
    x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    for ddof in (3, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = QF.mzscore(x, 3, dim=0, ddof=ddof)
        assert ((result == 0) | torch.isnan(result)).all()


def test_mzscore_negative_dimension() -> None:
    """Test moving z-score with negative dimension indexing."""
    torch.manual_seed(1)
    x = torch.randn(5, 10)

    result_neg = QF.mzscore(x, 3, dim=-1)
    result_pos = QF.mzscore(x, 3, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_mzscore_dim_consistency_across_shapes() -> None:
    """Test that batched dimensions match per-series 1-D computations."""
    torch.manual_seed(2)
    x = torch.randn(4, 15, dtype=torch.float64)
    result = QF.mzscore(x, 5, dim=1)
    for i in range(4):
        np.testing.assert_allclose(
            result[i].numpy(),
            QF.mzscore(x[i], 5, dim=0).numpy(),
            rtol=1e-12,
        )

    x3 = torch.randn(2, 3, 12, dtype=torch.float64)
    result3 = QF.mzscore(x3, 4, dim=2)
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(
                result3[i, j].numpy(),
                QF.mzscore(x3[i, j], 4, dim=0).numpy(),
                rtol=1e-12,
            )


def test_mzscore_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mzscore(x, 3, dim=dim)
            assert_basic_properties(result, x)
