import math
import warnings

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mcovar(
    x: torch.Tensor, y: torch.Tensor, span: int, ddof: int
) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        xw = x[i - span + 1 : i + 1]
        yw = y[i - span + 1 : i + 1]
        result[i] = ((xw - xw.mean()) * (yw - yw.mean())).sum() / (span - ddof)
    return result


def test_mcovar_basic_functionality() -> None:
    """Test basic moving covariance functionality against pandas."""
    a = QF.randn(100, 6, dtype=torch.float64)
    b = QF.randn(100, 6, dtype=torch.float64)
    df_a = pd.DataFrame(a.numpy())
    df_b = pd.DataFrame(b.numpy())
    for w in (2, 5, 10):
        for ddof in (0, 1):
            np.testing.assert_allclose(
                QF.mcovar(a, b, w, dim=0, ddof=ddof).numpy(),
                df_a.rolling(w).cov(df_b, ddof=ddof).to_numpy(),
                rtol=1e-8,
                atol=1e-10,
            )


def test_mcovar_simple_case() -> None:
    """Test moving covariance with simple known data."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

    result = QF.mcovar(x, y, 3, dim=0)

    # Every window of y equals 2 * x, so Cov(x, y) = 2 * Var(x) = 2.
    expected = torch.tensor([math.nan, math.nan, 2.0, 2.0, 2.0])

    finite_mask = torch.isfinite(result)
    np.testing.assert_allclose(
        result[finite_mask].numpy(), expected[finite_mask].numpy()
    )


def test_mcovar_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    The implementation splits the data into span-sized chunks, so this
    exercises all relative positions of windows and chunk boundaries,
    including data shorter than, equal to, and longer than the window.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 25):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            y = torch.randn(n, dtype=torch.float64) * 2 - 50
            for ddof in (0, 1):
                if ddof >= span:
                    continue
                result = QF.mcovar(x, y, span, dim=0, ddof=ddof)
                expected = _reference_mcovar(x, y, span, ddof)
                np.testing.assert_allclose(
                    result.numpy(),
                    expected.numpy(),
                    rtol=1e-10,
                    atol=1e-12,
                )


def test_mcovar_self_covariance_matches_mvar() -> None:
    """Test that the covariance of a series with itself is its variance."""
    torch.manual_seed(1)
    x = torch.randn(50, dtype=torch.float64) + 10
    for span in (2, 5, 10):
        for ddof in (0, 1):
            np.testing.assert_allclose(
                QF.mcovar(x, x, span, dim=0, ddof=ddof).numpy(),
                QF.mvar(x, span, dim=0, ddof=ddof).numpy(),
                rtol=1e-12,
                atol=1e-14,
            )


def test_mcovar_symmetry() -> None:
    """Test that swapping the arguments does not change the result."""
    torch.manual_seed(2)
    x = torch.randn(50, dtype=torch.float64)
    y = torch.randn(50, dtype=torch.float64)
    for span in (2, 5, 10):
        np.testing.assert_allclose(
            QF.mcovar(x, y, span, dim=0).numpy(),
            QF.mcovar(y, x, span, dim=0).numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


def test_mcovar_bilinearity() -> None:
    """Test that scaling one argument scales the covariance linearly."""
    torch.manual_seed(3)
    x = torch.randn(40, dtype=torch.float64)
    y = torch.randn(40, dtype=torch.float64)
    base = QF.mcovar(x, y, 5, dim=0)
    for c in (-3.0, 0.5, 10.0):
        np.testing.assert_allclose(
            QF.mcovar(x, y * c, 5, dim=0).numpy(),
            (base * c).numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


def test_mcovar_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it.

    A NaN is swept through every position of either input series to verify
    that no window outside the NaN's reach is affected.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        clean = QF.mcovar(x, y, span, dim=0)
        for target in (0, 1):
            for position in range(n):
                xp = x.clone()
                yp = y.clone()
                (xp, yp)[target][position] = math.nan
                result = QF.mcovar(xp, yp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert torch.isnan(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mcovar_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity contaminates exactly the windows containing
    it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mcovar(x, y, span, dim=0)
        for target in (0, 1):
            for value in (math.inf, -math.inf):
                for position in range(n):
                    xp = x.clone()
                    yp = y.clone()
                    (xp, yp)[target][position] = value
                    result = QF.mcovar(xp, yp, span, dim=0)
                    for i in range(n):
                        if i < span - 1 or i - span + 1 <= position <= i:
                            assert not torch.isfinite(result[i])
                        else:
                            assert result[i] == clean[i]


def test_mcovar_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with large offsets relative to the covariance.

    The naive sum-of-products formula loses all significant digits in this
    setting.  The result is compared with a float64 computation on the same
    (already quantized) inputs, so the tolerance covers only the error of
    the algorithm itself.
    """
    torch.manual_seed(0)
    base = torch.randn(1000, dtype=torch.float64)
    noise = torch.randn(1000, dtype=torch.float64)
    x = (base + 1e6).to(torch.float32)
    y = (base * 2 + noise * 1e-3 + 5e5).to(torch.float32)

    result = QF.mcovar(x, y, 20, dim=0)
    expected = QF.mcovar(x.to(torch.float64), y.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    relative_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / expected[mask].abs()
    assert relative_error.max().item() < 1e-4


def test_mcovar_numerical_stability_drift() -> None:
    """Test float32 accuracy on two correlated random walks.

    Unlike a constant offset, a drift cannot be fixed by subtracting a
    global constant, so this checks that the computation is locally
    centered.  Since the sample covariance itself may cross zero, the error
    is normalized by the natural scale of the covariance (the product of
    the two moving standard deviations) instead of the covariance itself.
    """
    torch.manual_seed(1)
    steps_x = torch.randn(10000, dtype=torch.float64) * 0.01
    steps_y = steps_x * 0.6 + torch.randn(10000, dtype=torch.float64) * 0.008
    x = (steps_x.cumsum(dim=0) + 1000).to(torch.float32)
    y = (steps_y.cumsum(dim=0) + 2000).to(torch.float32)

    result = QF.mcovar(x, y, 50, dim=0)
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    expected = QF.mcovar(x64, y64, 50, dim=0)
    scale = (QF.mvar(x64, 50, dim=0) * QF.mvar(y64, 50, dim=0)).sqrt()

    mask = torch.isfinite(expected)
    normalized_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / scale[mask]
    assert normalized_error.max().item() < 1e-4


def test_mcovar_broadcasting() -> None:
    """Test broadcasting of the two input tensors."""
    torch.manual_seed(4)
    x = torch.randn(11, dtype=torch.float64)
    y = torch.randn(3, 11, dtype=torch.float64)
    result = QF.mcovar(x, y, 4, dim=-1)
    assert result.shape == (3, 11)
    for i in range(3):
        np.testing.assert_allclose(
            result[i].numpy(),
            QF.mcovar(x, y[i], 4, dim=0).numpy(),
            rtol=1e-12,
        )

    x2 = torch.randn(2, 1, 9, dtype=torch.float64)
    y2 = torch.randn(1, 4, 9, dtype=torch.float64)
    result2 = QF.mcovar(x2, y2, 3, dim=-1)
    assert result2.shape == (2, 4, 9)
    for i in range(2):
        for j in range(4):
            np.testing.assert_allclose(
                result2[i, j].numpy(),
                QF.mcovar(x2[i, 0], y2[0, j], 3, dim=0).numpy(),
                rtol=1e-12,
            )


def test_mcovar_span_one() -> None:
    """Test that a single-element window has zero population covariance."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([5.0, -1.0, 2.0])

    result = QF.mcovar(x, y, 1, dim=0, ddof=0)
    np.testing.assert_allclose(result.numpy(), np.zeros(3))

    # The sample covariance of a single element is undefined.
    assert torch.isnan(QF.mcovar(x, y, 1, dim=0, ddof=1)).all()


def test_mcovar_window_larger_than_data() -> None:
    """Test moving covariance when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 4.0, 6.0])

    result = QF.mcovar(x, y, 5, dim=0)

    # All values should be NaN when window > data length.
    assert torch.isnan(result).all()


def test_mcovar_ddof_greater_or_equal_to_span() -> None:
    """Test that ddof >= span yields non-finite values without warnings."""
    x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    y = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    for ddof in (3, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = QF.mcovar(x, y, 3, dim=0, ddof=ddof)
        assert not torch.isfinite(result).any()


def test_mcovar_negative_dimension() -> None:
    """Test moving covariance with negative dimension indexing."""
    torch.manual_seed(5)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)

    result_neg = QF.mcovar(x, y, 3, dim=-1)
    result_pos = QF.mcovar(x, y, 3, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_mcovar_dim_consistency_across_shapes() -> None:
    """Test that batched dimensions match per-series 1-D computations."""
    torch.manual_seed(6)
    x = torch.randn(4, 15, dtype=torch.float64)
    y = torch.randn(4, 15, dtype=torch.float64)
    result = QF.mcovar(x, y, 5, dim=1)
    for i in range(4):
        np.testing.assert_allclose(
            result[i].numpy(),
            QF.mcovar(x[i], y[i], 5, dim=0).numpy(),
            rtol=1e-12,
        )

    # dim=0 is the transpose of dim=1.
    result0 = QF.mcovar(x.t(), y.t(), 5, dim=0)
    np.testing.assert_allclose(result0.numpy(), result.t().numpy())

    x3 = torch.randn(2, 3, 12, dtype=torch.float64)
    y3 = torch.randn(2, 3, 12, dtype=torch.float64)
    result3 = QF.mcovar(x3, y3, 4, dim=2)
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(
                result3[i, j].numpy(),
                QF.mcovar(x3[i, j], y3[i, j], 4, dim=0).numpy(),
                rtol=1e-12,
            )


def test_mcovar_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            y = torch.randn(shape, dtype=dtype)
            result = QF.mcovar(x, y, 3, dim=dim)
            assert_basic_properties(result, x)
