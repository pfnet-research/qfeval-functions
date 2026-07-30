import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_mslope_basic_functionality() -> None:
    """Test basic moving slope functionality against pandas.

    The rolling OLS slope equals the rolling covariance divided by the
    rolling variance of the explanatory variable.
    """
    a = QF.randn(100, 6, dtype=torch.float64)
    b = QF.randn(100, 6, dtype=torch.float64)
    df_a = pd.DataFrame(a.numpy())
    df_b = pd.DataFrame(b.numpy())
    for w in (2, 5, 10):
        expected = (
            df_a.rolling(w).cov(df_b) / df_a.rolling(w).var()
        ).to_numpy()
        np.testing.assert_allclose(
            QF.mslope(a, b, w, dim=0).numpy(),
            expected,
            rtol=1e-8,
            atol=1e-10,
        )


def test_mslope_exact_linear_relationship() -> None:
    """Test that an exact linear relationship recovers its slope."""
    torch.manual_seed(0)
    x = torch.randn(50, dtype=torch.float64)
    y = 2 * x + 3
    for span in (2, 5, 10):
        result = QF.mslope(x, y, span, dim=0)
        assert torch.isnan(result[: span - 1]).all()
        np.testing.assert_allclose(
            result[span - 1 :].numpy(),
            np.full(50 - span + 1, 2.0),
            rtol=1e-10,
        )


def test_mslope_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size
    against a per-window ``numpy.polyfit`` reference."""
    torch.manual_seed(0)
    for span in range(2, 9):
        for n in range(1, 25):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            y = torch.randn(n, dtype=torch.float64) * 2 - 50
            result = QF.mslope(x, y, span, dim=0)
            for i in range(n):
                if i < span - 1:
                    assert torch.isnan(result[i])
                    continue
                xw = x[i - span + 1 : i + 1].numpy()
                yw = y[i - span + 1 : i + 1].numpy()
                if np.std(xw) == 0:
                    continue
                np.testing.assert_allclose(
                    result[i].item(),
                    np.polyfit(xw, yw, 1)[0],
                    rtol=1e-8,
                    atol=1e-8,
                )


def test_mslope_rolling_beta() -> None:
    """Test that the moving slope recovers the beta of a noisy linear
    model, as used for rolling betas in finance."""
    torch.manual_seed(2)
    x = torch.randn(2000, dtype=torch.float64)
    y = 1.5 * x + 0.1 * torch.randn(2000, dtype=torch.float64)
    result = QF.mslope(x, y, 100, dim=0)
    betas = result[torch.isfinite(result)]
    assert betas.shape == (1901,)
    assert (betas - 1.5).abs().max().item() < 0.1


def test_mslope_constant_x_is_undefined() -> None:
    """Test that windows with a constant explanatory variable yield a
    non-finite slope."""
    torch.manual_seed(3)
    x = torch.randn(30, dtype=torch.float64)
    y = torch.randn(30, dtype=torch.float64)
    clean = QF.mslope(x, y, 4, dim=0)

    x2 = x.clone()
    x2[10:20] = 2.0
    result = QF.mslope(x2, y, 4, dim=0)
    # Windows fully inside the constant stretch are undefined.
    assert not torch.isfinite(result[13:20]).any()
    # Windows fully outside the stretch are unchanged (the first
    # ``span - 1`` warm-up values are NaN and compared separately).
    assert torch.equal(result[3:10], clean[3:10])
    assert torch.isnan(result[:3]).all()

    # A fully constant series is undefined everywhere.
    x3 = torch.full((10,), 7.0, dtype=torch.float64)
    assert not torch.isfinite(QF.mslope(x3, y[:10], 3, dim=0)).any()


def test_mslope_relation_to_mcorrel() -> None:
    """Test the identity slope = corr * std(y) / std(x)."""
    torch.manual_seed(4)
    x = torch.randn(60, dtype=torch.float64) + 5
    y = torch.randn(60, dtype=torch.float64) - 3
    for span in (2, 5, 10):
        expected = (
            QF.mcorrel(x, y, span, dim=0)
            * QF.mstd(y, span, dim=0, ddof=0)
            / QF.mstd(x, span, dim=0, ddof=0)
        )
        np.testing.assert_allclose(
            QF.mslope(x, y, span, dim=0).numpy(),
            expected.numpy(),
            rtol=1e-9,
            atol=1e-12,
        )


def test_mslope_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN in either series contaminates exactly the windows
    containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mslope(x, y, span, dim=0)
        for target in (0, 1):
            for position in range(n):
                xp = x.clone()
                yp = y.clone()
                (xp, yp)[target][position] = math.nan
                result = QF.mslope(xp, yp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert torch.isnan(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mslope_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity in either series makes exactly the windows
    containing it non-finite."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mslope(x, y, span, dim=0)
        for target in (0, 1):
            for value in (math.inf, -math.inf):
                for position in range(n):
                    xp = x.clone()
                    yp = y.clone()
                    (xp, yp)[target][position] = value
                    result = QF.mslope(xp, yp, span, dim=0)
                    for i in range(n):
                        if i < span - 1 or i - span + 1 <= position <= i:
                            assert not torch.isfinite(result[i])
                        else:
                            assert result[i] == clean[i]


def test_mslope_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with large offsets relative to the variances.

    The result is compared with a float64 computation on the same (already
    quantized) inputs, so the tolerance covers only the error of the
    algorithm itself.
    """
    torch.manual_seed(0)
    base = torch.randn(1000, dtype=torch.float64)
    noise = torch.randn(1000, dtype=torch.float64)
    x = (base + 1e6).to(torch.float32)
    y = (base * 0.5 + noise * 0.5 + 5e5).to(torch.float32)

    result = QF.mslope(x, y, 20, dim=0)
    expected = QF.mslope(x.to(torch.float64), y.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    relative_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / expected[mask].abs()
    assert relative_error.max().item() < 1e-4


def test_mslope_broadcasting() -> None:
    """Test broadcasting of the two input tensors."""
    torch.manual_seed(5)
    x = torch.randn(11, dtype=torch.float64)
    y = torch.randn(3, 11, dtype=torch.float64)
    result = QF.mslope(x, y, 4, dim=-1)
    assert result.shape == (3, 11)
    for i in range(3):
        np.testing.assert_allclose(
            result[i].numpy(),
            QF.mslope(x, y[i], 4, dim=0).numpy(),
            rtol=1e-12,
        )


def test_mslope_window_larger_than_data() -> None:
    """Test moving slope when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 4.0, 6.0])
    assert torch.isnan(QF.mslope(x, y, 5, dim=0)).all()


def test_mslope_negative_dimension() -> None:
    """Test moving slope with negative dimension indexing."""
    torch.manual_seed(6)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)

    result_neg = QF.mslope(x, y, 3, dim=-1)
    result_pos = QF.mslope(x, y, 3, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_mslope_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            y = torch.randn(shape, dtype=dtype)
            result = QF.mslope(x, y, 3, dim=dim)
            assert_basic_properties(result, x)
