import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_mcorrel_basic_functionality() -> None:
    """Test basic moving correlation functionality against pandas."""
    a = QF.randn(100, 6, dtype=torch.float64)
    b = QF.randn(100, 6, dtype=torch.float64)
    df_a = pd.DataFrame(a.numpy())
    df_b = pd.DataFrame(b.numpy())
    for w in (2, 5, 10):
        np.testing.assert_allclose(
            QF.mcorrel(a, b, w, dim=0).numpy(),
            df_a.rolling(w).corr(df_b).to_numpy(),
            rtol=1e-8,
            atol=1e-8,
        )


def test_mcorrel_perfect_linear_relationship() -> None:
    """Test that a perfect linear relationship yields a correlation of
    +-1."""
    torch.manual_seed(0)
    x = torch.randn(50, dtype=torch.float64)
    for span in (2, 5, 10):
        result = QF.mcorrel(x, 2 * x + 1, span, dim=0)
        np.testing.assert_allclose(
            result[span - 1 :].numpy(),
            np.ones(50 - span + 1),
            rtol=0,
            atol=1e-10,
        )
        result = QF.mcorrel(x, -3 * x, span, dim=0)
        np.testing.assert_allclose(
            result[span - 1 :].numpy(),
            -np.ones(50 - span + 1),
            rtol=0,
            atol=1e-10,
        )


def test_mcorrel_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size
    against a per-window ``numpy.corrcoef`` reference."""
    torch.manual_seed(0)
    for span in range(2, 9):
        for n in range(1, 25):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            y = torch.randn(n, dtype=torch.float64) * 2 - 50
            result = QF.mcorrel(x, y, span, dim=0)
            for i in range(n):
                if i < span - 1:
                    assert torch.isnan(result[i])
                    continue
                xw = x[i - span + 1 : i + 1].numpy()
                yw = y[i - span + 1 : i + 1].numpy()
                if np.std(xw) == 0 or np.std(yw) == 0:
                    continue
                np.testing.assert_allclose(
                    result[i].item(),
                    np.corrcoef(xw, yw)[0, 1],
                    rtol=1e-10,
                    atol=1e-10,
                )


def test_mcorrel_range_and_constant_window() -> None:
    """Test that finite results stay within [-1, 1] up to floating-point
    error and that constant windows yield NaN."""
    torch.manual_seed(1)
    x = torch.randn(200, dtype=torch.float64)
    y = torch.randn(200, dtype=torch.float64)
    result = QF.mcorrel(x, y, 5, dim=0)
    finite = result[torch.isfinite(result)]
    assert (finite.abs() <= 1 + 1e-6).all()

    # Windows fully inside a constant stretch of either series are NaN.
    y2 = y.clone()
    y2[50:80] = 2.0
    result2 = QF.mcorrel(x, y2, 5, dim=0)
    assert torch.isnan(result2[54:80]).all()
    # Windows fully outside the stretch are unchanged (the first
    # ``span - 1`` warm-up values are NaN and compared separately).
    assert torch.equal(result2[4:50], result[4:50])
    assert torch.isnan(result2[:4]).all()


def test_mcorrel_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN in either series contaminates exactly the windows
    containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mcorrel(x, y, span, dim=0)
        for target in (0, 1):
            for position in range(n):
                xp = x.clone()
                yp = y.clone()
                (xp, yp)[target][position] = math.nan
                result = QF.mcorrel(xp, yp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert torch.isnan(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mcorrel_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity in either series makes exactly the windows
    containing it non-finite."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5):
        clean = QF.mcorrel(x, y, span, dim=0)
        for target in (0, 1):
            for value in (math.inf, -math.inf):
                for position in range(n):
                    xp = x.clone()
                    yp = y.clone()
                    (xp, yp)[target][position] = value
                    result = QF.mcorrel(xp, yp, span, dim=0)
                    for i in range(n):
                        if i < span - 1 or i - span + 1 <= position <= i:
                            assert not torch.isfinite(result[i])
                        else:
                            assert result[i] == clean[i]


def test_mcorrel_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with large offsets relative to the variances.

    The result is compared with a float64 computation on the same (already
    quantized) inputs, so the tolerance covers only the error of the
    algorithm itself.  Since the correlation is dimensionless and bounded,
    the absolute error is checked.
    """
    torch.manual_seed(0)
    base = torch.randn(1000, dtype=torch.float64)
    noise = torch.randn(1000, dtype=torch.float64)
    x = (base + 1e6).to(torch.float32)
    y = (base * 0.5 + noise * 0.5 + 5e5).to(torch.float32)

    result = QF.mcorrel(x, y, 20, dim=0)
    expected = QF.mcorrel(x.to(torch.float64), y.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    absolute_error = (result.to(torch.float64) - expected)[mask].abs()
    assert absolute_error.max().item() < 1e-3


def test_mcorrel_broadcasting() -> None:
    """Test broadcasting of the two input tensors."""
    torch.manual_seed(2)
    x = torch.randn(11, dtype=torch.float64)
    y = torch.randn(3, 11, dtype=torch.float64)
    result = QF.mcorrel(x, y, 4, dim=-1)
    assert result.shape == (3, 11)
    for i in range(3):
        np.testing.assert_allclose(
            result[i].numpy(),
            QF.mcorrel(x, y[i], 4, dim=0).numpy(),
            rtol=1e-12,
        )


def test_mcorrel_span_one_is_undefined() -> None:
    """Test that a single-element window has an undefined correlation."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([5.0, -1.0, 2.0])
    assert torch.isnan(QF.mcorrel(x, y, 1, dim=0)).all()


def test_mcorrel_window_larger_than_data() -> None:
    """Test moving correlation when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([2.0, 4.0, 6.0])
    assert torch.isnan(QF.mcorrel(x, y, 5, dim=0)).all()


def test_mcorrel_negative_dimension() -> None:
    """Test moving correlation with negative dimension indexing."""
    torch.manual_seed(3)
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)

    result_neg = QF.mcorrel(x, y, 3, dim=-1)
    result_pos = QF.mcorrel(x, y, 3, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_mcorrel_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            y = torch.randn(shape, dtype=dtype)
            result = QF.mcorrel(x, y, 3, dim=dim)
            assert_basic_properties(result, x)
