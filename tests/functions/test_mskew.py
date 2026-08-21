import math

import numpy as np
import pandas as pd
import scipy.stats
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mskew(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1].numpy()
        result[i] = scipy.stats.skew(window, bias=False)
    return result


def test_mskew_basic_functionality() -> None:
    """Test basic moving skewness functionality against pandas."""
    torch.manual_seed(0)
    a = torch.randn(100, 10, dtype=torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (3, 5, 10, 20):
        np.testing.assert_allclose(
            QF.mskew(a, span, dim=0).numpy(),
            df.rolling(span).skew().to_numpy(),
            1e-7,
            1e-8,
        )


def test_mskew_simple_case() -> None:
    """Test moving skewness with simple known data."""
    x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])

    result = QF.mskew(x, 3, dim=0)

    # Every window is a doubling triple [a, 2a, 4a], whose skewness is
    # scale-invariant: G1([1, 2, 4]) = 0.9352195...
    expected = torch.tensor(
        [math.nan, math.nan, 0.93521953, 0.93521953, 0.93521953]
    )

    assert torch.isnan(result[:2]).all()
    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), rtol=1e-6, atol=0
    )


def test_mskew_symmetric_windows_are_zero() -> None:
    """Test that symmetric windows have zero skewness."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)

    # Every window of equally spaced values is symmetric around its mean.
    result = QF.mskew(x, 3, dim=0)

    np.testing.assert_allclose(
        result[2:].numpy(), np.zeros(4), rtol=0, atol=1e-12
    )


def test_mskew_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    The implementation splits the data into span-sized chunks, so this
    exercises all relative positions of windows and chunk boundaries,
    including data shorter than, equal to, and longer than the window.
    """
    torch.manual_seed(0)
    for span in range(3, 10):
        for n in range(1, 26):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            result = QF.mskew(x, span, dim=0)
            expected = _reference_mskew(x, span)
            np.testing.assert_allclose(
                result.numpy(),
                expected.numpy(),
                rtol=1e-9,
                atol=1e-10,
            )


def test_mskew_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with a large offset relative to the scale.

    The naive sum-of-powers formula loses all significant digits in this
    setting.  The result is compared with a float64 computation on the same
    (already quantized) input, so the tolerance covers only the error of the
    algorithm itself.
    """
    torch.manual_seed(0)
    x = (torch.randn(1000, dtype=torch.float64) + 1e6).to(torch.float32)

    result = QF.mskew(x, 20, dim=0)
    expected = QF.mskew(x.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    error = (result.to(torch.float64) - expected)[mask].abs()
    assert error.max().item() < 2e-3


def test_mskew_numerical_stability_drift() -> None:
    """Test float32 accuracy on a drifting series (random walk).

    Unlike a constant offset, a drift cannot be fixed by subtracting a
    global constant, so this checks that the computation is locally
    centered.
    """
    torch.manual_seed(1)
    steps = torch.randn(10000, dtype=torch.float64) * 0.01
    x = (steps.cumsum(dim=0) + 1000).to(torch.float32)

    result = QF.mskew(x, 50, dim=0)
    expected = QF.mskew(x.to(torch.float64), 50, dim=0)

    mask = torch.isfinite(expected)
    error = (result.to(torch.float64) - expected)[mask].abs()
    assert error.max().item() < 2e-3


def test_mskew_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it.

    The chunked implementation centers each partial sum on a chunk-boundary
    element, so this sweeps a NaN through every position to verify that no
    window outside the NaN's reach is affected.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (3, 4, 5, 7):
        clean = QF.mskew(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mskew(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mskew_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (3, 4, 5, 7):
        clean = QF.mskew(x, span, dim=0)
        for value in (math.inf, -math.inf):
            for position in range(n):
                xp = x.clone()
                xp[position] = value
                result = QF.mskew(xp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert not torch.isfinite(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mskew_distribution_sanity() -> None:
    """Test the sign of the skewness for known distributions."""
    torch.manual_seed(0)
    span = 1000

    # Symmetric (normal) data: skewness should be close to 0.
    x = torch.randn(5000, dtype=torch.float64)
    result = QF.mskew(x, span, dim=0)
    result = result[torch.isfinite(result)]
    assert result.abs().mean().item() < 0.2

    # Right-skewed (lognormal) data: skewness should be clearly positive.
    y = torch.randn(5000, dtype=torch.float64).exp()
    result = QF.mskew(y, span, dim=0)
    result = result[torch.isfinite(result)]
    assert (result > 0).all()
    assert result.mean().item() > 1.0


def test_mskew_constant_windows() -> None:
    """Test that windows with zero variance yield NaN.

    Note that this deviates from pandas, which special-cases perfectly
    uniform windows to 0.0 while returning NaN for near-constant windows.
    """
    x = torch.tensor([1.0, 5.0, 5.0, 5.0, 2.0])

    result = QF.mskew(x, 3, dim=0)

    assert torch.isfinite(result[2])  # Window [1, 5, 5].
    assert torch.isnan(result[3])  # Window [5, 5, 5] has zero variance.
    assert torch.isfinite(result[4])  # Window [5, 5, 2].

    # All-constant input yields NaN everywhere.
    result = QF.mskew(torch.full((8,), 3.0), 3, dim=0)
    assert torch.isnan(result).all()


def test_mskew_small_span_returns_nan() -> None:
    """Test that spans smaller than 3 produce all-NaN results.

    The bias correction of the sample skewness requires at least 3
    observations; pandas rolling skew behaves in the same way.
    """
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        for span in (1, 2):
            result = QF.mskew(x, span, dim=0)
            assert_basic_properties(result, x)
            assert torch.isnan(result).all()


def test_mskew_window_larger_than_data() -> None:
    """Test moving skewness when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])

    result = QF.mskew(x, 5, dim=0)

    assert torch.isnan(result).all()


def test_mskew_span_equals_data_length() -> None:
    """Test moving skewness when the window matches the data length."""
    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64)

    result = QF.mskew(x, 5, dim=0)

    assert torch.isnan(result[:4]).all()
    np.testing.assert_allclose(
        result[4].item(),
        float(scipy.stats.skew(x.numpy(), bias=False)),
        rtol=1e-9,
        atol=1e-10,
    )


def test_mskew_dimension_equivalences() -> None:
    """Test that dim variants match applying the function along rows."""
    torch.manual_seed(0)
    x = torch.randn(4, 9, dtype=torch.float64)

    expected = torch.stack([_reference_mskew(row, 3) for row in x])
    np.testing.assert_allclose(
        QF.mskew(x, 3, dim=1).numpy(),
        expected.numpy(),
        rtol=1e-9,
        atol=1e-10,
    )

    # Negative dimension indexing.
    np.testing.assert_allclose(
        QF.mskew(x, 3, dim=-1).numpy(),
        QF.mskew(x, 3, dim=1).numpy(),
        rtol=0,
        atol=0,
    )

    # The same computation along the other dimension.
    np.testing.assert_allclose(
        QF.mskew(x.t(), 3, dim=0).numpy(),
        QF.mskew(x, 3, dim=1).numpy().T,
        rtol=0,
        atol=0,
    )


def test_mskew_high_dimensional() -> None:
    """Test moving skewness with a 3D tensor."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 30, dtype=torch.float64)

    result = QF.mskew(x, 5, dim=2)

    assert result.shape == (2, 3, 30)
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(
                result[i, j].numpy(),
                _reference_mskew(x[i, j], 5).numpy(),
                rtol=1e-9,
                atol=1e-10,
            )


def test_mskew_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mskew(x, 3, dim=dim)
            assert_basic_properties(result, x)
