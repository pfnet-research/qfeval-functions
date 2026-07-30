import math

import numpy as np
import pandas as pd
import scipy.stats
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mkurtosis(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1].numpy()
        result[i] = scipy.stats.kurtosis(window, bias=False, fisher=True)
    return result


def test_mkurtosis_basic_functionality() -> None:
    """Test basic moving kurtosis functionality against pandas."""
    torch.manual_seed(0)
    a = torch.randn(100, 10, dtype=torch.float64)
    df = pd.DataFrame(a.numpy())
    for span in (4, 5, 10, 20):
        np.testing.assert_allclose(
            QF.mkurtosis(a, span, dim=0).numpy(),
            df.rolling(span).kurt().to_numpy(),
            1e-7,
            1e-8,
        )


def test_mkurtosis_simple_case() -> None:
    """Test moving kurtosis with simple known data."""
    x = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])

    result = QF.mkurtosis(x, 4, dim=0)

    # Every window is a doubling quadruple [a, 2a, 4a, 8a], whose kurtosis
    # is scale-invariant: G2([1, 2, 4, 8]) = 0.7576559...
    expected = torch.tensor(
        [math.nan, math.nan, math.nan, 0.75765595, 0.75765595]
    )

    assert torch.isnan(result[:3]).all()
    np.testing.assert_allclose(
        result[3:].numpy(), expected[3:].numpy(), rtol=1e-5, atol=0
    )


def test_mkurtosis_light_tailed_windows_are_negative() -> None:
    """Test that equally spaced (light-tailed) windows have G2 < 0."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)

    # Every window of equally spaced values is platykurtic, and the
    # bias-corrected excess kurtosis of [1, 2, 3, 4, 5] is exactly -1.2.
    result = QF.mkurtosis(x, 5, dim=0)

    np.testing.assert_allclose(
        result[4:].numpy(), np.full(3, -1.2), rtol=1e-12, atol=0
    )


def test_mkurtosis_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    The implementation splits the data into span-sized chunks, so this
    exercises all relative positions of windows and chunk boundaries,
    including data shorter than, equal to, and longer than the window.
    """
    torch.manual_seed(0)
    for span in range(4, 10):
        for n in range(1, 26):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            result = QF.mkurtosis(x, span, dim=0)
            expected = _reference_mkurtosis(x, span)
            np.testing.assert_allclose(
                result.numpy(),
                expected.numpy(),
                rtol=1e-9,
                atol=1e-10,
            )


def test_mkurtosis_numerical_stability_large_offset() -> None:
    """Test float32 accuracy with a large offset relative to the scale.

    The naive sum-of-powers formula loses all significant digits in this
    setting.  The result is compared with a float64 computation on the same
    (already quantized) input, so the tolerance covers only the error of the
    algorithm itself.
    """
    torch.manual_seed(0)
    x = (torch.randn(1000, dtype=torch.float64) + 1e6).to(torch.float32)

    result = QF.mkurtosis(x, 20, dim=0)
    expected = QF.mkurtosis(x.to(torch.float64), 20, dim=0)

    mask = torch.isfinite(expected)
    error = (result.to(torch.float64) - expected)[mask].abs()
    assert error.max().item() < 2e-3


def test_mkurtosis_numerical_stability_drift() -> None:
    """Test float32 accuracy on a drifting series (random walk).

    Unlike a constant offset, a drift cannot be fixed by subtracting a
    global constant, so this checks that the computation is locally
    centered.
    """
    torch.manual_seed(1)
    steps = torch.randn(10000, dtype=torch.float64) * 0.01
    x = (steps.cumsum(dim=0) + 1000).to(torch.float32)

    result = QF.mkurtosis(x, 50, dim=0)
    expected = QF.mkurtosis(x.to(torch.float64), 50, dim=0)

    mask = torch.isfinite(expected)
    error = (result.to(torch.float64) - expected)[mask].abs()
    assert error.max().item() < 2e-3


def test_mkurtosis_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it.

    The chunked implementation centers each partial sum on a chunk-boundary
    element, so this sweeps a NaN through every position to verify that no
    window outside the NaN's reach is affected.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (4, 5, 7):
        clean = QF.mkurtosis(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mkurtosis(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mkurtosis_inf_affects_only_windows_containing_it() -> None:
    """Test that an infinity contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (4, 5, 7):
        clean = QF.mkurtosis(x, span, dim=0)
        for value in (math.inf, -math.inf):
            for position in range(n):
                xp = x.clone()
                xp[position] = value
                result = QF.mkurtosis(xp, span, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert not torch.isfinite(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mkurtosis_distribution_sanity() -> None:
    """Test the sign of the excess kurtosis for known distributions."""
    torch.manual_seed(0)
    span = 1000

    # Normal data: excess kurtosis should be close to 0.
    x = torch.randn(5000, dtype=torch.float64)
    result = QF.mkurtosis(x, span, dim=0)
    result = result[torch.isfinite(result)]
    assert result.abs().mean().item() < 0.3

    # Heavy-tailed (cubed normal) data: kurtosis should be clearly
    # positive.
    y = torch.randn(5000, dtype=torch.float64) ** 3
    result = QF.mkurtosis(y, span, dim=0)
    result = result[torch.isfinite(result)]
    assert (result > 0).all()
    assert result.mean().item() > 5.0


def test_mkurtosis_constant_windows() -> None:
    """Test that windows with zero variance yield NaN.

    Note that this deviates from pandas, which special-cases perfectly
    uniform windows to -3.0 while returning NaN for near-constant windows.
    """
    x = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 2.0])

    result = QF.mkurtosis(x, 4, dim=0)

    assert torch.isfinite(result[3])  # Window [1, 5, 5, 5].
    assert torch.isnan(result[4])  # Window [5, 5, 5, 5] has zero variance.
    assert torch.isfinite(result[5])  # Window [5, 5, 5, 2].

    # All-constant input yields NaN everywhere.
    result = QF.mkurtosis(torch.full((8,), 3.0), 4, dim=0)
    assert torch.isnan(result).all()


def test_mkurtosis_small_span_returns_nan() -> None:
    """Test that spans smaller than 4 produce all-NaN results.

    The bias correction of the sample excess kurtosis requires at least 4
    observations; pandas rolling kurtosis behaves in the same way.
    """
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
        for span in (1, 2, 3):
            result = QF.mkurtosis(x, span, dim=0)
            assert_basic_properties(result, x)
            assert torch.isnan(result).all()


def test_mkurtosis_window_larger_than_data() -> None:
    """Test moving kurtosis when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])

    result = QF.mkurtosis(x, 5, dim=0)

    assert torch.isnan(result).all()


def test_mkurtosis_span_equals_data_length() -> None:
    """Test moving kurtosis when the window matches the data length."""
    torch.manual_seed(0)
    x = torch.randn(6, dtype=torch.float64)

    result = QF.mkurtosis(x, 6, dim=0)

    assert torch.isnan(result[:5]).all()
    np.testing.assert_allclose(
        result[5].item(),
        float(scipy.stats.kurtosis(x.numpy(), bias=False, fisher=True)),
        rtol=1e-9,
        atol=1e-10,
    )


def test_mkurtosis_dimension_equivalences() -> None:
    """Test that dim variants match applying the function along rows."""
    torch.manual_seed(0)
    x = torch.randn(4, 9, dtype=torch.float64)

    expected = torch.stack([_reference_mkurtosis(row, 4) for row in x])
    np.testing.assert_allclose(
        QF.mkurtosis(x, 4, dim=1).numpy(),
        expected.numpy(),
        rtol=1e-9,
        atol=1e-10,
    )

    # Negative dimension indexing.
    np.testing.assert_allclose(
        QF.mkurtosis(x, 4, dim=-1).numpy(),
        QF.mkurtosis(x, 4, dim=1).numpy(),
        rtol=0,
        atol=0,
    )

    # The same computation along the other dimension.
    np.testing.assert_allclose(
        QF.mkurtosis(x.t(), 4, dim=0).numpy(),
        QF.mkurtosis(x, 4, dim=1).numpy().T,
        rtol=0,
        atol=0,
    )


def test_mkurtosis_high_dimensional() -> None:
    """Test moving kurtosis with a 3D tensor."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 30, dtype=torch.float64)

    result = QF.mkurtosis(x, 5, dim=2)

    assert result.shape == (2, 3, 30)
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(
                result[i, j].numpy(),
                _reference_mkurtosis(x[i, j], 5).numpy(),
                rtol=1e-9,
                atol=1e-10,
            )


def test_mkurtosis_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mkurtosis(x, 4, dim=dim)
            assert_basic_properties(result, x)
