import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _reference_mprod(x: torch.Tensor, span: int) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        result[i] = x[i - span + 1 : i + 1].prod()
    return result


def test_mprod_pandas_reference() -> None:
    """Test moving product against pandas rolling apply with np.prod."""
    torch.manual_seed(0)
    a = torch.randn(50, 5, dtype=torch.float64) * 0.1 + 1.0
    df = pd.DataFrame(a.numpy())
    for span in (1, 2, 3, 5, 10):
        expected0 = df.rolling(span).apply(np.prod, raw=True).to_numpy()
        np.testing.assert_allclose(
            QF.mprod(a, span, dim=0).numpy(), expected0, rtol=1e-12
        )
        expected1 = df.T.rolling(span).apply(np.prod, raw=True).to_numpy().T
        np.testing.assert_allclose(
            QF.mprod(a, span, dim=1).numpy(), expected1, rtol=1e-12
        )


def test_mprod_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    The implementation splits the data into span-sized chunks, so this
    exercises all relative positions of windows and chunk boundaries,
    including data shorter than, equal to, and longer than the window.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 26):
            x = torch.randn(n, dtype=torch.float64) * 0.1 + 1.0
            result = QF.mprod(x, span, dim=0)
            expected = _reference_mprod(x, span)
            np.testing.assert_allclose(
                result.numpy(), expected.numpy(), rtol=1e-12
            )


def test_mprod_zero_affects_only_windows_containing_it() -> None:
    """Test that a zero yields exactly zero without contaminating neighbors.

    Unlike division-based running products, the chunked cumulative products
    never divide by a window element, so a window containing a zero is
    exactly ``0.0`` and every other window is bit-identical to the result
    computed on the zero-free data.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64) + 3.0
    for span in (2, 3, 5, 7):
        clean = QF.mprod(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = 0.0
            result = QF.mprod(xp, span, dim=0)
            for i in range(n):
                if i < span - 1:
                    assert torch.isnan(result[i])
                elif i - span + 1 <= position <= i:
                    assert result[i].item() == 0.0
                else:
                    assert result[i] == clean[i]


def test_mprod_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it."""
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64) + 3.0
    for span in (2, 3, 5, 7):
        clean = QF.mprod(x, span, dim=0)
        for position in range(n):
            xp = x.clone()
            xp[position] = math.nan
            result = QF.mprod(xp, span, dim=0)
            for i in range(n):
                if i < span - 1 or i - span + 1 <= position <= i:
                    assert torch.isnan(result[i])
                else:
                    assert result[i] == clean[i]


def test_mprod_infinity_sign_propagation() -> None:
    """Test that infinity in a window produces a correctly signed infinity."""
    x = torch.tensor([1.0, -2.0, math.inf, 3.0, 4.0])
    result = QF.mprod(x, 3, dim=0)
    # Window [1, -2, inf] -> -inf; [-2, inf, 3] -> -inf; [inf, 3, 4] -> inf.
    assert result[2] == -math.inf
    assert result[3] == -math.inf
    assert result[4] == math.inf


def test_mprod_zero_and_infinity_window() -> None:
    """Test that a window containing both zero and infinity yields NaN."""
    x = torch.tensor([0.0, math.inf, 2.0, 3.0, 4.0])
    result = QF.mprod(x, 2, dim=0)
    assert torch.isnan(result[0])  # Window includes the NaN padding.
    assert torch.isnan(result[1])  # 0 * inf is NaN per IEEE 754.
    assert result[2] == math.inf  # Window [inf, 2].
    assert result[3] == 6.0
    assert result[4] == 12.0


def test_mprod_sign_correctness() -> None:
    """Test moving product signs with negative values."""
    x = torch.tensor([-1.0, 2.0, -3.0, -4.0, 5.0])
    result = QF.mprod(x, 3, dim=0)
    expected = torch.tensor([math.nan, math.nan, 6.0, 24.0, 60.0])
    np.testing.assert_allclose(result[2:].numpy(), expected[2:].numpy())
    assert torch.isnan(result[:2]).all()


def test_mprod_float32_long_series_stability() -> None:
    """Test float32 accuracy of long chains of multiplications.

    The result is compared with a float64 computation on the same (already
    quantized) input, so the tolerance covers only the error of the
    algorithm itself.
    """
    torch.manual_seed(0)
    x = (1.0 + 0.001 * torch.randn(5000, dtype=torch.float64)).to(torch.float32)
    result = QF.mprod(x, 20, dim=0)
    expected = QF.mprod(x.to(torch.float64), 20, dim=0)
    mask = torch.isfinite(expected)
    relative_error = (result.to(torch.float64) - expected)[
        mask
    ].abs() / expected[mask].abs()
    assert relative_error.max().item() < 1e-4


def test_mprod_span_one_identity() -> None:
    """Test that span=1 returns the input values unchanged."""
    x = torch.tensor([3.0, -1.5, 0.0, 7.0])
    result = QF.mprod(x, 1, dim=0)
    np.testing.assert_allclose(result.numpy(), x.numpy())


def test_mprod_window_larger_than_data() -> None:
    """Test moving product when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mprod(x, 5, dim=0)
    assert torch.isnan(result).all()


def test_mprod_single_element() -> None:
    """Test moving product with a single-element input."""
    x = torch.tensor([42.0])
    result = QF.mprod(x, 1, dim=0)
    assert result.shape == x.shape
    assert result.item() == 42.0


def test_mprod_2d_and_3d_dims() -> None:
    """Test moving product on 2D/3D tensors against per-series results."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 15, dtype=torch.float64) * 0.1 + 1.0
    result = QF.mprod(x, 4, dim=2)
    assert result.shape == x.shape
    for i in range(3):
        for j in range(4):
            expected = _reference_mprod(x[i, j], 4)
            np.testing.assert_allclose(
                result[i, j].numpy(), expected.numpy(), rtol=1e-12
            )
    result0 = QF.mprod(x, 2, dim=0)
    for j in range(4):
        for t in range(15):
            expected = _reference_mprod(x[:, j, t], 2)
            np.testing.assert_allclose(
                result0[:, j, t].numpy(), expected.numpy(), rtol=1e-12
            )


def test_mprod_negative_dim() -> None:
    """Test moving product with negative dimension indexing."""
    torch.manual_seed(0)
    x = torch.randn(5, 12)
    np.testing.assert_allclose(
        QF.mprod(x, 3, dim=-1).numpy(), QF.mprod(x, 3, dim=1).numpy()
    )
    x3 = torch.randn(2, 6, 4)
    np.testing.assert_allclose(
        QF.mprod(x3, 2, dim=-2).numpy(), QF.mprod(x3, 2, dim=1).numpy()
    )


def test_mprod_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mprod(x, 3, dim=dim)
            assert_basic_properties(result, x)
