import math

import numpy as np
import pandas as pd
import pytest
import torch

import qfeval_functions.functions as QF
from qfeval_functions.functions.mquantile import MQuantileAlgorithm

from .test_utils import assert_basic_properties

QS = (0.0, 0.25, 0.5, 0.9, 1.0)
ALGORITHMS: tuple[MQuantileAlgorithm, ...] = (
    "sort",
    "wavelet",
)
ACCELERATOR_DEVICES = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


def _reference_mquantile(x: torch.Tensor, span: int, q: float) -> torch.Tensor:
    """Naive per-window reference implementation for 1-D tensors."""
    result = torch.full_like(x, math.nan)
    for i in range(span - 1, x.shape[0]):
        window = x[i - span + 1 : i + 1].numpy()
        if not np.isnan(window).any():
            result[i] = float(np.quantile(window, q, method="linear"))
    return result


def test_mquantile_basic_functionality() -> None:
    """Test basic moving quantile functionality against pandas."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    for span in (2, 5, 10):
        for q in QS:
            result = QF.mquantile(a, span, q, dim=0)
            expected = df.rolling(span).quantile(q, interpolation="linear")
            np.testing.assert_allclose(
                result.numpy(),
                expected.to_numpy(),
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )


def test_mquantile_all_length_span_alignments() -> None:
    """Test every alignment of the data length relative to the window size.

    This exercises all relative positions of windows and data boundaries,
    including data shorter than, equal to, and longer than the window,
    against a naive per-window NumPy reference.
    """
    torch.manual_seed(0)
    for span in range(1, 9):
        for n in range(1, 25):
            x = torch.randn(n, dtype=torch.float64) * 3 + 100
            for q in QS:
                result = QF.mquantile(x, span, q, dim=0)
                expected = _reference_mquantile(x, span, q)
                np.testing.assert_allclose(
                    result.numpy(),
                    expected.numpy(),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )


def test_mquantile_wavelet_long_monotone_and_tied_series() -> None:
    """Exercise wavelet range selection on structured and tied inputs."""
    span = 64
    increasing = torch.arange(512, dtype=torch.float64)
    for x in (increasing, increasing.flip(0), torch.ones_like(increasing)):
        for q in (0.0, 0.37, 0.5, 1.0):
            torch.testing.assert_close(
                QF.mquantile(x, span, q, dim=0, algorithm="wavelet"),
                _reference_mquantile(x, span, q),
                equal_nan=True,
            )


def test_mquantile_wavelet_matches_sort() -> None:
    """Wavelet range selection matches sort on varied exact cases."""
    torch.manual_seed(19)
    x = torch.randn(3, 79, dtype=torch.float64)
    x[0, 11] = math.nan
    x[1, 23] = math.inf
    x[1, 51] = -math.inf
    x[2] = torch.arange(79, dtype=x.dtype).remainder(7)

    for span in (1, 2, 7, 16, 32, 83):
        for q in (0.0, 0.13, 0.5, 0.91, 1.0):
            expected = QF.mquantile(x, span, q, dim=1, algorithm="sort")
            actual = QF.mquantile(x, span, q, dim=1, algorithm="wavelet")
            torch.testing.assert_close(actual, expected, equal_nan=True)


def test_mquantile_auto_large_window_matches_sort() -> None:
    """The automatic large-window path preserves the reference semantics."""
    torch.manual_seed(29)
    x = torch.randn(2, 400, dtype=torch.float64)
    x[0, 211] = math.nan
    x[1] = torch.arange(400, dtype=x.dtype).remainder(11)
    for q in (0.0, 0.37, 0.5, 1.0):
        expected = QF.mquantile(x, 256, q, dim=1, algorithm="sort")
        actual = QF.mquantile(x, 256, q, dim=1, algorithm="auto")
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_mquantile_auto_selection_thresholds() -> None:
    """The measured crossover policy keeps small workloads on sort."""
    from qfeval_functions.functions.mquantile import _choose_mquantile_algorithm

    assert _choose_mquantile_algorithm(torch.empty(1, 4_096), 64) == "sort"
    assert _choose_mquantile_algorithm(torch.empty(1, 4_096), 128) == "wavelet"
    assert _choose_mquantile_algorithm(torch.empty(1, 260), 256) == "sort"


def test_mquantile_auto_endpoint_fast_path_matches_sort() -> None:
    """Endpoint quantiles use linear moving extrema without changing results."""
    torch.manual_seed(31)
    x = torch.randn(3, 400, dtype=torch.float64)
    x[0, 211] = math.nan
    x[1, 170] = math.inf
    x[2] = torch.arange(400, dtype=x.dtype).remainder(13)
    for q in (0.0, 1.0):
        torch.testing.assert_close(
            QF.mquantile(x, 256, q, dim=1),
            QF.mquantile(x, 256, q, dim=1, algorithm="sort"),
            equal_nan=True,
        )


def test_mquantile_known_values() -> None:
    """Test hand-computed quantile values."""
    x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    # Exact order statistics: minimum, median, and maximum.
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.0, dim=0),
        torch.tensor([math.nan, math.nan, 1.0, 2.0, 2.0]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.5, dim=0),
        torch.tensor([math.nan, math.nan, 2.0, 3.0, 4.0]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.mquantile(x, 3, 1.0, dim=0),
        torch.tensor([math.nan, math.nan, 3.0, 5.0, 5.0]),
        equal_nan=True,
    )
    # Interpolated positions for span=2: h = 0.25 * (2 - 1) = 0.25.
    torch.testing.assert_close(
        QF.mquantile(x, 2, 0.25, dim=0),
        torch.tensor([math.nan, 1.5, 2.25, 2.75, 4.25]),
        equal_nan=True,
    )
    # Interpolated positions for span=3: h = 0.75 * (3 - 1) = 1.5, i.e.,
    # the mean of the two upper order statistics of each sorted window.
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.75, dim=0),
        torch.tensor([math.nan, math.nan, 2.5, 4.0, 4.5]),
        equal_nan=True,
    )


def test_mquantile_nan_affects_only_windows_containing_it() -> None:
    """Test that a NaN contaminates exactly the windows containing it.

    Sorting pushes NaNs to the end of each window, so without explicit
    masking a contaminated window would silently yield the quantile of its
    non-NaN values.  This sweeps a NaN through every position.
    """
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, dtype=torch.float64)
    for span in (2, 3, 5, 7):
        for q in (0.0, 0.5, 1.0):
            clean = QF.mquantile(x, span, q, dim=0)
            for position in range(n):
                xp = x.clone()
                xp[position] = math.nan
                result = QF.mquantile(xp, span, q, dim=0)
                for i in range(n):
                    if i < span - 1 or i - span + 1 <= position <= i:
                        assert torch.isnan(result[i])
                    else:
                        assert result[i] == clean[i]


def test_mquantile_positive_inf_handling() -> None:
    """Test that +inf behaves like an ordinary largest order statistic."""
    x = torch.tensor([1.0, math.inf, 2.0, 3.0, 4.0])
    # q=1.0 selects the maximum, so windows containing +inf yield +inf
    # (not NaN, thanks to the exact-index shortcut).
    torch.testing.assert_close(
        QF.mquantile(x, 3, 1.0, dim=0),
        torch.tensor([math.nan, math.nan, math.inf, math.inf, 4.0]),
        equal_nan=True,
    )
    # q=0.0 selects the minimum, which is unaffected by +inf.
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.0, dim=0),
        torch.tensor([math.nan, math.nan, 1.0, 2.0, 2.0]),
        equal_nan=True,
    )
    # q=0.5 picks the middle order statistic of the sorted windows
    # [1, 2, inf], [2, 3, inf], and [2, 3, 4].
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.5, dim=0),
        torch.tensor([math.nan, math.nan, 2.0, 3.0, 3.0]),
        equal_nan=True,
    )


def test_mquantile_negative_inf_handling() -> None:
    """Test that -inf behaves like an ordinary smallest order statistic."""
    x = torch.tensor([1.0, -math.inf, 2.0, 3.0, 4.0])
    torch.testing.assert_close(
        QF.mquantile(x, 3, 0.0, dim=0),
        torch.tensor([math.nan, math.nan, -math.inf, -math.inf, 2.0]),
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.mquantile(x, 3, 1.0, dim=0),
        torch.tensor([math.nan, math.nan, 2.0, 3.0, 4.0]),
        equal_nan=True,
    )


def test_mquantile_inf_interpolation() -> None:
    """Test interpolation involving infinite endpoints.

    A single infinite endpoint keeps its sign because both interpolation
    weights are strictly positive; only opposite infinities produce NaN
    (``-inf + inf`` is undefined).
    """
    result = QF.mquantile(torch.tensor([1.0, math.inf]), 2, 0.25, dim=0)
    assert result[1] == math.inf
    result = QF.mquantile(torch.tensor([-math.inf, 1.0]), 2, 0.25, dim=0)
    assert result[1] == -math.inf
    result = QF.mquantile(torch.tensor([-math.inf, math.inf]), 2, 0.5, dim=0)
    assert torch.isnan(result[1])


def test_mquantile_span_one_returns_input() -> None:
    """Test that span=1 returns the input for every quantile."""
    torch.manual_seed(0)
    x = torch.randn(10)
    for q in QS:
        torch.testing.assert_close(QF.mquantile(x, 1, q, dim=0), x)


def test_mquantile_window_larger_than_data() -> None:
    """Test moving quantile when the window is larger than the data."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mquantile(x, 5, 0.5, dim=0)
    assert result.shape == x.shape
    assert torch.isnan(result).all()


def test_mquantile_single_element() -> None:
    """Test moving quantile with single-element input."""
    x = torch.tensor([42.0])
    torch.testing.assert_close(QF.mquantile(x, 1, 0.5, dim=0), x)
    assert torch.isnan(QF.mquantile(x, 2, 0.5, dim=0)).all()


def test_mquantile_2d_dims() -> None:
    """Test that dim=0 and dim=1 match per-column/per-row 1-D results."""
    torch.manual_seed(0)
    x = torch.randn(6, 7, dtype=torch.float64)
    result0 = QF.mquantile(x, 3, 0.25, dim=0)
    result1 = QF.mquantile(x, 3, 0.25, dim=1)
    for j in range(x.shape[1]):
        torch.testing.assert_close(
            result0[:, j],
            _reference_mquantile(x[:, j], 3, 0.25),
            equal_nan=True,
        )
    for i in range(x.shape[0]):
        torch.testing.assert_close(
            result1[i],
            _reference_mquantile(x[i], 3, 0.25),
            equal_nan=True,
        )


def test_mquantile_3d_and_negative_dim() -> None:
    """Test 3D tensors and negative dimension equivalence."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 20)
    result = QF.mquantile(x, 5, 0.9, dim=2)
    assert result.shape == x.shape
    torch.testing.assert_close(
        result, QF.mquantile(x, 5, 0.9, dim=-1), equal_nan=True
    )
    for i in range(3):
        for j in range(4):
            torch.testing.assert_close(
                result[i, j],
                QF.mquantile(x[i, j], 5, 0.9, dim=0),
                equal_nan=True,
            )
    torch.testing.assert_close(
        QF.mquantile(x, 5, 0.9, dim=1),
        QF.mquantile(x, 5, 0.9, dim=-2),
        equal_nan=True,
    )


def test_mquantile_dtype_and_shape_preservation() -> None:
    """Test dtype, device, and shape preservation across dimensions."""
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        for shape, dim in (((7,), 0), ((4, 9), 1), ((2, 3, 11), -1)):
            x = torch.randn(shape, dtype=dtype)
            result = QF.mquantile(x, 3, 0.75, dim=dim)
            assert_basic_properties(result, x)


@pytest.mark.parametrize("device", ACCELERATOR_DEVICES)
def test_mquantile_accelerator_device(device: str) -> None:
    """Order-statistic indices are gathered on the input device."""
    x_cpu = torch.tensor(
        [[3.0, 1.0, 4.0, 2.0, 5.0], [1.0, math.nan, 3.0, 4.0, 2.0]]
    )
    expected = QF.mquantile(x_cpu, 3, 0.25, dim=1)
    result = QF.mquantile(x_cpu.to(device), 3, 0.25, dim=1)
    assert result.device.type == device
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mquantile_preserves_autograd_path(
    algorithm: MQuantileAlgorithm,
) -> None:
    """Gradients flow to the selected adjacent order statistics."""
    values = [8.0, 1.0, 6.0, 3.0, 7.0, 2.0, 5.0, 4.0]
    span = 4
    q = 0.25

    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    QF.mquantile(x, span, q, dim=0, algorithm=algorithm)[
        span - 1 :
    ].sum().backward()
    actual_grad = x.grad

    reference_x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    sorted_windows = reference_x.unfold(0, span, 1).sort(dim=-1).values
    pos = q * (span - 1)
    lo = math.floor(pos)
    frac = pos - lo
    reference = (
        sorted_windows[:, lo] * (1 - frac) + sorted_windows[:, lo + 1] * frac
    )
    reference.sum().backward()

    torch.testing.assert_close(actual_grad, reference_x.grad)


def test_mquantile_empty_batch() -> None:
    """An empty batch preserves its shape without constructing indices."""
    x = torch.empty((0, 8), dtype=torch.float64)
    result = QF.mquantile(x, 3, 0.5, dim=1)
    assert_basic_properties(result, x)


def test_mquantile_q_out_of_range_raises_value_error() -> None:
    """``q`` must be in the range [0, 1]."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for q in (-0.001, 1.001, math.nan):
        with pytest.raises(ValueError, match="q must be"):
            QF.mquantile(x, 2, q)


def test_mquantile_invalid_span_raises_value_error() -> None:
    """``span`` must be a positive integer."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for span in (0, -1):
        with pytest.raises(ValueError, match="span must be a positive"):
            QF.mquantile(x, span, 0.5)


def test_mquantile_non_integer_span_raises_type_error() -> None:
    """Non-integer ``span`` values are rejected; ``bool`` is a subclass of
    ``int`` and must not be silently accepted as 1."""
    x = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mquantile(x, 1.5, 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="span must be an integer"):
        QF.mquantile(x, True, 0.5)


def test_mquantile_non_floating_point_input_raises_type_error() -> None:
    """Non-floating-point inputs are rejected."""
    for dtype in (torch.int32, torch.int64, torch.bool):
        x = torch.ones(10, dtype=dtype)
        with pytest.raises(TypeError, match="floating point"):
            QF.mquantile(x, 3, 0.5)


def test_mquantile_invalid_algorithm_raises_value_error() -> None:
    """Unknown implementations are rejected instead of silently falling back."""
    x = torch.tensor([1.0, 2.0, 3.0])
    for algorithm in ("select", "tree"):
        with pytest.raises(ValueError, match="algorithm must be one of"):
            QF.mquantile(
                x, 2, 0.5, algorithm=algorithm  # type: ignore[arg-type]
            )
