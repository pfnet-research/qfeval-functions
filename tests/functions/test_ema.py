import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_ema_basic_pandas_comparison() -> None:
    """Test basic EMA functionality against pandas implementation."""
    a = QF.randn(10, 100)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.ema(a, 0.1, dim=0).numpy(),
        df.ewm(alpha=0.1).mean().to_numpy(),
        1e-6,
        1e-6,
    )


def test_ema_1d_tensor() -> None:
    """Test EMA on 1D tensor with known values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.3
    result = QF.ema(x, alpha, dim=0)

    # Compare with pandas
    expected = pd.Series(x.numpy()).ewm(alpha=alpha).mean().to_numpy()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-7, atol=1e-7)


def test_ema_2d_tensor_dim0() -> None:
    """Test EMA on 2D tensor along axis 0."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    alpha = 0.2
    result = QF.ema(x, alpha, dim=0)

    # Compare column by column with pandas
    for col in range(x.shape[1]):
        expected = (
            pd.Series(x[:, col].numpy()).ewm(alpha=alpha).mean().to_numpy()
        )
        np.testing.assert_allclose(
            result[:, col].numpy(), expected, rtol=1e-7, atol=1e-7
        )


def test_ema_2d_tensor_dim1() -> None:
    """Test EMA on 2D tensor along axis 1."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    alpha = 0.4
    result = QF.ema(x, alpha, dim=1)

    # Compare row by row with pandas
    for row in range(x.shape[0]):
        expected = (
            pd.Series(x[row, :].numpy()).ewm(alpha=alpha).mean().to_numpy()
        )
        np.testing.assert_allclose(
            result[row, :].numpy(), expected, rtol=1e-7, atol=1e-7
        )


def test_ema_3d_tensor() -> None:
    """Test EMA on 3D tensor."""
    x = torch.randn(5, 4, 6)
    alpha = 0.15

    # Test along different dimensions
    for dim in range(3):
        result = QF.ema(x, alpha, dim=dim)
        assert result.shape == x.shape

        # Verify that EMA is computed correctly along the specified dimension
        if dim == 0:
            # Test a slice
            slice_data = x[:, 0, 0]
            slice_result = QF.ema(slice_data, alpha, dim=0)
            np.testing.assert_allclose(
                result[:, 0, 0].numpy(), slice_result.numpy(), rtol=1e-7
            )


def test_ema_negative_dim() -> None:
    """Test EMA with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    alpha = 0.3

    result_neg = QF.ema(x, alpha, dim=-1)
    result_pos = QF.ema(x, alpha, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_ema_alpha_boundary_values() -> None:
    """Test EMA with boundary alpha values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Alpha very close to 0 (should behave like cumulative mean)
    alpha_small = 1e-6
    result_small = QF.ema(x, alpha_small, dim=0)
    # With very small alpha, result should be close to cumulative mean
    cum_mean = torch.cumsum(x, dim=0) / torch.arange(
        1, len(x) + 1, dtype=x.dtype
    )
    # Allow some tolerance due to numerical precision
    assert torch.allclose(result_small, cum_mean, rtol=1e-3, atol=1e-3)

    # Alpha close to 1 (should behave like the input itself)
    alpha_large = 0.999
    result_large = QF.ema(x, alpha_large, dim=0)
    # With large alpha, result should be close to input values
    assert torch.allclose(result_large, x, rtol=1e-2, atol=1e-2)


def test_ema_alpha_extreme_cases() -> None:
    """Test EMA with alpha exactly 0 and 1."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Alpha = 0 edge case
    try:
        result_zero = QF.ema(x, 0.0, dim=0)
        # Should behave like cumulative mean
        cum_mean = torch.cumsum(x, dim=0) / torch.arange(
            1, len(x) + 1, dtype=x.dtype
        )
        np.testing.assert_allclose(result_zero.numpy(), cum_mean.numpy())
    except ZeroDivisionError:
        # This is acceptable behavior for alpha=0
        pass

    # Alpha = 1 should give the input values themselves
    result_one = QF.ema(x, 1.0, dim=0)
    np.testing.assert_allclose(result_one.numpy(), x.numpy())


def test_ema_with_nan_values() -> None:
    """Test EMA behavior with NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0])
    alpha = 0.3
    result = QF.ema(x, alpha, dim=0)

    # NaN should propagate through the EMA
    assert torch.isnan(result[1])  # NaN input should produce NaN output
    assert torch.isnan(result[2])  # NaN should affect subsequent values
    assert torch.isnan(result[3])  # NaN should affect subsequent values


def test_ema_with_infinity() -> None:
    """Test EMA behavior with infinity values."""
    x = torch.tensor([1.0, math.inf, 3.0, 4.0])
    alpha = 0.3
    result = QF.ema(x, alpha, dim=0)

    # Infinity should propagate through the EMA
    assert torch.isinf(
        result[1]
    )  # Infinity input should produce infinity output
    assert torch.isinf(result[2])  # Infinity should affect subsequent values


def test_ema_large_tensor() -> None:
    """Test EMA with large tensor for performance verification."""
    x = torch.randn(1000, 50)
    alpha = 0.1
    result = QF.ema(x, alpha, dim=0)

    assert result.shape == x.shape
    # Test first and last values to ensure computation completed
    assert torch.isfinite(result[0]).all()
    assert torch.isfinite(result[-1]).all()


def test_ema_numerical_stability() -> None:
    """Test EMA numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10], dtype=torch.float64)
    result_small = QF.ema(x_small, 0.3, dim=0)
    assert torch.isfinite(result_small).all()

    # Very large values
    x_large = torch.tensor([1e10, 2e10, 3e10, 4e10], dtype=torch.float64)
    result_large = QF.ema(x_large, 0.3, dim=0)
    assert torch.isfinite(result_large).all()


def test_ema_monotonic_input() -> None:
    """Test EMA with monotonic input sequences."""
    # Increasing sequence
    x_inc = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result_inc = QF.ema(x_inc, 0.3, dim=0)

    # EMA of increasing sequence should also be increasing (but smoothed)
    assert (result_inc[1:] >= result_inc[:-1]).all()

    # Decreasing sequence
    x_dec = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    result_dec = QF.ema(x_dec, 0.3, dim=0)

    # EMA of decreasing sequence should also be decreasing (but smoothed)
    assert (result_dec[1:] <= result_dec[:-1]).all()


def test_ema_constant_input() -> None:
    """Test EMA with constant input values."""
    x = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    alpha = 0.3
    result = QF.ema(x, alpha, dim=0)

    # EMA of constant values should be the constant value
    expected = torch.full_like(x, 5.0)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_ema_different_alpha_values() -> None:
    """Test EMA with different alpha values to verify behavior."""
    x = torch.tensor([1.0, 10.0, 1.0, 10.0, 1.0])
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for alpha in alphas:
        result = QF.ema(x, alpha, dim=0)
        results.append(result)

    # Higher alpha should react more quickly to changes
    # Check the second element (after the jump from 1 to 10)
    values_at_jump = [result[1].item() for result in results]

    # With higher alpha, the EMA should be closer to the new value (10)
    for i in range(1, len(values_at_jump)):
        assert values_at_jump[i] >= values_at_jump[i - 1]


def test_ema_comparison_with_pandas_calculation() -> None:
    """Test EMA against pandas implementation for consistency."""
    x = torch.tensor([2.0, 4.0, 6.0, 8.0])
    alpha = 0.5

    result = QF.ema(x, alpha, dim=0)
    expected = pd.Series(x.numpy()).ewm(alpha=alpha).mean().to_numpy()

    # Should match pandas calculation closely (accounting for float32 vs float64)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_ema_batch_processing() -> None:
    """Test EMA with batch processing (multiple series)."""
    batch_size = 20
    seq_length = 50
    x = torch.randn(seq_length, batch_size)
    alpha = 0.2

    result = QF.ema(x, alpha, dim=0)
    assert result.shape == x.shape

    # Verify each series independently
    for batch_idx in range(min(3, batch_size)):  # Test first few batches
        series = x[:, batch_idx]
        expected = pd.Series(series.numpy()).ewm(alpha=alpha).mean().to_numpy()
        np.testing.assert_allclose(
            result[:, batch_idx].numpy(), expected, rtol=1e-4, atol=1e-4
        )


def test_ema_step_function() -> None:
    """Test EMA response to step function input."""
    # Create step function: 0 for first half, 1 for second half
    x = torch.cat([torch.zeros(50), torch.ones(50)])
    alpha = 0.1
    result = QF.ema(x, alpha, dim=0)

    # EMA should start at 0, gradually increase after step
    assert result[0].item() == 0.0  # First value should be 0
    assert result[49].item() < 0.1  # Should still be small before step
    assert result[51].item() > result[49].item()  # Should increase after step
    assert result[-1].item() < 1.0  # Should not reach 1.0 immediately
    assert result[-1].item() > 0.8  # But should be approaching 1.0


def test_ema_high_frequency_oscillation() -> None:
    """Test EMA with high frequency oscillating input."""
    # Create oscillating signal
    t = torch.arange(100, dtype=torch.float32)
    x = torch.sin(t * 0.5)  # High frequency oscillation
    alpha = 0.1  # Low alpha for smoothing

    result = QF.ema(x, alpha, dim=0)

    # EMA should smooth out the oscillations
    # Check that EMA has smaller variance than input
    input_var = torch.var(x[10:])  # Skip first few values
    result_var = torch.var(result[10:])
    assert result_var < input_var
