import math

import torch

from qfeval_functions.functions.rci import rci
import pytest


def test_rci_known_values() -> None:
    """Test RCI with manually calculated known values."""
    x1 = torch.Tensor([500, 510, 515, 520, 530])
    y1 = torch.Tensor([math.nan] * 4 + [100.0])
    assert torch.allclose(
        rci(x1, period=5),
        y1,
        equal_nan=True,
    )

    x2 = torch.Tensor([530, 520, 515, 510, 500])
    y2 = torch.Tensor([math.nan] * 4 + [-100.0])
    assert torch.allclose(
        rci(x2, period=5),
        y2,
        equal_nan=True,
    )
    assert torch.allclose(
        rci(torch.stack((x1, x2)), period=5),
        torch.stack((y1, y2)),
        equal_nan=True,
    )

    x3 = torch.Tensor([float(i) for i in range(50)])
    y3 = torch.Tensor([math.nan] * 8 + [100.0] * 42)
    assert torch.allclose(
        rci(torch.stack((x3, -x3)), period=9),
        torch.stack((y3, -y3)),
        equal_nan=True,
    )

    # rci(x4)[4] : (1- 6*(3^2+1^2+1^2+1^2+0)/5/(5^2-1))*100 =  40
    # rci(-x4)[4]: (1- 6*(1^2+3^2+1^2+1^2+4^2)/5/(5^2-1))*100 =-40
    x4 = torch.Tensor([4, 1, 2, 3, 5])
    y4 = torch.Tensor([math.nan] * 4 + [40.0])

    assert torch.allclose(
        rci(torch.stack((x4, -x4)), period=5),
        torch.stack((y4, -y4)),
        equal_nan=True,
    )


def test_rci_basic_functionality() -> None:
    """Test basic RCI functionality."""
    # Perfect uptrend should give 100
    x_up = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result_up = rci(x_up, period=5)
    assert torch.isnan(result_up[:4]).all()
    assert torch.allclose(result_up[4:], torch.tensor([100.0]))

    # Perfect downtrend should give -100
    x_down = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    result_down = rci(x_down, period=5)
    assert torch.isnan(result_down[:4]).all()
    assert torch.allclose(result_down[4:], torch.tensor([-100.0]))


def test_rci_period_validation() -> None:
    """Test RCI with different period values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Test period=3
    result_3 = rci(x, period=3)
    assert torch.isnan(result_3[:2]).all()
    assert not torch.isnan(result_3[2:]).any()

    # Test period=5
    result_5 = rci(x, period=5)
    assert torch.isnan(result_5[:4]).all()
    assert not torch.isnan(result_5[4:]).any()

    # Test period=9 (default)
    result_9 = rci(x, period=9)
    assert torch.isnan(result_9[:8]).all()
    assert not torch.isnan(result_9[8:]).any()


def test_rci_default_period() -> None:
    """Test RCI with default period (9)."""
    x = torch.tensor([float(i) for i in range(15)])
    result = rci(x)  # Should use period=9 by default

    # First 8 values should be NaN
    assert torch.isnan(result[:8]).all()
    # Remaining values should not be NaN
    assert not torch.isnan(result[8:]).any()

    # Perfect uptrend should give 100
    assert torch.allclose(result[8:], torch.tensor([100.0] * 7))


@pytest.mark.random
def test_rci_shape_preservation() -> None:
    """Test that RCI preserves input tensor shape."""
    # 1D tensor
    x_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result_1d = rci(x_1d, period=3)
    assert result_1d.shape == x_1d.shape

    # 2D tensor
    x_2d = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    result_2d = rci(x_2d, period=3)
    assert result_2d.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(2, 3, 10)
    result_3d = rci(x_3d, period=5)
    assert result_3d.shape == x_3d.shape


def test_rci_dim_parameter() -> None:
    """Test RCI with different dimension parameters."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]])

    # Test dim=-1 (default, along last dimension)
    result_dim_neg1 = rci(x, period=3, dim=-1)

    # Test dim=1 (same as dim=-1 for 2D tensor)
    result_dim_1 = rci(x, period=3, dim=1)

    torch.testing.assert_close(result_dim_neg1, result_dim_1, equal_nan=True)

    # Test dim=0 (along first dimension)
    x_transposed = x.T
    result_dim_0 = rci(x_transposed, period=3, dim=0)
    assert result_dim_0.shape == x_transposed.shape


def test_rci_edge_cases() -> None:
    """Test RCI with edge cases."""
    # All equal values (no change) - RCI gives 100 due to ties in ranking
    x_equal = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    result_equal = rci(x_equal, period=5)
    assert torch.isnan(result_equal[:4]).all()
    # For all equal values, RCI typically gives a specific value (100 in this case)
    assert not torch.isnan(result_equal[4])

    # Single value tensor
    x_single = torch.tensor([1.0])
    result_single = rci(x_single, period=1)
    assert torch.isnan(result_single[0])

    # Two values
    x_two = torch.tensor([1.0, 2.0])
    result_two = rci(x_two, period=2)
    assert torch.isnan(result_two[0])
    assert torch.allclose(result_two[1:], torch.tensor([100.0]))


@pytest.mark.random
def test_rci_mathematical_properties() -> None:
    """Test mathematical properties of RCI."""
    # RCI should be in range [-100, 100]
    x = torch.randn(100)
    result = rci(x, period=9)
    valid_values = result[~torch.isnan(result)]
    assert torch.all(valid_values >= -100)
    assert torch.all(valid_values <= 100)

    # Perfect correlation should give ±100
    perfect_up = torch.tensor([float(i) for i in range(10)])
    perfect_down = torch.tensor([float(9 - i) for i in range(10)])

    rci_up = rci(perfect_up, period=5)
    rci_down = rci(perfect_down, period=5)

    assert torch.allclose(rci_up[4:], torch.tensor([100.0] * 6))
    assert torch.allclose(rci_down[4:], torch.tensor([-100.0] * 6))


def test_rci_with_duplicates() -> None:
    """Test RCI with duplicate values."""
    # Some duplicate values
    x = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0])
    result = rci(x, period=5)
    assert torch.isnan(result[:4]).all()
    assert not torch.isnan(result[4])
    assert -100 <= result[4].item() <= 100


def test_rci_with_negative_values() -> None:
    """Test RCI with negative values."""
    x = torch.tensor([-5.0, -3.0, -1.0, 1.0, 3.0])
    result = rci(x, period=5)
    assert torch.isnan(result[:4]).all()
    assert torch.allclose(result[4:], torch.tensor([100.0]))

    # Reverse trend
    x_reverse = torch.tensor([3.0, 1.0, -1.0, -3.0, -5.0])
    result_reverse = rci(x_reverse, period=5)
    assert torch.allclose(result_reverse[4:], torch.tensor([-100.0]))


def test_rci_with_large_values() -> None:
    """Test RCI with large values."""
    x = torch.tensor([1e6, 2e6, 3e6, 4e6, 5e6])
    result = rci(x, period=5)
    assert torch.isnan(result[:4]).all()
    assert torch.allclose(result[4:], torch.tensor([100.0]))


def test_rci_with_small_values() -> None:
    """Test RCI with very small values."""
    x = torch.tensor([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
    result = rci(x, period=5)
    assert torch.isnan(result[:4]).all()
    assert torch.allclose(result[4:], torch.tensor([100.0]))


@pytest.mark.random
def test_rci_multidimensional() -> None:
    """Test RCI with multidimensional tensors."""
    batch_size = 3
    seq_length = 15
    x = torch.randn(batch_size, seq_length)

    result = rci(x, period=5)
    assert result.shape == (batch_size, seq_length)

    # First 4 values should be NaN for each batch
    assert torch.isnan(result[:, :4]).all()

    # Values should be in valid range
    valid_values = result[:, 4:]
    assert torch.all(valid_values >= -100)
    assert torch.all(valid_values <= 100)


@pytest.mark.random
def test_rci_with_random_data() -> None:
    """Test RCI with random data for stability."""
    torch.manual_seed(42)
    x = torch.randn(50)

    result = rci(x, period=9)

    # Should not have any NaN except for the first period-1 values
    assert torch.isnan(result[:8]).all()
    assert not torch.isnan(result[8:]).any()

    # Values should be in valid range
    valid_values = result[8:]
    assert torch.all(valid_values >= -100)
    assert torch.all(valid_values <= 100)


def test_rci_formula_verification() -> None:
    """Test RCI formula with manually calculated example."""
    # Example from the comment in original test
    x = torch.tensor([4.0, 1.0, 2.0, 3.0, 5.0])
    result = rci(x, period=5)

    # Manual calculation:
    # Ranks: [3, 0, 1, 2, 4] (0-indexed ranks of [-prices])
    # Time indices: [0, 1, 2, 3, 4]
    # d = sum((time_index - rank)^2) = (0-3)^2 + (1-0)^2 + (2-1)^2 + (3-2)^2 + (4-4)^2
    #   = 9 + 1 + 1 + 1 + 0 = 12
    # RCI = (1 - 6*12/(5*(25-1))) * 100 = (1 - 72/120) * 100 = (1 - 0.6) * 100 = 40

    assert torch.isnan(result[:4]).all()
    assert torch.allclose(result[4:], torch.tensor([40.0]))


def test_rci_extremes() -> None:
    """Test RCI with extreme patterns."""
    # Zigzag pattern
    x_zigzag = torch.tensor([1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0])
    result_zigzag = rci(x_zigzag, period=5)

    # Should not be NaN for valid positions
    assert torch.isnan(result_zigzag[:4]).all()
    assert not torch.isnan(result_zigzag[4:]).any()

    # Values should be in valid range
    valid_values = result_zigzag[4:]
    assert torch.all(valid_values >= -100)
    assert torch.all(valid_values <= 100)


def test_rci_with_inf_values() -> None:
    """Test RCI behavior with infinite values."""
    x = torch.tensor([1.0, 2.0, math.inf, 4.0, 5.0])
    result = rci(x, period=5)

    # Should handle inf gracefully (behavior may vary)
    assert result.shape == x.shape
    assert torch.isnan(result[:4]).all()


def test_rci_with_nan_values() -> None:
    """Test RCI behavior with NaN input values."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])
    result = rci(x, period=5)

    # Should handle NaN in input (exact behavior may vary)
    assert result.shape == x.shape
    assert torch.isnan(result[:4]).all()


def test_rci_spearman_correlation() -> None:
    """Test that RCI essentially measures Spearman rank correlation."""
    # RCI is based on Spearman rank correlation formula
    # For perfect positive correlation: RCI = 100
    # For perfect negative correlation: RCI = -100
    # For no correlation: RCI ≈ 0

    period = 7

    # Perfect positive correlation (prices increase with time)
    x_pos = torch.tensor([float(i) for i in range(period)])
    rci_pos = rci(x_pos, period=period)
    assert torch.allclose(rci_pos[period - 1 :], torch.tensor([100.0]))

    # Perfect negative correlation (prices decrease with time)
    x_neg = torch.tensor([float(period - 1 - i) for i in range(period)])
    rci_neg = rci(x_neg, period=period)
    assert torch.allclose(rci_neg[period - 1 :], torch.tensor([-100.0]))


def test_rci_sliding_window() -> None:
    """Test RCI sliding window behavior."""
    # Create data where different windows should give different results
    x = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 7.0, 8.0])
    result = rci(x, period=5)

    # Each position after period-1 should be calculated independently
    assert torch.isnan(result[:4]).all()
    assert not torch.isnan(result[4:]).any()

    # Different windows should potentially give different results
    # (unless there's a consistent pattern)
    assert result.shape == x.shape


def test_rci_monotonic_sequences() -> None:
    """Test RCI with various monotonic sequences."""
    # Strictly increasing
    x_inc = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    rci_inc = rci(x_inc, period=5)
    assert torch.allclose(rci_inc[4:], torch.tensor([100.0]))

    # Strictly decreasing
    x_dec = torch.tensor([9.0, 7.0, 5.0, 3.0, 1.0])
    rci_dec = rci(x_dec, period=5)
    assert torch.allclose(rci_dec[4:], torch.tensor([-100.0]))

    # Non-strictly increasing (with duplicates)
    x_inc_dup = torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0])
    rci_inc_dup = rci(x_inc_dup, period=5)
    assert (
        rci_inc_dup[4].item() >= 0
    )  # Should be positive but not necessarily 100


def test_rci_numerical_stability() -> None:
    """Test numerical stability of RCI calculation."""
    # Very close values
    x_close = torch.tensor([1.0, 1.0001, 1.0002, 1.0003, 1.0004])
    result_close = rci(x_close, period=5)
    assert torch.isnan(result_close[:4]).all()
    assert torch.isfinite(result_close[4:]).all()
    assert torch.allclose(result_close[4:], torch.tensor([100.0]))

    # Very large differences
    x_large = torch.tensor([1.0, 1000.0, 2000.0, 3000.0, 4000.0])
    result_large = rci(x_large, period=5)
    assert torch.allclose(result_large[4:], torch.tensor([100.0]))


@pytest.mark.random
def test_rci_batch_processing() -> None:
    """Test RCI with batch processing."""
    batch_size = 4
    seq_length = 20
    x = torch.randn(batch_size, seq_length)

    result = rci(x, period=9)
    assert result.shape == (batch_size, seq_length)

    # Check each batch individually
    for i in range(batch_size):
        batch_result = rci(x[i], period=9)
        torch.testing.assert_close(result[i], batch_result, equal_nan=True)


def test_rci_period_edge_cases() -> None:
    """Test RCI with edge case periods."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Period equals sequence length
    result_equal = rci(x, period=7)
    assert torch.isnan(result_equal[:6]).all()
    assert not torch.isnan(result_equal[6])

    # Period larger than sequence length
    result_larger = rci(x, period=10)
    assert torch.isnan(result_larger).all()  # All should be NaN

    # Minimum meaningful period
    result_min = rci(x, period=2)
    assert torch.isnan(result_min[0])
    assert not torch.isnan(result_min[1:]).any()
