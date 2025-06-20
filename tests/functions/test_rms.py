import math

import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_rms_simple_calculation() -> None:
    """Test RMS with simple manually calculated examples."""
    x = torch.tensor([[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]])
    np.testing.assert_allclose(
        QF.rms(x, dim=1).numpy(),
        np.array([(14 / 3) ** 0.5, (14 / 3) ** 0.5 / 10]),
    )


def test_rms_basic_functionality() -> None:
    """Test basic RMS functionality."""
    # Simple case: RMS of [3, 4] should be sqrt((9+16)/2) = sqrt(12.5) = 5/sqrt(2)
    x = torch.tensor([3.0, 4.0])
    result = QF.rms(x)
    expected = math.sqrt((9 + 16) / 2)
    assert torch.allclose(result, torch.tensor(expected))

    # RMS of zeros should be zero
    x_zeros = torch.zeros(5)
    result_zeros = QF.rms(x_zeros)
    assert torch.allclose(result_zeros, torch.tensor(0.0))

    # RMS of single value should be absolute value
    x_single = torch.tensor([-3.0])
    result_single = QF.rms(x_single)
    assert torch.allclose(result_single, torch.tensor(3.0))


def test_rms_mathematical_properties() -> None:
    """Test mathematical properties of RMS."""
    # RMS is always non-negative
    x = torch.tensor([-5.0, -3.0, 0.0, 3.0, 5.0])
    result = QF.rms(x)
    assert result >= 0

    # RMS of constant values equals absolute value of constant
    x_const = torch.tensor([2.0, 2.0, 2.0, 2.0])
    result_const = QF.rms(x_const)
    assert torch.allclose(result_const, torch.tensor(2.0))

    # RMS of negative constant
    x_neg_const = torch.tensor([-3.0, -3.0, -3.0])
    result_neg_const = QF.rms(x_neg_const)
    assert torch.allclose(result_neg_const, torch.tensor(3.0))


def test_rms_dim_parameter() -> None:
    """Test RMS with different dimension parameters."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Default (reduce all dimensions)
    result_all = QF.rms(x)
    expected_all = math.sqrt((1 + 4 + 9 + 16 + 25 + 36) / 6)
    assert torch.allclose(result_all, torch.tensor(expected_all))

    # Along dimension 0 (columns)
    result_dim0 = QF.rms(x, dim=0)
    expected_dim0 = torch.tensor(
        [
            math.sqrt((1 + 16) / 2),  # sqrt((1^2 + 4^2)/2)
            math.sqrt((4 + 25) / 2),  # sqrt((2^2 + 5^2)/2)
            math.sqrt((9 + 36) / 2),  # sqrt((3^2 + 6^2)/2)
        ]
    )
    torch.testing.assert_close(result_dim0, expected_dim0)

    # Along dimension 1 (rows)
    result_dim1 = QF.rms(x, dim=1)
    expected_dim1 = torch.tensor(
        [
            math.sqrt((1 + 4 + 9) / 3),  # sqrt((1^2 + 2^2 + 3^2)/3)
            math.sqrt((16 + 25 + 36) / 3),  # sqrt((4^2 + 5^2 + 6^2)/3)
        ]
    )
    torch.testing.assert_close(result_dim1, expected_dim1)


@pytest.mark.random
def test_rms_multiple_dims() -> None:
    """Test RMS with multiple dimensions."""
    x = torch.randn(2, 3, 4)

    # Reduce over dimensions (0, 2)
    result_02 = QF.rms(x, dim=(0, 2))
    assert result_02.shape == (3,)

    # Reduce over dimensions (1, 2)
    result_12 = QF.rms(x, dim=(1, 2))
    assert result_12.shape == (2,)

    # Reduce over all dimensions
    result_all = QF.rms(x, dim=(0, 1, 2))
    assert result_all.shape == ()


def test_rms_keepdim() -> None:
    """Test RMS with keepdim parameter."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # keepdim=False (default)
    result_no_keepdim = QF.rms(x, dim=1, keepdim=False)
    assert result_no_keepdim.shape == (2,)

    # keepdim=True
    result_keepdim = QF.rms(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)

    # Values should be the same
    torch.testing.assert_close(result_no_keepdim, result_keepdim.squeeze())


def test_rms_edge_cases() -> None:
    """Test RMS with edge cases."""
    # Empty tensor
    x_empty = torch.empty(0)
    result_empty = QF.rms(x_empty)
    # RMS of empty tensor should be NaN
    assert torch.isnan(result_empty)

    # Single element
    x_single = torch.tensor([5.0])
    result_single = QF.rms(x_single)
    assert torch.allclose(result_single, torch.tensor(5.0))

    # All zeros
    x_zeros = torch.zeros(10)
    result_zeros = QF.rms(x_zeros)
    assert torch.allclose(result_zeros, torch.tensor(0.0))


@pytest.mark.random
def test_rms_multidimensional() -> None:
    """Test RMS with high-dimensional tensors."""
    x = torch.randn(2, 3, 4, 5)

    # Different dimension combinations
    result_3 = QF.rms(x, dim=3)
    assert result_3.shape == (2, 3, 4)

    result_23 = QF.rms(x, dim=(2, 3))
    assert result_23.shape == (2, 3)

    result_all = QF.rms(x)
    assert result_all.shape == ()

    # All results should be non-negative
    assert torch.all(result_3 >= 0)
    assert torch.all(result_23 >= 0)
    assert result_all >= 0


def test_rms_with_negative_values() -> None:
    """Test RMS with negative values."""
    # Mixed positive and negative
    x_mixed = torch.tensor([-3.0, 4.0, -5.0])
    result_mixed = QF.rms(x_mixed)
    expected_mixed = math.sqrt((9 + 16 + 25) / 3)
    assert torch.allclose(result_mixed, torch.tensor(expected_mixed))

    # All negative
    x_negative = torch.tensor([-1.0, -2.0, -3.0])
    result_negative = QF.rms(x_negative)
    expected_negative = math.sqrt((1 + 4 + 9) / 3)
    assert torch.allclose(result_negative, torch.tensor(expected_negative))


def test_rms_with_large_values() -> None:
    """Test RMS with large values."""
    x_large = torch.tensor([1e6, 2e6, 3e6])
    result_large = QF.rms(x_large)
    expected_large = math.sqrt((1e12 + 4e12 + 9e12) / 3)
    assert torch.allclose(result_large, torch.tensor(expected_large))


def test_rms_with_small_values() -> None:
    """Test RMS with very small values."""
    x_small = torch.tensor([1e-6, 2e-6, 3e-6])
    result_small = QF.rms(x_small)
    expected_small = math.sqrt((1e-12 + 4e-12 + 9e-12) / 3)
    assert torch.allclose(result_small, torch.tensor(expected_small))


def test_rms_numerical_stability() -> None:
    """Test numerical stability of RMS calculation."""
    # Very close values
    x_close = torch.tensor([1.0, 1.0001, 1.0002])
    result_close = QF.rms(x_close)
    assert torch.isfinite(result_close)
    assert result_close > 1.0
    assert result_close < 1.1

    # Mix of very different scales
    x_mixed_scale = torch.tensor([1e-10, 1e10])
    result_mixed_scale = QF.rms(x_mixed_scale)
    assert torch.isfinite(result_mixed_scale)
    assert result_mixed_scale > 0


def test_rms_with_inf_values() -> None:
    """Test RMS behavior with infinite values."""
    # With positive infinity
    x_pos_inf = torch.tensor([1.0, math.inf, 3.0])
    result_pos_inf = QF.rms(x_pos_inf)
    assert torch.isinf(result_pos_inf)

    # With negative infinity
    x_neg_inf = torch.tensor([1.0, -math.inf, 3.0])
    result_neg_inf = QF.rms(x_neg_inf)
    assert torch.isinf(result_neg_inf)


def test_rms_with_nan_values() -> None:
    """Test RMS behavior with NaN values."""
    x_nan = torch.tensor([1.0, math.nan, 3.0])
    result_nan = QF.rms(x_nan)
    assert torch.isnan(result_nan)


def test_rms_comparison_with_norm() -> None:
    """Test relationship between RMS and L2 norm."""
    x = torch.tensor([3.0, 4.0])

    # RMS is L2 norm divided by sqrt(n)
    rms_result = QF.rms(x)
    l2_norm = torch.norm(x)
    expected_rms = l2_norm / math.sqrt(len(x))

    assert torch.allclose(rms_result, expected_rms)


def test_rms_scaling_property() -> None:
    """Test scaling property of RMS."""
    x = torch.tensor([1.0, 2.0, 3.0])
    scale = 2.5

    rms_original = QF.rms(x)
    rms_scaled = QF.rms(x * scale)

    # RMS(scale * x) = scale * RMS(x)
    assert torch.allclose(rms_scaled, scale * rms_original)


@pytest.mark.random
def test_rms_batch_processing() -> None:
    """Test RMS with batch processing."""
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length)

    # RMS along sequence dimension
    result = QF.rms(x, dim=1)
    assert result.shape == (batch_size,)

    # All values should be non-negative
    assert torch.all(result >= 0)

    # Check individual batches
    for i in range(batch_size):
        individual_rms = QF.rms(x[i])
        assert torch.allclose(result[i], individual_rms)


@pytest.mark.random
def test_rms_statistical_properties() -> None:
    """Test statistical properties of RMS."""
    # For normal distribution, RMS ≈ standard deviation
    torch.manual_seed(42)
    x = torch.randn(10000)  # Standard normal distribution

    rms_val = QF.rms(x)
    std_val = x.std()

    # For large samples from standard normal, RMS should be close to 1
    assert abs(rms_val.item() - 1.0) < 0.1
    # RMS should be close to standard deviation for zero-mean data
    assert abs(rms_val.item() - std_val.item()) < 0.1


def test_rms_symmetry() -> None:
    """Test symmetry properties of RMS."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0])

    # RMS should be the same for x and -x
    rms_positive = QF.rms(torch.abs(x))
    rms_original = QF.rms(x)

    assert torch.allclose(rms_positive, rms_original)


def test_rms_monotonicity() -> None:
    """Test that RMS increases when adding larger absolute values."""
    x = torch.tensor([1.0, 2.0])
    x_extended = torch.tensor([1.0, 2.0, 5.0])

    rms_original = QF.rms(x)
    rms_extended = QF.rms(x_extended)

    # Adding a larger value should increase RMS
    assert rms_extended > rms_original


def test_rms_formula_verification() -> None:
    """Test RMS formula with manually calculated examples."""
    # Example 1: [1, 2, 3]
    x1 = torch.tensor([1.0, 2.0, 3.0])
    rms1 = QF.rms(x1)
    expected1 = math.sqrt((1 + 4 + 9) / 3)  # sqrt(14/3)
    assert torch.allclose(rms1, torch.tensor(expected1))

    # Example 2: [0, 1, -1]
    x2 = torch.tensor([0.0, 1.0, -1.0])
    rms2 = QF.rms(x2)
    expected2 = math.sqrt((0 + 1 + 1) / 3)  # sqrt(2/3)
    assert torch.allclose(rms2, torch.tensor(expected2))


@pytest.mark.random
def test_rms_performance() -> None:
    """Test RMS performance with large tensors."""
    for size in [1000, 5000]:
        x = torch.randn(size)
        result = QF.rms(x)

        # Verify result properties
        assert torch.isfinite(result)
        assert result >= 0

        # Clean up
        del x, result


def test_rms_broadcasting_compatibility() -> None:
    """Test RMS with broadcasting scenarios."""
    # Different shapes that can be broadcasted
    x1 = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    x2 = torch.tensor([[1.0], [2.0]])  # (2, 1)

    # RMS should work on each tensor individually
    rms1 = QF.rms(x1)
    rms2 = QF.rms(x2)

    assert torch.isfinite(rms1)
    assert torch.isfinite(rms2)


def test_rms_gradient_compatibility() -> None:
    """Test that RMS works with gradient computation."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = QF.rms(x)

    # Should be able to compute gradients
    result.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_rms_empty_dim_cases() -> None:
    """Test RMS with empty dimension specifications."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Empty tuple should reduce all dimensions
    result_empty = QF.rms(x, dim=())
    result_all = QF.rms(x)

    assert torch.allclose(result_empty, result_all)


def test_rms_negative_dim() -> None:
    """Test RMS with negative dimension indices."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # dim=-1 should be same as dim=1
    result_neg1 = QF.rms(x, dim=-1)
    result_pos1 = QF.rms(x, dim=1)

    torch.testing.assert_close(result_neg1, result_pos1)

    # dim=-2 should be same as dim=0
    result_neg2 = QF.rms(x, dim=-2)
    result_pos0 = QF.rms(x, dim=0)

    torch.testing.assert_close(result_neg2, result_pos0)
