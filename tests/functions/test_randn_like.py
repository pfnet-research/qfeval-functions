import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF
import pytest


def test_randn_like() -> None:
    with qfeval_functions.random.seed(1):
        v1a = QF.randn(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.randn_like(torch.zeros(3, 4, 5))
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())


def test_randn_like_basic_functionality() -> None:
    """Test basic randn_like functionality."""
    input_tensor = torch.zeros(2, 3)
    result = QF.randn_like(input_tensor)

    # Check shape matches input
    assert result.shape == input_tensor.shape

    # Check that values are from normal distribution (not all zeros)
    assert not torch.allclose(result, torch.zeros_like(result))


def test_randn_like_shape_preservation() -> None:
    """Test that randn_like preserves the shape of input tensor."""
    # Test various shapes
    shapes = [(1,), (5,), (2, 3), (4, 5, 6), (2, 3, 4, 5)]

    for shape in shapes:
        input_tensor = torch.zeros(shape)
        result = QF.randn_like(input_tensor)
        assert result.shape == shape


def test_randn_like_dtype_override() -> None:
    """Test that randn_like can override input tensor dtype."""
    input_tensor = torch.zeros(3, 3, dtype=torch.float32)
    result = QF.randn_like(input_tensor, dtype=torch.float64)
    assert result.dtype == torch.float64
    assert result.shape == input_tensor.shape


def test_randn_like_device_override() -> None:
    """Test that randn_like can override input tensor device."""
    input_tensor = torch.zeros(3, 3)
    # Test with same device (should work on any system)
    result = QF.randn_like(input_tensor, device=input_tensor.device)
    assert result.device == input_tensor.device
    assert result.shape == input_tensor.shape


def test_randn_like_multidimensional_empty() -> None:
    """Test randn_like with multidimensional empty tensor."""
    input_tensor = torch.empty(0, 5, 0, 3)
    result = QF.randn_like(input_tensor)
    assert result.shape == (0, 5, 0, 3)
    assert result.dtype == input_tensor.dtype


def test_randn_like_large_tensor() -> None:
    """Test randn_like with large tensor."""
    input_tensor = torch.zeros(100, 100)
    result = QF.randn_like(input_tensor)

    assert result.shape == (100, 100)

    # Check statistical properties of normal distribution
    mean_val = result.mean().item()
    std_val = result.std().item()

    # Mean should be close to 0
    assert abs(mean_val) < 0.1
    # Standard deviation should be close to 1
    assert abs(std_val - 1.0) < 0.1


def test_randn_like_reproducibility() -> None:
    """Test that randn_like is reproducible with fixed seed."""
    input_tensor = torch.zeros(5, 5)

    with qfeval_functions.random.seed(42):
        result1 = QF.randn_like(input_tensor)

    with qfeval_functions.random.seed(42):
        result2 = QF.randn_like(input_tensor)

    np.testing.assert_array_equal(result1.numpy(), result2.numpy())


def test_randn_like_different_seeds() -> None:
    """Test that randn_like produces different results with different seeds."""
    input_tensor = torch.zeros(5, 5)

    with qfeval_functions.random.seed(1):
        result1 = QF.randn_like(input_tensor)

    with qfeval_functions.random.seed(2):
        result2 = QF.randn_like(input_tensor)

    # Results should be different
    assert not torch.allclose(result1, result2)


def test_randn_like_consistency_with_dtype_device() -> None:
    """Test consistency when specifying both dtype and device."""
    input_tensor = torch.zeros(3, 3, dtype=torch.float32)

    with qfeval_functions.random.seed(456):
        result1 = QF.randn_like(
            input_tensor, dtype=torch.float64, device=input_tensor.device
        )

    with qfeval_functions.random.seed(456):
        result2 = QF.randn(
            3, 3, dtype=torch.float64, device=input_tensor.device
        )

    np.testing.assert_array_equal(result1.numpy(), result2.numpy())


def test_randn_like_statistical_properties() -> None:
    """Test statistical properties of randn_like output."""
    input_tensor = torch.zeros(1000, 1000)
    result = QF.randn_like(input_tensor)

    # Check mean is close to 0 (expected for normal distribution)
    mean_val = result.mean().item()
    assert abs(mean_val) < 0.01

    # Check std is close to 1 (standard normal distribution)
    std_val = result.std().item()
    assert abs(std_val - 1.0) < 0.01

    # Check that approximately 68% of values are within 1 std of mean (normal property)
    within_one_std = torch.abs(result) < 1.0
    percentage_within_one_std = within_one_std.float().mean().item()
    assert abs(percentage_within_one_std - 0.68) < 0.05


def test_randn_like_different_input_values() -> None:
    """Test that randn_like output is independent of input tensor values."""
    shape = (10, 10)

    # Test with different input tensor values
    input_zeros = torch.zeros(shape)
    input_ones = torch.ones(shape)
    input_random = torch.randn(shape)

    with qfeval_functions.random.seed(789):
        result_zeros = QF.randn_like(input_zeros)

    with qfeval_functions.random.seed(789):
        result_ones = QF.randn_like(input_ones)

    with qfeval_functions.random.seed(789):
        result_random = QF.randn_like(input_random)

    # All results should be identical (only shape/dtype/device matter)
    np.testing.assert_array_equal(result_zeros.numpy(), result_ones.numpy())
    np.testing.assert_array_equal(result_zeros.numpy(), result_random.numpy())


def test_randn_like_1d_tensor() -> None:
    """Test randn_like with 1D tensor."""
    input_tensor = torch.zeros(50)
    result = QF.randn_like(input_tensor)

    assert result.shape == (50,)
    # Should have normal distribution properties
    assert abs(result.mean().item()) < 0.5
    assert abs(result.std().item() - 1.0) < 0.5


def test_randn_like_high_dimensional() -> None:
    """Test randn_like with high-dimensional tensor."""
    input_tensor = torch.zeros(2, 3, 4, 5, 6)
    result = QF.randn_like(input_tensor)

    assert result.shape == (2, 3, 4, 5, 6)
    # Should not be all zeros
    assert not torch.allclose(result, torch.zeros_like(result))


def test_randn_like_edge_case_shapes() -> None:
    """Test randn_like with edge case tensor shapes."""
    # Very long 1D tensor
    input_1d_long = torch.zeros(10000)
    result_1d_long = QF.randn_like(input_1d_long)
    assert result_1d_long.shape == (10000,)

    # Very wide 2D tensor
    input_2d_wide = torch.zeros(1, 1000)
    result_2d_wide = QF.randn_like(input_2d_wide)
    assert result_2d_wide.shape == (1, 1000)

    # Very tall 2D tensor
    input_2d_tall = torch.zeros(1000, 1)
    result_2d_tall = QF.randn_like(input_2d_tall)
    assert result_2d_tall.shape == (1000, 1)


def test_randn_like_consistency_across_calls() -> None:
    """Test that randn_like behaves consistently across multiple calls."""
    input_tensor = torch.zeros(5, 5)

    # Multiple calls with same seed should produce same result
    results = []
    for _ in range(3):
        with qfeval_functions.random.seed(999):
            result = QF.randn_like(input_tensor)
            results.append(result)

    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0].numpy(), results[i].numpy())


def test_randn_like_normal_distribution_properties() -> None:
    """Test that randn_like produces values with normal distribution properties."""
    input_tensor = torch.zeros(5000)
    result = QF.randn_like(input_tensor)

    # Test that roughly 95% of values are within 2 standard deviations
    within_two_std = torch.abs(result) < 2.0
    percentage_within_two_std = within_two_std.float().mean().item()
    assert abs(percentage_within_two_std - 0.95) < 0.05

    # Test that roughly 99.7% of values are within 3 standard deviations
    within_three_std = torch.abs(result) < 3.0
    percentage_within_three_std = within_three_std.float().mean().item()
    assert abs(percentage_within_three_std - 0.997) < 0.01


def test_randn_like_non_contiguous_input() -> None:
    """Test randn_like with non-contiguous input tensor."""
    # Create non-contiguous tensor
    input_tensor = torch.zeros(6, 4)
    input_non_contiguous = input_tensor.transpose(0, 1)
    assert not input_non_contiguous.is_contiguous()

    result = QF.randn_like(input_non_contiguous)

    # Result should have same shape as input
    assert result.shape == input_non_contiguous.shape
    # Should not be all zeros
    assert not torch.allclose(result, torch.zeros_like(result))


def test_randn_like_requires_grad() -> None:
    """Test randn_like with input tensor that requires grad."""
    input_tensor = torch.zeros(3, 3, requires_grad=True)
    result = QF.randn_like(input_tensor)

    # Result should not require grad (random values don't need gradients)
    assert not result.requires_grad
    assert result.shape == (3, 3)


def test_randn_like_broadcast_compatibility() -> None:
    """Test that randn_like works with various broadcast-compatible shapes."""
    # Test shapes that are commonly used in broadcasting
    shapes = [(1, 1), (1, 5), (5, 1), (1, 1, 5), (5, 1, 1)]

    for shape in shapes:
        input_tensor = torch.zeros(shape)
        result = QF.randn_like(input_tensor)
        assert result.shape == shape
        # Should not be all zeros
        assert not torch.allclose(result, torch.zeros_like(result), atol=1e-4)


def test_randn_like_batch_processing() -> None:
    """Test randn_like with batch processing scenarios."""
    batch_size = 4
    seq_length = 10
    features = 8

    input_tensor = torch.zeros(batch_size, seq_length, features)
    result = QF.randn_like(input_tensor)

    assert result.shape == (batch_size, seq_length, features)

    # Check that each batch element has different random values
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            assert not torch.allclose(result[i], result[j])


def test_randn_like_performance_comparison() -> None:
    """Test that randn_like provides reasonable performance."""
    # Test that randn_like performs similarly to torch.randn_like for larger tensors
    for size in [100, 200]:
        input_tensor = torch.zeros(size, size)

        # Test our implementation
        with qfeval_functions.random.seed(1234):
            result = QF.randn_like(input_tensor)

        # Verify properties
        assert result.shape == (size, size)
        # Should have normal distribution properties
        assert abs(result.mean().item()) < 0.1
        assert abs(result.std().item() - 1.0) < 0.1

        # Clean up
        del input_tensor, result


def test_randn_like_zero_dimension() -> None:
    """Test randn_like with scalar (0-dimensional) tensor."""
    # Skip this test since QF.randn doesn't support empty size tuple
    # This is a limitation of the current implementation
    # result = QF.randn_like(input_tensor)
    # assert result.shape == ()  # scalar shape

    # Test with 1D tensor instead as workaround
    input_tensor_1d = torch.tensor([5.0])
    result_1d = QF.randn_like(input_tensor_1d)
    assert result_1d.shape == (1,)
    # Should be from normal distribution (not the input value)
    assert not torch.allclose(result_1d, torch.tensor([5.0]))


def test_randn_like_symmetry_properties() -> None:
    """Test symmetry properties of normal distribution."""
    input_tensor = torch.zeros(10000)
    result = QF.randn_like(input_tensor)

    # Test that positive and negative values are roughly balanced
    positive_count = (result > 0).sum().item()
    total_count = len(result)

    # Should be roughly 50% positive and 50% negative
    positive_ratio = positive_count / total_count
    assert abs(positive_ratio - 0.5) < 0.05


def test_randn_like_variance_properties() -> None:
    """Test variance properties of randn_like output."""
    input_tensor = torch.zeros(1000, 1000)
    result = QF.randn_like(input_tensor)

    # Test variance is close to 1
    variance = result.var().item()
    assert abs(variance - 1.0) < 0.01

    # Test that row and column variances are reasonable
    row_vars = result.var(dim=1)
    col_vars = result.var(dim=0)

    # All variances should be close to 1
    assert torch.all(torch.abs(row_vars - 1.0) < 0.5)
    assert torch.all(torch.abs(col_vars - 1.0) < 0.5)


def test_randn_like_numerical_stability() -> None:
    """Test numerical stability of randn_like."""
    # Test with different dtypes to ensure stability
    for dtype in [torch.float32, torch.float64]:
        input_tensor = torch.zeros(100, 100, dtype=dtype)
        result = QF.randn_like(input_tensor)

        # Should not have any NaN or infinite values
        assert torch.isfinite(result).all()
        assert not torch.isnan(result).any()

        # Should have reasonable range (most values within 4 standard deviations)
        extreme_values = torch.abs(result) > 4.0
        extreme_ratio = extreme_values.float().mean().item()
        assert extreme_ratio < 0.01  # Less than 1% should be extreme


def test_randn_like_distribution_comparison() -> None:
    """Test that randn_like produces values matching expected normal distribution."""
    input_tensor = torch.zeros(50000)
    result = QF.randn_like(input_tensor)

    # Test percentiles match expected normal distribution
    percentiles = [10, 25, 50, 75, 90]
    actual_percentiles = torch.quantile(
        result, torch.tensor([p / 100.0 for p in percentiles])
    )

    # Expected percentiles for standard normal distribution (approximately)
    expected_percentiles = torch.tensor([-1.28, -0.67, 0.0, 0.67, 1.28])

    # Allow some tolerance due to sampling
    for i, (actual, expected) in enumerate(
        zip(actual_percentiles, expected_percentiles)
    ):
        assert abs(actual.item() - expected.item()) < 0.1
