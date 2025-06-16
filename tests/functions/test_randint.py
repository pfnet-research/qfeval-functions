import hashlib

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_randint() -> None:
    RAND_SHAPE = (5, 7, 11)
    with qfeval_functions.random.seed(1):
        v1a = QF.randint(0, 100, RAND_SHAPE)
    with qfeval_functions.random.seed(1):
        v1b = QF.randint(0, 100, RAND_SHAPE)
    with qfeval_functions.random.seed(2):
        v2 = QF.randint(0, 100, RAND_SHAPE)
    assert v1a.shape == RAND_SHAPE
    assert v1a.amax() == 99
    assert v1a.amin() == 0
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.mean(np.not_equal(v1a.numpy(), v2.numpy())) > 0.95
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "b8b48c4017604f3fc2afebad57ad27a88cf33b8a"


def test_randint_basic_functionality() -> None:
    """Test basic randint functionality."""
    result = QF.randint(0, 10, (3, 4))

    # Check shape
    assert result.shape == (3, 4)

    # Check range [0, 10)
    assert result.min().item() >= 0
    assert result.max().item() < 10

    # Check dtype is int64 by default
    assert result.dtype == torch.int64


def test_randint_range_validation() -> None:
    """Test that randint respects the specified range."""
    low, high = 5, 15
    result = QF.randint(low, high, (100,))

    assert result.min().item() >= low
    assert result.max().item() < high
    assert result.min().item() >= 5
    assert result.max().item() <= 14


def test_randint_shape_preservation() -> None:
    """Test that randint generates tensors with correct shapes."""
    shapes = [(1,), (5,), (2, 3), (4, 5, 6), (2, 3, 4, 5)]

    for shape in shapes:
        result = QF.randint(0, 100, shape)
        assert result.shape == shape


def test_randint_dtype_specification() -> None:
    """Test that randint respects specified dtype."""
    # Test int32
    result_int32 = QF.randint(0, 10, (5,), dtype=torch.int32)
    assert result_int32.dtype == torch.int32

    # Test int64 (default)
    result_int64 = QF.randint(0, 10, (5,))
    assert result_int64.dtype == torch.int64

    # Test int8
    result_int8 = QF.randint(0, 10, (5,), dtype=torch.int8)
    assert result_int8.dtype == torch.int8


def test_randint_device_specification() -> None:
    """Test that randint respects specified device."""
    # Test with CPU device
    result = QF.randint(0, 10, (5,), device=torch.device("cpu"))
    assert result.device.type == "cpu"


def test_randint_reproducibility() -> None:
    """Test that randint is reproducible with fixed seed."""
    shape = (10, 10)

    with qfeval_functions.random.seed(42):
        result1 = QF.randint(0, 100, shape)

    with qfeval_functions.random.seed(42):
        result2 = QF.randint(0, 100, shape)

    np.testing.assert_array_equal(result1.numpy(), result2.numpy())


def test_randint_different_seeds() -> None:
    """Test that randint produces different results with different seeds."""
    shape = (10, 10)

    with qfeval_functions.random.seed(1):
        result1 = QF.randint(0, 100, shape)

    with qfeval_functions.random.seed(2):
        result2 = QF.randint(0, 100, shape)

    # Results should be different
    assert not torch.equal(result1, result2)


def test_randint_large_tensor() -> None:
    """Test randint with large tensor."""
    result = QF.randint(0, 1000, (100, 100))

    assert result.shape == (100, 100)
    assert result.min().item() >= 0
    assert result.max().item() < 1000


def test_randint_multidimensional_empty() -> None:
    """Test randint with multidimensional empty tensor."""
    result = QF.randint(0, 10, (0, 5, 0, 3))
    assert result.shape == (0, 5, 0, 3)
    assert result.dtype == torch.int64


def test_randint_negative_ranges() -> None:
    """Test randint with negative ranges."""
    # Test negative to positive
    result1 = QF.randint(-10, 10, (100,))
    assert result1.min().item() >= -10
    assert result1.max().item() < 10

    # Test all negative
    result2 = QF.randint(-20, -10, (100,))
    assert result2.min().item() >= -20
    assert result2.max().item() < -10


def test_randint_single_value_range() -> None:
    """Test randint with single value range."""
    result = QF.randint(5, 6, (10,))
    assert torch.all(result == 5)


def test_randint_large_range() -> None:
    """Test randint with large range."""
    result = QF.randint(0, 1000000, (1000,))
    assert result.min().item() >= 0
    assert result.max().item() < 1000000

    # Check that we get good distribution across the range
    unique_values = torch.unique(result)
    assert len(unique_values) > 500  # Should have many unique values


def test_randint_statistical_properties() -> None:
    """Test statistical properties of randint output."""
    low, high = 0, 100
    result = QF.randint(low, high, (10000,))

    # Check mean is close to expected value (low + high) / 2
    expected_mean = (low + high - 1) / 2  # -1 because high is exclusive
    actual_mean = result.float().mean().item()
    assert abs(actual_mean - expected_mean) < 5  # Allow some tolerance

    # Check that all values in range are represented reasonably
    unique_count = len(torch.unique(result))
    assert unique_count > 80  # Should have most values in [0, 100)


def test_randint_uniform_distribution() -> None:
    """Test that randint produces approximately uniform distribution."""
    result = QF.randint(0, 10, (10000,))

    # Count occurrences of each value
    counts = torch.bincount(result)

    # Each value should appear roughly 1000 times (10000/10)
    expected_count = 1000
    tolerance = 200  # Allow some variation

    for count in counts:
        assert abs(count.item() - expected_count) < tolerance


def test_randint_1d_tensor() -> None:
    """Test randint with 1D tensor."""
    result = QF.randint(0, 50, (100,))

    assert result.shape == (100,)
    assert result.min().item() >= 0
    assert result.max().item() < 50


def test_randint_high_dimensional() -> None:
    """Test randint with high-dimensional tensor."""
    result = QF.randint(0, 20, (2, 3, 4, 5, 6))

    assert result.shape == (2, 3, 4, 5, 6)
    assert result.min().item() >= 0
    assert result.max().item() < 20


def test_randint_edge_case_shapes() -> None:
    """Test randint with edge case tensor shapes."""
    # Very long 1D tensor
    result_1d_long = QF.randint(0, 100, (10000,))
    assert result_1d_long.shape == (10000,)

    # Very wide 2D tensor
    result_2d_wide = QF.randint(0, 100, (1, 1000))
    assert result_2d_wide.shape == (1, 1000)

    # Very tall 2D tensor
    result_2d_tall = QF.randint(0, 100, (1000, 1))
    assert result_2d_tall.shape == (1000, 1)


def test_randint_zero_range() -> None:
    """Test randint with zero as lower bound."""
    result = QF.randint(0, 1, (100,))
    assert torch.all(result == 0)


def test_randint_boundary_values() -> None:
    """Test randint boundary value behavior."""
    # Test that high value is exclusive
    result = QF.randint(0, 1, (1000,))
    assert torch.all(result == 0)  # Only 0 should be generated

    # Test with small range
    result_small = QF.randint(98, 100, (1000,))
    unique_vals = torch.unique(result_small)
    assert len(unique_vals) <= 2  # Should only have 98 and 99
    assert torch.all((unique_vals == 98) | (unique_vals == 99))


def test_randint_batch_processing() -> None:
    """Test randint with batch processing scenarios."""
    batch_size = 4
    seq_length = 10
    vocab_size = 1000

    result = QF.randint(0, vocab_size, (batch_size, seq_length))

    assert result.shape == (batch_size, seq_length)
    assert result.min().item() >= 0
    assert result.max().item() < vocab_size

    # Check that each batch element has different values
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            assert not torch.equal(result[i], result[j])


def test_randint_different_ranges() -> None:
    """Test randint with various range sizes."""
    ranges = [(0, 2), (0, 10), (0, 100), (0, 1000), (-50, 50), (100, 200)]

    for low, high in ranges:
        result = QF.randint(low, high, (100,))
        assert result.min().item() >= low
        assert result.max().item() < high

        # Check that range is utilized
        if high - low > 1:
            unique_count = len(torch.unique(result))
            assert unique_count > 1  # Should have multiple unique values


def test_randint_dtype_range_compatibility() -> None:
    """Test that randint works with different dtypes and appropriate ranges."""
    # Test int8 with small range
    result_int8 = QF.randint(0, 100, (50,), dtype=torch.int8)
    assert result_int8.dtype == torch.int8
    assert result_int8.min().item() >= 0
    assert result_int8.max().item() < 100

    # Test int32
    result_int32 = QF.randint(0, 1000000, (50,), dtype=torch.int32)
    assert result_int32.dtype == torch.int32
    assert result_int32.min().item() >= 0
    assert result_int32.max().item() < 1000000


def test_randint_performance_comparison() -> None:
    """Test that randint provides reasonable performance."""
    for size in [100, 200]:
        with qfeval_functions.random.seed(1234):
            result = QF.randint(0, 1000, (size, size))

        # Verify properties
        assert result.shape == (size, size)
        assert result.min().item() >= 0
        assert result.max().item() < 1000

        # Clean up
        del result


def test_randint_hash_consistency() -> None:
    """Test that randint produces consistent hash for same seed."""
    shape = (5, 7, 11)

    with qfeval_functions.random.seed(1):
        result = QF.randint(0, 100, shape)

    # Verify the hash matches expected value (from original test)
    m = hashlib.sha1()
    m.update(result.numpy().tobytes())
    assert m.hexdigest() == "b8b48c4017604f3fc2afebad57ad27a88cf33b8a"


def test_randint_range_validation_edge_cases() -> None:
    """Test randint with edge cases for range validation."""
    # Test with very large numbers
    result_large = QF.randint(1000000, 1000001, (10,))
    assert torch.all(result_large == 1000000)

    # Test with negative ranges
    result_neg = QF.randint(-1000, -999, (10,))
    assert torch.all(result_neg == -1000)


def test_randint_seed_independence() -> None:
    """Test that different seeds produce statistically independent results."""
    shape = (1000,)

    with qfeval_functions.random.seed(100):
        result1 = QF.randint(0, 100, shape)

    with qfeval_functions.random.seed(200):
        result2 = QF.randint(0, 100, shape)

    # Results should be different
    assert not torch.equal(result1, result2)

    # Check that they are reasonably different (>90% different values)
    diff_ratio = (result1 != result2).float().mean().item()
    assert diff_ratio > 0.9


def test_randint_multidimensional_statistics() -> None:
    """Test statistical properties of randint in multidimensional tensors."""
    result = QF.randint(0, 10, (100, 100))

    # Check that mean is approximately correct across entire tensor
    expected_mean = 4.5  # (0 + 9) / 2
    actual_mean = result.float().mean().item()
    assert abs(actual_mean - expected_mean) < 0.5

    # Check that all dimensions have reasonable variance
    row_means = result.float().mean(dim=1)
    col_means = result.float().mean(dim=0)

    # Standard deviation of means should be reasonable
    assert row_means.std().item() < 1.0
    assert col_means.std().item() < 1.0
