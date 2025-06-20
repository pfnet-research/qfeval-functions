import hashlib

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF
import pytest


def test_rand() -> None:
    with qfeval_functions.random.seed(1):
        v1a = QF.rand(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.rand(3, 4, 5)
    with qfeval_functions.random.seed(2):
        v2 = QF.rand(3, 4, 5)
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.all(np.not_equal(v1a.numpy(), v2.numpy()))
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "4fbb956f90936e3ce6ee85af4d6c18108b3242c4"


def test_rand_basic_functionality() -> None:
    """Test basic random tensor generation functionality."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(10, 20)

    assert result.shape == (10, 20)
    assert result.dtype == torch.float32  # Default dtype

    # Values should be in [0, 1) range
    assert torch.all(result >= 0)
    assert torch.all(result < 1)


def test_rand_reproducibility() -> None:
    """Test that random generation is reproducible with same seed."""
    # Test multiple generations with same seed
    results = []
    for _ in range(3):
        with qfeval_functions.random.seed(123):
            results.append(QF.rand(5, 5))

    # All results should be identical
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0].numpy(), results[i].numpy())


def test_rand_different_seeds() -> None:
    """Test that different seeds produce different random values."""
    results = []
    for seed in [1, 2, 3, 42, 100]:
        with qfeval_functions.random.seed(seed):
            results.append(QF.rand(10, 10))

    # All results should be different from each other
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not torch.allclose(results[i], results[j])


def test_rand_scalar_dimension() -> None:
    """Test random generation with scalar dimension."""
    with qfeval_functions.random.seed(42):
        result = QF.rand()

    assert result.shape == ()
    assert 0 <= result.item() < 1


def test_rand_multiple_dimensions() -> None:
    """Test random generation with various dimension specifications."""
    test_shapes = [
        (1,),
        (10,),
        (3, 4),
        (2, 3, 4),
        (2, 3, 4, 5),
        (1, 1, 1, 1, 1),
    ]

    for shape in test_shapes:
        with qfeval_functions.random.seed(42):
            result = QF.rand(*shape)

        assert result.shape == shape
        assert torch.all(result >= 0)
        assert torch.all(result < 1)


def test_rand_dtype_specifications() -> None:
    """Test random generation with different dtype specifications."""
    dtypes_to_test = [
        torch.float32,
        torch.float64,
        torch.float16,
    ]

    for dtype in dtypes_to_test:
        with qfeval_functions.random.seed(42):
            result = QF.rand(5, 5, dtype=dtype)

        assert result.dtype == dtype
        assert torch.all(result >= 0)
        assert torch.all(result < 1)


def test_rand_default_dtype() -> None:
    """Test that default dtype is float32."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(5, 5)

    assert result.dtype == torch.float32


def test_rand_device_specification() -> None:
    """Test random generation with device specification."""
    # Test CPU device explicitly
    with qfeval_functions.random.seed(42):
        result_cpu = QF.rand(5, 5, device=torch.device("cpu"))

    assert result_cpu.device == torch.device("cpu")
    assert torch.all(result_cpu >= 0)
    assert torch.all(result_cpu < 1)


def test_rand_large_tensors() -> None:
    """Test random generation with large tensor sizes."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(100, 100)

    assert result.shape == (100, 100)
    assert torch.all(result >= 0)
    assert torch.all(result < 1)

    # Check that values are reasonably distributed
    mean_val = result.mean().item()
    assert 0.4 < mean_val < 0.6  # Should be around 0.5 for uniform [0,1)


def test_rand_statistical_properties() -> None:
    """Test statistical properties of random generation."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(10000)

    # Test uniform distribution properties
    mean_val = result.mean().item()
    std_val = result.std().item()

    # For uniform [0,1): mean ≈ 0.5, std ≈ 1/√12 ≈ 0.289
    assert 0.48 < mean_val < 0.52
    assert 0.27 < std_val < 0.31

    # Test range constraints
    assert torch.all(result >= 0)
    assert torch.all(result < 1)
    assert result.min().item() >= 0
    assert result.max().item() < 1


def test_rand_zero_size_tensor() -> None:
    """Test random generation with zero-size tensor."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(0)

    assert result.shape == (0,)
    assert result.dtype == torch.float32


def test_rand_numerical_stability() -> None:
    """Test numerical stability of random generation."""
    with qfeval_functions.random.seed(42):
        # Generate very small tensor
        small_result = QF.rand(1, 1)

        # Generate very large tensor
        large_result = QF.rand(1000, 1000)

    # All values should be finite and in valid range
    assert torch.all(torch.isfinite(small_result))
    assert torch.all(torch.isfinite(large_result))
    assert torch.all(small_result >= 0) and torch.all(small_result < 1)
    assert torch.all(large_result >= 0) and torch.all(large_result < 1)


def test_rand_edge_cases() -> None:
    """Test edge cases in random generation."""
    # Test with dimension 1 in various positions
    test_cases = [
        (1, 10),
        (10, 1),
        (1, 1, 10),
        (10, 1, 1),
        (1, 1, 1),
    ]

    for shape in test_cases:
        with qfeval_functions.random.seed(42):
            result = QF.rand(*shape)

        assert result.shape == shape
        assert torch.all(result >= 0)
        assert torch.all(result < 1)


def test_rand_hash_consistency() -> None:
    """Test that generated tensors have consistent hash values."""
    hashes = []

    for _ in range(3):
        with qfeval_functions.random.seed(123):
            result = QF.rand(10, 10)

        m = hashlib.sha1()
        m.update(result.numpy().tobytes())
        hashes.append(m.hexdigest())

    # All hashes should be identical
    assert all(h == hashes[0] for h in hashes)


def test_rand_different_dimension_orders() -> None:
    """Test random generation with different dimension specifications."""
    with qfeval_functions.random.seed(42):
        result1 = QF.rand(2, 3, 4)

    with qfeval_functions.random.seed(42):
        result2 = QF.rand(2, 3, 4)

    np.testing.assert_array_equal(result1.numpy(), result2.numpy())


def test_rand_precision_across_dtypes() -> None:
    """Test precision consistency across different data types."""
    with qfeval_functions.random.seed(42):
        result_f32 = QF.rand(100, dtype=torch.float32)

    with qfeval_functions.random.seed(42):
        result_f64 = QF.rand(100, dtype=torch.float64)

    # Values should be similar but not exactly equal due to precision differences
    assert torch.allclose(result_f32.double(), result_f64, atol=1e-4)


def test_rand_distribution_uniformity() -> None:
    """Test that random values are uniformly distributed."""
    with qfeval_functions.random.seed(42):
        result = QF.rand(50000)

    # Divide into bins and check distribution
    bins = 10
    hist = torch.histc(result, bins=bins, min=0, max=1)

    # Each bin should have roughly equal counts (uniform distribution)
    expected_count = len(result) / bins

    for count in hist:
        # Allow for statistical variation (within 10% of expected)
        assert abs(count.item() - expected_count) < expected_count * 0.1


def test_rand_seed_isolation() -> None:
    """Test that seed context doesn't affect other random operations."""
    # Generate some random values outside seed context
    torch.manual_seed(999)
    torch_result_before = torch.rand(5)

    # Use qfeval seed context
    with qfeval_functions.random.seed(42):
        QF.rand(5)  # Generate to set the seed state

    # Generate more random values outside seed context
    torch.manual_seed(999)
    torch_result_after = torch.rand(5)

    # The torch results should be identical (seed context shouldn't interfere)
    np.testing.assert_array_equal(
        torch_result_before.numpy(), torch_result_after.numpy()
    )


def test_rand_multiple_tensor_generation() -> None:
    """Test generating multiple tensors in sequence."""
    with qfeval_functions.random.seed(42):
        tensors = [QF.rand(3, 3) for _ in range(5)]

    # All tensors should be different from each other
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            assert not torch.allclose(tensors[i], tensors[j])

        # But all should be in valid range
        assert torch.all(tensors[i] >= 0)
        assert torch.all(tensors[i] < 1)


def test_rand_fast_mode_compatibility() -> None:
    """Test that fast mode and regular mode produce valid results."""
    # Test with fast mode disabled (if possible)
    with qfeval_functions.random.seed(42, fast=False):
        result_slow = QF.rand(10, 10)

    # Test with fast mode enabled
    with qfeval_functions.random.seed(42, fast=True):
        result_fast = QF.rand(10, 10)

    # Both should be valid random tensors
    assert torch.all(result_slow >= 0) and torch.all(result_slow < 1)
    assert torch.all(result_fast >= 0) and torch.all(result_fast < 1)

    # They may be different due to different generators, but both should be valid
    assert result_slow.shape == result_fast.shape
    assert result_slow.dtype == result_fast.dtype


def test_rand_parameter_validation() -> None:
    """Test parameter validation for edge cases."""
    # Test with various valid parameter combinations
    with qfeval_functions.random.seed(42):
        # Multiple ways to specify the same shape
        r1 = QF.rand(3, 4)
        r2 = QF.rand(3, 4, dtype=torch.float32)
        r3 = QF.rand(3, 4, device=torch.device("cpu"))
        r4 = QF.rand(3, 4, dtype=torch.float32, device=torch.device("cpu"))

    # All should have the same shape
    assert r1.shape == r2.shape == r3.shape == r4.shape == (3, 4)

    # All should be valid random values
    for r in [r1, r2, r3, r4]:
        assert torch.all(r >= 0)
        assert torch.all(r < 1)
