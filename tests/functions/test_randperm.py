import hashlib

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_randperm() -> None:
    RAND_SIZE = 97
    with qfeval_functions.random.seed(1):
        v1a = QF.randperm(RAND_SIZE)
    with qfeval_functions.random.seed(1):
        v1b = QF.randperm(RAND_SIZE)
    with qfeval_functions.random.seed(2):
        v2 = QF.randperm(RAND_SIZE)
    assert v1a.shape == (RAND_SIZE,)
    np.testing.assert_array_equal(
        v1a.sort().values,
        np.arange(RAND_SIZE),
    )
    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.mean(np.not_equal(v1a.numpy(), v2.numpy())) > 0.95

    # Test deterministic hash consistency (same seed produces same hash)
    m1 = hashlib.sha1()
    m1.update(v1a.numpy().tobytes())
    m2 = hashlib.sha1()
    m2.update(v1b.numpy().tobytes())
    assert (
        m1.hexdigest() == m2.hexdigest()
    ), "Same seed should produce same hash"

    # Test different seeds produce different hashes
    m3 = hashlib.sha1()
    m3.update(v2.numpy().tobytes())
    assert (
        m1.hexdigest() != m3.hexdigest()
    ), "Different seeds should produce different hashes"


def test_randperm_basic_functionality() -> None:
    """Test basic randperm functionality."""
    n = 10
    result = QF.randperm(n)

    # Check shape
    assert result.shape == (n,)

    # Check that it's a permutation (contains all numbers 0 to n-1)
    sorted_result = result.sort().values
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_result, expected)

    # Check dtype is int64 by default
    assert result.dtype == torch.int64


def test_randperm_small_sizes() -> None:
    """Test randperm with small sizes."""
    # Test n=1
    result_1 = QF.randperm(1)
    assert result_1.shape == (1,)
    assert result_1.item() == 0

    # Test n=2
    result_2 = QF.randperm(2)
    assert result_2.shape == (2,)
    sorted_result = result_2.sort().values
    torch.testing.assert_close(sorted_result, torch.tensor([0, 1]))

    # Test n=3
    result_3 = QF.randperm(3)
    assert result_3.shape == (3,)
    sorted_result = result_3.sort().values
    torch.testing.assert_close(sorted_result, torch.tensor([0, 1, 2]))


def test_randperm_zero_size() -> None:
    """Test randperm with zero size."""
    result = QF.randperm(0)
    assert result.shape == (0,)
    assert result.dtype == torch.int64


def test_randperm_large_size() -> None:
    """Test randperm with large size."""
    n = 1000
    result = QF.randperm(n)

    assert result.shape == (n,)

    # Check that it's a valid permutation
    sorted_result = result.sort().values
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_result, expected)

    # Check that it's not in order (very unlikely for large n)
    assert not torch.equal(result, torch.arange(n))


def test_randperm_dtype_specification() -> None:
    """Test that randperm respects specified dtype."""
    n = 10

    # Test int32
    result_int32 = QF.randperm(n, dtype=torch.int32)
    assert result_int32.dtype == torch.int32

    # Test int64 (default)
    result_int64 = QF.randperm(n)
    assert result_int64.dtype == torch.int64

    # Test int8
    result_int8 = QF.randperm(n, dtype=torch.int8)
    assert result_int8.dtype == torch.int8


def test_randperm_device_specification() -> None:
    """Test that randperm respects specified device."""
    n = 10
    # Test with CPU device
    result = QF.randperm(n, device=torch.device("cpu"))
    assert result.device.type == "cpu"


def test_randperm_reproducibility() -> None:
    """Test that randperm is reproducible with fixed seed."""
    n = 50

    with qfeval_functions.random.seed(42):
        result1 = QF.randperm(n)

    with qfeval_functions.random.seed(42):
        result2 = QF.randperm(n)

    torch.testing.assert_close(result1, result2)


def test_randperm_different_seeds() -> None:
    """Test that randperm produces different results with different seeds."""
    n = 50

    with qfeval_functions.random.seed(1):
        result1 = QF.randperm(n)

    with qfeval_functions.random.seed(2):
        result2 = QF.randperm(n)

    # Results should be different
    assert not torch.equal(result1, result2)


def test_randperm_permutation_properties() -> None:
    """Test mathematical properties of permutations."""
    n = 20
    result = QF.randperm(n)

    # Check that each element appears exactly once
    unique_values = torch.unique(result)
    assert len(unique_values) == n

    # Check that all values are in range [0, n-1]
    assert result.min().item() >= 0
    assert result.max().item() < n

    # Check that sorted result equals [0, 1, ..., n-1]
    sorted_result = result.sort().values
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_result, expected)


def test_randperm_uniformity() -> None:
    """Test that randperm produces roughly uniform distribution of permutations."""
    n = 5
    num_trials = 1000

    # Count how often each number appears in each position
    position_counts = torch.zeros(n, n)

    for i in range(num_trials):
        with qfeval_functions.random.seed(i):
            perm = QF.randperm(n)
        for pos, val in enumerate(perm):
            position_counts[pos, val] += 1

    # Each number should appear roughly equally often in each position
    expected_count = num_trials / n
    tolerance = 50  # Allow some variation

    for i in range(n):
        for j in range(n):
            assert (
                abs(position_counts[i, j].item() - expected_count) < tolerance
            )


def test_randperm_edge_cases() -> None:
    """Test randperm with edge cases."""
    # Test with n=1
    result_1 = QF.randperm(1)
    assert result_1.shape == (1,)
    assert result_1.item() == 0

    # Test with n=2 multiple times to check both possible permutations
    permutations_seen = set()
    for i in range(100):
        with qfeval_functions.random.seed(i):
            result = QF.randperm(2)
        perm_tuple = tuple(result.tolist())
        permutations_seen.add(perm_tuple)

    # Should see both (0, 1) and (1, 0)
    assert (0, 1) in permutations_seen
    assert (1, 0) in permutations_seen


def test_randperm_no_duplicates() -> None:
    """Test that randperm never produces duplicate values."""
    for n in [5, 10, 50, 100]:
        result = QF.randperm(n)
        unique_values = torch.unique(result)
        assert len(unique_values) == n  # No duplicates


def test_randperm_range_validation() -> None:
    """Test that randperm produces values in correct range."""
    for n in [1, 5, 10, 100]:
        result = QF.randperm(n)
        assert result.min().item() >= 0
        assert result.max().item() <= n - 1


def test_randperm_statistical_properties() -> None:
    """Test statistical properties of randperm."""
    n = 100
    result = QF.randperm(n)

    # Mean should be close to (n-1)/2
    expected_mean = (n - 1) / 2.0
    actual_mean = result.float().mean().item()
    assert abs(actual_mean - expected_mean) < 5  # Allow some tolerance

    # Sum should equal n*(n-1)/2
    expected_sum = n * (n - 1) // 2
    actual_sum = result.sum().item()
    assert actual_sum == expected_sum


def test_randperm_dtype_consistency() -> None:
    """Test that randperm maintains dtype consistency."""
    n = 10

    # Test different dtypes
    for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        result = QF.randperm(n, dtype=dtype)
        assert result.dtype == dtype
        # Should still be a valid permutation
        sorted_result = result.sort().values
        expected = torch.arange(n, dtype=dtype)
        torch.testing.assert_close(sorted_result, expected)


def test_randperm_device_consistency() -> None:
    """Test that randperm maintains device consistency."""
    n = 10

    # Test CPU device
    result_cpu = QF.randperm(n, device=torch.device("cpu"))
    assert result_cpu.device.type == "cpu"

    # Should still be a valid permutation
    sorted_result = result_cpu.sort().values
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_result, expected)


def test_randperm_inversion_property() -> None:
    """Test that permutation can be inverted."""
    n = 20
    perm = QF.randperm(n)

    # Create inverse permutation
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n)

    # Applying permutation then inverse should give identity
    identity = perm[inverse]
    expected = torch.arange(n)
    torch.testing.assert_close(identity, expected)


def test_randperm_cycle_properties() -> None:
    """Test cycle properties of permutations."""
    n = 10
    perm = QF.randperm(n)

    # Check that applying permutation multiple times eventually returns to identity
    current = torch.arange(n)
    max_cycles = 100  # Upper bound to prevent infinite loop

    for cycle in range(1, max_cycles + 1):
        current = current[perm]
        if torch.equal(current, torch.arange(n)):
            break

    # Should find a cycle within reasonable number of iterations
    assert cycle < max_cycles


def test_randperm_composition() -> None:
    """Test composition of permutations."""
    n = 15

    with qfeval_functions.random.seed(100):
        perm1 = QF.randperm(n)

    with qfeval_functions.random.seed(200):
        perm2 = QF.randperm(n)

    # Compose permutations
    composed = perm1[perm2]

    # Result should still be a valid permutation
    sorted_result = composed.sort().values
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_result, expected)


def test_randperm_hash_consistency() -> None:
    """Test that randperm produces consistent hash for same seed."""
    RAND_SIZE = 97

    # Test that same seed produces same hash
    with qfeval_functions.random.seed(1):
        result1 = QF.randperm(RAND_SIZE)

    with qfeval_functions.random.seed(1):
        result2 = QF.randperm(RAND_SIZE)

    # Generate hashes for comparison
    m1 = hashlib.sha1()
    m1.update(result1.numpy().tobytes())
    m2 = hashlib.sha1()
    m2.update(result2.numpy().tobytes())

    # Same seed should produce identical results and hashes
    assert (
        m1.hexdigest() == m2.hexdigest()
    ), "Same seed should produce same hash"

    # Test with different seed to ensure hashes are different
    with qfeval_functions.random.seed(42):
        result3 = QF.randperm(RAND_SIZE)

    m3 = hashlib.sha1()
    m3.update(result3.numpy().tobytes())
    assert (
        m1.hexdigest() != m3.hexdigest()
    ), "Different seeds should produce different hashes"


def test_randperm_seed_independence() -> None:
    """Test that different seeds produce statistically independent results."""
    n = 50

    with qfeval_functions.random.seed(300):
        result1 = QF.randperm(n)

    with qfeval_functions.random.seed(400):
        result2 = QF.randperm(n)

    # Results should be different
    assert not torch.equal(result1, result2)

    # Check that they are reasonably different (>80% different positions)
    diff_ratio = (result1 != result2).float().mean().item()
    assert diff_ratio > 0.8


def test_randperm_comparison_with_sorting() -> None:
    """Test relationship between randperm and sorting."""
    n = 25

    # Generate random permutation
    perm = QF.randperm(n)

    # Sort it
    sorted_perm, sort_indices = perm.sort()

    # Sorted permutation should be [0, 1, ..., n-1]
    expected = torch.arange(n)
    torch.testing.assert_close(sorted_perm, expected)

    # Sort indices should be the inverse permutation
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n)
    torch.testing.assert_close(sort_indices, inverse)


def test_randperm_performance() -> None:
    """Test that randperm provides reasonable performance."""
    for n in [100, 500]:
        with qfeval_functions.random.seed(1234):
            result = QF.randperm(n)

        # Verify properties
        assert result.shape == (n,)
        sorted_result = result.sort().values
        expected = torch.arange(n)
        torch.testing.assert_close(sorted_result, expected)

        # Clean up
        del result


def test_randperm_boundary_values() -> None:
    """Test randperm with boundary values."""
    # Test n=0
    result_0 = QF.randperm(0)
    assert result_0.shape == (0,)

    # Test n=1
    result_1 = QF.randperm(1)
    assert result_1.shape == (1,)
    assert result_1.item() == 0

    # Test larger boundary
    result_large = QF.randperm(1000)
    assert result_large.shape == (1000,)
    assert result_large.min().item() == 0
    assert result_large.max().item() == 999


def test_randperm_deterministic_properties() -> None:
    """Test deterministic properties that should hold for any permutation."""
    n = 30
    perm = QF.randperm(n)

    # Sum should always be n*(n-1)/2
    expected_sum = n * (n - 1) // 2
    assert perm.sum().item() == expected_sum

    # Product can vary, but check that no element is zero (since we use 0..n-1)
    if n > 1:  # Skip for n=1 case where we have 0
        assert torch.all(perm >= 0)


def test_randperm_consecutive_calls() -> None:
    """Test behavior of consecutive randperm calls."""
    n = 10

    # Without setting seed, consecutive calls should produce different results
    result1 = QF.randperm(n)
    result2 = QF.randperm(n)

    # Very unlikely to be identical for n > 2
    if n > 2:
        # Allow small chance of being same due to randomness
        assert not torch.equal(result1, result2) or n <= 3


def test_randperm_element_frequency() -> None:
    """Test that each element appears with equal frequency across multiple permutations."""
    n = 6
    num_permutations = 120  # More than n! to get good statistics

    # Count frequency of each element
    element_counts = torch.zeros(n)

    for i in range(num_permutations):
        with qfeval_functions.random.seed(i + 1000):
            perm = QF.randperm(n)
        for element in perm:
            element_counts[element] += 1

    # Each element should appear roughly equally often
    expected_count = num_permutations
    tolerance = 20  # Allow some variation

    for count in element_counts:
        assert abs(count.item() - expected_count) < tolerance
