import hashlib
import math

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_randn_seed_consistency() -> None:
    """Test that randn produces consistent results with same seed."""
    with qfeval_functions.random.seed(1):
        v1a = QF.randn(3, 4, 5)
    with qfeval_functions.random.seed(1):
        v1b = QF.randn(3, 4, 5)
    with qfeval_functions.random.seed(2):
        v2 = QF.randn(3, 4, 5)

    np.testing.assert_array_equal(v1a.numpy(), v1b.numpy())
    assert np.all(np.not_equal(v1a.numpy(), v2.numpy()))

    # Verify specific hash for reproducibility
    m = hashlib.sha1()
    m.update(v1a.numpy().tobytes())
    assert m.hexdigest() == "e94e5a0bfab0c4fae459f1a2a4b6dea0171c54ea"


def test_randn_different_shapes() -> None:
    """Test randn with various tensor shapes."""
    with qfeval_functions.random.seed(42):
        # 1D tensor
        x1d = QF.randn(10)
        assert x1d.shape == (10,)

        # 2D tensor
        x2d = QF.randn(5, 6)
        assert x2d.shape == (5, 6)

        # 3D tensor
        x3d = QF.randn(2, 3, 4)
        assert x3d.shape == (2, 3, 4)

        # 4D tensor
        x4d = QF.randn(2, 3, 4, 5)
        assert x4d.shape == (2, 3, 4, 5)


def test_randn_single_element() -> None:
    """Test randn with single element tensor."""
    with qfeval_functions.random.seed(123):
        x = QF.randn(1)
        assert x.shape == (1,)
        assert torch.is_tensor(x)


def test_randn_empty_tensor() -> None:
    """Test randn with zero-size dimensions."""
    with qfeval_functions.random.seed(456):
        x = QF.randn(0)
        assert x.shape == (0,)

        x2d = QF.randn(0, 5)
        assert x2d.shape == (0, 5)

        x3d = QF.randn(3, 0, 4)
        assert x3d.shape == (3, 0, 4)


def test_randn_dtype_specification() -> None:
    """Test randn with different dtype specifications."""
    with qfeval_functions.random.seed(789):
        # Default dtype (should be float32)
        x_default = QF.randn(5)
        assert x_default.dtype in [torch.float32, torch.float64]

        # Explicit float32
        x_float32 = QF.randn(5, dtype=torch.float32)
        assert x_float32.dtype == torch.float32

        # Explicit float64
        x_float64 = QF.randn(5, dtype=torch.float64)
        assert x_float64.dtype == torch.float64


def test_randn_device_specification() -> None:
    """Test randn device handling."""
    with qfeval_functions.random.seed(101):
        # CPU device (default)
        x_cpu = QF.randn(5)
        assert x_cpu.device.type == "cpu"

        # Test device parameter if supported
        try:
            x_cpu_explicit = QF.randn(5, device="cpu")
            assert x_cpu_explicit.device.type == "cpu"
        except TypeError:
            # Device parameter might not be supported
            pass


def test_randn_statistical_properties() -> None:
    """Test that randn produces values with expected statistical properties."""
    with qfeval_functions.random.seed(999):
        # Large sample for statistical testing
        x = QF.randn(10000)

        # Mean should be close to 0
        mean = torch.mean(x)
        assert abs(mean) < 0.1, f"Mean {mean} is too far from 0"

        # Standard deviation should be close to 1
        std = torch.std(x, unbiased=True)
        assert abs(std - 1.0) < 0.1, f"Std {std} is too far from 1"

        # Should contain both positive and negative values
        assert torch.any(x > 0)
        assert torch.any(x < 0)


def test_randn_value_range() -> None:
    """Test that randn produces reasonable value ranges."""
    with qfeval_functions.random.seed(555):
        x = QF.randn(1000)

        # Most values should be within 3 standard deviations
        within_3_sigma = torch.abs(x) < 3.0
        proportion_within_3_sigma = torch.mean(within_3_sigma.float())
        assert (
            proportion_within_3_sigma > 0.99
        ), "Too many outliers beyond 3 sigma"

        # Should have some variety (not all the same)
        assert torch.var(x) > 0.5


def test_randn_reproducibility_across_calls() -> None:
    """Test reproducibility when called multiple times with same seed."""
    # Generate reference sequence
    with qfeval_functions.random.seed(777):
        ref1 = QF.randn(5)
        ref2 = QF.randn(3)
        ref3 = QF.randn(7)

    # Generate same sequence again
    with qfeval_functions.random.seed(777):
        test1 = QF.randn(5)
        test2 = QF.randn(3)
        test3 = QF.randn(7)

    np.testing.assert_array_equal(ref1.numpy(), test1.numpy())
    np.testing.assert_array_equal(ref2.numpy(), test2.numpy())
    np.testing.assert_array_equal(ref3.numpy(), test3.numpy())


def test_randn_different_seeds_produce_different_values() -> None:
    """Test that different seeds produce different random sequences."""
    seeds = [1, 2, 3, 4, 5]
    results = []

    for seed in seeds:
        with qfeval_functions.random.seed(seed):
            x = QF.randn(100)
            results.append(x)

    # All results should be different
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not torch.equal(results[i], results[j])


def test_randn_large_tensor() -> None:
    """Test randn with large tensor dimensions."""
    with qfeval_functions.random.seed(888):
        # Test large 1D tensor
        x_large = QF.randn(100000)
        assert x_large.shape == (100000,)

        # Test moderately large multi-dimensional tensor
        x_multi = QF.randn(100, 100)
        assert x_multi.shape == (100, 100)


def test_randn_edge_case_dimensions() -> None:
    """Test randn with edge case dimensions."""
    with qfeval_functions.random.seed(333):
        # Single dimension with size 1
        x1 = QF.randn(1, 1, 1)
        assert x1.shape == (1, 1, 1)

        # Mix of 1s and larger dimensions
        x2 = QF.randn(1, 10, 1, 5)
        assert x2.shape == (1, 10, 1, 5)


def test_randn_numerical_properties() -> None:
    """Test numerical properties of generated values."""
    with qfeval_functions.random.seed(666):
        x = QF.randn(5000)

        # No NaN values should be generated
        assert not torch.any(torch.isnan(x))

        # No infinite values should be generated
        assert not torch.any(torch.isinf(x))

        # Values should be finite
        assert torch.all(torch.isfinite(x))


def test_randn_memory_efficiency() -> None:
    """Test that randn doesn't cause memory issues with repeated calls."""
    with qfeval_functions.random.seed(444):
        # Multiple calls shouldn't cause memory leaks
        for i in range(10):
            x = QF.randn(1000)
            # Force deletion to test memory cleanup
            del x


def test_randn_distribution_shape() -> None:
    """Test that the distribution roughly follows normal distribution shape."""
    with qfeval_functions.random.seed(111):
        x = QF.randn(50000)

        # Rough test of normal distribution properties
        # Values within 1 sigma (~68%)
        within_1_sigma = torch.abs(x) < 1.0
        prop_1_sigma = torch.mean(within_1_sigma.float())
        assert 0.6 < prop_1_sigma < 0.75

        # Values within 2 sigma (~95%)
        within_2_sigma = torch.abs(x) < 2.0
        prop_2_sigma = torch.mean(within_2_sigma.float())
        assert 0.9 < prop_2_sigma < 0.98
