import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF
from tests.functions.test_utils import generic_test_memory_efficiency
from tests.functions.test_utils import generic_test_single_element


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
            x_cpu_explicit = QF.randn(5, device=torch.device("cpu"))
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


def test_randn_single_element() -> None:
    """Test randn with single element generation."""

    def randn_wrapper(x: torch.Tensor) -> torch.Tensor:
        # Use the shape of input to generate randn with same shape
        return QF.randn(*x.shape)

    generic_test_single_element(randn_wrapper)


def test_randn_consistency() -> None:
    """Test that randn produces consistent results with same seed."""
    # Test consistency with seeded generation
    with qfeval_functions.random.seed(42):
        result1 = QF.randn(5, 3)
    with qfeval_functions.random.seed(42):
        result2 = QF.randn(5, 3)

    torch.testing.assert_close(result1, result2)


def test_randn_memory_efficiency() -> None:
    """Test memory efficiency of randn generation."""

    def randn_wrapper(x: torch.Tensor) -> torch.Tensor:
        # Generate tensor with same shape as input
        return QF.randn(*x.shape)

    generic_test_memory_efficiency(randn_wrapper)


def test_randn_dtype_preservation() -> None:
    """Test that randn respects dtype parameter."""
    test_dtypes = [torch.float32, torch.float64]

    for dtype in test_dtypes:
        result = QF.randn(5, 3, dtype=dtype)
        assert result.dtype == dtype


def test_randn_device_preservation() -> None:
    """Test that randn respects device parameter."""
    # Test CPU device (default)
    result_cpu = QF.randn(5, 3)
    assert result_cpu.device.type == "cpu"

    # Test explicit CPU device if device parameter is supported
    try:
        result_cpu_explicit = QF.randn(5, 3, device=torch.device("cpu"))
        assert result_cpu_explicit.device.type == "cpu"
    except TypeError:
        # Device parameter might not be supported in this implementation
        pass
