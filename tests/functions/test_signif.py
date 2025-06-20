import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_signif() -> None:
    x = torch.tensor(
        [0.123456789, 123456789, 1.23456789e-20], dtype=torch.float64
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x).numpy(),
        [0.123457, 123457000, 1.23457e-20],
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x.float(), 3).numpy(),
        [0.123, 123000000, 1.23e-20],
        decimal=8,
    )
    x = torch.tensor(
        [
            0.1234,
            -12.34,
            9.876,
            -9.999,
            1.234e-100,
            1.234e100,
            0.0,
            math.nan,
            math.inf,
            -math.inf,
        ],
        dtype=torch.float64,
    )
    np.testing.assert_array_almost_equal(
        QF.signif(x, 3).numpy(),
        [
            0.123,
            -12.3,
            9.88,
            -10.00,
            1.23e-100,
            1.23e100,
            0.0,
            math.nan,
            math.inf,
            -math.inf,
        ],
        decimal=5,
    )


def test_signif_basic_functionality() -> None:
    """Test basic signif functionality."""
    # Test with default decimals (6)
    x = torch.tensor([1.234567890])
    result = QF.signif(x)
    expected = torch.tensor([1.23457])
    np.testing.assert_array_almost_equal(
        result.numpy(), expected.numpy(), decimal=10
    )

    # Test with different decimals
    result_3 = QF.signif(x, 3)
    expected_3 = torch.tensor([1.23])
    np.testing.assert_array_almost_equal(
        result_3.numpy(), expected_3.numpy(), decimal=10
    )

    result_1 = QF.signif(x, 1)
    expected_1 = torch.tensor([1.0])
    np.testing.assert_array_almost_equal(
        result_1.numpy(), expected_1.numpy(), decimal=10
    )


def test_signif_different_magnitudes() -> None:
    """Test signif with different order of magnitudes."""
    # Very small numbers
    x_small = torch.tensor([1.23456e-10, 9.87654e-15])
    result_small = QF.signif(x_small, 3)
    expected_small = torch.tensor([1.23e-10, 9.88e-15])
    np.testing.assert_array_almost_equal(
        result_small.numpy(), expected_small.numpy(), decimal=20
    )

    # Very large numbers
    x_large = torch.tensor([1.23456e10, 9.87654e15])
    result_large = QF.signif(x_large, 3)
    expected_large = torch.tensor([1.23e10, 9.88e15])
    np.testing.assert_array_almost_equal(
        result_large.numpy(), expected_large.numpy(), decimal=5
    )

    # Numbers around 1
    x_around_one = torch.tensor([0.12345, 1.2345, 12.345, 123.45])
    result_around_one = QF.signif(x_around_one, 3)
    expected_around_one = torch.tensor([0.123, 1.23, 12.3, 123.0])
    np.testing.assert_array_almost_equal(
        result_around_one.numpy(), expected_around_one.numpy(), decimal=10
    )


def test_signif_negative_numbers() -> None:
    """Test signif with negative numbers."""
    x = torch.tensor([-1.23456, -12.3456, -123.456, -0.00123456])
    result = QF.signif(x, 3)
    expected = torch.tensor([-1.23, -12.3, -123.0, -0.00123])
    np.testing.assert_array_almost_equal(
        result.numpy(), expected.numpy(), decimal=10
    )


def test_signif_zero_values() -> None:
    """Test signif with zero values."""
    x = torch.tensor([0.0, -0.0, 0.00000])
    result = QF.signif(x, 3)
    expected = torch.tensor([0.0, 0.0, 0.0])
    torch.testing.assert_close(result, expected)


def test_signif_special_values() -> None:
    """Test signif with special values (NaN, inf, -inf)."""
    x = torch.tensor([math.nan, math.inf, -math.inf])
    result = QF.signif(x, 3)

    # NaN should remain NaN
    assert torch.isnan(result[0])
    # Infinities should remain infinities
    assert torch.isinf(result[1]) and result[1] > 0
    assert torch.isinf(result[2]) and result[2] < 0


def test_signif_edge_decimals() -> None:
    """Test signif with edge case decimal values."""
    x = torch.tensor([1.23456])

    # Very small decimals
    result_1 = QF.signif(x, 1)
    expected_1 = torch.tensor([1.0])
    np.testing.assert_array_almost_equal(
        result_1.numpy(), expected_1.numpy(), decimal=10
    )

    # Larger decimals
    result_10 = QF.signif(x, 10)
    # Should be limited by float precision
    assert torch.isfinite(result_10).all()


def test_signif_shape_preservation() -> None:
    """Test that signif preserves tensor shape."""
    shapes = [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]

    for shape in shapes:
        x = torch.randn(shape)
        result = QF.signif(x, 3)
        assert result.shape == x.shape


def test_signif_rounding_behavior() -> None:
    """Test signif rounding behavior."""
    # Test rounding at exactly 0.5
    x = torch.tensor([1.25, 1.35, 1.45, 1.55])
    result = QF.signif(x, 2)
    # Just check that rounding occurred
    assert torch.all(result != x)
    assert torch.all(torch.isfinite(result))


def test_signif_multidimensional() -> None:
    """Test signif with multidimensional tensors."""
    x = torch.tensor([[1.23456, 2.34567, 3.45678], [4.56789, 5.67890, 6.78901]])
    result = QF.signif(x, 3)
    expected = torch.tensor([[1.23, 2.35, 3.46], [4.57, 5.68, 6.79]])
    np.testing.assert_array_almost_equal(
        result.numpy(), expected.numpy(), decimal=10
    )


def test_signif_batch_processing() -> None:
    """Test signif with batch processing."""
    batch_size = 4
    features = 6
    x = torch.randn(batch_size, features) * 100  # Scale to have variety

    result = QF.signif(x, 4)
    assert result.shape == (batch_size, features)

    # All values should be finite (except if input had special values)
    finite_mask = torch.isfinite(x)
    assert torch.isfinite(result[finite_mask]).all()


def test_signif_precision_limits() -> None:
    """Test signif behavior at precision limits."""
    # Very small numbers near machine epsilon
    x_tiny = torch.tensor([1e-30, 1e-100, 1e-300])
    result_tiny = QF.signif(x_tiny, 3)
    # Should handle gracefully
    assert result_tiny.shape == x_tiny.shape

    # Very large numbers
    x_huge = torch.tensor([1e30, 1e100, 1e200])
    result_huge = QF.signif(x_huge, 3)
    assert result_huge.shape == x_huge.shape


def test_signif_integer_like_values() -> None:
    """Test signif with integer-like values."""
    x = torch.tensor([1.0, 2.0, 10.0, 100.0, 1000.0])
    result = QF.signif(x, 2)
    expected = torch.tensor([1.0, 2.0, 10.0, 100.0, 1000.0])
    # Integer values should remain the same if they have fewer digits
    assert torch.allclose(result[:3], expected[:3])


def test_signif_fractional_values() -> None:
    """Test signif with purely fractional values."""
    x = torch.tensor([0.1, 0.01, 0.001, 0.0001])
    result = QF.signif(x, 1)
    expected = torch.tensor([0.1, 0.01, 0.001, 0.0001])
    torch.testing.assert_close(result, expected)


def test_signif_mixed_signs() -> None:
    """Test signif with mixed positive and negative values."""
    x = torch.tensor([1.23456, -1.23456, 0.0, 12.3456, -12.3456])
    result = QF.signif(x, 3)
    expected = torch.tensor([1.23, -1.23, 0.0, 12.3, -12.3])
    np.testing.assert_array_almost_equal(
        result.numpy(), expected.numpy(), decimal=10
    )


def test_signif_gradient_compatibility() -> None:
    """Test that signif works with gradient computation."""
    x = torch.tensor([1.23456, 2.34567], requires_grad=True)
    result = QF.signif(x, 3)

    # Should be able to compute gradients
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_signif_numerical_stability() -> None:
    """Test numerical stability of signif."""
    # Test with numbers that might cause numerical issues
    x = torch.tensor([1.0000001, 0.9999999, 1e-15, 1e15])
    result = QF.signif(x, 6)

    # Should not produce NaN or inf for finite inputs
    assert torch.isfinite(result).all()


def test_signif_comparison_with_manual_calculation() -> None:
    """Test signif against manual calculations."""
    # Test case: 123.456 with 3 significant digits should be 123
    x = torch.tensor([123.456])
    result = QF.signif(x, 3)
    expected = torch.tensor([123.0])
    torch.testing.assert_close(result, expected)

    # Test case: 0.001234 with 2 significant digits should be 0.0012
    x2 = torch.tensor([0.001234])
    result2 = QF.signif(x2, 2)
    expected2 = torch.tensor([0.0012])
    torch.testing.assert_close(result2, expected2)


def test_signif_edge_rounding_cases() -> None:
    """Test signif with edge cases for rounding."""
    # Numbers that are exactly at rounding boundaries
    x = torch.tensor([1.5, 2.5, 3.5, 4.5])
    result = QF.signif(x, 1)
    # Check that rounding occurred consistently
    assert torch.all(torch.isfinite(result))
    assert torch.all(result.int() == result)  # Should be integers


def test_signif_very_small_decimals() -> None:
    """Test signif with very small decimal counts."""
    x = torch.tensor([123.456789])

    # Test with decimals = 1
    result_1 = QF.signif(x, 1)
    expected_1 = torch.tensor([100.0])
    torch.testing.assert_close(result_1, expected_1)


def test_signif_large_decimals() -> None:
    """Test signif with large decimal counts."""
    x = torch.tensor([1.23456789])

    # Test with large decimals (limited by float precision)
    result_15 = QF.signif(x, 15)
    # Should not introduce artifacts
    assert torch.isfinite(result_15).all()
    # Result should be close to original for high precision
    assert torch.allclose(result_15, x)


def test_signif_performance() -> None:
    """Test signif performance with large tensors."""
    for size in [1000, 5000]:
        x = torch.randn(size) * 1000  # Scale for variety
        result = QF.signif(x, 5)

        # Verify properties
        assert result.shape == x.shape
        finite_mask = torch.isfinite(x)
        assert torch.isfinite(result[finite_mask]).all()

        # Clean up
        del x, result


def test_signif_with_repeated_digits() -> None:
    """Test signif with numbers having repeated digits."""
    x = torch.tensor([111.111, 222.222, 333.333])
    result = QF.signif(x, 3)
    expected = torch.tensor([111.0, 222.0, 333.0])
    torch.testing.assert_close(result, expected)


def test_signif_boundary_values() -> None:
    """Test signif with boundary values."""
    # Test with values at powers of 10
    x = torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0])
    result = QF.signif(x, 2)
    expected = torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0])
    torch.testing.assert_close(result, expected)


def test_signif_scientific_notation() -> None:
    """Test signif with numbers in scientific notation range."""
    x = torch.tensor([1.23e-5, 4.56e10, 7.89e-15, 1.23e20])
    result = QF.signif(x, 2)
    expected = torch.tensor([1.2e-5, 4.6e10, 7.9e-15, 1.2e20])
    np.testing.assert_array_almost_equal(
        result.numpy(), expected.numpy(), decimal=25
    )


def test_signif_symmetry() -> None:
    """Test that signif treats positive and negative values symmetrically."""
    x_pos = torch.tensor([1.23456, 12.3456, 123.456])
    x_neg = -x_pos

    result_pos = QF.signif(x_pos, 3)
    result_neg = QF.signif(x_neg, 3)

    # Results should be negatives of each other
    torch.testing.assert_close(result_neg, -result_pos)


def test_signif_default_parameter() -> None:
    """Test signif with default decimals parameter."""
    x = torch.tensor([1.234567890123456])
    result_default = QF.signif(x)  # Should use default decimals=6
    result_explicit = QF.signif(x, 6)

    torch.testing.assert_close(result_default, result_explicit)


def test_signif_consistency_across_dtypes() -> None:
    """Test signif consistency across different dtypes."""
    value = 1.23456
    x_float32 = torch.tensor([value], dtype=torch.float32)
    x_float64 = torch.tensor([value], dtype=torch.float64)

    result_32 = QF.signif(x_float32, 3)
    result_64 = QF.signif(x_float64, 3)

    # Results should be approximately equal (within dtype precision)
    assert torch.allclose(result_32.double(), result_64, rtol=1e-6)
