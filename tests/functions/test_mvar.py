import math

import numpy as np
import pandas as pd
import torch

import qfeval_functions.functions as QF


def test_mvar_basic_functionality() -> None:
    """Test basic moving variance functionality against pandas."""
    a = QF.randn(100, 10)
    df = pd.DataFrame(a.numpy())
    np.testing.assert_allclose(
        QF.mvar(a, 10, dim=0).numpy(),
        df.rolling(10).var().to_numpy(),
        1e-6,
        1e-6,
    )
    np.testing.assert_allclose(
        QF.mvar(a, 10, dim=0, ddof=0).numpy(),
        df.rolling(10).var(ddof=0).to_numpy(),
        1e-6,
        1e-6,
    )


def test_mvar_simple_case() -> None:
    """Test moving variance with simple known data."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Moving variance with window size 3
    result = QF.mvar(x, 3, dim=0)

    # For window [1,2,3]: var = ((1-2)² + (2-2)² + (3-2)²) / 2 = 1.0
    # For window [2,3,4]: var = ((2-3)² + (3-3)² + (4-3)²) / 2 = 1.0
    # For window [3,4,5]: var = ((3-4)² + (4-4)² + (5-4)²) / 2 = 1.0
    expected = torch.tensor([math.nan, math.nan, 1.0, 1.0, 1.0])

    # Compare only finite values
    finite_mask = torch.isfinite(result)
    np.testing.assert_allclose(
        result[finite_mask].numpy(), expected[finite_mask].numpy()
    )


def test_mvar_constant_values() -> None:
    """Test moving variance with constant values."""
    x = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])

    result = QF.mvar(x, 3, dim=0)

    # Variance of constant values should be 0
    expected = torch.tensor([math.nan, math.nan, 0.0, 0.0, 0.0])

    finite_mask = torch.isfinite(result)
    np.testing.assert_allclose(
        result[finite_mask].numpy(), expected[finite_mask].numpy()
    )


def test_mvar_2d_tensors() -> None:
    """Test moving variance with 2D tensors."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]])

    # Test along dimension 1
    result = QF.mvar(x, 3, dim=1)

    assert result.shape == (2, 5)

    # Both rows should have the same pattern since they're linear
    finite_mask = torch.isfinite(result)
    assert finite_mask.sum() > 0  # Should have some finite values


def test_mvar_different_window_sizes() -> None:
    """Test moving variance with different window sizes."""
    x = torch.randn(20)

    for window_size in [2, 5, 10]:
        result = QF.mvar(x, window_size, dim=0)

        assert result.shape == x.shape
        # First (window_size - 1) values should be NaN
        assert torch.isnan(result[: window_size - 1]).all()
        # Remaining values should be finite
        assert torch.isfinite(result[window_size - 1 :]).all()


def test_mvar_ddof_values() -> None:
    """Test moving variance with different ddof values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Test ddof=0 (population variance)
    result_ddof0 = QF.mvar(x, 3, dim=0, ddof=0)

    # Test ddof=1 (sample variance)
    result_ddof1 = QF.mvar(x, 3, dim=0, ddof=1)

    # ddof=0 should be smaller than ddof=1
    finite_mask = torch.isfinite(result_ddof0) & torch.isfinite(result_ddof1)
    if finite_mask.any():
        assert torch.all(result_ddof0[finite_mask] < result_ddof1[finite_mask])


def test_mvar_window_larger_than_data() -> None:
    """Test moving variance when window is larger than data."""
    x = torch.tensor([1.0, 2.0, 3.0])

    result = QF.mvar(x, 5, dim=0)

    # All values should be NaN when window > data length
    assert torch.isnan(result).all()


def test_mvar_negative_dimension() -> None:
    """Test moving variance with negative dimension indexing."""
    x = torch.randn(5, 10)

    result_neg = QF.mvar(x, 3, dim=-1)
    result_pos = QF.mvar(x, 3, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_mvar_batch_processing() -> None:
    """Test moving variance with batch processing."""
    batch_size = 5
    seq_length = 20
    window_size = 5

    x = torch.randn(batch_size, seq_length)
    result = QF.mvar(x, window_size, dim=1)

    assert result.shape == (batch_size, seq_length)

    # Verify against pandas for each batch
    for i in range(batch_size):
        df_row = pd.DataFrame(x[i].numpy().reshape(1, -1)).T
        expected = df_row.rolling(window_size).var().to_numpy().flatten()

        finite_mask = np.isfinite(expected)
        np.testing.assert_allclose(
            result[i][finite_mask].numpy(), expected[finite_mask], rtol=1e-5
        )


def test_mvar_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10], dtype=torch.float64)
    result_small = QF.mvar(x_small, 3, dim=0)

    # Should handle small values without numerical issues
    finite_results = result_small[torch.isfinite(result_small)]
    assert torch.all(finite_results >= 0)  # Variance should be non-negative

    # Very large values
    x_large = torch.tensor([1e10, 2e10, 3e10, 4e10], dtype=torch.float64)
    result_large = QF.mvar(x_large, 3, dim=0)

    finite_results = result_large[torch.isfinite(result_large)]
    assert torch.all(finite_results >= 0)
    assert torch.all(torch.isfinite(finite_results))


def test_mvar_with_nan_values() -> None:
    """Test moving variance behavior with NaN values."""
    x = torch.tensor([1.0, 2.0, math.nan, 4.0, 5.0])

    result = QF.mvar(x, 3, dim=0)

    # NaN should propagate through moving window
    assert torch.isnan(result[2])  # Window containing NaN
    assert torch.isnan(result[3])  # Window containing NaN
    assert torch.isnan(result[4])  # Window containing NaN


def test_mvar_with_infinity() -> None:
    """Test moving variance behavior with infinity values."""
    x = torch.tensor([1.0, math.inf, 3.0, 4.0, 5.0])

    result = QF.mvar(x, 3, dim=0)

    # Infinity should cause NaN in variance calculation
    assert torch.isnan(result[2]) or torch.isinf(result[2])


def test_mvar_zero_ddof_edge_case() -> None:
    """Test moving variance with ddof=0 and small windows."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = QF.mvar(x, 2, dim=0, ddof=0)

    # Should compute population variance correctly
    # Window [1,2]: var = 0.25, Window [2,3]: var = 0.25, Window [3,4]: var = 0.25
    expected = torch.tensor([math.nan, 0.25, 0.25, 0.25])

    finite_mask = torch.isfinite(result)
    np.testing.assert_allclose(
        result[finite_mask].numpy(), expected[finite_mask].numpy()
    )


def test_mvar_high_dimensional() -> None:
    """Test moving variance with high-dimensional tensors."""
    x = torch.randn(3, 4, 20)

    result = QF.mvar(x, 5, dim=2)

    assert result.shape == (3, 4, 20)

    # Each element should follow the same pattern
    for i in range(3):
        for j in range(4):
            # First 4 values should be NaN, rest should be finite
            assert torch.isnan(result[i, j, :4]).all()
            assert torch.isfinite(result[i, j, 4:]).all()


def test_mvar_pandas_comparison_extended() -> None:
    """Extended comparison with pandas rolling variance."""
    # Test with different data patterns
    test_cases = [
        torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ),  # Triangular
        torch.tensor(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0]
        ),  # Step function
        torch.randn(15),  # Random data
    ]

    for window_size in [3, 5]:
        for test_data in test_cases:
            result = QF.mvar(test_data, window_size, dim=0)

            df = pd.DataFrame(test_data.numpy())
            expected = df.rolling(window_size).var().to_numpy().flatten()

            finite_mask = np.isfinite(expected)
            np.testing.assert_allclose(
                result[finite_mask].numpy(),
                expected[finite_mask],
                rtol=1e-5,
                atol=1e-8,
            )


def test_mvar_mathematical_properties() -> None:
    """Test mathematical properties of moving variance."""
    x = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

    # Test that variance is always non-negative
    result = QF.mvar(x, 3, dim=0)
    finite_results = result[torch.isfinite(result)]
    assert torch.all(finite_results >= 0)

    # Test that variance of identical values is zero
    x_constant = torch.tensor([5.0, 5.0, 5.0, 5.0])
    result_constant = QF.mvar(x_constant, 3, dim=0)
    finite_results = result_constant[torch.isfinite(result_constant)]
    assert torch.all(torch.abs(finite_results) < 1e-10)


def test_mvar_window_boundary_conditions() -> None:
    """Test moving variance at window boundaries."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    window_size = 4

    result = QF.mvar(x, window_size, dim=0)

    # First (window_size - 1) values should be NaN
    assert torch.isnan(result[: window_size - 1]).all()

    # Remaining values should be computed correctly
    assert torch.isfinite(result[window_size - 1 :]).all()


def test_mvar_different_ddof_comprehensive() -> None:
    """Comprehensive test of different ddof values."""
    x = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    window_size = 4

    for ddof in [0, 1, 2]:
        if ddof < window_size:  # Valid ddof
            result = QF.mvar(x, window_size, dim=0, ddof=ddof)

            # Should have finite values where expected
            finite_results = result[torch.isfinite(result)]
            assert len(finite_results) == len(x) - window_size + 1
            assert torch.all(finite_results >= 0)


def test_mvar_edge_case_single_value() -> None:
    """Test moving variance with single value input."""
    x = torch.tensor([5.0])

    result = QF.mvar(x, 1, dim=0, ddof=0)

    # Single value variance should be 0
    assert result.item() == 0.0


def test_mvar_alternating_pattern() -> None:
    """Test moving variance with alternating pattern."""
    x = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    result = QF.mvar(x, 2, dim=0)

    # Each pair [1, -1] should have the same variance
    finite_results = result[torch.isfinite(result)]
    if len(finite_results) > 1:
        # All variances should be equal for alternating pattern
        assert torch.allclose(
            finite_results, finite_results[0] * torch.ones_like(finite_results)
        )


def test_mvar_precision_validation() -> None:
    """Test precision with known variance calculations."""
    # Data where we can calculate variance manually
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

    result = QF.mvar(x, 3, dim=0, ddof=1)

    # Manual calculation for window [0,1,2]: mean=1, var=((0-1)²+(1-1)²+(2-1)²)/2 = 1
    # Manual calculation for window [1,2,3]: mean=2, var=((1-2)²+(2-2)²+(3-2)²)/2 = 1
    # Manual calculation for window [2,3,4]: mean=3, var=((2-3)²+(3-3)²+(4-3)²)/2 = 1
    expected_finite = torch.tensor([1.0, 1.0, 1.0])

    finite_results = result[torch.isfinite(result)]
    np.testing.assert_allclose(finite_results.numpy(), expected_finite.numpy())
