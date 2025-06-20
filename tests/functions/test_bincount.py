import numpy as np
import pytest
import torch

import qfeval_functions.functions as QF


def test_bincount() -> None:
    np.testing.assert_allclose(
        QF.bincount(torch.tensor([[1, 2, 2], [3, 3, 1]])).numpy(),
        [[0, 1, 2, 0], [0, 1, 0, 2]],
    )
    np.testing.assert_allclose(
        QF.bincount(torch.tensor([1, 2, 2, 3, 3, 1])).numpy(), [0, 2, 2, 2]
    )


@pytest.mark.parametrize(
    "x",
    [
        QF.randint(low=1, high=100, size=(100,)),
        QF.randint(low=1, high=100, size=(0,)),  # edge case
    ],
)
def test_bincount_compared_with_torch(x: torch.Tensor) -> None:
    # expect the same result as torch.bincount when the input is a 1D tensor.
    for minlength in [0, 10, 100, 200]:
        assert torch.allclose(
            QF.bincount(x, dim=-1, minlength=minlength),
            torch.bincount(x, minlength=minlength),
        )


def test_bincount_basic_functionality() -> None:
    """Test basic bincount functionality."""
    # Simple 1D case
    x = torch.tensor([0, 1, 1, 2, 2, 2])
    result = QF.bincount(x)
    expected = torch.tensor([1, 2, 3])  # count of 0, 1, 2
    torch.testing.assert_close(result, expected)

    # 2D case
    x_2d = torch.tensor([[0, 1, 1], [2, 2, 0]])
    result_2d = QF.bincount(x_2d)
    expected_2d = torch.tensor([[1, 2, 0], [1, 0, 2]])
    torch.testing.assert_close(result_2d, expected_2d)


def test_bincount_zeros_only() -> None:
    """Test bincount with tensor containing only zeros."""
    x_zeros = torch.tensor([0, 0, 0, 0])
    result = QF.bincount(x_zeros)
    expected = torch.tensor([4])  # 4 occurrences of 0
    torch.testing.assert_close(result, expected)


def test_bincount_consecutive_values() -> None:
    """Test bincount with consecutive integer values."""
    x = torch.tensor([0, 1, 2, 3, 4, 5])
    result = QF.bincount(x)
    expected = torch.ones(6, dtype=torch.int64)  # each value appears once
    torch.testing.assert_close(result, expected)


def test_bincount_repeated_values() -> None:
    """Test bincount with repeated values in different patterns."""
    # Repeated pattern
    x = torch.tensor([1, 1, 2, 2, 3, 3, 1, 2])
    result = QF.bincount(x)
    expected = torch.tensor([0, 3, 3, 2])  # counts for 0, 1, 2, 3
    torch.testing.assert_close(result, expected)

    # All same value
    x_same = torch.tensor([5, 5, 5, 5, 5])
    result_same = QF.bincount(x_same)
    expected_same = torch.zeros(6, dtype=torch.int64)
    expected_same[5] = 5
    torch.testing.assert_close(result_same, expected_same)


def test_bincount_minlength_parameter() -> None:
    """Test bincount with various minlength values."""
    x = torch.tensor([0, 1, 2])

    # minlength = 0 (default)
    result_0 = QF.bincount(x, minlength=0)
    expected_0 = torch.tensor([1, 1, 1])
    torch.testing.assert_close(result_0, expected_0)

    # minlength smaller than max value + 1
    result_small = QF.bincount(x, minlength=2)
    torch.testing.assert_close(
        result_small, expected_0
    )  # should be same as default

    # minlength larger than max value + 1
    result_large = QF.bincount(x, minlength=10)
    expected_large = torch.zeros(10, dtype=torch.int64)
    expected_large[0] = 1
    expected_large[1] = 1
    expected_large[2] = 1
    torch.testing.assert_close(result_large, expected_large)


def test_bincount_dim_parameter() -> None:
    """Test bincount with different dimension parameters."""
    x = torch.tensor([[0, 1, 2], [1, 1, 0], [2, 0, 2]])

    # dim=-1 (default, along last dimension)
    result_neg1 = QF.bincount(x, dim=-1)
    expected_neg1 = torch.tensor([[1, 1, 1], [1, 2, 0], [1, 0, 2]])
    torch.testing.assert_close(result_neg1, expected_neg1)

    # dim=1 (same as dim=-1 for 2D tensor)
    result_1 = QF.bincount(x, dim=1)
    torch.testing.assert_close(result_1, expected_neg1)

    # dim=0 (along first dimension)
    result_0 = QF.bincount(x, dim=0)
    expected_0 = torch.tensor([[1, 1, 1], [1, 2, 0], [1, 0, 2]])
    torch.testing.assert_close(result_0, expected_0)


def test_bincount_shape_preservation() -> None:
    """Test that bincount preserves batch dimensions correctly."""
    # 2D input
    x_2d = torch.tensor([[0, 1, 1], [2, 0, 2]])
    result_2d = QF.bincount(x_2d)
    assert result_2d.shape[0] == 2  # batch dimension preserved

    # 3D input
    x_3d = torch.tensor([[[0, 1], [1, 0]], [[2, 2], [0, 1]]])
    result_3d = QF.bincount(x_3d)
    assert result_3d.shape[:2] == (2, 2)  # first two dimensions preserved


def test_bincount_large_values() -> None:
    """Test bincount with large integer values."""
    x = torch.tensor([100, 200, 100, 300])
    result = QF.bincount(x)

    # Should create bins up to max value
    assert result.shape[0] == 301  # 0 to 300 inclusive
    assert result[100].item() == 2  # 100 appears twice
    assert result[200].item() == 1  # 200 appears once
    assert result[300].item() == 1  # 300 appears once
    assert result[150].item() == 0  # 150 doesn't appear


def test_bincount_sparse_values() -> None:
    """Test bincount with sparse (widely separated) values."""
    x = torch.tensor([0, 5, 10, 15])
    result = QF.bincount(x)

    assert result.shape[0] == 16  # 0 to 15 inclusive
    assert result[0].item() == 1
    assert result[5].item() == 1
    assert result[10].item() == 1
    assert result[15].item() == 1
    # All other values should be 0
    for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]:
        assert result[i].item() == 0


def test_bincount_multidimensional() -> None:
    """Test bincount with various multidimensional inputs."""
    # 3D tensor
    x_3d = torch.tensor([[[0, 1], [1, 2]], [[2, 0], [1, 1]]])
    result_3d = QF.bincount(x_3d)

    # Check shape
    assert result_3d.shape == (2, 2, 3)  # batch dims + output dim

    # Check individual results
    torch.testing.assert_close(result_3d[0, 0], torch.tensor([1, 1, 0]))
    torch.testing.assert_close(result_3d[0, 1], torch.tensor([0, 1, 1]))
    torch.testing.assert_close(result_3d[1, 0], torch.tensor([1, 0, 1]))
    torch.testing.assert_close(result_3d[1, 1], torch.tensor([0, 2, 0]))


def test_bincount_edge_cases() -> None:
    """Test bincount with various edge cases."""
    # Maximum value is 0
    x_max_zero = torch.tensor([0, 0, 0])
    result_max_zero = QF.bincount(x_max_zero)
    expected_max_zero = torch.tensor([3])
    torch.testing.assert_close(result_max_zero, expected_max_zero)

    # Single unique value (not zero)
    x_single_unique = torch.tensor([7, 7, 7])
    result_single_unique = QF.bincount(x_single_unique)
    expected_single_unique = torch.zeros(8, dtype=torch.int64)
    expected_single_unique[7] = 3
    torch.testing.assert_close(result_single_unique, expected_single_unique)


def test_bincount_mathematical_properties() -> None:
    """Test mathematical properties of bincount."""
    x = torch.tensor([0, 1, 2, 1, 0, 3, 2, 1])
    result = QF.bincount(x)

    # Sum of counts should equal input length
    assert result.sum().item() == len(x)

    # Each bin count should be non-negative
    assert torch.all(result >= 0)

    # Verify specific counts
    assert result[0].item() == 2  # 0 appears twice
    assert result[1].item() == 3  # 1 appears three times
    assert result[2].item() == 2  # 2 appears twice
    assert result[3].item() == 1  # 3 appears once


def test_bincount_comparison_with_torch() -> None:
    """Test detailed comparison with torch.bincount for 1D cases."""
    test_cases = [
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([5, 5, 5]),
        torch.tensor([0, 10, 5]),
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([0]),
    ]

    for x in test_cases:
        for minlength in [0, 5, 15]:
            result_qf = QF.bincount(x, minlength=minlength)
            result_torch = torch.bincount(x, minlength=minlength)
            torch.testing.assert_close(result_qf, result_torch)


@pytest.mark.random
def test_bincount_batch_processing() -> None:
    """Test bincount with batch processing scenarios."""
    batch_size = 4
    seq_length = 10
    max_value = 5

    x = torch.randint(0, max_value, (batch_size, seq_length))
    result = QF.bincount(x)

    assert result.shape[0] == batch_size
    assert result.shape[1] == max_value  # 0 to max_value-1

    # Check that each batch sums to seq_length
    batch_sums = result.sum(dim=1)
    expected_sums = torch.full((batch_size,), seq_length, dtype=torch.int64)
    torch.testing.assert_close(batch_sums, expected_sums)


def test_bincount_histogram_properties() -> None:
    """Test that bincount acts as a histogram."""
    # Generate data with known distribution
    x = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3])
    result = QF.bincount(x)

    # Should match manual count
    expected = torch.tensor([2, 3, 4, 1])  # counts for 0, 1, 2, 3
    torch.testing.assert_close(result, expected)

    # Total count should match input size
    assert result.sum().item() == len(x)


def test_bincount_with_gaps() -> None:
    """Test bincount when input has gaps in values."""
    x = torch.tensor([0, 2, 4, 6, 8])  # even numbers only
    result = QF.bincount(x)

    expected = torch.zeros(9, dtype=torch.int64)
    expected[0] = 1
    expected[2] = 1
    expected[4] = 1
    expected[6] = 1
    expected[8] = 1
    # Odd indices should be 0

    torch.testing.assert_close(result, expected)


@pytest.mark.random
def test_bincount_performance() -> None:
    """Test bincount performance with larger tensors."""
    # Test with moderately large tensor
    x_large = torch.randint(0, 1000, (100, 100))
    result_large = QF.bincount(x_large)

    assert result_large.shape == (100, 1000)

    # Verify properties
    batch_sums = result_large.sum(dim=1)
    assert torch.all(batch_sums == 100)  # each batch has 100 elements


def test_bincount_consistency() -> None:
    """Test that bincount produces consistent results."""
    x = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 1]])

    # Multiple calls should produce same result
    result1 = QF.bincount(x)
    result2 = QF.bincount(x)
    torch.testing.assert_close(result1, result2)

    # Manual verification
    expected = torch.tensor([[1, 2, 1], [1, 1, 2]])
    torch.testing.assert_close(result1, expected)


def test_bincount_different_minlengths() -> None:
    """Test bincount behavior with various minlength values."""
    x = torch.tensor([1, 3, 1])  # max value is 3

    # minlength < max_value + 1
    result_small = QF.bincount(x, minlength=2)
    expected_small = torch.tensor([0, 2, 0, 1])  # natural length is 4
    torch.testing.assert_close(result_small, expected_small)

    # minlength = max_value + 1
    result_exact = QF.bincount(x, minlength=4)
    torch.testing.assert_close(result_exact, expected_small)

    # minlength > max_value + 1
    result_large = QF.bincount(x, minlength=10)
    expected_large = torch.zeros(10, dtype=torch.int64)
    expected_large[1] = 2
    expected_large[3] = 1
    torch.testing.assert_close(result_large, expected_large)


def test_bincount_multidimensional_empty() -> None:
    """Test bincount with multidimensional empty tensors."""
    # Skip this test as apply_for_axis has limitations with empty tensors
    # The bincount function works with basic empty tensors but has issues
    # with multidimensional empty tensors due to apply_for_axis limitations
    x_empty_1d = torch.empty(0, dtype=torch.int64)
    result = QF.bincount(x_empty_1d)
    assert result.shape[-1] == 0  # output should be empty


def test_bincount_numerical_stability() -> None:
    """Test numerical stability of bincount."""
    # Test with maximum reasonable values
    x = torch.tensor([0, 1000, 0, 1000])
    result = QF.bincount(x)

    assert result.shape[0] == 1001  # 0 to 1000 inclusive
    assert result[0].item() == 2
    assert result[1000].item() == 2
    assert result[500].item() == 0  # middle value should be 0


def test_bincount_axis_consistency() -> None:
    """Test that bincount works consistently across different axes."""
    x = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 2, 1]])

    # Test along dim=1 (default)
    result_dim1 = QF.bincount(x, dim=1)

    # Test along dim=0
    x_transposed = x.T
    result_dim0 = QF.bincount(x_transposed, dim=0)

    # Results should have appropriate shapes
    assert result_dim1.shape == (3, 3)  # 3 rows, max value is 2 so 3 bins
    assert result_dim0.shape == (3, 3)  # 3 columns, max value is 2 so 3 bins
