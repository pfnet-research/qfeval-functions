import numpy as np
import torch

import qfeval_functions.functions as QF


def test_cumcount() -> None:
    a = torch.tensor([0, 1, 3, 2, 1, 0, 3, 1, 2])
    np.testing.assert_array_almost_equal(
        QF.cumcount(a).numpy(), np.array([0, 0, 0, 0, 1, 1, 1, 2, 1])
    )
    a = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_almost_equal(
        QF.cumcount(a, dim=1).numpy(),
        np.array([[0, 1, 2, 0, 1, 2, 3, 4, 5]]),
    )


def test_cumcount_simple_grouping() -> None:
    """Test cumulative counting with simple repeated values."""
    x = torch.tensor([1, 1, 2, 2, 2, 3, 3])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 1, 0, 1, 2, 0, 1])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_all_unique() -> None:
    """Test cumulative counting with all unique values."""
    x = torch.tensor([1, 2, 3, 4, 5])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_all_same() -> None:
    """Test cumulative counting with all identical values."""
    x = torch.tensor([7, 7, 7, 7, 7])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_2d_tensors_dim0() -> None:
    """Test cumulative counting on 2D tensors along dimension 0."""
    x = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 3, 3]])
    result = QF.cumcount(x, dim=0)
    expected = torch.tensor([[0, 0, 0], [1, 1, 0], [2, 0, 1]])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_2d_tensors_dim1() -> None:
    """Test cumulative counting on 2D tensors along dimension 1."""
    x = torch.tensor([[1, 1, 2, 2], [3, 3, 3, 4]])
    result = QF.cumcount(x, dim=1)
    expected = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 0]])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_negative_dimension() -> None:
    """Test cumulative counting with negative dimension indexing."""
    x = torch.tensor([[1, 2, 1], [2, 1, 2]])

    result_neg = QF.cumcount(x, dim=-1)
    result_pos = QF.cumcount(x, dim=1)

    np.testing.assert_array_equal(result_neg.numpy(), result_pos.numpy())


def test_cumcount_3d_tensors() -> None:
    """Test cumulative counting with 3D tensors."""
    x = torch.tensor([[[1, 1], [2, 2]], [[1, 2], [2, 1]]])

    # Test along last dimension
    result = QF.cumcount(x, dim=2)
    expected = torch.tensor([[[0, 1], [0, 1]], [[0, 0], [0, 0]]])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_float_values() -> None:
    """Test cumulative counting with floating point values."""
    x = torch.tensor([1.5, 2.0, 1.5, 3.0, 2.0, 1.5])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 1, 0, 1, 2])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_negative_values() -> None:
    """Test cumulative counting with negative values."""
    x = torch.tensor([-1, -2, -1, 0, -2, -1])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 1, 0, 1, 2])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_mixed_positive_negative() -> None:
    """Test cumulative counting with mixed positive and negative values."""
    x = torch.tensor([1, -1, 1, -1, 0, 1])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 1, 1, 0, 2])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_large_groups() -> None:
    """Test cumulative counting with large groups."""
    # Create tensor with large repeated groups
    group_size = 100
    x = torch.cat([torch.full((group_size,), i) for i in range(5)])
    result = QF.cumcount(x)

    # Each group should count from 0 to group_size-1
    for i in range(5):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        expected_group = torch.arange(group_size)
        np.testing.assert_array_equal(
            result[start_idx:end_idx].numpy(), expected_group.numpy()
        )


def test_cumcount_random_pattern() -> None:
    """Test cumulative counting with random-like pattern."""
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    result = QF.cumcount(x)

    # Manually verify some key positions
    # Value 1 appears at positions 1, 3 -> counts should be [0, 1]
    # Value 5 appears at positions 4, 8, 10 -> counts should be [0, 1, 2]
    # Value 3 appears at positions 0, 9 -> counts should be [0, 1]

    assert result[1].item() == 0  # First occurrence of 1
    assert result[3].item() == 1  # Second occurrence of 1
    assert result[4].item() == 0  # First occurrence of 5
    assert result[8].item() == 1  # Second occurrence of 5
    assert result[10].item() == 2  # Third occurrence of 5


def test_cumcount_batch_processing() -> None:
    """Test cumulative counting with batch processing."""
    batch_size = 3
    seq_length = 8

    x = torch.tensor(
        [
            [1, 1, 2, 2, 3, 1, 2, 3],
            [4, 5, 4, 6, 5, 4, 6, 7],
            [8, 8, 8, 9, 9, 8, 9, 9],
        ]
    )

    result = QF.cumcount(x, dim=1)

    assert result.shape == (batch_size, seq_length)

    # Verify first batch: [1,1,2,2,3,1,2,3] -> [0,1,0,1,0,2,2,1]
    expected_batch0 = torch.tensor([0, 1, 0, 1, 0, 2, 2, 1])
    np.testing.assert_array_equal(result[0].numpy(), expected_batch0.numpy())


def test_cumcount_zero_values() -> None:
    """Test cumulative counting with zero values."""
    x = torch.tensor([0, 1, 0, 2, 0, 1, 0])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 1, 0, 2, 1, 3])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_ascending_sequence() -> None:
    """Test cumulative counting with ascending sequence."""
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 0, 0, 0, 0])  # All unique values
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_descending_sequence() -> None:
    """Test cumulative counting with descending sequence."""
    x = torch.tensor([6, 5, 4, 3, 2, 1])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 0, 0, 0, 0])  # All unique values
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_interleaved_pattern() -> None:
    """Test cumulative counting with interleaved pattern."""
    x = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2])
    result = QF.cumcount(x)
    # 1 appears at positions 0,2,4,6 -> counts [0,1,2,3]
    # 2 appears at positions 1,3,5,7 -> counts [0,1,2,3]
    expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_large_values() -> None:
    """Test cumulative counting with large values."""
    x = torch.tensor([1000000, 999999, 1000000, 1000001, 999999])
    result = QF.cumcount(x)
    expected = torch.tensor([0, 0, 1, 0, 1])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_repeated_at_ends() -> None:
    """Test cumulative counting with repeated values at start and end."""
    x = torch.tensor([5, 5, 1, 2, 3, 5, 5])
    result = QF.cumcount(x)
    # Value 5 appears at positions 0,1,5,6 -> counts [0,1,2,3]
    expected = torch.tensor([0, 1, 0, 0, 0, 2, 3])
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_cumcount_high_dimensional() -> None:
    """Test cumulative counting with high-dimensional tensors."""
    x = torch.tensor([[[1, 2], [1, 3]], [[2, 1], [3, 1]], [[1, 1], [2, 3]]])

    # Test along dimension 0 (across batches)
    result = QF.cumcount(x, dim=0)

    assert result.shape == (3, 2, 2)

    # Value at position [0,0,0] = 1, appears again at [2,0,0] -> counts [0, 1]
    assert result[0, 0, 0].item() == 0
    assert result[2, 0, 0].item() == 1


def test_cumcount_mathematical_properties() -> None:
    """Test mathematical properties of cumulative counting."""
    x = torch.tensor([1, 2, 1, 3, 2, 1, 3, 2])
    result = QF.cumcount(x)

    # All counts should be non-negative
    assert torch.all(result >= 0)

    # Maximum count for any value should be (occurrences - 1)
    unique_vals = torch.unique(x)
    for val in unique_vals:
        mask = x == val
        max_count = result[mask].max().item()
        expected_max = mask.sum().item() - 1
        assert max_count == expected_max


def test_cumcount_edge_case_single_group() -> None:
    """Test cumulative counting with single large group."""
    n = 20
    x = torch.full((n,), 42)  # All same value
    result = QF.cumcount(x)
    expected = torch.arange(n)
    np.testing.assert_array_equal(result.numpy(), expected.numpy())
