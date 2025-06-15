import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_msum_basic() -> None:
    """Test basic moving sum with span=3 on a simple sequence."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.msum(x, span=3, dim=0)
    expected = torch.tensor([math.nan, math.nan, 6.0, 9.0, 12.0])

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_msum_span_2() -> None:
    """Test moving sum with span=2 to verify different window sizes."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.msum(x, span=2, dim=0)
    expected = torch.tensor([math.nan, 3.0, 5.0, 7.0, 9.0])

    np.testing.assert_allclose(
        result[1:].numpy(), expected[1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[0])


def test_msum_2d_tensor() -> None:
    """Test moving sum on 2D tensor along axis 1."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    result = QF.msum(x, span=2, dim=1)
    expected = torch.tensor(
        [[math.nan, 3.0, 5.0, 7.0], [math.nan, 11.0, 13.0, 15.0]]
    )

    np.testing.assert_allclose(
        result[:, 1:].numpy(), expected[:, 1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:, 0]).all()


def test_msum_along_axis_0() -> None:
    """Test moving sum along axis 0 of a 2D tensor."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = QF.msum(x, span=2, dim=0)
    expected = torch.tensor([[math.nan, math.nan], [4.0, 6.0], [8.0, 10.0]])

    np.testing.assert_allclose(
        result[1:].numpy(), expected[1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[0]).all()


def test_msum_negative_dim() -> None:
    """Test moving sum with negative dimension indexing (last dimension)."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = QF.msum(x, span=2, dim=-1)
    expected = torch.tensor([[math.nan, 3.0, 5.0], [math.nan, 9.0, 11.0]])

    np.testing.assert_allclose(
        result[:, 1:].numpy(), expected[:, 1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:, 0]).all()


def test_msum_span_equals_length() -> None:
    """Test moving sum when span equals tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.msum(x, span=3, dim=0)
    expected = torch.tensor([math.nan, math.nan, 6.0])

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_msum_large_span() -> None:
    """Test moving sum when span is larger than tensor length (should return NaN)."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.msum(x, span=5, dim=0)
    expected = torch.tensor([math.nan, math.nan, math.nan])

    assert torch.isnan(result).all()


def test_msum_single_element() -> None:
    """Test moving sum with single element tensor and span=1."""
    x = torch.tensor([5.0])
    result = QF.msum(x, span=1, dim=0)
    expected = torch.tensor([5.0])

    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-6)


def test_msum_with_zeros() -> None:
    """Test moving sum with tensor containing zero values."""
    x = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
    result = QF.msum(x, span=3, dim=0)
    expected = torch.tensor([math.nan, math.nan, 1.0, 3.0, 2.0])

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_msum_with_negative_values() -> None:
    """Test moving sum with tensor containing negative values."""
    x = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
    result = QF.msum(x, span=3, dim=0)
    expected = torch.tensor([math.nan, math.nan, -2.0, 3.0, -4.0])

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_msum_with_nan_values() -> None:
    """Test moving sum behavior when input contains NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0])
    result = QF.msum(x, span=3, dim=0)

    # Results should be NaN when NaN is within the current window
    assert torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert torch.isnan(result[2])
    assert torch.isnan(result[3])
    # Position 4 has window [3.0, 4.0, 5.0] which doesn't contain NaN
    assert not torch.isnan(result[4])
    assert result[4] == 12.0  # 3.0 + 4.0 + 5.0


def test_msum_dtype_preservation() -> None:
    """Test that moving sum preserves input tensor's dtype."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = QF.msum(x, span=2, dim=0)
    assert result.dtype == torch.float64


def test_msum_3d_tensor() -> None:
    """Test moving sum on 3D tensor along axis 0."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = QF.msum(x, span=2, dim=0)
    expected = torch.tensor(
        [
            [[math.nan, math.nan], [math.nan, math.nan]],
            [[6.0, 8.0], [10.0, 12.0]],
        ]
    )

    np.testing.assert_allclose(
        result[1:].numpy(), expected[1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[0]).all()


def test_msum_very_large_span() -> None:
    """Test moving sum with span much larger than tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.msum(x, span=10, dim=0)
    expected = torch.tensor([math.nan, math.nan, math.nan])

    assert torch.isnan(result).all()


def test_msum_inf_and_nan_mixed() -> None:
    """Test moving sum with both infinity and NaN values."""
    x = torch.tensor([1.0, math.inf, math.nan, 4.0, 5.0])
    result = QF.msum(x, span=3, dim=0)

    # All windows containing inf or nan should be inf or nan
    assert torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert torch.isnan(result[2])
    assert torch.isnan(result[3])
    # Window at position 4: [nan, 4.0, 5.0] - still contains NaN
    assert torch.isnan(result[4])


def test_msum_edge_case_span_equals_one() -> None:
    """Test moving sum with span=1 (should return original values)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.msum(x, span=1, dim=0)
    expected = x.clone()

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_msum_alternating_signs() -> None:
    """Test moving sum with alternating positive and negative values."""
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    result = QF.msum(x, span=2, dim=0)
    expected = torch.tensor([math.nan, 0.0, 1.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(result[1:].numpy(), expected[1:].numpy())
    assert torch.isnan(result[0])


def test_msum_very_small_values() -> None:
    """Test moving sum with very small values to test numerical precision."""
    x = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10], dtype=torch.float64)
    result = QF.msum(x, span=2, dim=0)
    expected = torch.tensor(
        [math.nan, 3e-10, 5e-10, 7e-10], dtype=torch.float64
    )

    np.testing.assert_allclose(
        result[1:].numpy(), expected[1:].numpy(), atol=1e-15
    )
    assert torch.isnan(result[0])
