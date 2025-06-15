import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_mstd_basic() -> None:
    """Test basic moving standard deviation with span=3 on a simple sequence."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.mstd(x, span=3, dim=0)

    expected = torch.tensor(
        [math.nan, math.nan, math.sqrt(1.0), math.sqrt(1.0), math.sqrt(1.0)]
    )

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_mstd_with_ddof_0() -> None:
    """Test moving standard deviation with ddof=0 (population standard deviation)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.mstd(x, span=3, dim=0, ddof=0)

    expected_variance = torch.tensor(
        [math.nan, math.nan, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )
    expected = torch.sqrt(expected_variance)

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_mstd_2d_tensor() -> None:
    """Test moving standard deviation on 2D tensor along axis 1."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    result = QF.mstd(x, span=2, dim=1)

    expected = torch.tensor(
        [
            [math.nan, math.sqrt(0.5), math.sqrt(0.5), math.sqrt(0.5)],
            [math.nan, math.sqrt(0.5), math.sqrt(0.5), math.sqrt(0.5)],
        ]
    )

    np.testing.assert_allclose(
        result[:, 1:].numpy(), expected[:, 1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:, 0]).all()


def test_mstd_with_zeros() -> None:
    """Test moving standard deviation with tensor containing zeros."""
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    result = QF.mstd(x, span=3, dim=0)

    expected = torch.tensor(
        [math.nan, math.nan, 0.0, math.sqrt(1.0 / 3.0), math.sqrt(1.0 / 3.0)]
    )

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_mstd_negative_dim() -> None:
    """Test moving standard deviation with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = QF.mstd(x, span=2, dim=-1)

    expected = torch.tensor(
        [
            [math.nan, math.sqrt(0.5), math.sqrt(0.5)],
            [math.nan, math.sqrt(0.5), math.sqrt(0.5)],
        ]
    )

    np.testing.assert_allclose(
        result[:, 1:].numpy(), expected[:, 1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:, 0]).all()


def test_mstd_span_equals_length() -> None:
    """Test moving standard deviation when span equals tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mstd(x, span=3, dim=0)

    expected = torch.tensor([math.nan, math.nan, 1.0])

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_mstd_large_span() -> None:
    """Test moving standard deviation when span is larger than tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0])
    result = QF.mstd(x, span=5, dim=0)

    expected = torch.tensor([math.nan, math.nan, math.nan])

    assert torch.isnan(result).all()


def test_mstd_single_element() -> None:
    """Test moving standard deviation with single element and span=1 (should be NaN due to ddof=1)."""
    x = torch.tensor([5.0])
    result = QF.mstd(x, span=1, dim=0)

    # With span=1 and ddof=1, the result should be NaN (not enough data points)
    assert torch.isnan(result[0])


def test_mstd_with_nan_values() -> None:
    """Test moving standard deviation behavior when input contains NaN values."""
    x = torch.tensor([1.0, math.nan, 3.0, 4.0, 5.0])
    result = QF.mstd(x, span=3, dim=0)

    # Results should be NaN when NaN is within the current window
    assert torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert torch.isnan(result[2])
    assert torch.isnan(result[3])
    # Position 4 has window [3.0, 4.0, 5.0] which doesn't contain NaN
    assert not torch.isnan(result[4])


def test_mstd_dtype_preservation() -> None:
    """Test that moving standard deviation preserves input tensor's dtype."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = QF.mstd(x, span=2, dim=0)
    assert result.dtype == torch.float64


def test_mstd_different_ddof_values() -> None:
    """Test moving standard deviation with various ddof values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Test ddof=0 (population std)
    result_ddof0 = QF.mstd(x, span=3, dim=0, ddof=0)

    # Test ddof=1 (sample std)
    result_ddof1 = QF.mstd(x, span=3, dim=0, ddof=1)

    # Test ddof=2
    result_ddof2 = QF.mstd(x, span=3, dim=0, ddof=2)

    # ddof=0 should give larger std than ddof=1, which should be larger than ddof=2
    # for the same data (when not NaN)
    assert result_ddof0[3] > result_ddof1[3] > result_ddof2[3]


def test_mstd_very_small_span() -> None:
    """Test moving standard deviation with span=1 (should be 0 or NaN depending on ddof)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # With ddof=0, span=1 should give 0
    result_ddof0 = QF.mstd(x, span=1, dim=0, ddof=0)
    expected_ddof0 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result_ddof0.numpy(), expected_ddof0.numpy())

    # With ddof=1, span=1 should give NaN (not enough degrees of freedom)
    result_ddof1 = QF.mstd(x, span=1, dim=0, ddof=1)
    assert torch.isnan(result_ddof1).all()


def test_mstd_identical_values() -> None:
    """Test moving standard deviation with identical values in window."""
    x = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
    result = QF.mstd(x, span=3, dim=0)

    # Standard deviation of identical values should be 0
    expected = torch.tensor([math.nan, math.nan, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(result[2:].numpy(), expected[2:].numpy())
    assert torch.isnan(result[:2]).all()


def test_mstd_alternating_pattern() -> None:
    """Test moving standard deviation with alternating pattern."""
    x = torch.tensor([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
    result = QF.mstd(x, span=2, dim=0)

    # Standard deviation of [1,3] should be sqrt(2)/sqrt(2-1) = sqrt(2)
    expected_std = math.sqrt(2.0)
    expected = torch.tensor(
        [
            math.nan,
            expected_std,
            expected_std,
            expected_std,
            expected_std,
            expected_std,
        ]
    )

    np.testing.assert_allclose(
        result[1:].numpy(), expected[1:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[0])


def test_mstd_large_span_relative_to_tensor() -> None:
    """Test moving standard deviation with span close to tensor length."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result = QF.mstd(x, span=4, dim=0)

    # Only position 4 should have a valid result
    assert torch.isnan(result[:4]).all()
    assert not torch.isnan(result[4])

    # Manual calculation for [2,3,4,5]
    manual_std = torch.std(torch.tensor([2.0, 3.0, 4.0, 5.0]), unbiased=True)
    np.testing.assert_allclose(result[4].numpy(), manual_std.numpy(), atol=1e-6)


def test_mstd_3d_tensor_different_axes() -> None:
    """Test moving standard deviation on 3D tensor along different axes."""
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )

    # Test along axis 0
    result_axis0 = QF.mstd(x, span=2, dim=0)
    assert result_axis0.shape == x.shape
    assert torch.isnan(result_axis0[0]).all()
    assert not torch.isnan(result_axis0[1:]).any()

    # Test along axis 2
    result_axis2 = QF.mstd(x, span=2, dim=2)
    assert result_axis2.shape == x.shape
    assert torch.isnan(result_axis2[:, :, 0]).all()
    assert not torch.isnan(result_axis2[:, :, 1]).any()


def test_mstd_floating_point_precision() -> None:
    """Test moving standard deviation with values requiring high precision."""
    x = torch.tensor(
        [1.0000001, 1.0000002, 1.0000003, 1.0000004], dtype=torch.float64
    )
    result = QF.mstd(x, span=3, dim=0, ddof=1)

    # Should be able to compute std of very small differences
    assert not torch.isnan(result[2:]).any()
    assert torch.all(result[2:] > 0)  # Should be positive


def test_mstd_mixed_positive_negative() -> None:
    """Test moving standard deviation with mixed positive and negative values."""
    x = torch.tensor([-5.0, -2.0, 1.0, 4.0, 7.0])
    result = QF.mstd(x, span=3, dim=0)

    expected = torch.tensor(
        [
            math.nan,
            math.nan,
            math.sqrt(
                (
                    (
                        (-5.0 - (-2.0)) ** 2
                        + ((-2.0) - (-2.0)) ** 2
                        + (1.0 - (-2.0)) ** 2
                    )
                    / 2
                )
            ),
            math.sqrt(
                (((-2.0 - 1.0) ** 2 + (1.0 - 1.0) ** 2 + (4.0 - 1.0) ** 2) / 2)
            ),
            math.sqrt(
                ((1.0 - 4.0) ** 2 + (4.0 - 4.0) ** 2 + (7.0 - 4.0) ** 2) / 2
            ),
        ]
    )

    np.testing.assert_allclose(
        result[2:].numpy(), expected[2:].numpy(), atol=1e-6
    )
    assert torch.isnan(result[:2]).all()


def test_mstd_outlier_handling() -> None:
    """Test moving standard deviation behavior with outliers."""
    x = torch.tensor([1.0, 1.0, 100.0, 1.0, 1.0])  # One large outlier
    result = QF.mstd(x, span=3, dim=0)

    # The window containing the outlier should have large std
    outlier_std = result[2]  # Window [1, 1, 100]
    normal_std = result[4]  # Window [1, 1, 1]

    assert outlier_std > normal_std
    assert normal_std == 0.0  # Identical values


def test_mstd_inf_values() -> None:
    """Test moving standard deviation behavior with infinite values."""
    x = torch.tensor([1.0, math.inf, 3.0, 4.0, 5.0])
    result = QF.mstd(x, span=3, dim=0)

    # Windows containing infinity should produce NaN or infinity
    assert torch.isnan(result[0])
    assert torch.isnan(result[1])
    assert torch.isnan(result[2]) or torch.isinf(result[2])
    assert torch.isnan(result[3]) or torch.isinf(result[3])
    assert not torch.isnan(result[4]) and not torch.isinf(result[4])


def test_mstd_very_large_values() -> None:
    """Test moving standard deviation with very large values."""
    x = torch.tensor([1e10, 2e10, 3e10, 4e10], dtype=torch.float64)
    result = QF.mstd(x, span=3, dim=0)

    # Should handle large values without overflow
    assert not torch.isnan(result[2:]).any()
    assert not torch.isinf(result[2:]).any()
    assert torch.all(result[2:] > 0)


def test_mstd_batch_consistency() -> None:
    """Test moving standard deviation consistency across multiple calls."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Multiple calls should give same result
    result1 = QF.mstd(x, span=3, dim=0)
    result2 = QF.mstd(x, span=3, dim=0)

    np.testing.assert_allclose(result1.numpy(), result2.numpy())

    # Different tensors with same values should give same result
    x2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result3 = QF.mstd(x2, span=3, dim=0)

    np.testing.assert_allclose(result1.numpy(), result3.numpy())
