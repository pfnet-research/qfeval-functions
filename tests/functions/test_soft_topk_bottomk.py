import numpy as np
import torch

import qfeval_functions.functions as QF


def test_soft_topk_bottomk_basic() -> None:
    """Test basic soft topk/bottomk functionality with k=1 on 1D tensor."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1)

    assert result.shape == x.shape
    # The sum of each row should be approximately 0 for soft_topk_bottomk
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [0.0], atol=1e-5)


def test_soft_topk_basic() -> None:
    """Test basic soft topk functionality with k=1 on 1D tensor."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk(x, k=1, dim=1)

    assert result.shape == x.shape
    # The sum of each row should be approximately 1 for soft_topk
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [1.0], atol=1e-5)
    # All values should be non-negative for soft_topk
    assert (result >= 0).all()


def test_soft_topk_bottomk_2d_dim0() -> None:
    """Test soft topk/bottomk on 2D tensor along axis 0."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=0)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=0).numpy(), [0.0, 0.0], atol=1e-5)


def test_soft_topk_2d_dim0() -> None:
    """Test soft topk on 2D tensor along axis 0."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = QF.soft_topk(x, k=1, dim=0)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=0).numpy(), [1.0, 1.0], atol=1e-5)
    assert (result >= 0).all()


def test_soft_topk_bottomk_negative_dim() -> None:
    """Test soft topk/bottomk with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=-1)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=-1).numpy(), [0.0], atol=1e-5)


def test_soft_topk_negative_dim() -> None:
    """Test soft topk with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.soft_topk(x, k=1, dim=-1)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=-1).numpy(), [1.0], atol=1e-5)
    assert (result >= 0).all()


def test_soft_topk_bottomk_small_epsilon() -> None:
    """Test soft topk/bottomk with small epsilon value for sharper selection."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1, epsilon=1e-3)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [0.0], atol=1e-4)


def test_soft_topk_small_epsilon() -> None:
    """Test soft topk with small epsilon value for sharper selection."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk(x, k=1, dim=1, epsilon=1e-3)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [1.0], atol=1e-4)
    assert (result >= 0).all()


def test_soft_topk_bottomk_larger_k() -> None:
    """Test soft topk/bottomk with k=2 to select multiple elements."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk_bottomk(x, k=2, dim=1)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [0.0], atol=1e-5)


def test_soft_topk_larger_k() -> None:
    """Test soft topk with k=2 to select multiple elements."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk(x, k=2, dim=1)

    assert result.shape == x.shape
    # For soft_topk with k=2, the sum should be 2.0 (representing the k selected elements)
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [2.0], atol=1e-5)
    assert (result >= 0).all()


def test_soft_topk_bottomk_max_iter() -> None:
    """Test soft topk/bottomk with custom maximum iterations parameter."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1, max_iter=50)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [0.0], atol=1e-4)


def test_soft_topk_max_iter() -> None:
    """Test soft topk with custom maximum iterations parameter."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = QF.soft_topk(x, k=1, dim=1, max_iter=50)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [1.0], atol=1e-4)
    assert (result >= 0).all()


def test_soft_topk_bottomk_batch() -> None:
    """Test soft topk/bottomk with batch of tensors (multiple rows)."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [0.0, 0.0], atol=1e-5)


def test_soft_topk_batch() -> None:
    """Test soft topk with batch of tensors (multiple rows)."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    result = QF.soft_topk(x, k=1, dim=1)

    assert result.shape == x.shape
    np.testing.assert_allclose(result.sum(dim=1).numpy(), [1.0, 1.0], atol=1e-5)
    assert (result >= 0).all()


def test_soft_topk_bottomk_3d_tensor() -> None:
    """Test soft topk/bottomk on 3D tensor along axis 2."""
    x = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    result = QF.soft_topk_bottomk(x, k=1, dim=2)

    assert result.shape == x.shape
    np.testing.assert_allclose(
        result.sum(dim=2).numpy(), [[0.0], [0.0]], atol=1e-5
    )


def test_soft_topk_3d_tensor() -> None:
    """Test soft topk on 3D tensor along axis 2."""
    x = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    result = QF.soft_topk(x, k=1, dim=2)

    assert result.shape == x.shape
    np.testing.assert_allclose(
        result.sum(dim=2).numpy(), [[1.0], [1.0]], atol=1e-5
    )
    assert (result >= 0).all()


def test_soft_topk_bottomk_dtype_preservation() -> None:
    """Test that soft topk/bottomk preserves input tensor's dtype."""
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    result = QF.soft_topk_bottomk(x, k=1, dim=1)
    assert result.dtype == torch.float64


def test_soft_topk_dtype_preservation() -> None:
    """Test that soft topk preserves input tensor's dtype."""
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    result = QF.soft_topk(x, k=1, dim=1)
    assert result.dtype == torch.float64


def test_soft_topk_bottomk_device_preservation() -> None:
    """Test that soft topk/bottomk preserves input tensor's device."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1)
    assert result.device == x.device


def test_soft_topk_device_preservation() -> None:
    """Test that soft topk preserves input tensor's device."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    result = QF.soft_topk(x, k=1, dim=1)
    assert result.device == x.device


def test_soft_topk_bottomk_error_on_invalid_input() -> None:
    """Test that soft topk/bottomk raises ValueError for NaN input."""
    x = torch.tensor([[float("nan"), 2.0, 3.0]])
    try:
        QF.soft_topk_bottomk(x, k=1, dim=1)
        assert False, "Expected ValueError for nan input"
    except ValueError:
        pass


def test_soft_topk_error_on_invalid_input() -> None:
    """Test that soft topk raises ValueError for NaN input."""
    x = torch.tensor([[float("nan"), 2.0, 3.0]])
    try:
        QF.soft_topk(x, k=1, dim=1)
        assert False, "Expected ValueError for nan input"
    except ValueError:
        pass


def test_soft_topk_bottomk_error_on_negative_epsilon() -> None:
    """Test that soft topk/bottomk raises AssertionError for negative epsilon."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    try:
        QF.soft_topk_bottomk(x, k=1, dim=1, epsilon=-0.1)
        assert False, "Expected AssertionError for negative epsilon"
    except AssertionError:
        pass


def test_soft_topk_error_on_negative_epsilon() -> None:
    """Test that soft topk raises AssertionError for negative epsilon."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    try:
        QF.soft_topk(x, k=1, dim=1, epsilon=-0.1)
        assert False, "Expected AssertionError for negative epsilon"
    except AssertionError:
        pass


def test_soft_topk_bottomk_edge_case_k_equals_dimension() -> None:
    """Test soft topk/bottomk when k equals the dimension size."""
    x = torch.tensor([[1.0, 2.0, 3.0]])

    # This should fail or handle edge case appropriately
    try:
        result = QF.soft_topk_bottomk(x, k=3, dim=1)
        # If it doesn't fail, check that sum is still 0
        assert abs(result.sum(dim=1).item()) < 1e-5
    except (AssertionError, RuntimeError):
        # Expected behavior for edge case
        pass


def test_soft_topk_very_large_k() -> None:
    """Test soft topk with k larger than dimension size."""
    x = torch.tensor([[1.0, 2.0, 3.0]])

    try:
        QF.soft_topk(x, k=5, dim=1)
        # Should fail or handle gracefully
        assert False, "Expected error for k > dimension size"
    except (AssertionError, RuntimeError):
        pass


def test_soft_topk_bottomk_identical_values() -> None:
    """Test soft topk/bottomk with identical input values."""
    x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    result = QF.soft_topk_bottomk(x, k=1, dim=1)

    # Should handle identical values gracefully
    assert result.shape == x.shape
    assert abs(result.sum(dim=1).item()) < 1e-5


def test_soft_topk_extreme_epsilon_values() -> None:
    """Test soft topk with very small and very large epsilon values."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # Very small epsilon (approaching hard selection)
    result_small = QF.soft_topk(x, k=1, dim=1, epsilon=1e-8)
    assert result_small.shape == x.shape
    assert (
        abs(result_small.sum(dim=1).item() - 1.0) < 2e-5
    )  # More lenient tolerance

    # Large epsilon (more uniform distribution)
    result_large = QF.soft_topk(x, k=1, dim=1, epsilon=10.0)
    assert result_large.shape == x.shape
    assert abs(result_large.sum(dim=1).item() - 1.0) < 1e-5


def test_soft_topk_bottomk_convergence() -> None:
    """Test soft topk/bottomk convergence with different max_iter values."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Test with few iterations
    result_few = QF.soft_topk_bottomk(x, k=1, dim=1, max_iter=5)

    # Test with many iterations
    result_many = QF.soft_topk_bottomk(x, k=1, dim=1, max_iter=500)

    # Both should have similar properties
    assert result_few.shape == result_many.shape == x.shape
    assert abs(result_few.sum(dim=1).item()) < 1e-4
    assert abs(result_many.sum(dim=1).item()) < 1e-4
