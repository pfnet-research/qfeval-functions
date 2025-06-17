import math

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_orthogonalize_basic_functionality() -> None:
    """Test basic orthogonalization functionality with random vectors."""
    qfeval_functions.random.seed(1)
    for _ in range(10):
        x = QF.randn(10, 32)
        y = QF.randn(10, 32)
        orth_x = QF.orthogonalize(x, y)

        # Inner products between orthogonalized x and y should be zero (i.e.,
        # orthogonalized x and y should be orthogonal).
        orth_x_dot_y = (orth_x * y).sum(dim=-1)
        np.testing.assert_allclose(orth_x_dot_y, torch.zeros(10), atol=1e-5)

        # (x - orth_x) and y should be on the same axis.
        np.testing.assert_allclose(
            QF.correl(x - orth_x, y, dim=-1).abs(), torch.ones(10), atol=1e-5
        )


def test_orthogonalize_simple_2d_vectors() -> None:
    """Test orthogonalization with simple 2D vectors."""
    # Simple test case: orthogonalize [1, 1] with respect to [1, 0]
    x = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0]])

    result = QF.orthogonalize(x, y, dim=1)
    expected = torch.tensor([[0.0, 1.0]])  # Should be [0, 1]

    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-10)

    # Verify orthogonality
    dot_product = (result * y).sum(dim=1)
    np.testing.assert_allclose(dot_product.numpy(), 0.0, atol=1e-10)


def test_orthogonalize_3d_vectors() -> None:
    """Test orthogonalization with 3D vectors."""
    # Orthogonalize [1, 1, 1] with respect to [1, 0, 0]
    x = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])

    result = QF.orthogonalize(x, y, dim=1)
    expected = torch.tensor([[0.0, 1.0, 1.0]])  # Should be [0, 1, 1]

    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-10)

    # Verify orthogonality
    dot_product = (result * y).sum(dim=1)
    np.testing.assert_allclose(dot_product.numpy(), 0.0, atol=1e-10)


def test_orthogonalize_already_orthogonal() -> None:
    """Test orthogonalization when vectors are already orthogonal."""
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])

    result = QF.orthogonalize(x, y, dim=1)

    # Should remain unchanged since they're already orthogonal
    np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-10)


def test_orthogonalize_parallel_vectors() -> None:
    """Test orthogonalization when vectors are parallel."""
    # Parallel vectors
    x = torch.tensor([[2.0, 4.0]])
    y = torch.tensor([[1.0, 2.0]])  # y is parallel to x

    result = QF.orthogonalize(x, y, dim=1)

    # Result should be zero vector since x is in the span of y
    np.testing.assert_allclose(
        result.numpy(), torch.zeros(1, 2).numpy(), atol=1e-10
    )


def test_orthogonalize_antiparallel_vectors() -> None:
    """Test orthogonalization when vectors are antiparallel."""
    # Antiparallel vectors
    x = torch.tensor([[2.0, 4.0]])
    y = torch.tensor([[-1.0, -2.0]])  # y is antiparallel to x

    result = QF.orthogonalize(x, y, dim=1)

    # Result should be zero vector since x is in the span of y
    np.testing.assert_allclose(
        result.numpy(), torch.zeros(1, 2).numpy(), atol=1e-10
    )


def test_orthogonalize_different_dimensions() -> None:
    """Test orthogonalization along different dimensions."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    y = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

    # Test along dim=2 (last dimension)
    result_dim2 = QF.orthogonalize(x, y, dim=2)

    # Verify orthogonality
    dot_products = (result_dim2 * y).sum(dim=2)
    np.testing.assert_allclose(
        dot_products.numpy(), torch.zeros(2, 2).numpy(), atol=1e-10
    )

    # Test along dim=1
    x_1d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_1d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result_dim1 = QF.orthogonalize(x_1d, y_1d, dim=1)
    dot_products_1d = (result_dim1 * y_1d).sum(dim=1)
    np.testing.assert_allclose(
        dot_products_1d.numpy(), torch.zeros(2).numpy(), atol=1e-10
    )


def test_orthogonalize_negative_dim() -> None:
    """Test orthogonalization with negative dimension indexing."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result_neg = QF.orthogonalize(x, y, dim=-1)
    result_pos = QF.orthogonalize(x, y, dim=1)

    np.testing.assert_allclose(result_neg.numpy(), result_pos.numpy())


def test_orthogonalize_gram_schmidt_property() -> None:
    """Test that orthogonalization follows Gram-Schmidt process properties."""
    # Create a set of linearly independent vectors
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[1.0, 1.0, 0.0]])
    v3 = torch.tensor([[1.0, 1.0, 1.0]])

    # Orthogonalize v2 with respect to v1
    u2 = QF.orthogonalize(v2, v1, dim=1)

    # Orthogonalize v3 with respect to v1
    u3_step1 = QF.orthogonalize(v3, v1, dim=1)
    # Then orthogonalize with respect to u2
    u3 = QF.orthogonalize(u3_step1, u2, dim=1)

    # Check that u2 and v1 are orthogonal
    dot_u2_v1 = (u2 * v1).sum(dim=1)
    np.testing.assert_allclose(dot_u2_v1.numpy(), 0.0, atol=1e-10)

    # Check that u3 is orthogonal to both v1 and u2
    dot_u3_v1 = (u3 * v1).sum(dim=1)
    dot_u3_u2 = (u3 * u2).sum(dim=1)
    np.testing.assert_allclose(dot_u3_v1.numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(dot_u3_u2.numpy(), 0.0, atol=1e-10)


def test_orthogonalize_projection_formula() -> None:
    """Test that orthogonalization implements correct projection formula."""
    x = torch.tensor([[3.0, 4.0]])
    y = torch.tensor([[1.0, 0.0]])

    # Manual calculation: proj_y(x) = (x·y / y·y) * y
    dot_xy = (x * y).sum(dim=1, keepdim=True)
    dot_yy = (y * y).sum(dim=1, keepdim=True)
    projection = (dot_xy / dot_yy) * y
    expected_result = x - projection

    result = QF.orthogonalize(x, y, dim=1)

    np.testing.assert_allclose(
        result.numpy(), expected_result.numpy(), atol=1e-10
    )


def test_orthogonalize_with_zero_vector() -> None:
    """Test orthogonalization when one vector is zero."""
    # Zero x vector
    x_zero = torch.tensor([[0.0, 0.0]])
    y = torch.tensor([[1.0, 0.0]])
    result_zero_x = QF.orthogonalize(x_zero, y, dim=1)
    np.testing.assert_allclose(
        result_zero_x.numpy(), x_zero.numpy(), atol=1e-10
    )

    # Zero y vector (should cause division by zero, handle gracefully)
    x = torch.tensor([[1.0, 2.0]])
    y_zero = torch.tensor([[0.0, 0.0]])
    result_zero_y = QF.orthogonalize(x, y_zero, dim=1)
    # This should result in NaN or inf due to division by zero
    assert torch.isnan(result_zero_y).any() or torch.isinf(result_zero_y).any()


def test_orthogonalize_with_nan_values() -> None:
    """Test orthogonalization behavior with NaN values."""
    x = torch.tensor([[1.0, math.nan]])
    y = torch.tensor([[1.0, 0.0]])

    result = QF.orthogonalize(x, y, dim=1)

    # NaN should propagate
    assert torch.isnan(result).any()


def test_orthogonalize_with_infinity() -> None:
    """Test orthogonalization behavior with infinity values."""
    x = torch.tensor([[math.inf, 1.0]])
    y = torch.tensor([[1.0, 0.0]])

    result = QF.orthogonalize(x, y, dim=1)

    # Infinity should propagate or result in NaN
    assert torch.isinf(result).any() or torch.isnan(result).any()


def test_orthogonalize_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    x_small = torch.tensor([[1e-10, 2e-10]], dtype=torch.float64)
    y_small = torch.tensor([[1e-10, 0.0]], dtype=torch.float64)
    result_small = QF.orthogonalize(x_small, y_small, dim=1)

    # Should still maintain orthogonality
    dot_product_small = (result_small * y_small).sum(dim=1)
    assert abs(dot_product_small.item()) < 1e-15

    # Very large values
    x_large = torch.tensor([[1e10, 2e10]], dtype=torch.float64)
    y_large = torch.tensor([[1e10, 0.0]], dtype=torch.float64)
    result_large = QF.orthogonalize(x_large, y_large, dim=1)

    # Should still maintain orthogonality
    dot_product_large = (result_large * y_large).sum(dim=1)
    assert (
        abs(dot_product_large.item()) < 1e5
    )  # Allow some tolerance for large numbers


def test_orthogonalize_batch_processing() -> None:
    """Test orthogonalization with batch processing."""
    batch_size = 5
    vector_dim = 10

    x = torch.randn(batch_size, vector_dim)
    y = torch.randn(batch_size, vector_dim)

    result = QF.orthogonalize(x, y, dim=1)

    assert result.shape == x.shape

    # Check orthogonality for each batch
    dot_products = (result * y).sum(dim=1)
    np.testing.assert_allclose(
        dot_products.numpy(), torch.zeros(batch_size).numpy(), atol=1e-6
    )


def test_orthogonalize_high_dimensional() -> None:
    """Test orthogonalization with high-dimensional vectors."""
    dim = 1000
    x = torch.randn(1, dim)
    y = torch.randn(1, dim)

    result = QF.orthogonalize(x, y, dim=1)

    # Check orthogonality
    dot_product = (result * y).sum(dim=1)
    assert abs(dot_product.item()) < 1e-5


def test_orthogonalize_linearity_property() -> None:
    """Test linearity property: orthogonalize(ax + by, z) = a*orthogonalize(x,z) + b*orthogonalize(y,z)."""
    x1 = torch.tensor([[1.0, 2.0]])
    x2 = torch.tensor([[3.0, 1.0]])
    y = torch.tensor([[1.0, 0.0]])
    a, b = 2.0, 3.0

    # Left side: orthogonalize(ax1 + bx2, y)
    combined = a * x1 + b * x2
    result_combined = QF.orthogonalize(combined, y, dim=1)

    # Right side: a*orthogonalize(x1,y) + b*orthogonalize(x2,y)
    result_x1 = QF.orthogonalize(x1, y, dim=1)
    result_x2 = QF.orthogonalize(x2, y, dim=1)
    result_linear = a * result_x1 + b * result_x2

    np.testing.assert_allclose(
        result_combined.numpy(), result_linear.numpy(), atol=1e-10
    )


def test_orthogonalize_invariance_under_scaling() -> None:
    """Test that orthogonalization result direction is invariant under scaling of y."""
    x = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0]])

    result1 = QF.orthogonalize(x, y, dim=1)
    result2 = QF.orthogonalize(x, 2.0 * y, dim=1)
    result3 = QF.orthogonalize(x, 0.5 * y, dim=1)

    # Results should be identical (direction invariant under scaling)
    np.testing.assert_allclose(result1.numpy(), result2.numpy(), atol=1e-10)
    np.testing.assert_allclose(result1.numpy(), result3.numpy(), atol=1e-10)


def test_orthogonalize_idempotency() -> None:
    """Test that orthogonalizing an already orthogonalized vector gives the same result."""
    x = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])

    # First orthogonalization
    result1 = QF.orthogonalize(x, y, dim=1)

    # Second orthogonalization (should be idempotent)
    result2 = QF.orthogonalize(result1, y, dim=1)

    np.testing.assert_allclose(result1.numpy(), result2.numpy(), atol=1e-10)


def test_orthogonalize_complex_gram_schmidt() -> None:
    """Test multi-step Gram-Schmidt orthogonalization process."""
    # Create 4 linearly independent vectors in 4D space
    v1 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    v2 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    v3 = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    v4 = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

    # Gram-Schmidt process
    u1 = v1  # First vector unchanged
    u2 = QF.orthogonalize(v2, u1, dim=1)
    u3_temp = QF.orthogonalize(v3, u1, dim=1)
    u3 = QF.orthogonalize(u3_temp, u2, dim=1)
    u4_temp = QF.orthogonalize(v4, u1, dim=1)
    u4_temp2 = QF.orthogonalize(u4_temp, u2, dim=1)
    u4 = QF.orthogonalize(u4_temp2, u3, dim=1)

    # Check all pairs are orthogonal
    vectors = [u1, u2, u3, u4]
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dot_product = (vectors[i] * vectors[j]).sum(dim=1)
            assert (
                abs(dot_product.item()) < 1e-7
            ), f"Vectors {i} and {j} are not orthogonal"


def test_orthogonalize_span_preservation() -> None:
    """Test that orthogonalization preserves the span of the vector space."""
    # Original vector
    x = torch.tensor([[3.0, 4.0]])
    y = torch.tensor([[1.0, 0.0]])

    # Orthogonalized vector
    orth_x = QF.orthogonalize(x, y, dim=1)

    # The projection of x onto y
    dot_xy = (x * y).sum(dim=1, keepdim=True)
    dot_yy = (y * y).sum(dim=1, keepdim=True)
    proj_x_onto_y = (dot_xy / dot_yy) * y

    # x should equal orth_x + proj_x_onto_y (decomposition)
    reconstructed = orth_x + proj_x_onto_y
    np.testing.assert_allclose(reconstructed.numpy(), x.numpy(), atol=1e-10)
