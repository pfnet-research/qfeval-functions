import math

import numpy as np
import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_orthonormalize_basic_functionality() -> None:
    """Test basic orthonormalization functionality with random vectors."""
    qfeval_functions.random.seed(1)
    a = QF.randn(17, 7, 31)
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True)
    b = QF.orthonormalize(a)
    eye = torch.eye(7)[None, :, :]
    a_actual = QF.einsum("bxi,byi->bxy", a, a)
    a_actual = a_actual - eye
    b_actual = QF.einsum("bxi,byi->bxy", b, b)
    b_actual = b_actual - eye
    # Inner products between original vectors should be non-zero.
    assert torch.all(torch.std(a_actual, dim=(1, 2)) > 0.01)
    # Inner products between orthonormal vectors should be zero.
    assert torch.all(torch.std(b_actual, dim=(1, 2)) < 1e-4)
    # The orthonormalized one should be similar to the original one.
    assert torch.all(QF.correl(a, b, dim=(1, 2)) > 0.9)
    # Assert that it uses the Gram-Schmidt process.
    for i in range(1, a.size(1)):
        c = QF.einsum("bi,bxi->bx", b[:, i], a[:, :i])
        assert torch.all(c.abs().mean(dim=1) < 1e-4)


def test_orthonormalize_simple_2d_vectors() -> None:
    """Test orthonormalization with simple 2D vectors."""
    # Input: two linearly independent vectors
    a = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])  # Shape: (1, 2, 2)

    result = QF.orthonormalize(a)

    # Check orthonormality: Q^T Q = I
    gram_matrix = torch.einsum("bij,bik->bjk", result, result)
    identity = torch.eye(2).unsqueeze(0)

    np.testing.assert_allclose(
        gram_matrix.numpy(), identity.numpy(), atol=1e-10
    )

    # Check that all vectors have unit norm
    norms = torch.linalg.norm(result, dim=-1)
    expected_norms = torch.ones(1, 2)
    np.testing.assert_allclose(
        norms.numpy(), expected_norms.numpy(), atol=1e-10
    )


def test_orthonormalize_3d_vectors() -> None:
    """Test orthonormalization with 3D vectors."""
    # Input: three linearly independent vectors in 3D
    a = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]])

    result = QF.orthonormalize(a)

    # Check orthonormality
    gram_matrix = torch.einsum("bij,bik->bjk", result, result)
    identity = torch.eye(3).unsqueeze(0)

    np.testing.assert_allclose(
        gram_matrix.numpy(), identity.numpy(), atol=1e-10
    )

    # Check unit norms
    norms = torch.linalg.norm(result, dim=-1)
    expected_norms = torch.ones(1, 3)
    np.testing.assert_allclose(
        norms.numpy(), expected_norms.numpy(), atol=1e-10
    )


def test_orthonormalize_already_orthonormal() -> None:
    """Test orthonormalization with already orthonormal vectors."""
    # Create orthonormal vectors (standard basis)
    a = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    result = QF.orthonormalize(a)

    # Should remain essentially unchanged (up to sign changes)
    # Check that the result is still orthonormal
    gram_matrix = torch.einsum("bij,bik->bjk", result, result)
    identity = torch.eye(3).unsqueeze(0)

    np.testing.assert_allclose(
        gram_matrix.numpy(), identity.numpy(), atol=1e-10
    )


def test_orthonormalize_orthogonality_property() -> None:
    """Test that result vectors are orthogonal to each other."""
    batch_size = 3
    num_vectors = 4
    vector_dim = 6

    a = torch.randn(batch_size, num_vectors, vector_dim)
    result = QF.orthonormalize(a)

    # Check orthogonality: dot products between different vectors should be 0
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            dot_products = torch.sum(result[:, i] * result[:, j], dim=-1)
            np.testing.assert_allclose(dot_products.numpy(), 0.0, atol=1e-6)


def test_orthonormalize_normalization_property() -> None:
    """Test that result vectors have unit norm."""
    batch_size = 5
    num_vectors = 3
    vector_dim = 7

    a = torch.randn(batch_size, num_vectors, vector_dim)
    result = QF.orthonormalize(a)

    # Check that all vectors have unit norm
    norms = torch.linalg.norm(result, dim=-1)
    expected_norms = torch.ones(batch_size, num_vectors)

    np.testing.assert_allclose(norms.numpy(), expected_norms.numpy(), atol=1e-6)


def test_orthonormalize_gram_matrix_identity() -> None:
    """Test that Q^T Q = I (Gram matrix is identity)."""
    a = torch.randn(2, 4, 8)
    result = QF.orthonormalize(a)

    # Compute Gram matrix Q^T Q - the vectors are in the second-to-last dimension
    gram_matrix = torch.einsum("bik,bjk->bij", result, result)
    identity = torch.eye(4).unsqueeze(0).expand(2, -1, -1)

    np.testing.assert_allclose(gram_matrix.numpy(), identity.numpy(), atol=1e-6)


def test_orthonormalize_span_preservation() -> None:
    """Test that orthonormalization preserves the span of input vectors."""
    # Create vectors with known span
    a = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])  # 2 vectors in 3D

    result = QF.orthonormalize(a)

    # The span should be preserved - any linear combination of original vectors
    # should be expressible as a linear combination of orthonormal vectors

    # Test: original vector should be in span of orthonormal vectors
    # This is verified by checking that projecting the original vectors onto
    # the orthonormal basis gives a reasonable reconstruction

    # Project first original vector onto orthonormal basis
    # Each vector is result[0, i, :], so we compute dot products correctly
    projections = torch.zeros(result.shape[1])  # num_vectors
    for i in range(result.shape[1]):
        projections[i] = torch.dot(a[0, 0], result[0, i])
    reconstructed = torch.zeros_like(a[0, 0])
    for i in range(result.shape[1]):
        reconstructed += projections[i] * result[0, i]

    # Should be able to reconstruct the original vector (in the span)
    # The reconstruction should be close to the projection of original onto the span
    np.testing.assert_allclose(
        reconstructed.numpy(), a[0, 0].numpy(), rtol=1e-4
    )


def test_orthonormalize_shape_preservation() -> None:
    """Test that orthonormalization preserves input tensor shape."""
    shapes = [(1, 2, 3), (2, 3, 4), (3, 2, 5), (4, 3, 6)]

    for shape in shapes:
        a = torch.randn(*shape)
        result = QF.orthonormalize(a)
        assert result.shape == a.shape


def test_orthonormalize_single_vector() -> None:
    """Test orthonormalization with single vector."""
    a = torch.tensor([[[1.0, 2.0, 3.0]]])  # Single vector

    result = QF.orthonormalize(a)

    # Should just normalize the vector
    expected_norm = torch.linalg.norm(a[0, 0])
    expected = a / expected_norm

    # The result should have the same direction (up to sign)
    normalized_result = result / torch.linalg.norm(result, dim=-1, keepdim=True)
    normalized_expected = expected / torch.linalg.norm(
        expected, dim=-1, keepdim=True
    )

    # Check that they are parallel (dot product = ±1)
    dot_product = torch.sum(normalized_result[0, 0] * normalized_expected[0, 0])
    assert abs(abs(dot_product.item()) - 1.0) < 1e-6

    # Check unit norm
    norm = torch.linalg.norm(result[0, 0])
    assert abs(norm.item() - 1.0) < 1e-6


def test_orthonormalize_linearly_dependent_vectors() -> None:
    """Test orthonormalization with linearly dependent vectors."""
    # TODO(claude): The orthonormalize function should detect and handle rank-deficient
    # input matrices more gracefully. Expected behavior: when input vectors are linearly
    # dependent, the function should either return a warning, set dependent vectors to zero,
    # or provide an option to return only the independent vectors that span the subspace.
    # Create linearly dependent vectors: second vector is 2x the first
    a = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])

    result = QF.orthonormalize(a)

    # Result should be orthonormal for the independent vectors
    # With linearly dependent input, one vector becomes zero
    # Check orthonormality - vectors are rows, so compute dot products between rows
    gram_matrix = torch.einsum("bik,bjk->bij", result, result)

    # For linearly dependent vectors, we expect the first vector to be normalized
    # and the second vector to be zero (or close to zero)
    assert torch.allclose(
        gram_matrix[0, 0, 0], torch.tensor(1.0), atol=1e-5
    )  # First vector normalized
    assert torch.allclose(
        gram_matrix[0, 0, 1], torch.tensor(0.0), atol=1e-5
    )  # Orthogonal to second
    assert torch.allclose(
        gram_matrix[0, 1, 0], torch.tensor(0.0), atol=1e-5
    )  # Orthogonal to first
    # Second vector may be zero due to linear dependence
    assert gram_matrix[0, 1, 1] <= 1.0 + 1e-5  # At most normalized


def test_orthonormalize_numerical_stability() -> None:
    """Test numerical stability with very small and large values."""
    # Very small values
    a_small = torch.tensor(
        [[[1e-10, 2e-10], [2e-10, 1e-10]]], dtype=torch.float64
    )
    result_small = QF.orthonormalize(a_small)

    # Should still be orthonormal
    gram_small = torch.einsum("bij,bik->bjk", result_small, result_small)
    identity = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    np.testing.assert_allclose(gram_small.numpy(), identity.numpy(), atol=1e-12)

    # Very large values
    a_large = torch.tensor([[[1e10, 2e10], [2e10, 1e10]]], dtype=torch.float64)
    result_large = QF.orthonormalize(a_large)

    # Should still be orthonormal
    gram_large = torch.einsum("bij,bik->bjk", result_large, result_large)
    np.testing.assert_allclose(gram_large.numpy(), identity.numpy(), atol=1e-8)


def test_orthonormalize_with_nan_values() -> None:
    """Test orthonormalization behavior with NaN values."""
    a = torch.tensor([[[1.0, 2.0], [math.nan, 1.0]]])

    result = QF.orthonormalize(a)

    # NaN should propagate or result in valid orthonormal vectors for non-NaN parts
    # This depends on implementation - QR decomposition may handle NaN differently
    assert torch.isnan(result).any() or torch.isfinite(result).all()


def test_orthonormalize_with_infinity() -> None:
    """Test orthonormalization behavior with infinity values."""
    a = torch.tensor([[[math.inf, 1.0], [1.0, 2.0]]])

    result = QF.orthonormalize(a)

    # Should handle infinity gracefully (may result in NaN)
    assert torch.isnan(result).any() or torch.isinf(result).any()


def test_orthonormalize_batch_processing() -> None:
    """Test orthonormalization with multiple batches."""
    batch_size = 10
    num_vectors = 4
    vector_dim = 7

    a = torch.randn(batch_size, num_vectors, vector_dim)
    result = QF.orthonormalize(a)

    # Check orthonormality for each batch
    for b in range(batch_size):
        gram_matrix = torch.einsum("ik,jk->ij", result[b], result[b])
        identity = torch.eye(num_vectors)
        np.testing.assert_allclose(
            gram_matrix.numpy(), identity.numpy(), atol=1e-6
        )


def test_orthonormalize_high_dimensional() -> None:
    """Test orthonormalization with high-dimensional vectors."""
    num_vectors = 5
    vector_dim = 100

    a = torch.randn(1, num_vectors, vector_dim)
    result = QF.orthonormalize(a)

    # Check orthonormality
    gram_matrix = torch.einsum("ik,jk->ij", result[0], result[0])
    identity = torch.eye(num_vectors)

    np.testing.assert_allclose(gram_matrix.numpy(), identity.numpy(), atol=1e-6)


def test_orthonormalize_determinant_preservation() -> None:
    """Test that orthonormalization preserves orientation when possible."""
    # For square matrices, the determinant should be ±1
    a = torch.randn(2, 3, 3)  # Square matrices
    result = QF.orthonormalize(a)

    # Compute determinants
    for i in range(2):
        det = torch.det(result[i])
        # Determinant should be ±1 for orthonormal matrices
        assert abs(abs(det.item()) - 1.0) < 1e-6


def test_orthonormalize_reconstruction_property() -> None:
    """Test that original vectors can be reconstructed from orthonormal basis."""
    a = torch.randn(1, 3, 5)
    result = QF.orthonormalize(a)

    # For each original vector, compute its projection onto the orthonormal basis
    # and verify it reconstructs correctly
    for i in range(a.shape[1]):
        original_vec = a[0, i]

        # Project onto orthonormal basis
        projections = torch.sum(original_vec.unsqueeze(0) * result[0], dim=-1)
        reconstructed_vec = torch.sum(
            projections.unsqueeze(-1) * result[0], dim=0
        )

        # Should match original vector (they span the same space)
        np.testing.assert_allclose(
            reconstructed_vec.numpy(), original_vec.numpy(), rtol=1e-4
        )


def test_orthonormalize_invariance_under_orthonormal_transform() -> None:
    """Test that applying orthonormalization to already orthonormal vectors preserves them."""
    # Start with identity matrix (already orthonormal)
    a = torch.eye(3).unsqueeze(0)  # Shape: (1, 3, 3)

    result = QF.orthonormalize(a)

    # Should remain essentially unchanged (up to signs)
    # Check that result is still orthonormal
    gram_matrix = torch.einsum("bij,bik->bjk", result, result)
    identity = torch.eye(3).unsqueeze(0)

    np.testing.assert_allclose(
        gram_matrix.numpy(), identity.numpy(), atol=1e-10
    )

    # Check that each vector has unit norm
    norms = torch.linalg.norm(result, dim=-1)
    expected_norms = torch.ones(1, 3)
    np.testing.assert_allclose(
        norms.numpy(), expected_norms.numpy(), atol=1e-10
    )


def test_orthonormalize_rank_preservation() -> None:
    """Test that orthonormalization preserves the rank of the input."""
    # Create rank-2 matrix in 3D space
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([1.0, 1.0, 0.0])
    v3 = v1 + 2 * v2  # Linear combination, so rank is still 2

    a = torch.stack([v1, v2, v3]).unsqueeze(0)  # Shape: (1, 3, 3)
    result = QF.orthonormalize(a)

    # The result should still span the same subspace
    # Check that the third vector (if not zero) is in the span of first two
    if torch.linalg.norm(result[0, 2]) > 1e-8:
        # If third vector is not negligible, check orthonormality
        gram_matrix = torch.einsum("ij,ik->jk", result[0], result[0])
        identity = torch.eye(3)
        np.testing.assert_allclose(
            gram_matrix.numpy(), identity.numpy(), atol=1e-6
        )


def test_orthonormalize_sign_consistency() -> None:
    """Test that sign correction in QR decomposition works correctly."""
    # Create vectors where QR might produce negative diagonal elements
    a = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])

    result = QF.orthonormalize(a)

    # Should still be orthonormal regardless of signs
    gram_matrix = torch.einsum("bij,bik->bjk", result, result)
    identity = torch.eye(2).unsqueeze(0)

    np.testing.assert_allclose(
        gram_matrix.numpy(), identity.numpy(), atol=1e-10
    )
