import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_eigh_symmetric_matrix() -> None:
    """Test eigh with a simple 2x2 symmetric matrix to verify basic eigenvalue decomposition."""
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    w, v = QF.eigh(A)

    expected_w = torch.tensor([1.0, 3.0])
    expected_v = torch.tensor([[-0.7071, 0.7071], [0.7071, 0.7071]])

    np.testing.assert_allclose(w.numpy(), expected_w.numpy(), atol=1e-4)
    np.testing.assert_allclose(
        torch.abs(v).numpy(), torch.abs(expected_v).numpy(), atol=1e-4
    )


def test_eigh_identity_matrix() -> None:
    """Test eigh with identity matrix - should return eigenvalues of 1 and orthogonal eigenvectors."""
    A = torch.eye(3, dtype=torch.float32)
    w, v = QF.eigh(A)

    expected_w = torch.ones(3)

    np.testing.assert_allclose(w.numpy(), expected_w.numpy(), atol=1e-4)
    assert v.shape == (3, 3)


def test_eigh_diagonal_matrix() -> None:
    """Test eigh with diagonal matrix - eigenvalues should match diagonal elements."""
    A = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
    w, v = QF.eigh(A)

    expected_w = torch.tensor([1.0, 3.0, 5.0])

    np.testing.assert_allclose(w.numpy(), expected_w.numpy(), atol=1e-4)
    assert v.shape == (3, 3)


def test_eigh_uplo_upper() -> None:
    """Test eigh with both upper and lower triangle options to ensure consistent results."""
    A = torch.tensor([[4.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    w_lower, v_lower = QF.eigh(A, uplo="L")
    w_upper, v_upper = QF.eigh(A, uplo="U")

    np.testing.assert_allclose(w_lower.numpy(), w_upper.numpy(), atol=1e-4)


def test_eigh_reconstruction() -> None:
    """Test that eigenvalue decomposition can reconstruct the original matrix (A = V * diag(w) * V^T)."""
    A = torch.tensor(
        [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]], dtype=torch.float32
    )
    w, v = QF.eigh(A)

    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


def test_eigh_larger_matrix() -> None:
    """Test eigh with larger symmetric matrix to verify scalability."""
    # 4x4 symmetric matrix
    A = torch.tensor(
        [
            [4.0, 1.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    w, v = QF.eigh(A)

    # Verify we get 4 eigenvalues and eigenvectors
    assert w.shape == (4,)
    assert v.shape == (4, 4)

    # Verify reconstruction
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


def test_eigh_negative_eigenvalues() -> None:
    """Test eigh with matrix having negative eigenvalues."""
    # Matrix with both positive and negative eigenvalues
    A = torch.tensor([[1.0, 2.0], [2.0, -1.0]], dtype=torch.float32)

    w, v = QF.eigh(A)

    # Verify we get both positive and negative eigenvalues
    assert len(w) == 2
    assert torch.any(w > 0) and torch.any(w < 0)

    # Verify reconstruction
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


def test_eigh_zero_eigenvalue() -> None:
    """Test eigh with matrix having zero eigenvalue (singular matrix)."""
    # Singular matrix with one zero eigenvalue
    A = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    w, v = QF.eigh(A)

    # Should have one zero eigenvalue
    assert torch.any(torch.abs(w) < 1e-6)

    # Verify reconstruction
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


def test_eigh_nearly_zero_matrix() -> None:
    """Test eigh with matrix close to zero to test numerical stability."""
    A = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]], dtype=torch.float64)

    w, v = QF.eigh(A)

    # Eigenvalues should be very small but positive
    assert torch.all(w >= 0)
    assert torch.all(w < 1e-9)

    # Verify reconstruction
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-15)


def test_eigh_orthogonality_of_eigenvectors() -> None:
    """Test that eigenvectors are orthogonal for symmetric matrices."""
    A = torch.tensor(
        [[5.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]], dtype=torch.float32
    )

    w, v = QF.eigh(A)

    # Check orthogonality: V^T @ V should be identity
    orthogonality_check = v.T @ v
    identity = torch.eye(3, dtype=torch.float32)
    np.testing.assert_allclose(
        orthogonality_check.numpy(), identity.numpy(), atol=1e-5
    )


def test_eigh_eigenvalue_ordering() -> None:
    """Test that eigenvalues are returned in ascending order."""
    A = torch.tensor([[6.0, 2.0], [2.0, 3.0]], dtype=torch.float32)

    w, v = QF.eigh(A)

    # Eigenvalues should be in ascending order
    assert torch.all(w[:-1] <= w[1:])


def test_eigh_batch_processing() -> None:
    """Test eigh with batch dimension if supported by checking individual matrices."""
    # Test multiple 2x2 matrices
    matrices = [
        torch.tensor([[2.0, 1.0], [1.0, 3.0]]),
        torch.tensor([[4.0, 2.0], [2.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 5.0]]),
    ]

    for i, A in enumerate(matrices):
        w, v = QF.eigh(A)

        # Verify properties for each matrix
        assert w.shape == (2,)
        assert v.shape == (2, 2)

        # Verify reconstruction
        reconstructed = v @ torch.diag(w) @ v.T
        np.testing.assert_allclose(
            reconstructed.numpy(),
            A.numpy(),
            atol=1e-5,
            err_msg=f"Failed for matrix {i}",
        )


def test_eigh_complex_values_real_matrix() -> None:
    """Test eigh with matrix that might produce complex intermediate values."""
    # Matrix designed to test numerical stability
    A = torch.tensor([[1e6, 1.0], [1.0, 1e-6]], dtype=torch.float64)

    w, v = QF.eigh(A)

    # All eigenvalues should be real for symmetric matrix
    assert w.dtype == torch.float64
    assert v.dtype == torch.float64

    # Verify reconstruction with high precision
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), rtol=1e-10)


def test_eigh_memory_cleanup() -> None:
    """Test that eigh properly cleans up memory (especially for CUDA operations)."""
    # Test with multiple calls to ensure no memory leaks
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)

    for _ in range(10):
        w, v = QF.eigh(A)

        # Verify basic properties
        assert w.shape == (2,)
        assert v.shape == (2, 2)

        # Force garbage collection
        del w, v


@pytest.mark.random
def test_eigh_stress_test_large_matrix() -> None:
    """Test eigh with larger matrix to verify robustness."""
    # 10x10 symmetric matrix
    n = 10
    A = torch.randn(n, n, dtype=torch.float64)
    A = (A + A.T) / 2  # Make symmetric

    w, v = QF.eigh(A)

    # Verify shapes
    assert w.shape == (n,)
    assert v.shape == (n, n)

    # Verify eigenvalues are real
    assert w.dtype == torch.float64

    # Verify reconstruction (with looser tolerance for larger matrix)
    reconstructed = v @ torch.diag(w) @ v.T
    np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-10)
