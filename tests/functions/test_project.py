import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_project_basic_functionality() -> None:
    """Test basic projection functionality."""
    a = torch.tensor([[1, 1, 0], [1, 0, 1]])  # (2, 3) projection matrix
    x = torch.tensor([[1, 10, 200], [2, 30, 100], [3, 20, 300]])  # (3, 3) input

    result = QF.project(a, x)
    expected = torch.tensor([[11, 201], [32, 102], [23, 303]])

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_simple_2d() -> None:
    """Test projection with simple 2D matrices."""
    # Identity projection
    a_identity = torch.eye(3)  # (3, 3) identity
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    result = QF.project(a_identity, x)

    # Should be unchanged with identity projection
    np.testing.assert_allclose(result.numpy(), x.numpy())


def test_project_dimension_reduction() -> None:
    """Test projection that reduces dimensions."""
    # Project from 3D to 2D by taking first two components
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    result = QF.project(a, x)
    expected = torch.tensor([[1.0, 2.0], [4.0, 5.0]])  # First two components

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_dimension_expansion() -> None:
    """Test projection that expands dimensions."""
    # Project from 2D to 3D by adding zero component
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # (3, 2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    result = QF.project(a, x)
    expected = torch.tensor(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]
    )  # Add zero component

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_linear_combination() -> None:
    """Test projection as linear combination."""
    # Projection matrix that combines components
    a = torch.tensor([[1.0, 1.0], [2.0, -1.0]])  # (2, 2)
    x = torch.tensor([[3.0, 4.0], [1.0, 2.0]])  # (2, 2)

    result = QF.project(a, x)

    # Manual calculation:
    # First output: 1*3 + 1*4 = 7, 1*1 + 1*2 = 3
    # Second output: 2*3 + (-1)*4 = 2, 2*1 + (-1)*2 = 0
    expected = torch.tensor([[7.0, 2.0], [3.0, 0.0]])

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_batch_processing() -> None:
    """Test projection with batch dimensions."""
    batch_size = 3
    a = torch.randn(batch_size, 2, 4)  # (3, 2, 4) projection matrices
    x = torch.randn(batch_size, 5, 4)  # (3, 5, 4) input tensors

    result = QF.project(a, x)

    assert result.shape == (batch_size, 5, 2)

    # Verify each batch individually
    for i in range(batch_size):
        expected_i = torch.matmul(x[i], a[i].T)
        np.testing.assert_allclose(
            result[i].numpy(), expected_i.numpy(), rtol=1e-6
        )


def test_project_broadcasting() -> None:
    """Test projection with broadcasting."""
    # Single projection matrix, multiple input batches
    a = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])  # (2, 3)
    x = torch.randn(4, 5, 3)  # (4, 5, 3) - 4 batches

    result = QF.project(a, x)

    assert result.shape == (4, 5, 2)

    # Verify that same projection is applied to all batches
    for i in range(4):
        expected_i = torch.matmul(x[i], a.T)
        np.testing.assert_allclose(
            result[i].numpy(), expected_i.numpy(), rtol=1e-6
        )


def test_project_orthogonal_projection() -> None:
    """Test projection onto orthogonal subspace."""
    # Create orthogonal projection matrix for 2D subspace in 3D
    # Project onto xy-plane (zero z-component)
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    result = QF.project(a, x)

    # Should get only x and y components
    expected = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_matrix_multiplication_property() -> None:
    """Test that projection follows matrix multiplication properties."""
    a = torch.tensor([[2.0, 1.0], [1.0, 3.0]])  # (2, 2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    result = QF.project(a, x)

    # Should be equivalent to x @ a.T
    expected = torch.matmul(x, a.T)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_zero_matrix() -> None:
    """Test projection with zero matrix."""
    a = torch.zeros(2, 3)  # Zero projection matrix
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = QF.project(a, x)

    # Should result in zero tensor
    expected = torch.zeros(2, 2)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_large_tensors() -> None:
    """Test projection with large tensors for performance verification."""
    a = torch.randn(100, 50)  # (100, 50) projection matrix
    x = torch.randn(200, 50)  # (200, 50) input

    result = QF.project(a, x)

    assert result.shape == (200, 100)
    assert torch.isfinite(result).all()


def test_project_with_nan_values() -> None:
    """Test projection behavior with NaN values."""
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = torch.tensor([[1.0, math.nan]])

    result = QF.project(a, x)

    # NaN should propagate
    assert torch.isnan(result).any()


def test_project_with_infinity() -> None:
    """Test projection behavior with infinity values."""
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = torch.tensor([[math.inf, 1.0]])

    result = QF.project(a, x)

    # Infinity should propagate
    assert torch.isinf(result).any()


def test_project_numerical_precision() -> None:
    """Test projection with values requiring high numerical precision."""
    a = torch.tensor(
        [[1.0000001, 0.0000001], [0.0000001, 1.0000001]], dtype=torch.float64
    )
    x = torch.tensor([[1.0000002, 2.0000003]], dtype=torch.float64)

    result = QF.project(a, x)
    expected = torch.matmul(x, a.T)

    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-12)


def test_project_dimension_mismatch_error() -> None:
    """Test that dimension mismatch raises appropriate error."""
    a = torch.tensor([[1.0, 0.0]])  # (1, 2)
    x = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3) - mismatch in last dimension

    try:
        QF.project(a, x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "last dimension must match" in str(e).lower()


def test_project_very_small_values() -> None:
    """Test projection with very small values."""
    a = torch.tensor([[1e-10, 2e-10]], dtype=torch.float64)  # (1, 2)
    x = torch.tensor([[3e-10, 4e-10]], dtype=torch.float64)  # (1, 2)

    result = QF.project(a, x)
    expected = torch.matmul(x, a.T)

    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-10)


def test_project_very_large_values() -> None:
    """Test projection with very large values."""
    a = torch.tensor([[1e10, 2e10]], dtype=torch.float64)  # (1, 2)
    x = torch.tensor([[3e10, 4e10]], dtype=torch.float64)  # (1, 2)

    result = QF.project(a, x)
    expected = torch.matmul(x, a.T)

    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)


def test_project_negative_values() -> None:
    """Test projection with negative values."""
    a = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])  # (2, 2)
    x = torch.tensor([[-5.0, 6.0], [7.0, -8.0]])  # (2, 2)

    result = QF.project(a, x)
    expected = torch.matmul(x, a.T)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_project_complex_batch_shapes() -> None:
    """Test projection with complex batch shapes."""
    # Test 3D batch dimensions
    a = torch.randn(2, 3, 4, 5)  # (2, 3, 4, 5)
    x = torch.randn(2, 3, 6, 5)  # (2, 3, 6, 5)

    result = QF.project(a, x)

    assert result.shape == (2, 3, 6, 4)

    # Verify a specific batch element with relaxed tolerance
    expected_00 = torch.matmul(x[0, 0], a[0, 0].T)
    np.testing.assert_allclose(
        result[0, 0].numpy(), expected_00.numpy(), rtol=1e-4, atol=1e-4
    )


def test_project_linearity_property() -> None:
    """Test that projection is linear: project(a, cx + dy) = c*project(a,x) + d*project(a,y)."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    x = torch.tensor([[1.0, 0.0]])  # (1, 2)
    y = torch.tensor([[0.0, 1.0]])  # (1, 2)
    c, d = 2.0, 3.0

    # Left side: project(a, cx + dy)
    combined = c * x + d * y
    result_combined = QF.project(a, combined)

    # Right side: c*project(a,x) + d*project(a,y)
    result_x = QF.project(a, x)
    result_y = QF.project(a, y)
    result_linear = c * result_x + d * result_y

    np.testing.assert_allclose(
        result_combined.numpy(), result_linear.numpy(), rtol=1e-10
    )


def test_project_composition_property() -> None:
    """Test composition of projections: project(b, project(a, x)) = project(b@a, x)."""
    a = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])  # (2, 3)
    b = torch.tensor([[1.0, 1.0]])  # (1, 2)
    x = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)

    # Left side: project(b, project(a, x))
    intermediate = QF.project(a, x)
    result_composed = QF.project(b, intermediate)

    # Right side: project(b@a, x)
    combined_projection = torch.matmul(b, a)  # (1, 3)
    result_direct = QF.project(combined_projection, x)

    np.testing.assert_allclose(
        result_composed.numpy(), result_direct.numpy(), rtol=1e-10
    )
