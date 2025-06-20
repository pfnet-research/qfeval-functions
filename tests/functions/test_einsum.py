import math

import numpy as np
import torch

import qfeval_functions.functions as QF


def test_einsum_basic_matrix_multiplication() -> None:
    """Test basic matrix multiplication with einsum."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    result = QF.einsum("ij,jk->ik", a, b)
    expected = torch.einsum("ij,jk->ik", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_batch_matrix_multiplication() -> None:
    """Test batch matrix multiplication."""
    a = QF.randn(5, 10, 20)
    b = QF.randn(5, 20, 15)

    result = QF.einsum("bij,bjk->bik", a, b)
    expected = torch.einsum("bij,bjk->bik", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_element_wise_multiplication() -> None:
    """Test element-wise multiplication with einsum."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    result = QF.einsum("i,i->i", a, b)
    expected = torch.einsum("i,i->i", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_dot_product() -> None:
    """Test dot product with einsum."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    result = QF.einsum("i,i->", a, b)
    expected = torch.einsum("i,i->", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_trace() -> None:
    """Test matrix trace calculation with einsum."""
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    result = QF.einsum("ii->", a)
    expected = torch.einsum("ii->", a)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_transpose() -> None:
    """Test matrix transpose with einsum."""
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = QF.einsum("ij->ji", a)
    expected = torch.einsum("ij->ji", a)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_sum_over_axis() -> None:
    """Test sum over specific axis with einsum."""
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Sum over rows
    result = QF.einsum("ij->j", a)
    expected = torch.einsum("ij->j", a)
    np.testing.assert_allclose(result.numpy(), expected.numpy())

    # Sum over columns
    result = QF.einsum("ij->i", a)
    expected = torch.einsum("ij->i", a)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_outer_product() -> None:
    """Test outer product with einsum."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0])

    result = QF.einsum("i,j->ij", a, b)
    expected = torch.einsum("i,j->ij", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_broadcasting() -> None:
    """Test broadcasting behavior with einsum."""
    a = QF.randn(10, 1, 30)
    b = QF.randn(1, 20, 30)

    result = QF.einsum("abc,adc->abd", a, b)
    expected = torch.einsum("abc,adc->abd", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_higher_dimensional() -> None:
    """Test einsum with higher dimensional tensors."""
    a = QF.randn(2, 3, 4, 5)
    b = QF.randn(2, 5, 6)

    result = QF.einsum("ijkl,ilm->ijkm", a, b)
    expected = torch.einsum("ijkl,ilm->ijkm", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_bilinear_form() -> None:
    """Test bilinear form computation with einsum."""
    a = torch.tensor([1.0, 2.0])
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    c = torch.tensor([1.0, 1.0])

    # Compute a^T B c
    result = QF.einsum("i,ij,j->", a, B, c)
    expected = torch.einsum("i,ij,j->", a, B, c)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_tensor_contraction() -> None:
    """Test tensor contraction with einsum."""
    a = QF.randn(3, 4, 5)
    b = QF.randn(4, 5, 6)

    result = QF.einsum("ijk,jkl->il", a, b)
    expected = torch.einsum("ijk,jkl->il", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_complex_pattern() -> None:
    """Test complex einsum pattern with multiple contractions."""
    a = QF.randn(2, 3, 4)
    b = QF.randn(3, 4, 5)
    c = QF.randn(5, 6)

    result = QF.einsum("ijk,jkl,lm->im", a, b, c)
    expected = torch.einsum("ijk,jkl,lm->im", a, b, c)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_repeated_indices() -> None:
    """Test einsum with repeated indices (diagonal operations)."""
    # Test diagonal extraction
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    result = QF.einsum("ii->i", a)
    expected = torch.einsum("ii->i", a)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_single_operand() -> None:
    """Test einsum with single operand operations."""
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Sum all elements
    result = QF.einsum("ij->", a)
    expected = torch.einsum("ij->", a)
    np.testing.assert_allclose(result.numpy(), expected.numpy())

    # Reshape operation
    result = QF.einsum("ij->ji", a)
    expected = torch.einsum("ij->ji", a)
    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_large_tensors() -> None:
    """Test einsum with large tensors for performance verification."""
    a = QF.randn(100, 200)
    b = QF.randn(200, 150)

    result = QF.einsum("ij,jk->ik", a, b)
    expected = torch.einsum("ij,jk->ik", a, b)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_einsum_numerical_precision() -> None:
    """Test einsum with values requiring high numerical precision."""
    a = torch.tensor(
        [[1.0000001, 1.0000002], [1.0000003, 1.0000004]], dtype=torch.float64
    )
    b = torch.tensor(
        [[1.0000001, 1.0000002], [1.0000003, 1.0000004]], dtype=torch.float64
    )

    result = QF.einsum("ij,jk->ik", a, b)
    expected = torch.einsum("ij,jk->ik", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-15, atol=1e-15)


def test_einsum_with_infinity() -> None:
    """Test einsum behavior with infinity values."""
    a = torch.tensor([[1.0, math.inf], [2.0, 3.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = QF.einsum("ij,jk->ik", a, b)
    expected = torch.einsum("ij,jk->ik", a, b)

    # Check that infinity is preserved appropriately
    assert torch.isinf(result).any() == torch.isinf(expected).any()

    # Check finite values
    finite_mask = torch.isfinite(result) & torch.isfinite(expected)
    if finite_mask.any():
        np.testing.assert_allclose(
            result[finite_mask].numpy(), expected[finite_mask].numpy()
        )


def test_einsum_with_nan() -> None:
    """Test einsum behavior with NaN values."""
    a = torch.tensor([[1.0, math.nan], [2.0, 3.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = QF.einsum("ij,jk->ik", a, b)
    expected = torch.einsum("ij,jk->ik", a, b)

    # Check that NaN propagation matches
    assert torch.isnan(result).sum() == torch.isnan(expected).sum()

    # Check finite values
    finite_mask = torch.isfinite(result) & torch.isfinite(expected)
    if finite_mask.any():
        np.testing.assert_allclose(
            result[finite_mask].numpy(), expected[finite_mask].numpy()
        )


def test_einsum_batch_outer_product() -> None:
    """Test batch outer product operations."""
    a = QF.randn(5, 3)
    b = QF.randn(5, 4)

    result = QF.einsum("bi,bj->bij", a, b)
    expected = torch.einsum("bi,bj->bij", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())


def test_einsum_advanced_broadcasting() -> None:
    """Test advanced broadcasting scenarios."""
    a = QF.randn(1, 5, 3)
    b = QF.randn(4, 1, 3)

    result = QF.einsum("aij,bij->abij", a, b)
    expected = torch.einsum("aij,bij->abij", a, b)

    np.testing.assert_allclose(result.numpy(), expected.numpy())
