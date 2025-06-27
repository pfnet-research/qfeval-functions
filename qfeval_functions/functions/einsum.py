import torch


def einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    r"""Sums the product of tensor elements over specified indices using
    Einstein notation.

    This function provides a typed wrapper around ``torch.einsum``, enabling
    better static type analysis. Einstein summation convention allows for
    expressing many tensor operations (including matrix multiplication, batch
    matrix multiplication, dot products, broadcasting, and more) in a compact
    notation.

    Args:
        equation (str):
            A string describing the subscripts for summation. The string
            contains comma-separated subscript labels for each operand,
            followed by ``->`` and the subscript labels for the output.
        *operands (Tensor):
            The input tensors to operate on. The number of operands must match
            the number of comma-separated groups in the equation.

    Returns:
        Tensor:
            The result of the Einstein summation, with shape determined by the
            output subscript labels in the equation.

    Example:

        >>> # Matrix multiplication: "ij,jk->ik"
        >>> A = torch.randn(3, 4)
        >>> B = torch.randn(4, 5)
        >>> C = QF.einsum("ij,jk->ik", A, B)
        >>> C.shape
        torch.Size([3, 5])

        >>> # Batch matrix multiplication: "bij,bjk->bik"
        >>> A = torch.randn(10, 3, 4)
        >>> B = torch.randn(10, 4, 5)
        >>> C = QF.einsum("bij,bjk->bik", A, B)
        >>> C.shape
        torch.Size([10, 3, 5])

        >>> # Trace of a matrix: "ii->"
        >>> A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> trace = QF.einsum("ii->", A)
        >>> trace
        tensor(5.)

        >>> # Transpose: "ij->ji"
        >>> A = torch.randn(3, 4)
        >>> A_T = QF.einsum("ij->ji", A)
        >>> torch.allclose(A_T, A.T)
        True
    """
    return torch.einsum(equation, *operands)  # type: ignore
