import typing

import numpy as np
import torch

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


def eigh(
    tensor: torch.Tensor, uplo: str = "L"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix.

    This function computes the eigenvalues and eigenvectors of a real symmetric
    or complex Hermitian matrix. The eigenvalues are returned in ascending
    order, and the corresponding eigenvectors are normalized. This
    implementation uses NumPy for CPU tensors and CuPy for CUDA tensors.

    .. note::
        This function does not support automatic differentiation (autograd).

    Args:
        tensor (Tensor):
            A symmetric (if real) or Hermitian (if complex) matrix of shape
            ``(..., N, N)``. Only the upper or lower triangular part is used,
            depending on the :attr:`uplo` parameter.
        uplo (str, optional):
            Indicates which triangular part of the matrix is used:

            - 'L' or 'l': Use the lower triangular part (default)
            - 'U' or 'u': Use the upper triangular part

    Returns:
        Tuple[Tensor, Tensor]:
            A tuple containing:

            - Eigenvalues: A tensor of shape ``(..., N)`` containing
              eigenvalues in ascending order
            - Eigenvectors: A tensor of shape ``(..., N, N)`` where the columns
              are the normalized eigenvectors corresponding to the eigenvalues

    Example:
        >>> # Create a symmetric matrix
        >>> A = torch.tensor([[4.0, -2.0],
        ...                   [-2.0, 3.0]])
        >>> eigenvalues, eigenvectors = QF.eigh(A)
        >>> eigenvalues
        tensor([1.4384, 5.5616])

        >>> # Verify: A @ v = λ @ v
        >>> torch.allclose(A @ eigenvectors, eigenvectors @ torch.diag(eigenvalues))
        True

        >>> # Using upper triangular part
        >>> B = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [0.0, 4.0, 5.0],
        ...                   [0.0, 0.0, 6.0]])
        >>> eigenvalues, eigenvectors = QF.eigh(B, uplo='U')
    """

    def calculate() -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = tensor.cpu().numpy()
        if tensor.device.type == "cpu":
            w, v = np.linalg.eigh(x, uplo)  # type: ignore
        else:
            w, v = cp.linalg.eigh(cp.array(x), uplo)
            w = cp.asnumpy(w)
            v = cp.asnumpy(v)
        w = torch.tensor(w, device=tensor.device)
        v = torch.tensor(v, device=tensor.device)
        return w, v

    w, v = calculate()
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    return w, v
