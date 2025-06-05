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
    """Computes eigenvalues and eigenvectors.

    NOTE: This does not support differentiable computation.
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
