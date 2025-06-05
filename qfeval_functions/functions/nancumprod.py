import math

import torch


def nancumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.where(
        x.isnan(),
        torch.as_tensor(math.nan, dtype=x.dtype, device=x.device),
        x.nan_to_num(1.0).cumprod(dim=dim),
    )
