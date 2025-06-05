import math

import torch


def nancumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.where(
        x.isnan(),
        torch.as_tensor(math.nan, dtype=x.dtype, device=x.device),
        x.nan_to_num().cumsum(dim=dim),
    )
