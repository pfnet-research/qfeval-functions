import math

import torch


def nanones(like: torch.Tensor) -> torch.Tensor:
    r"""Returns a tensor having ones with the same shape of the given tensor,
    preserving Not a Numbers (NaNs).
    """
    return torch.where(
        like.isnan(), torch.as_tensor(math.nan).to(like), torch.ones_like(like)
    )
