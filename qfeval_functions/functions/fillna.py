import math

import torch


def fillna(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float = math.inf,
    neginf: float = -math.inf,
) -> torch.Tensor:
    r"""Replaces NaN, positive infinity, and negative infinity values with
    specified numbers.

    NOTE: Differently from torch.nan_to_num, this preserves infinity values as
    is.
    """
    return x.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
