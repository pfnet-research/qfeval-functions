import torch


def _exponential_weighted_sum(
    x: torch.Tensor, alpha: float, dim: int = -1
) -> torch.Tensor:
    r"""This returns :math:`ews[i]=\sum_{j=0}^{i}v[j]*(1-\alpha)^(i-j)` over a
    given dimension `dim`.

    NOTE: This uses a nature of a geometrical progression:
    :math:`a[i]/a[i-d]==(1-\alpha)^d`.  In each iteration, this calculates a
    geometrical progression of `2^i` elements internally for each original
    element.
    """

    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)

    # 2. In i-th step, exponential weighted sum with the window size of
    # (2 ** i) is calculated.  This enables to calculate exponential weighted
    # sum in O(log n) where n is the length of the array.
    shift = 1
    decay = 1 - alpha
    while shift < len(x) and abs(decay) > 1e-8:
        x = torch.cat((x[:shift], x[shift:] + x[:-shift] * decay), dim=0)
        shift *= 2
        decay = decay**2

    # 3. Restore the order of dimensions.
    return x.transpose(0, dim)


def ema(x: torch.Tensor, alpha: float, dim: int = -1) -> torch.Tensor:
    r"""Returns the exponential weighted moving average of the given tensor."""
    ew_weight = _exponential_weighted_sum(
        torch.ones_like(x), alpha=alpha, dim=dim
    )
    ew_sum = _exponential_weighted_sum(x, alpha=alpha, dim=dim)
    return ew_sum / ew_weight
