import torch


def rcumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Returns the reversely cumulative sum of elements of `input` in the
    dimension `dim`.
    """
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])
