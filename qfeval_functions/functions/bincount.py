import torch

from .apply_for_axis import apply_for_axis


def bincount(
    x: torch.Tensor, minlength: int = 0, dim: int = -1
) -> torch.Tensor:
    r"""Computes the frequency of each value in a tensor along a specified
    dimension, with support for a minimum length of the output tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([[1, 2, 2], [3, 3, 1]])
        >>> bincount(x)
        tensor([[0, 1, 2, 0],
                [0, 1, 0, 2]])

        >>> x = torch.tensor([1, 2, 2, 3, 3, 1])
        >>> bincount(x)
        tensor([0, 2, 2, 2])
    """

    def _bincount(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            # handle edge case where x is empty
            # https://github.com/pytorch/pytorch/blob/5fb11cda4fe60c1a7b30e6c844f84ce8933ef953/torch/_numpy/_funcs_impl.py#L630
            return torch.zeros((1, minlength), dtype=torch.int64)
        n = max(minlength, int(torch.amax(x)) + 1)
        zeros = torch.zeros_like(x[:1, :1]).expand(x.shape[0], n)
        ones = torch.ones_like(x[:1, :1]).expand(x.shape)
        return torch.scatter_add(zeros, dim, x, ones)

    return apply_for_axis(_bincount, x, dim)
