import torch

from .apply_for_axis import apply_for_axis


def bincount(
    x: torch.Tensor, minlength: int = 0, dim: int = -1
) -> torch.Tensor:
    r"""Count number of occurrences of each value in a tensor.

    This function computes the frequency of each non-negative integer value
    in the input tensor along the specified dimension. The output tensor's
    size along the counted dimension is the maximum of :attr:`minlength` and
    the largest value in the input plus one. Each position in the output
    contains the count of that index value in the input.

    Args:
        x (Tensor):
            The input tensor containing non-negative integer values.
        minlength (int, optional):
            Minimum length of the output tensor along the counted dimension. If
            the maximum value in :attr:`x` is less than :attr:`minlength` - 1,
            the output is padded with zeros.
            Default is 0.
        dim (int, optional):
            The dimension along which to count values.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor where the value at index `i` along the specified dimension
            contains the count of occurrences of value `i` in the corresponding
            slice of the input tensor.

    Example:

        >>> x = torch.tensor([1, 2, 2, 3, 3, 1])
        >>> QF.bincount(x)
        tensor([0, 2, 2, 2])

        >>> x = torch.tensor([[1, 2, 2], [3, 3, 1]])
        >>> QF.bincount(x, dim=1)
        tensor([[0, 1, 2, 0],
                [0, 1, 0, 2]])

        >>> x = torch.tensor([1, 2])
        >>> QF.bincount(x, minlength=5)
        tensor([0, 1, 1, 0, 0])
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
