import typing

import torch


def rms(
    x: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the root mean square of each row of the input tensor in the
    given dimension `dim`.  If `dim` is a list of dimensions, reduce over all
    of them.

    Args:
        x (Tensor): The input tensor.
        dim (int or tuple of ints): The dimension or dimensions to reduce.
        keepdim (bool): Whether the output tensor has `dim` retained or not.

    Returns:
        y (Tensor): The output tensor.
    """
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()
