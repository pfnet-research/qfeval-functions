import typing

import numpy as np
import torch


def apply_for_axis(
    f: typing.Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    r"""Apply a function expecting 2D input to a tensor along a specified
    dimension.

    This utility function allows applying functions that expect 2D tensors of
    shape ``(batch, n)`` to tensors of arbitrary dimensions. It handles the
    reshaping by flattening all dimensions except the specified one, applying
    the function, and then restoring the original shape. This is particularly
    useful for implementing dimension-aware operations without explicitly
    handling different tensor shapes.

    Args:
        f (Callable[[Tensor], Tensor]):
            A function that expects a 2D tensor of shape ``(batch, n)`` as
            input and returns a tensor where the batch dimension is preserved.
            The function must be dimension-preserving along the batch dimension.
        x (Tensor):
            The input tensor of arbitrary dimensions to process.
        dim (int, optional):
            The dimension along which to apply the function.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            The result of applying function :attr:`f` along the specified
            dimension, with the same shape as the input tensor.

    Example:

        >>> # Apply to a 3D tensor along dimension 1
        >>> x = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]],
        ...                   [[7., 8.], [9., 10.], [11., 12.]]])
        >>> QF.apply_for_axis(lambda x: x.cumsum(dim=1), x, dim=1)
        tensor([[[ 1.,  2.],
                 [ 4.,  6.],
                 [ 9., 12.]],
        <BLANKLINE>
                [[ 7.,  8.],
                 [16., 18.],
                 [27., 30.]]])

        >>> # Simple 2D example: apply function along columns (dim=0)
        >>> x = torch.tensor([[1., 2., 3.],
        ...                   [4., 5., 6.]])
        >>> def normalize_columns(x):
        ...     # Normalize each column to sum to 1
        ...     return x / x.sum(dim=0, keepdim=True)
        >>> QF.apply_for_axis(normalize_columns, x, dim=0)
        tensor([[0.1667, 0.3333, 0.5000],
                [0.2667, 0.3333, 0.4000]])

    .. note::
        The function :attr:`f` must preserve the batch dimension size. Functions
        that change the batch dimension size will cause shape mismatch errors
        during the reshape operation.
    """

    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    # NOTE: This does not use -1 intentionally because it fails if the given
    # tensor has one or more 0-length axes.
    x = x.reshape((shape[0], int(np.prod(shape[1:]))))

    # 2. Apply the given function.
    x = f(x.t()).t()

    # 3. Restore the shape and the order of dimensions.
    return x.reshape(x.shape[:1] + shape[1:]).transpose(0, dim)
