import typing

import numpy as np
import torch


def apply_for_axis(
    f: typing.Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Applies the given data `x` into the given function `f`, which expects a
    tensor of `(batch, n)` shape.

    This flattens the unspecified dimensions of `x` and unflattens the result
    of the given function `f`.  This is useful to implement a function
    manipulating a tensor along an axis without thinking of the number of
    dimensions.
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
