import math
import typing

import torch

from .cumcount import cumcount


def groupby(
    x: torch.Tensor,
    group_id: torch.Tensor,
    dim: int = -1,
    empty_value: typing.Any = math.nan,
) -> torch.Tensor:
    r"""Groups items of `x` by `group_id`.

    This adds a new dimension just after the dimension specified by `dim`.
    The additional dimension represents items of the group.  Since a tensor
    requires every dimension to be a fixed size, this fills `empty_value` if
    the numbers of the items of groups are varying.
    """

    # 1. Resolve dimension and keep subsidiary dimensions.
    dim = dim + len(x.shape) if dim < 0 else dim
    shape_l, shape_r = x.shape[:dim], x.shape[dim + 1 :]

    # 2. Flatten subsidiary dimensions.
    x = x.transpose(0, dim)
    x = x.reshape(x.shape[:1] + (-1,))

    # 3. Calculate group shape.
    size = int(group_id.max() + 1)
    group_cumcount = cumcount(group_id)
    depth = int(group_cumcount.max() + 1)

    # 4. Scatter values.
    y_shape = (size * depth, x.shape[1])
    y = torch.full(y_shape, empty_value, dtype=x.dtype, device=x.device)
    indices = group_id * depth + group_cumcount
    y.scatter_(0, indices[:, None].expand(x.shape), x)

    # 5. Restore the shape.
    return y.transpose(0, dim).reshape(shape_l + (size, depth) + shape_r)
