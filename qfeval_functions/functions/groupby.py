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
    r"""Group tensor elements by group identifiers along a dimension.

    This function reorganizes elements of a tensor into groups based on
    provided group identifiers. It creates a new dimension that contains all
    elements belonging to each group. This is similar to SQL's GROUP BY or
    pandas' groupby operation, but adapted for tensor operations.

    The function adds a new dimension immediately after the specified
    dimension. This new dimension represents the items within each group.
    Since tensors require fixed-size dimensions, groups with fewer elements are
    padded with :attr:`empty_value`.

    Args:
        x (Tensor):
            The input tensor to be grouped.
        group_id (Tensor):
            A 1D tensor of integer group identifiers with the same
            length as ``x.shape[dim]``. Each element specifies which group the
            corresponding element in :attr:`x` belongs to. Group IDs should be
            non-negative integers.
        dim (int, optional):
            The dimension along which to group elements.
            Default is -1 (the last dimension).
        empty_value (scalar, optional):
            The value used to pad groups that have fewer elements than the
            maximum group size.
            Default is ``nan``.

    Returns:
        Tensor:
            A tensor with one additional dimension compared to the input.
            The shape is the same as :attr:`x` except at dimension :attr:`dim`,
            which is replaced by two dimensions:
            ``(num_groups, max_group_size)``. Elements are rearranged according
            to their group membership.

    Example:

        >>> # Group a 1D tensor
        >>> x = torch.tensor([10., 20., 30., 40., 50.])
        >>> group_id = torch.tensor([0, 1, 0, 1, 0])
        >>> grouped = QF.groupby(x, group_id)
        >>> grouped
        tensor([[10., 30., 50.],
                [20., 40., nan]])

        >>> # Group along a specific dimension of 2D tensor
        >>> x = torch.tensor([[1., 2., 3., 4.],
        ...                   [5., 6., 7., 8.]])
        >>> group_id = torch.tensor([0, 1, 0, 2])
        >>> grouped = QF.groupby(x, group_id, dim=1)
        >>> grouped
        tensor([[[1., 3.],
                 [2., nan],
                 [4., nan]],
        <BLANKLINE>
                [[5., 7.],
                 [6., nan],
                 [8., nan]]])

        >>> # Using custom empty value
        >>> x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int)
        >>> group_id = torch.tensor([0, 0, 1, 1, 1])
        >>> grouped = QF.groupby(x, group_id, empty_value=-1)
        >>> grouped
        tensor([[ 1,  2, -1],
                [ 3,  4,  5]], dtype=torch.int32)
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
