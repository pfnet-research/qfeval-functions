import torch

from .shift import shift as _shift


def nanshift(
    x: torch.Tensor,
    shift: int = 1,
    dim: int = -1,
) -> torch.Tensor:
    r"""Shift tensor elements along a dimension while preserving NaN positions.

    This function shifts the valid (non-NaN) elements of a tensor along the
    specified dimension while keeping NaN values in their original positions.
    Unlike standard shifting operations, NaN values act as "immovable" elements
    that do not participate in the shifting process, allowing only valid data
    to be shifted around them.

    This is particularly useful for time series data where missing values
    (represented as NaN) should remain in their temporal positions while
    valid observations are shifted for analysis purposes.

    Args:
        x (Tensor):
            The input tensor to be shifted.
        shift (int, optional):
            Number of positions to shift. Positive values
            shift towards higher indices, negative values shift towards lower
            indices. Default is 1.
        dim (int, optional):
            The dimension along which to perform the shift.
            Default is -1 (last dimension).

    Returns:
        Tensor:
            A tensor with the same shape as the input, where valid elements
            have been shifted along the specified dimension while NaN positions
            remain unchanged.

    Example:

        >>> # Simple 1D shift with NaN values
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0, nan])
        >>> QF.nanshift(x, shift=1)
        tensor([nan, nan, 1., 3., nan])

        >>> # Negative shift
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0, nan])
        >>> QF.nanshift(x, shift=-1)
        tensor([3., nan, 4., nan, nan])

        >>> # 2D tensor shift along rows
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan],
        ...                   [nan, 8.0, 9.0]])
        >>> QF.nanshift(x, shift=1, dim=0)
        tensor([[nan, nan, nan],
                [1., nan, nan],
                [nan, 5., 3.]])

        >>> # 2D tensor shift along columns
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanshift(x, shift=1, dim=1)
        tensor([[nan, nan, 1.],
                [nan, 4., nan]])

        >>> # Large shift (wraps around valid elements)
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0, nan])
        >>> QF.nanshift(x, shift=2)
        tensor([nan, nan, nan, 1., nan])

        >>> # All NaN tensor (no change)
        >>> x = torch.tensor([nan, nan, nan])
        >>> QF.nanshift(x, shift=1)
        tensor([nan, nan, nan])

    .. warning::
        The shift operation wraps around the valid elements. For example, if
        there are 3 valid elements and ``shift=1``, the last valid element
        becomes the first, and all others shift by one position.

    .. seealso::
        :func:`shift`: Standard shift function without NaN handling.
        :func:`group_shift`: Shift operation within groups.
        ``torch.roll``: PyTorch's standard tensor rolling function.
    """

    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape((shape[0], -1))

    # 2. Build a mapping to gather/scatter valid values from/to a sparse tensor.
    priority = torch.arange(shape[0], device=x.device)[:, None]
    priority = priority + x.isnan().int() * shape[0]
    index = torch.argsort(priority, dim=0)
    reversed_index = torch.argsort(index, dim=0)

    # 3. Gather valid values, apply shift, and scatter them.
    y = x[index, torch.arange(x.shape[1])[None, :]]
    y = torch.where(y.isnan(), torch.as_tensor(y).to(y), _shift(y, shift, 0))
    x = y[reversed_index, torch.arange(y.shape[1])[None, :]]

    # 4. Restore the shape and the order of dimensions.
    return x.reshape(shape).transpose(0, dim)
