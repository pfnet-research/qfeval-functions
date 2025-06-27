import math

import torch


def nanones(like: torch.Tensor) -> torch.Tensor:
    r"""Create a tensor filled with ones, preserving NaN positions from the input tensor.

    This function creates a new tensor with the same shape, dtype, and device as
    the input tensor, filled with ones. However, positions where the input tensor
    contains NaN values are preserved as NaN in the output tensor. This is useful
    for creating mask-like tensors that maintain the NaN structure of the original
    data while providing ones for valid positions.

    The function performs the following operation:

    .. math::
        \text{nanones}(X)_i = \begin{cases}
        \text{NaN} & \text{if } X_i = \text{NaN} \\
        1 & \text{otherwise}
        \end{cases}

    Args:
        like (Tensor):
            The input tensor whose shape, dtype, and device will be
            used for the output tensor. NaN positions in this tensor will be
            preserved in the output.

    Returns:
        Tensor:
            A tensor with the same shape, dtype, and device as :attr:`like`,
            filled with ones except for positions where :attr:`like` contains
            NaN values, which remain NaN.

    Example:

        >>> # Simple 1D tensor with NaN
        >>> x = torch.tensor([1.0, nan, 3.0, 4.0])
        >>> QF.nanones(x)
        tensor([1., nan, 1., 1.])

        >>> # 2D tensor with mixed NaN positions
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [nan, 5.0, 6.0]])
        >>> QF.nanones(x)
        tensor([[1., nan, 1.],
                [nan, 1., 1.]])

        >>> # All NaN tensor
        >>> x = torch.tensor([nan, nan, nan])
        >>> QF.nanones(x)
        tensor([nan, nan, nan])

        >>> # No NaN tensor
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> QF.nanones(x)
        tensor([1., 1., 1.])

        >>> # Different dtypes preserved
        >>> x = torch.tensor([1.0, nan, 3.0], dtype=torch.float32)
        >>> result = QF.nanones(x)
        >>> result.dtype
        torch.float32

    .. seealso::
        :func:`nanzeros`: Create zeros tensor preserving NaN positions.
        ``torch.ones_like``: Create ones tensor with same properties.
        ``torch.where``: Conditional tensor selection.
    """
    return torch.where(
        like.isnan(), torch.as_tensor(math.nan).to(like), torch.ones_like(like)
    )
