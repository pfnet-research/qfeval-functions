import math
import typing

import torch


class NanmaxResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def nanmax(x: torch.Tensor, dim: int, keepdim: bool = False) -> NanmaxResult:
    r"""Return the maximum values and indices along a dimension, ignoring NaN values.

    This function computes the maximum value along the specified dimension,
    excluding NaN values from consideration. It returns both the maximum values
    and their corresponding indices in the original tensor. This is similar to
    ``torch.max`` but with NaN-aware behavior.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int):
            The dimension along which to find the maximum values.
        keepdim (bool, optional):
            Whether the output tensors have :attr:`dim`
            retained or not. Default is False.

    Returns:
        NanmaxResult: A named tuple containing:

            - ``values`` (Tensor): The maximum values along the specified dimension,
              with NaN values ignored. If a slice contains only NaN values,
              the result is NaN.
            - ``indices`` (Tensor): The indices of the maximum values in the
              original tensor.

    Example:

        >>> # Simple maximum with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> result = QF.nanmax(x, dim=1)
        >>> result.values
        tensor([3., 5.])
        >>> result.indices
        tensor([2, 1])

        >>> # All NaN slice
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> result = QF.nanmax(x, dim=1)
        >>> result.values
        tensor([2., nan])
        >>> result.indices
        tensor([1, 0])

        >>> # With negative infinity
        >>> x = torch.tensor([[1.0, -inf, 3.0],
        ...                   [nan, 2.0, -inf]])
        >>> result = QF.nanmax(x, dim=1)
        >>> result.values
        tensor([3., 2.])
        >>> result.indices
        tensor([2, 1])

        >>> # With keepdim
        >>> x = torch.tensor([[nan, 2.0, 1.0],
        ...                   [3.0, nan, 4.0]])
        >>> result = QF.nanmax(x, dim=1, keepdim=True)
        >>> result.values
        tensor([[2.],
                [4.]])
        >>> result.indices
        tensor([[1],
                [2]])

    .. seealso::
        :func:`nanmin`: NaN-aware minimum function.
        :func:`nanargmax`: NaN-aware argument maximum function.
        ``torch.max``: Standard maximum function (NaN propagates).
    """

    # 1. Replace NaN -> -inf and name it `a`.
    a = x.nan_to_num(-math.inf, math.inf, -math.inf)

    # 2. Apply `a` to torch.max.
    a_v, a_idx = a.max(dim=dim, keepdim=keepdim)
    # If x has no negative inf values, no conflicts of -inf should happen and
    # the result can be computed by restoring NaN from -inf.
    if not x.isneginf().any():
        return NanmaxResult(a_v.nan_to_num(0, math.inf, math.nan), a_idx)

    # 3. Create b representing -1 == NaN, 0 == -inf, 1 == others.  This enables
    # `(b * -inf)` to have NaN and -inf like `a` has and inf for the others.
    b = (x.detach() * math.inf).nan_to_num(0, 2, 1)
    b_v, b_idx = b.max(dim=dim, keepdim=keepdim)

    # 4. Build the final result.
    use_b = a_v.isneginf()
    return NanmaxResult(
        torch.where(use_b, b_v * -math.inf, a_v),
        torch.where(use_b, b_idx, a_idx),
    )
