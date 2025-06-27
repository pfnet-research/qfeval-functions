import typing

import torch

from .nanmax import nanmax


class NanminResult(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def nanmin(x: torch.Tensor, dim: int, keepdim: bool = False) -> NanminResult:
    r"""Return the minimum values and indices along a dimension, ignoring NaN values.

    This function computes the minimum value along the specified dimension,
    excluding NaN values from consideration. It returns both the minimum values
    and their corresponding indices in the original tensor. This is similar to
    ``torch.min`` but with NaN-aware behavior.

    The function handles edge cases including:

    - Pure NaN slices (returns NaN for values)
    - Mixed NaN and positive infinity values
    - All positive infinity values (returns positive infinity)

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (int):
            The dimension along which to find the minimum values.
        keepdim (bool, optional):
            Whether the output tensors have :attr:`dim`
            retained or not. Default is False.

    Returns:
        NanminResult: A named tuple containing:

            - ``values`` (Tensor): The minimum values along the specified dimension,
              with NaN values ignored. If a slice contains only NaN values,
              the result is NaN.
            - ``indices`` (Tensor): The indices of the minimum values in the
              original tensor.

    Example:

        >>> # Simple minimum with NaN values
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> result = QF.nanmin(x, dim=1)
        >>> result.values
        tensor([1., 4.])
        >>> result.indices
        tensor([0, 0])

        >>> # All NaN slice
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> result = QF.nanmin(x, dim=1)
        >>> result.values
        tensor([1., nan])
        >>> result.indices
        tensor([0, 0])

        >>> # With positive infinity
        >>> x = torch.tensor([[1.0, inf, 3.0],
        ...                   [nan, 2.0, inf]])
        >>> result = QF.nanmin(x, dim=1)
        >>> result.values
        tensor([1., 2.])
        >>> result.indices
        tensor([0, 1])

        >>> # With keepdim
        >>> x = torch.tensor([[nan, 2.0, 1.0],
        ...                   [3.0, nan, 4.0]])
        >>> result = QF.nanmin(x, dim=1, keepdim=True)
        >>> result.values
        tensor([[1.],
                [3.]])
        >>> result.indices
        tensor([[2],
                [0]])

    .. seealso::
        - :func:`nanmax`: NaN-aware maximum function.
        - :func:`nanargmin`: NaN-aware argument minimum function.
        - :func:`torch.min`: Standard minimum function (NaN propagates).
    """

    v, idx = nanmax(-x, dim=dim, keepdim=keepdim)
    return NanminResult(-v, idx)
