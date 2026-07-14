import typing

import torch

from .nanamax import nanamax


def nanamin(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the minimum of tensor elements along specified dimensions,
    ignoring NaN values.

    This function calculates the minimum of all valid (non-NaN) elements
    in a tensor along the specified dimension(s). Unlike :func:`nanmin`,
    this function supports multiple dimensions but does not return indices.
    When no valid elements are found along a dimension, the result is NaN.

    The NaN-aware minimum is computed as:

    .. math::
        \text{nanamin}(X) = \min_{i \text{ valid}} X_i

    where the minimum is over all valid (non-NaN) values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the minimum. If None
            (default), the minimum is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The minimum values computed only over valid (non-NaN) values.
            When no valid values exist along a dimension, the result is NaN.
            The shape depends on the input dimensions, :attr:`dim`, and
            :attr:`keepdim` parameters.

    Example:

        >>> # Simple minimum with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanamin(x)
        tensor(1.)

        >>> # All NaN returns NaN
        >>> all_nan = torch.tensor([nan, nan, nan])
        >>> QF.nanamin(all_nan)
        tensor(nan)

        >>> # 2D tensor with min along columns
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanamin(x, dim=0)
        tensor([1., 5., 3.])

        >>> # Min along rows
        >>> QF.nanamin(x, dim=1)
        tensor([1., 4.])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, nan], [3.0, 4.0]],
        ...                   [[nan, 6.0], [7.0, nan]]])
        >>> QF.nanamin(x, dim=(1, 2))
        tensor([1., 6.])

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanamin(x, dim=1, keepdim=True)
        tensor([[1.],
                [4.]])

        >>> # All NaN slice returns NaN
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nanamin(x, dim=1)
        tensor([1., nan])

        >>> # With positive infinity
        >>> x = torch.tensor([[1.0, inf, 3.0],
        ...                   [nan, 2.0, inf]])
        >>> QF.nanamin(x, dim=1)
        tensor([1., 2.])

    .. seealso::
        - :func:`nanmin`: NaN-aware minimum with indices (single dimension
          only).
        - :func:`nanamax`: NaN-aware maximum over multiple dimensions.
        - ``torch.amin``: Standard minimum over multiple dimensions (NaN
          propagates).
    """
    # Delegate to nanamax via sign negation: min(x) == -max(-x).
    return -nanamax(-x, dim=dim, keepdim=keepdim)
