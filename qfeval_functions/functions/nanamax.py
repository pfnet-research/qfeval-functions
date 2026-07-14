import math
import typing

import torch


def nanamax(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute the maximum of tensor elements along specified dimensions,
    ignoring NaN values.

    This function calculates the maximum of all valid (non-NaN) elements
    in a tensor along the specified dimension(s). Unlike :func:`nanmax`,
    this function supports multiple dimensions but does not return indices.
    When no valid elements are found along a dimension, the result is NaN.

    The NaN-aware maximum is computed as:

    .. math::
        \text{nanamax}(X) = \max_{i \text{ valid}} X_i

    where the maximum is over all valid (non-NaN) values.

    Args:
        x (Tensor):
            The input tensor containing values.
        dim (None, int, or tuple of ints, optional):
            The dimension(s) along which to compute the maximum. If None
            (default), the maximum is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.

    Returns:
        Tensor:
            The maximum values computed only over valid (non-NaN) values.
            When no valid values exist along a dimension, the result is NaN.
            The shape depends on the input dimensions, :attr:`dim`, and
            :attr:`keepdim` parameters.

    Example:

        >>> # Simple maximum with NaN values
        >>> x = torch.tensor([1.0, 2.0, nan, 4.0, 5.0])
        >>> QF.nanamax(x)
        tensor(5.)

        >>> # All NaN returns NaN
        >>> all_nan = torch.tensor([nan, nan, nan])
        >>> QF.nanamax(all_nan)
        tensor(nan)

        >>> # 2D tensor with max along columns
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanamax(x, dim=0)
        tensor([4., 5., 3.])

        >>> # Max along rows
        >>> QF.nanamax(x, dim=1)
        tensor([3., 5.])

        >>> # Multiple dimensions
        >>> x = torch.tensor([[[1.0, nan], [3.0, 4.0]],
        ...                   [[nan, 6.0], [7.0, nan]]])
        >>> QF.nanamax(x, dim=(1, 2))
        tensor([4., 7.])

        >>> # With keepdim
        >>> x = torch.tensor([[1.0, nan, 3.0],
        ...                   [4.0, 5.0, nan]])
        >>> QF.nanamax(x, dim=1, keepdim=True)
        tensor([[3.],
                [5.]])

        >>> # All NaN slice returns NaN
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [nan, nan]])
        >>> QF.nanamax(x, dim=1)
        tensor([2., nan])

        >>> # With negative infinity
        >>> x = torch.tensor([[1.0, -inf, 3.0],
        ...                   [nan, 2.0, -inf]])
        >>> QF.nanamax(x, dim=1)
        tensor([3., 2.])

    .. seealso::
        - :func:`nanmax`: NaN-aware maximum with indices (single dimension
          only).
        - :func:`nanamin`: NaN-aware minimum over multiple dimensions.
        - ``torch.amax``: Standard maximum over multiple dimensions (NaN
          propagates).
    """
    # 1. Handle empty tensor (amax raises RuntimeError for numel() == 0).
    if x.numel() == 0:
        raise RuntimeError(
            "nanamax(): Expected reduction dim to be specified for "
            "input.numel() == 0. Specify the reduction dim with the "
            "'dim' argument."
        )

    # 2. Check for all-NaN slices.
    is_invalid = x.isnan().all(dim=dim, keepdim=keepdim)

    # 3. Replace NaN -> -inf and compute amax.
    y = x.nan_to_num(-math.inf, math.inf, -math.inf).amax(
        dim=dim, keepdim=keepdim  # type: ignore[arg-type]
    )

    # 4. Restore NaN for all-NaN slices.
    return torch.where(is_invalid, torch.as_tensor(math.nan).to(y), y)
