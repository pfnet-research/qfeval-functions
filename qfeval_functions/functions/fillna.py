import math

import torch


def fillna(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float = math.inf,
    neginf: float = -math.inf,
) -> torch.Tensor:
    r"""Replace NaN and infinity values with specified numbers.

    This function replaces NaN (Not a Number), positive infinity, and negative
    infinity values in a tensor with user-specified values. By default, NaN
    values are replaced with 0, while infinity values are preserved. This
    behavior differs from ``torch.nan_to_num``, which replaces infinities with
    the largest/smallest representable finite values by default.

    Args:
        x (Tensor):
            The input tensor containing values to be replaced.
        nan (float, optional):
            The value to replace NaN with.
            Default is 0.0.
        posinf (float, optional):
            The value to replace positive infinity with.
            Default is ``math.inf`` (preserves positive infinity).
        neginf (float, optional):
            The value to replace negative infinity with.
            Default is ``-math.inf`` (preserves negative infinity).

    Returns:
        Tensor:
            A new tensor with the same shape and dtype as the input, where NaN
            and infinity values are replaced according to the specified
            parameters.

    Example:

        >>> # Replace NaN values with 0 (default behavior)
        >>> x = torch.tensor([1.0, nan, 3.0, nan])
        >>> QF.fillna(x)
        tensor([1., 0., 3., 0.])

        >>> # Replace NaN with -1
        >>> x = torch.tensor([[nan, 2.0], [3.0, nan]])
        >>> QF.fillna(x, nan=-1.0)
        tensor([[-1.,  2.],
                [ 3., -1.]])

        >>> # Handle infinity values
        >>> x = torch.tensor([1.0, inf, -inf, nan])
        >>> QF.fillna(x, nan=0.0, posinf=999.0, neginf=-999.0)
        tensor([   1.,  999., -999.,    0.])

        >>> # Preserve infinity by default
        >>> x = torch.tensor([inf, -inf, nan])
        >>> QF.fillna(x)
        tensor([inf, -inf, 0.])
    """
    return x.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
