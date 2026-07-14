import typing

import torch


class Result(typing.NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def rcummax(x: torch.Tensor, dim: int) -> Result:
    r"""Returns the reversely cumulative max of elements of ``x`` in the
    dimension ``dim``.

    This is the reverse-direction counterpart of :func:`torch.cummax`: the
    :math:`i`-th output element is the maximum of all elements at or after
    position :math:`i` along the dimension, together with the index of the
    first position achieving that maximum.  This is useful, for example, to
    find the highest future price at each time step.

    Args:
        x (Tensor):
            The input tensor.
        dim (int):
            The dimension along which to compute the reversely cumulative
            max.

    Returns:
        Result: A named tuple of ``(values, indices)``:

            - ``values`` (Tensor): A tensor of the same shape as the input,
              where each element is the maximum of the input elements at or
              after its position along the dimension.
            - ``indices`` (Tensor): The index of the first occurrence of
              each maximum value along the dimension.

    Example:

        >>> x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
        >>> result = QF.rcummax(x, dim=0)
        >>> result.values
        tensor([5., 5., 5., 5., 4.])
        >>> result.indices
        tensor([3, 3, 3, 3, 4])

    .. note::
        NaN values propagate in the direction of accumulation: because the
        accumulation runs from the end of the dimension, all positions at or
        before a NaN become NaN.

    .. seealso::
        - :func:`rcumsum`: Reversely cumulative sum function.
        - :func:`mmax`: Moving maximum function.
        - ``torch.cummax``: Standard (forward) cumulative maximum.
    """
    result = torch.cummax(torch.flip(x, [dim]), dim)
    return Result(
        values=torch.flip(result.values, [dim]),
        indices=x.shape[dim] - 1 - torch.flip(result.indices, [dim]),
    )
