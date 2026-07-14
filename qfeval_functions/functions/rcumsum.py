import torch


def rcumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Returns the reversely cumulative sum of elements of ``x`` in the
    dimension ``dim``.

    This is the reverse-direction counterpart of :func:`torch.cumsum`: the
    :math:`i`-th output element is the sum of all elements at or after
    position :math:`i` along the dimension.  This is useful, for example,
    to compute the total of future values at each time step.

    Args:
        x (Tensor):
            The input tensor.
        dim (int):
            The dimension along which to compute the reversely cumulative
            sum.

    Returns:
        Tensor:
            A tensor of the same shape as the input, where each element is
            the sum of the input elements at or after its position along
            the dimension.

    Example:

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> QF.rcumsum(x, dim=0)
        tensor([10.,  9.,  7.,  4.])

    .. note::
        NaN values propagate in the direction of accumulation: because the
        accumulation runs from the end of the dimension, all positions at or
        before a NaN become NaN.

    .. seealso::
        - :func:`rcummax`: Reversely cumulative maximum function.
        - :func:`nancumsum`: NaN-aware cumulative sum function.
        - ``torch.cumsum``: Standard (forward) cumulative sum.
    """
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])
