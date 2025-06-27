import torch

from .mmax import mmax


def mmin(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) minimum of a tensor.

    This function calculates the minimum value within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1``
    elements along the selected dimension), the result is computed using
    available values by padding with the first element.

    The moving minimum is computed as:

    .. math::
        \text{MMIN}[i] = \min_{j=\max(0, i-\text{span}+1)}^{i} x[j]

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving minimum.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            minimum values.

    Example:

        >>> # Simple moving minimum with window size 3
        >>> x = torch.tensor([5.0, 1.0, 3.0, 2.0, 8.0])
        >>> QF.mmin(x, span=3)
        tensor([5., 1., 1., 1., 2.])

        >>> # 2D tensor with moving minimum along columns
        >>> x = torch.tensor([[4.0, 1.0, 6.0, 2.0],
        ...                   [2.0, 5.0, 1.0, 4.0],
        ...                   [7.0, 2.0, 4.0, 3.0]])
        >>> QF.mmin(x, span=2, dim=1)
        tensor([[4., 1., 1., 2.],
                [2., 2., 1., 1.],
                [7., 2., 2., 3.]])

        >>> # Moving minimum along the first dimension
        >>> x = torch.tensor([[4.0, 5.0],
        ...                   [1.0, 3.0],
        ...                   [3.0, 2.0],
        ...                   [2.0, 4.0]])
        >>> QF.mmin(x, span=3, dim=0)
        tensor([[4., 5.],
                [1., 3.],
                [1., 2.],
                [1., 2.]])

        >>> # Handling negative values
        >>> x = torch.tensor([-1.0, -5.0, -2.0, -4.0, -1.0])
        >>> QF.mmin(x, span=2)
        tensor([-1., -5., -5., -4., -4.])

    .. note::
        This function is implemented as ``-mmax(-x, span, dim)``, leveraging
        the duality between minimum and maximum operations. This approach
        ensures consistent behavior and performance with the moving maximum
        function while avoiding code duplication.

    .. seealso::
        :func:`mmax`: Moving maximum function.
        :func:`msum`: Moving sum function.
        :func:`ma`: Moving average function.
    """
    return -mmax(-x, span, dim)
