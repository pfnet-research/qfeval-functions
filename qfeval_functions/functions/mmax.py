import torch

from .rcummax import rcummax


def mmax(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Compute the moving (sliding window) maximum of a tensor.

    This function calculates the maximum value within a sliding window of
    size :attr:`span` along the specified dimension. The output tensor has the
    same shape as the input tensor. For positions where the sliding window
    cannot fully cover preceding elements (i.e., the first ``span - 1`` elements
    along the selected dimension), the result is computed using available values
    by padding with the first element.

    The moving maximum is computed as:

    .. math::
        \text{MMAX}[i] = \max_{j=\max(0, i-\text{span}+1)}^{i} x[j]

    Args:
        x (Tensor):
            The input tensor containing values.
        span (int):
            The size of the sliding window. Must be positive.
        dim (int, optional):
            The dimension along which to compute the moving maximum.
            Default is -1 (the last dimension).

    Returns:
        Tensor:
            A tensor of the same shape as the input, containing the moving
            maximum values.

    Example:

        >>> # Simple moving maximum with window size 3
        >>> x = torch.tensor([1.0, 5.0, 3.0, 8.0, 2.0])
        >>> QF.mmax(x, span=3)
        tensor([1., 5., 5., 8., 8.])

        >>> # 2D tensor with moving maximum along columns
        >>> x = torch.tensor([[1.0, 4.0, 2.0, 6.0],
        ...                   [3.0, 1.0, 5.0, 2.0],
        ...                   [2.0, 7.0, 3.0, 4.0]])
        >>> QF.mmax(x, span=2, dim=1)
        tensor([[1., 4., 4., 6.],
                [3., 3., 5., 5.],
                [2., 7., 7., 4.]])

        >>> # Moving maximum along the first dimension
        >>> x = torch.tensor([[1.0, 2.0],
        ...                   [4.0, 1.0],
        ...                   [2.0, 5.0],
        ...                   [3.0, 2.0]])
        >>> QF.mmax(x, span=3, dim=0)
        tensor([[1., 2.],
                [4., 2.],
                [4., 5.],
                [4., 5.]])

        >>> # Handling negative values
        >>> x = torch.tensor([-2.0, -1.0, -5.0, -3.0, -1.0])
        >>> QF.mmax(x, span=2)
        tensor([-2., -1., -1., -3., -1.])

    .. seealso::
        :func:`mmin`: Moving minimum function.
        :func:`msum`: Moving sum function.
    """
    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape((shape[0], -1))

    # 2. Reshape the target dimension into `(*, span)` with expanding the edge.
    pad_len = span * 2 - x.shape[0] % span
    pad = x[:1, :].expand(pad_len, -1)
    x = torch.cat((pad, x), dim=0)
    x = x.reshape((-1, span, x.shape[-1]))

    # 3. Calculate `max(x[i:i+span])` by splitting it into
    # `max(x[i:s], x[s:i+span])` where `s` is a multiple of `span`.  They can
    # be calculated by cummax and rcummax.
    a = x.cummax(dim=1).values
    b = rcummax(x, dim=1).values
    x = torch.cat((torch.max(a[1:, :-1], b[:-1, 1:]), a[1:, -1:]), dim=1)
    x = x.reshape((-1, x.shape[-1]))[-shape[0] :]

    # 4. Restore the shape and the order of dimensions.
    return x.reshape(shape).transpose(0, dim)
