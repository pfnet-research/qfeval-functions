import string
import typing

import numpy as np
import torch

from .einsum import einsum


def mulsum(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    mean: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Compute sum or mean of element-wise product in a memory-efficient way.

    This function calculates the sum (or mean) of the element-wise product of
    two tensors ``(x * y).sum(dim)`` or ``(x * y).mean(dim)`` without creating
    the intermediate product tensor in memory. This is particularly crucial when
    working with large tensors or when broadcasting would result in a very large
    intermediate tensor that could exceed available memory.

    Args:
        x (Tensor):
            The first input tensor.
        y (Tensor):
            The second input tensor. Must be broadcastable with :attr:`x`.
        dim (int or tuple of ints, optional):
            The dimension(s) along which to
            compute the sum or mean. If not specified (default is empty tuple),
            the operation is computed over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has :attr:`dim`
            retained or not. Default is False.
        mean (bool, optional):
            If True, computes the mean instead of sum.
            Default is False (computes sum).
        _ddof (int, optional):
            Delta degrees of freedom for mean calculations.
            The divisor used is ``N - _ddof``, where ``N`` is the number of
            elements. Default is 0. This is an internal parameter.

    Returns:
        Tensor:
            The sum or mean of the element-wise product. The shape depends
            on the input dimensions, :attr:`dim`, and :attr:`keepdim` parameters.

    Example:

        >>> # Simple element-wise product sum
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> y = torch.tensor([2.0, 3.0, 4.0])
        >>> QF.mulsum(x, y)
        tensor(20.)

        >>> # Equivalent to (x * y).sum()
        >>> torch.allclose(QF.mulsum(x, y), (x * y).sum())
        True

        >>> # Mean instead of sum
        >>> QF.mulsum(x, y, mean=True)
        tensor(6.6667)

        >>> # 2D tensors with specific dimension
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> y = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        >>> QF.mulsum(x, y, dim=1)
        tensor([ 4., 11.])

        >>> # Memory-efficient broadcasting
        >>> x = torch.randn(1000, 1)
        >>> y = torch.randn(1, 1000)
        >>> # Efficiently computes without creating 1000x1000 intermediate tensor
        >>> result = QF.mulsum(x, y)

        >>> # With keepdim
        >>> x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        >>> y = torch.tensor([[[2.0, 1.0]], [[1.0, 2.0]]])
        >>> QF.mulsum(x, y, dim=(1, 2), keepdim=True)
        tensor([[[ 4.]],
        <BLANKLINE>
                [[11.]]])

    .. seealso::
        :func:`mulmean`: Convenience function for computing means.
        :func:`einsum`: The underlying Einstein summation function.
        :func:`covar`: Uses this function for efficient covariance calculations.
    """

    # 1. Align the number of dimensions (c.f., NumPy broadcasting).
    length = max(len(x.shape), len(y.shape))
    x = x.reshape((1,) * (length - len(x.shape)) + x.shape)
    y = y.reshape((1,) * (length - len(y.shape)) + y.shape)

    # 2. Figure out the final shape.
    # NOTE: x[None][:0] is a trick to call a torch function without wasting
    # cpu/memory resources.
    mul_shape = (x[None][:0] * y[None][:0]).shape[1:]

    # 3. Parse dim.
    # `mask` should be a tuple of ints, each of which should represent whether
    # the dimension is aggregated (1) or not (0).
    mask = torch.sum(torch.zeros((0,) * length), dim=dim, keepdim=True).shape

    # 4. Prepare einsum parameters and apply them to einsum.
    input_eq = string.ascii_lowercase[: len(x.shape)]
    result_eq = "".join(c for c, m in zip(input_eq, mask) if m == 0)
    result = einsum(f"{input_eq},{input_eq}->{result_eq}", x, y)

    # 5. (If keepdim is enabled,) Restore aggregated dimensions.
    if keepdim:
        result = result.reshape(
            tuple(1 if m == 1 else s for s, m in zip(mul_shape, mask))
        )

    # 6. (If mean is enabled,) Divide the result by the number of aggregated
    # elements.
    if mean:
        result = result / (
            int(np.prod([s if m == 1 else 1 for s, m in zip(mul_shape, mask)]))
            - _ddof
        )

    return result
