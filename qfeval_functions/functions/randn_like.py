import typing

import torch

from .randn import randn


def randn_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor with the same size as ``input`` that is filled with
    random numbers from a normal distribution with mean 0 and variance 1.

    Unlike :func:`torch.randn_like`, if the seed is fixed with
    :func:`qfeval_functions.random.seed`, the result must be reproducible on
    any device.  See :func:`randn` for details.

    Args:
        input (Tensor):
            The input tensor whose shape, dtype, and device determine those
            of the output tensor.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor.
            Default is the dtype of ``input``.
        device (torch.device, optional):
            The desired device of the returned tensor.
            Default is the device of ``input``.

    Returns:
        Tensor:
            A tensor with the same shape as ``input`` filled with standard
            normal random numbers.

    Example:

        >>> import qfeval_functions
        >>> with qfeval_functions.random.seed(1):
        ...     x = QF.randn_like(torch.zeros(2, 3, dtype=torch.float64))
        >>> x
        tensor([[ 2.0468, -1.0365,  0.7300],
                [ 0.4158, -0.1923, -0.4909]], dtype=torch.float64)

    .. seealso::
        - :func:`randn`: Normal random tensor with a specified shape.
        - :func:`rand_like`: Uniform random tensor with the shape of another
          tensor.
    """
    return randn(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
