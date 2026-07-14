import typing

import torch

from .rand import rand


def rand_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a tensor with the same size as ``input`` that is filled with
    random numbers from a uniform distribution on the interval :math:`[0, 1)`.

    Unlike :func:`torch.rand_like`, if the seed is fixed with
    :func:`qfeval_functions.random.seed`, the result must be reproducible on
    any device.  See :func:`rand` for details.

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
            A tensor with the same shape as ``input`` filled with uniform
            random numbers in :math:`[0, 1)`.

    Example:

        >>> import qfeval_functions
        >>> with qfeval_functions.random.seed(1):
        ...     x = QF.rand_like(torch.zeros(2, 3, dtype=torch.float64))
        >>> x
        tensor([[0.2407, 0.7296, 0.5584],
                [0.5268, 0.1469, 0.9072]], dtype=torch.float64)

    .. seealso::
        - :func:`rand`: Uniform random tensor with a specified shape.
        - :func:`randn_like`: Normal random tensor with the shape of another
          tensor.
    """
    return rand(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
