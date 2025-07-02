import typing

import torch

from .randn import randn


def randn_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Generate a tensor with normal random numbers matching the shape of input tensor.

    This function creates a tensor with the same shape as the input tensor,
    filled with random numbers sampled from a standard normal distribution
    with mean 0 and standard deviation 1. The data type and device can be
    optionally overridden, otherwise they default to the input tensor's
    dtype and device.

    This function ensures reproducibility across different devices when a
    seed is fixed, making it ideal for creating random tensors that match
    the structure of existing data while maintaining deterministic behavior
    for statistical modeling and simulations.

    Args:
        input (Tensor):
            The input tensor whose shape will be used to determine the
            shape of the output tensor.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor. If not specified,
            defaults to the dtype of the input tensor.
        device (torch.device, optional):
            The desired device of the returned tensor. If not specified,
            defaults to the device of the input tensor.

    Returns:
        Tensor:
            A tensor with the same shape as :attr:`input`, filled with
            random numbers from the standard normal distribution N(0, 1).

    Example:
        >>> # Generate normal random tensor matching input shape
        >>> x = torch.zeros(3, 4)
        >>> random_x = QF.randn_like(x)
        >>> random_x.shape
        torch.Size([3, 4])
        >>> random_x.dtype
        torch.float32

        >>> # Override dtype while keeping shape
        >>> y = torch.zeros(2, 3, dtype=torch.int64)
        >>> random_y = QF.randn_like(y, dtype=torch.float64)
        >>> random_y.shape
        torch.Size([2, 3])
        >>> random_y.dtype
        torch.float64

        >>> # Override device while keeping shape and dtype
        >>> z = torch.zeros(5, device='cpu')
        >>> random_z = QF.randn_like(z, device='cpu')
        >>> random_z.device
        device(type='cpu')
        >>> random_z.dtype == z.dtype
        True

        >>> # Complex tensor shapes
        >>> complex_tensor = torch.zeros(2, 3, 4, 5)
        >>> random_complex = QF.randn_like(complex_tensor)
        >>> random_complex.shape
        torch.Size([2, 3, 4, 5])

        >>> # Statistical properties of large sample
        >>> large_sample = QF.randn_like(torch.zeros(10000))
        >>> abs(large_sample.mean()) < 0.1  # Should be close to 0
        tensor(True)
        >>> abs(large_sample.std() - 1.0) < 0.1  # Should be close to 1
        tensor(True)

    .. seealso::
        - :func:`randn`: Generate normal random tensor with specified shape.
        - :func:`rand_like`: Generate uniform random tensor with same shape as input.
        - :func:`torch.randn_like`: PyTorch's native normal random tensor generation.

    """
    return randn(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
