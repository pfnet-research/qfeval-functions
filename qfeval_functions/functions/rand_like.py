import typing

import torch

from .rand import rand


def rand_like(
    input: torch.Tensor,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Generate a tensor with random numbers matching the shape of input tensor.

    This function creates a tensor with the same shape as the input tensor,
    filled with random numbers sampled from a uniform distribution on the
    interval [0, 1). The data type and device can be optionally overridden,
    otherwise they default to the input tensor's dtype and device.

    This function ensures reproducibility across different devices when a
    seed is fixed, making it ideal for creating random tensors that match
    the structure of existing data while maintaining deterministic behavior.

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
            random numbers from a uniform distribution on [0, 1).

    Example:
        >>> # Generate random tensor matching input shape
        >>> x = torch.zeros(3, 4)
        >>> random_x = QF.rand_like(x)
        >>> random_x.shape
        torch.Size([3, 4])
        >>> random_x.dtype
        torch.float32

        >>> # Override dtype while keeping shape
        >>> y = torch.zeros(2, 3, dtype=torch.int64)
        >>> random_y = QF.rand_like(y, dtype=torch.float64)
        >>> random_y.shape
        torch.Size([2, 3])
        >>> random_y.dtype
        torch.float64

        >>> # Override device while keeping shape and dtype
        >>> z = torch.zeros(5, device='cpu')
        >>> random_z = QF.rand_like(z, device='cpu')
        >>> random_z.device
        device(type='cpu')
        >>> random_z.dtype == z.dtype
        True

        >>> # Complex tensor shapes
        >>> complex_tensor = torch.zeros(2, 3, 4, 5)
        >>> random_complex = QF.rand_like(complex_tensor)
        >>> random_complex.shape
        torch.Size([2, 3, 4, 5])

        >>> # Verify random values are in [0, 1)
        >>> result = QF.rand_like(torch.zeros(100))
        >>> (result >= 0).all() and (result < 1).all()
        tensor(True)

    See Also:
        :func:`rand`: Generate random tensor with specified shape.
        :func:`torch.rand_like`: PyTorch's native random tensor generation.

    .. note::
        This function is particularly useful in quantitative finance for:

        - Generating random shocks matching the structure of return data
        - Creating noise tensors for Monte Carlo simulations
        - Initializing random weights with the same shape as model parameters
        - Bootstrap sampling where random tensors must match data dimensions

    .. note::
        The function internally calls :func:`rand` with the appropriate
        parameters extracted from the input tensor, ensuring consistent
        behavior and reproducibility guarantees.
    """
    return rand(
        *input.shape, dtype=dtype or input.dtype, device=device or input.device
    )
