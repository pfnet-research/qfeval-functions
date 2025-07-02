import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def rand(
    *size: int,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Generate a tensor filled with random numbers from a uniform distribution.

    This function creates a tensor of the specified size filled with random
    numbers sampled from a uniform distribution on the interval [0, 1).
    The function ensures reproducibility across different devices when a
    seed is fixed, making it suitable for deterministic simulations and
    testing.

    When fast mode is enabled, this function delegates to PyTorch's native
    :func:`torch.rand` for optimal performance. Otherwise, it uses a custom
    random number generator to ensure cross-device reproducibility.

    Args:
        *size (int):
            Sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a single argument
            that is a sequence.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor. If not specified,
            defaults to ``torch.float32``.
        device (torch.device, optional):
            The desired device of the returned tensor. If not specified,
            uses the default device.

    Returns:
        Tensor:
            A tensor of shape ``size`` filled with random numbers from
            a uniform distribution on [0, 1).

    Example:
        >>> # Generate a 1D tensor
        >>> x = QF.rand(5)
        >>> x.shape
        torch.Size([5])
        >>> (x >= 0).all() and (x < 1).all()
        tensor(True)

        >>> # Generate a 2D tensor with specific dtype
        >>> y = QF.rand(3, 4, dtype=torch.float64)
        >>> y.shape
        torch.Size([3, 4])
        >>> y.dtype
        torch.float64

        >>> # Generate on specific device
        >>> z = QF.rand(2, 3, device='cpu')
        >>> z.device
        device(type='cpu')

        >>> # Generate with unpacked tuple
        >>> shape = (2, 3, 4)
        >>> w = QF.rand(*shape)
        >>> w.shape
        torch.Size([2, 3, 4])

    .. seealso::
        - :func:`rand_like`: Generate random tensor with same shape as input.
        - :func:`torch.rand`: PyTorch's native random tensor generation.

    .. note::
        This function provides deterministic behavior across devices when
        the random seed is fixed, which is particularly important for:

        - Monte Carlo simulations in quantitative finance
        - Reproducible backtesting and research
        - Cross-platform model validation
        - Distributed computing scenarios where consistency is required

    .. warning::
        When using this function in financial applications, ensure proper
        random seed management for reproducible results in production
        environments.
    """
    if is_fast():
        return torch.rand(*size, dtype=dtype or torch.float32, device=device)
    v = rng().random(size)
    return torch.tensor(v, dtype=dtype or torch.float32, device=device)
