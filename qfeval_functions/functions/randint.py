import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randint(
    low: int,
    high: int,
    size: typing.Tuple[int, ...],
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Generate a tensor filled with random integers from a uniform distribution.

    This function creates a tensor of the specified size filled with random
    integers sampled uniformly from the half-open interval [low, high).
    The function ensures reproducibility across different devices when a
    seed is fixed, making it suitable for deterministic simulations and
    testing scenarios.

    When fast mode is enabled, this function delegates to PyTorch's native
    :func:`torch.randint` for optimal performance. Otherwise, it uses a custom
    random number generator to ensure cross-device reproducibility.

    Args:
        low (int):
            Lowest integer to be drawn from the distribution (inclusive).
        high (int):
            One above the highest integer to be drawn from the distribution
            (exclusive). Must be greater than :attr:`low`.
        size (tuple of ints):
            Tuple defining the shape of the output tensor.
        dtype (torch.dtype, optional):
            The desired integer data type of the returned tensor. If not
            specified, defaults to ``torch.int64``.
        device (torch.device, optional):
            The desired device of the returned tensor. If not specified,
            uses the default device.

    Returns:
        Tensor:
            A tensor of shape ``size`` filled with random integers from
            the uniform distribution on [low, high).

    Raises:
        ValueError:
            If ``high <= low``.

    Example:
        >>> # Generate random integers between 0 and 10
        >>> x = QF.randint(0, 10, (3, 4))
        >>> x.shape
        torch.Size([3, 4])
        >>> x.dtype
        torch.int64
        >>> (x >= 0).all() and (x < 10).all()
        tensor(True)

        >>> # Generate with specific dtype
        >>> y = QF.randint(-5, 5, (2, 3), dtype=torch.int32)
        >>> y.dtype
        torch.int32
        >>> (y >= -5).all() and (y < 5).all()
        tensor(True)

        >>> # Generate on specific device
        >>> z = QF.randint(1, 7, (6,), device='cpu')  # Dice roll simulation
        >>> z.device
        device(type='cpu')
        >>> (z >= 1).all() and (z < 7).all()
        tensor(True)

        >>> # Generate binary values (0 or 1)
        >>> binary = QF.randint(0, 2, (5, 5))
        >>> torch.all((binary == 0) | (binary == 1))
        tensor(True)

    .. seealso::
        - :func:`rand`: Generate random floats from uniform distribution.
        - :func:`randperm`: Generate random permutation of integers.
        - :func:`torch.randint`: PyTorch's native random integer generation.

    """
    if is_fast():
        return torch.randint(
            low, high, size, dtype=dtype or torch.int64, device=device
        )
    v = rng().integers(low, high, size)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
