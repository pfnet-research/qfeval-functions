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
    r"""Returns a tensor filled with random integers generated uniformly
    between ``low`` (inclusive) and ``high`` (exclusive).

    Unlike :func:`torch.randint`, if the seed is fixed with
    :func:`qfeval_functions.random.seed`, the result must be reproducible on
    any device.  See :func:`rand` for details.

    Args:
        low (int):
            The lowest integer to be drawn from the distribution (inclusive).
        high (int):
            One above the highest integer to be drawn from the distribution
            (exclusive).
        size (tuple of ints):
            A tuple defining the shape of the output tensor.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor.
            Default is ``torch.int64``.
        device (torch.device, optional):
            The desired device of the returned tensor.
            Default uses the current device.

    Returns:
        Tensor:
            A tensor of the given shape filled with random integers in
            :math:`[\text{low}, \text{high})`.

    Example:

        >>> import qfeval_functions
        >>> with qfeval_functions.random.seed(1):
        ...     x = QF.randint(0, 10, (2, 3))
        >>> x
        tensor([[2, 9, 7],
                [1, 5, 9]])

    .. seealso::
        - :func:`rand`: Uniform random floats on the interval [0, 1).
        - :func:`randn`: Standard normal random numbers.
        - :func:`randperm`: Random permutation of integers.
        - ``torch.randint``: PyTorch's built-in random integer generator.
    """
    if is_fast():
        return torch.randint(
            low, high, size, dtype=dtype or torch.int64, device=device
        )
    v = rng().integers(low, high, size)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
