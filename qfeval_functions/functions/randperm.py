import typing

import torch

from qfeval_functions.random import is_fast
from qfeval_functions.random import rng


def randperm(
    n: int,
    *,
    dtype: typing.Optional[torch.dtype] = None,
    device: typing.Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Returns a random permutation of integers from ``0`` to ``n - 1``.

    Unlike :func:`torch.randperm`, if the seed is fixed with
    :func:`qfeval_functions.random.seed`, the result must be reproducible on
    any device.  See :func:`rand` for details.

    Args:
        n (int):
            The upper bound (exclusive) of the permutation.
        dtype (torch.dtype, optional):
            The desired data type of the returned tensor.
            Default is ``torch.int64``.
        device (torch.device, optional):
            The desired device of the returned tensor.
            Default uses the current device.

    Returns:
        Tensor:
            A 1-dimensional tensor of length ``n`` containing a random
            permutation of the integers from ``0`` to ``n - 1``.

    Example:

        >>> import qfeval_functions
        >>> with qfeval_functions.random.seed(1):
        ...     x = QF.randperm(10)
        >>> x
        tensor([1, 0, 9, 8, 6, 3, 7, 4, 2, 5])

    .. seealso::
        - :func:`randint`: Uniform random integers.
        - :func:`rand`: Uniform random floats on the interval [0, 1).
        - ``torch.randperm``: PyTorch's built-in random permutation.
    """
    if is_fast():
        return torch.randperm(n, dtype=dtype or torch.int64, device=device)
    v = rng().permutation(n)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
