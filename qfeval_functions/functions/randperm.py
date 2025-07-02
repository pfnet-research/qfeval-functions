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
    r"""Generate a random permutation of integers from 0 to n-1.

    This function creates a 1D tensor containing a random permutation of the
    integers from 0 to n-1 (inclusive). Each integer appears exactly once in
    the output, making this ideal for random sampling without replacement,
    shuffling operations, and creating random orderings.

    The function ensures reproducibility across different devices when a
    seed is fixed, making it suitable for deterministic shuffling and
    sampling procedures in research and production environments.

    When fast mode is enabled, this function delegates to PyTorch's native
    :func:`torch.randperm` for optimal performance. Otherwise, it uses a custom
    random number generator to ensure cross-device reproducibility.

    Args:
        n (int):
            The upper bound (exclusive) for the permutation. The permutation
            will contain integers from 0 to n-1. Must be non-negative.
        dtype (torch.dtype, optional):
            The desired integer data type of the returned tensor. If not
            specified, defaults to ``torch.int64``.
        device (torch.device, optional):
            The desired device of the returned tensor. If not specified,
            uses the default device.

    Returns:
        Tensor:
            A 1D tensor of length ``n`` containing a random permutation
            of integers from 0 to n-1.

    Raises:
        ValueError:
            If ``n < 0``.

    Example:
        >>> # Generate a random permutation of 5 elements
        >>> perm = QF.randperm(5)
        >>> perm.shape
        torch.Size([5])
        >>> perm.dtype
        torch.int64
        >>> torch.sort(perm)[0]  # Should contain 0, 1, 2, 3, 4
        tensor([0, 1, 2, 3, 4])

        >>> # Generate with specific dtype
        >>> perm32 = QF.randperm(3, dtype=torch.int32)
        >>> perm32.dtype
        torch.int32
        >>> len(torch.unique(perm32)) == 3  # All elements should be unique
        True

        >>> # Generate on specific device
        >>> perm_cpu = QF.randperm(4, device='cpu')
        >>> perm_cpu.device
        device(type='cpu')

        >>> # Random shuffle of indices
        >>> data = torch.tensor([10, 20, 30, 40, 50])
        >>> shuffled_indices = QF.randperm(len(data))
        >>> shuffled_data = data[shuffled_indices]
        >>> shuffled_data.shape
        torch.Size([5])

        >>> # Empty permutation
        >>> empty_perm = QF.randperm(0)
        >>> empty_perm.shape
        torch.Size([0])

    .. seealso::
        - :func:`randint`: Generate random integers from a range.
        - :func:`rand`: Generate random floats from uniform distribution.
        - :func:`torch.randperm`: PyTorch's native random permutation generation.

    .. note::
        This function is essential in quantitative finance for:

        - Bootstrap sampling and resampling techniques
        - Random portfolio rebalancing and asset selection
        - Cross-validation fold creation for time series models
        - Random shuffling of historical data for backtesting
        - Monte Carlo simulations requiring random orderings
        - Feature selection and random matrix permutations
        - Creating random train/test splits for financial datasets

    .. note::
        The permutation property guarantees that each integer from 0 to n-1
        appears exactly once, making this function perfect for sampling
        without replacement scenarios common in financial analysis.

    .. warning::
        When using this function for time series data, be mindful that
        random permutations can destroy temporal dependencies. Consider
        block-based or constrained permutations for time series applications.
    """
    if is_fast():
        return torch.randperm(n, dtype=dtype or torch.int64, device=device)
    v = rng().permutation(n)
    return torch.tensor(v, dtype=dtype or torch.int64, device=device)
