import torch

from .apply_for_axis import apply_for_axis


def bincount(
    x: torch.Tensor, minlength: int = 0, dim: int = -1
) -> torch.Tensor:
    r"""Computes the frequency of each value in a tensor along a specified dimension.

    This function extends PyTorch's :func:`torch.bincount` to work with multidimensional
    tensors. It counts the occurrences of each non-negative integer value along the
    specified dimension, preserving all other dimensions as batch dimensions.

    For each slice along the specified dimension, the function creates a histogram
    counting how many times each integer value appears. The output tensor has the
    same shape as the input except along the target dimension, which becomes the
    number of bins (determined by the maximum value in the input or ``minlength``).

    The mathematical operation for a 1-D tensor is::

        output[i] = count(x == i)  # for i in range(max(x) + 1)

    For multidimensional tensors, this operation is applied independently to each
    slice along the specified dimension::

        output[..., i] = count(x[..., :] == i)  # along the target dimension

    Note:
        - Input values must be non-negative integers
        - The function handles empty tensors by returning appropriately shaped zero tensors
        - All input values are treated as exact integer matches for counting

    Args:
        x (Tensor): Input tensor containing non-negative integer values to count
        minlength (int): Minimum number of bins in the output. If the maximum value
            in :attr:`x` is smaller than ``minlength - 1``, the output will be padded
            with zeros to reach this length. Default: 0
        dim (int): The dimension along which to compute the histogram. Default: -1
            (last dimension)

    Returns:
        Tensor: A tensor containing the frequency counts. The shape is the same as
        :attr:`x` except that ``x.shape[dim]`` is replaced by the number of bins
        (``max(minlength, max(x) + 1)``).

    Raises:
        RuntimeError: If input contains negative values (inherited from PyTorch's
            scatter operations).

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>>
        >>> # 2D example - histogram for each row
        >>> x = torch.tensor([[1, 2, 2], [3, 3, 1]])
        >>> QF.bincount(x, dim=1)
        tensor([[0, 1, 2, 0],
                [0, 1, 0, 2]])
        >>>
        >>> # Using minlength parameter
        >>> x = torch.tensor([[0, 1], [1, 2]])
        >>> QF.bincount(x, dim=1, minlength=4)
        tensor([[1, 1, 0, 0],
                [0, 1, 1, 0]])
        >>>
        >>> # Empty tensor handling
        >>> x = torch.empty(0, dtype=torch.int64)
        >>> QF.bincount(x, minlength=3, dim=0)
        tensor([0, 0, 0])
    """

    def _bincount(x: torch.Tensor) -> torch.Tensor:
        """Internal function to compute bincount for 2D tensors.

        Args:
            x: 2D tensor of shape (batch_size, sequence_length)

        Returns:
            2D tensor of shape (batch_size, num_bins) with frequency counts
        """
        # TODO(claude): This function only works for certain tensor configurations
        # when called via apply_for_axis. The function fails for 1D tensors and
        # tensors where dim != 1 due to mismatched tensor shapes and scatter dimensions.
        if x.numel() == 0:
            # Handle edge case where x is empty
            # Reference: PyTorch implementation for numpy compatibility
            return torch.zeros((1, minlength), dtype=torch.int64)

        # Determine number of bins: max of minlength and (max_value + 1)
        n = max(minlength, int(torch.amax(x)) + 1)

        # Create output tensor filled with zeros
        zeros = torch.zeros_like(x[:1, :1]).expand(x.shape[0], n)

        # Create tensor of ones for counting
        ones = torch.ones_like(x[:1, :1]).expand(x.shape)

        # TODO(claude): The scatter_add call has incorrect dim parameter.
        # When called via apply_for_axis, the tensor is transposed so that
        # the target dimension becomes dim=0, but scatter_add is called with
        # the original dim parameter. This causes runtime errors for 1D tensors
        # and incorrect behavior. The dim should be 0 when called from apply_for_axis.
        # Use scatter_add to accumulate counts
        return torch.scatter_add(zeros, dim, x, ones)

    return apply_for_axis(_bincount, x, dim)
