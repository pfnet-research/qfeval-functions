import typing

import torch


def rms(
    x: torch.Tensor,
    dim: typing.Union[None, int, typing.Tuple[int, ...]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Returns the root mean square of each row of the input tensor in the
    given dimension ``dim``.  If ``dim`` is a list of dimensions, reduce over all
    of them.

    The root mean square (RMS) is defined as:

    .. math::
        \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}

    It measures the magnitude of values regardless of their signs, and is
    commonly used to quantify the size of signals or returns.

    Args:
        x (Tensor):
            The input tensor.
        dim (None, int, or tuple of ints, optional):
            The dimension or dimensions to reduce. If ``None`` (default),
            reduces over all dimensions.
        keepdim (bool, optional):
            Whether the output tensor has ``dim`` retained or not.
            Default is ``False``.

    Returns:
        Tensor:
            The root mean square of the input tensor over the specified
            dimensions.

    Example:

        >>> x = torch.tensor([1.0, -2.0, 3.0, -4.0])
        >>> QF.rms(x)
        tensor(2.7386)

        >>> x = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        >>> QF.rms(x, dim=1)
        tensor([1.5811, 3.5355])

    .. seealso::
        - ``torch.std``: Standard deviation (RMS of deviations from the
          mean).
    """
    return x.square().mean(dim=dim, keepdim=keepdim).sqrt()
