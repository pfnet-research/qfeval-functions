import torch

from .drawdown import drawdown


def max_drawdown(
    x: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    r"""Compute the maximum drawdown along the specified dimension.

    The maximum drawdown is the largest peak-to-trough loss of a price or
    equity curve, i.e., the most negative value of the drawdown series:

    .. math::
        \text{MDD} = \min_i \left(
            \frac{x[i]}{\max_{j \le i} x[j]} - 1 \right)

    Following the convention of quantitative finance libraries such as
    empyrical, the result is returned as a non-positive number (e.g.,
    ``-0.35`` denotes a 35% peak-to-trough loss).  A monotonically
    non-decreasing series yields ``0.0``.

    Args:
        x (Tensor):
            The input tensor containing strictly positive prices.
        dim (int, optional):
            The dimension along which to compute the maximum drawdown.
            Default is -1 (the last dimension).
        keepdim (bool, optional):
            Whether the output tensor has ``dim`` retained or not.
            Default is ``False``.

    Returns:
        Tensor:
            A tensor with the specified dimension reduced (retained with
            size 1 if ``keepdim`` is ``True``), containing the most
            negative drawdown values as non-positive numbers.

    Example:

        >>> x = torch.tensor([100.0, 120.0, 90.0, 130.0, 65.0])
        >>> QF.max_drawdown(x)
        tensor(-0.5000)

        >>> # 2D example: one value per row.
        >>> x = torch.tensor([[100.0, 120.0, 90.0, 130.0, 65.0],
        ...                   [10.0, 20.0, 30.0, 40.0, 50.0]])
        >>> QF.max_drawdown(x, dim=1)
        tensor([-0.5000,  0.0000])

        >>> QF.max_drawdown(x, dim=1, keepdim=True)
        tensor([[-0.5000],
                [ 0.0000]])

    .. note::
        Any NaN along the reduced dimension makes the result NaN, because
        :func:`drawdown` propagates NaN stickily and ``amin`` propagates
        NaN as well.

    .. seealso::
        - :func:`drawdown`: The full drawdown series this function
          minimizes.
        - :func:`nanmin`: Minimum function that skips NaN values.
        - :func:`mmin`: Moving minimum function.
    """
    return drawdown(x, dim=dim).amin(dim=dim, keepdim=keepdim)
