import torch

from .stable_sort import stable_sort


def cumcount(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Numbers each item in each group from 0 to :math:`N-1`, where :math:`N`
    is the number of the items of the group.

    This behavior refers to pandas GroupBy.cumcount.
    """

    # 1. Flatten the input tensor.
    dim = len(x.shape) + dim if dim < 0 else dim
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape(x.shape[0], -1)

    # 2. Computes the index of a group for each sorted element.
    v, idx = stable_sort(x, dim=0)
    a = torch.arange(x.shape[0], device=x.device)[:, None]
    b = torch.where(
        torch.eq(v, v.roll(1, 0)),
        torch.zeros_like(a),
        a,
    )
    g_idx = a - b.cummax(0).values

    # 3. Distribute the indexes to the original locations.
    g_idx = idx.scatter(0, idx, g_idx)

    # 4. Restore the shape.
    return g_idx.reshape(shape).transpose(0, dim)  # type: ignore[no-any-return]
