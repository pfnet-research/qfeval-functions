import torch

from .rcummax import rcummax


def mmax(x: torch.Tensor, span: int, dim: int = -1) -> torch.Tensor:
    r"""Returns the moving max of the given tensor."""
    # 1. Move the target dimension to the top to make data manipulation easier.
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape((shape[0], -1))

    # 2. Reshape the target dimension into `(*, span)` with expanding the edge.
    pad_len = span * 2 - x.shape[0] % span
    pad = x[:1, :].expand(pad_len, -1)
    x = torch.cat((pad, x), dim=0)
    x = x.reshape((-1, span, x.shape[-1]))

    # 3. Calculate `max(x[i:i+span])` by splitting it into
    # `max(x[i:s], x[s:i+span])` where `s` is a multiple of `span`.  They can
    # be calculated by cummax and rcummax.
    a = x.cummax(dim=1).values
    b = rcummax(x, dim=1).values
    x = torch.cat((torch.max(a[1:, :-1], b[:-1, 1:]), a[1:, -1:]), dim=1)
    x = x.reshape((-1, x.shape[-1]))[-shape[0] :]

    # 4. Restore the shape and the order of dimensions.
    return x.reshape(shape).transpose(0, dim)
