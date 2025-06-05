from dataclasses import dataclass

import torch

from .eigh import eigh
from .nancovar import nancovar


@dataclass
class NanpcaResult:
    components: torch.Tensor
    explained_variance: torch.Tensor


def nanpca(data: torch.Tensor) -> NanpcaResult:
    """Computes principal components on the given data.

    The returned value represents principal components.  Specifically,
    `result[*, i, :]` represents the :math:`(i+1)`-th largest principal
    component of the batch specified by `*`.

    Shape:
        - data: :math:`(*, N, C)` where `*` means any number of additional
          dimensions, `N` means the number of samples, and `C` means the
          number of features.

    Return:
        - w (torch.Tensor): `w[i]` represents the eigenvalue of the
          :math:`i`-th component.
        - v (torch.Tensor): `v[i, j]` represents the :math:`i`-th component's
          weight for the `j`-th feature.
    """
    batch_shape = data.shape[:-2]
    data = data[None].flatten(end_dim=-3)
    w, v = eigh(nancovar(data[:, :, :, None], data[:, :, None, :], dim=-3))
    v = v.flip(-1).transpose(-1, -2)
    w = w.flip(-1)
    return NanpcaResult(
        components=v.reshape(batch_shape + v.shape[-2:]),
        explained_variance=w.reshape(batch_shape + w.shape[-1:]),
    )
