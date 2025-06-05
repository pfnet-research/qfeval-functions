from dataclasses import dataclass

import torch

from .covar import covar


@dataclass
class PcaResult:
    components: torch.Tensor
    explained_variance: torch.Tensor


def pca(x: torch.Tensor) -> PcaResult:
    """Computes principal components on the given input `x`.
    The returned value represents principal components.  Specifically,
    `result[*, i, :]` represents the :math:`(i+1)`-th largest principal
    component of the batch specified by `*`.
    Shape:
        - x: :math:`(*, S, D)` where `*` means any number of additional
          dimensions, `S` means the number of sections, and `D` means the
          number of dimensions.
    In qfeval, dimensions and sections often represent symbols and timestamps
    respectively.
    Return:
        - components (Tensor): `components[i]` represents the eigenvalue of the
            :math:`i`-th component.
        - explained_variance (Tensor): `explained_variance[i, j]` represents
            the :math:`i`-th component's weight for the `j`-th feature.
    """
    return pca_cov(covar(x[..., None], x[..., None, :], dim=-3))


def pca_cov(cov: torch.Tensor) -> PcaResult:
    """Computes principal components on the given covariance `cov`.
    The returned value represents principal components.  Specifically,
    `result[*, i, :]` represents the :math:`(i+1)`-th largest principal
    component of the batch specified by `*`.
    Shape:
        - cov: :math:`(*, D, D)` where `*` means any number of additional
          dimensions, and `D` means the number of dimensions.
    Return:
        - components (Tensor): `components[i]` represents the eigenvalue of the
            :math:`i`-th component.
        - explained_variance (Tensor): `explained_variance[i, j]` represents
            the :math:`i`-th component's weight for the `j`-th feature.
    """
    batch_shape = cov.shape[:-2]
    _, s, v = torch.linalg.svd(cov.unsqueeze(0).flatten(end_dim=-3))
    return PcaResult(
        components=v.reshape(batch_shape + v.shape[1:]),
        explained_variance=s.reshape(batch_shape + s.shape[1:]),
    )
