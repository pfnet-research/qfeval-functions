from dataclasses import dataclass

import torch

from .eigh import eigh
from .nancovar import nancovar


@dataclass
class NanpcaResult:
    components: torch.Tensor
    explained_variance: torch.Tensor


def nanpca(data: torch.Tensor) -> NanpcaResult:
    r"""Compute Principal Component Analysis (PCA) on data, ignoring NaN values.

    This function performs PCA by computing the eigendecomposition of the
    covariance matrix calculated with NaN-aware operations. PCA finds the
    principal components (eigenvectors) that capture the maximum variance in
    the data, ordered by their explained variance (eigenvalues).

    The function computes the covariance matrix using :func:`nancovar` to
    handle NaN values appropriately, then applies eigendecomposition via
    :func:`eigh` to obtain the principal components. The components are
    returned in descending order of explained variance.

    The mathematical formulation follows standard PCA:

    .. math::
        \mathbf{C} = \text{nancovar}(\mathbf{X})

    .. math::
        \mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i

    where :math:`\mathbf{C}` is the covariance matrix, :math:`\mathbf{v}_i`
    are the eigenvectors (principal components), and :math:`\lambda_i` are
    the eigenvalues (explained variance).

    Args:
        data (Tensor):
            Input tensor of shape :math:`(*, N, C)` where :math:`*`
            means any number of additional batch dimensions, :math:`N` is the
            number of samples, and :math:`C` is the number of features.

    Returns:
        NanpcaResult: A dataclass containing:

            - ``components`` (Tensor): Principal components of shape :math:`(*, C, C)`.
              ``components[..., i, :]`` represents the :math:`(i+1)`-th principal
              component (ordered by decreasing explained variance).
            - ``explained_variance`` (Tensor): Eigenvalues of shape :math:`(*, C)`
              representing the variance explained by each component, in descending order.

    Example:

        >>> # Simple 2D PCA with NaN values
        >>> data = torch.tensor([[[1.0, 2.0],
        ...                       [nan, 4.0],
        ...                       [3.0, 6.0],
        ...                       [4.0, nan]]])
        >>> result = QF.nanpca(data)
        >>> result.components.shape
        torch.Size([1, 2, 2])
        >>> result.explained_variance.shape
        torch.Size([1, 2])

        >>> # Batch processing multiple datasets
        >>> data = torch.randn(2, 10, 3)  # 2 batches, 10 samples, 3 features
        >>> # Introduce some NaN values
        >>> data[0, 2, 1] = nan
        >>> data[1, 5, :] = nan
        >>> result = QF.nanpca(data)
        >>> result.components.shape
        torch.Size([2, 3, 3])
        >>> result.explained_variance.shape
        torch.Size([2, 3])

        >>> # Access first principal component
        >>> first_pc = result.components[0, 0, :]  # First batch, first component
        >>> first_variance = result.explained_variance[0, 0]  # Corresponding variance

        >>> # Simple case without convergence issues
        >>> data = torch.tensor([[[1.0, 2.0],
        ...                       [3.0, 4.0],
        ...                       [5.0, nan]]])
        >>> result = QF.nanpca(data)
        >>> result.components.shape
        torch.Size([1, 2, 2])

    .. warning::
        If there are insufficient valid (non-NaN) observations to compute
        meaningful covariance estimates, the results may contain NaN values.
        Ensure adequate data coverage for reliable PCA results.

    .. seealso::
        :func:`nancovar`: NaN-aware covariance computation.
        :func:`eigh`: Eigendecomposition for symmetric matrices.
        :func:`nanmean`: NaN-aware mean used in covariance calculation.
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
