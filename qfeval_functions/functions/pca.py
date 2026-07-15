from dataclasses import dataclass

import torch

from .covar import covar


@dataclass
class PcaResult:
    components: torch.Tensor
    explained_variance: torch.Tensor


def pca(x: torch.Tensor) -> PcaResult:
    r"""Compute Principal Component Analysis (PCA) on the given input ``x``.

    This function computes the covariance matrix of the input observations
    along the section dimension and then extracts its principal components
    via singular value decomposition.  The components are returned in
    descending order of explained variance.  Specifically,
    ``components[*, i, :]`` represents the :math:`(i+1)`-th largest principal
    component of the batch specified by ``*``.

    In qfeval, dimensions and sections often represent symbols and timestamps
    respectively.

    Args:
        x (Tensor):
            The input tensor containing observations.

    Returns:
        PcaResult: A dataclass containing:

            - ``components`` (Tensor): Principal components (eigenvectors of
              the covariance matrix) of shape :math:`(*, D, D)`.
              ``components[..., i, j]`` represents the :math:`i`-th
              component's weight for the :math:`j`-th dimension.
            - ``explained_variance`` (Tensor): The variance explained by each
              component (eigenvalues of the covariance matrix) of shape
              :math:`(*, D)`, in descending order.

    Shape:
        - x: :math:`(*, S, D)` where ``*`` means any number of additional
          dimensions, ``S`` means the number of sections, and ``D`` means the
          number of dimensions.

    Example:

        >>> x = torch.tensor([[2.0, 0.0],
        ...                   [0.0, 1.0],
        ...                   [-2.0, 0.0],
        ...                   [0.0, -1.0]])
        >>> result = QF.pca(x)
        >>> result.components
        tensor([[1., 0.],
                [0., 1.]])
        >>> result.explained_variance
        tensor([2.6667, 0.6667])

    .. note::
        The sign of each principal component is arbitrary: components are
        defined only up to sign, so an equivalent input may yield components
        multiplied by -1.

    .. seealso::
        - :func:`pca_cov`: PCA on a precomputed covariance matrix, used in
          the implementation.
        - :func:`nanpca`: NaN-aware principal component analysis.
        - :func:`covar`: Covariance function used in the implementation.
    """
    return pca_cov(covar(x[..., None], x[..., None, :], dim=-3))


def pca_cov(cov: torch.Tensor) -> PcaResult:
    r"""Compute principal components on the given covariance matrix ``cov``.

    This function extracts principal components directly from a covariance
    matrix via singular value decomposition, which is useful when the
    covariance matrix is already available (e.g., computed with
    :func:`covar` or estimated by other means).  The components are returned
    in descending order of explained variance.  Specifically,
    ``components[*, i, :]`` represents the :math:`(i+1)`-th largest principal
    component of the batch specified by ``*``.

    Args:
        cov (Tensor):
            The input tensor containing symmetric covariance matrices.

    Returns:
        PcaResult: A dataclass containing:

            - ``components`` (Tensor): Principal components (eigenvectors of
              the covariance matrix) of shape :math:`(*, D, D)`.
              ``components[..., i, j]`` represents the :math:`i`-th
              component's weight for the :math:`j`-th dimension.
            - ``explained_variance`` (Tensor): The variance explained by each
              component (eigenvalues of the covariance matrix) of shape
              :math:`(*, D)`, in descending order.

    Shape:
        - cov: :math:`(*, D, D)` where ``*`` means any number of additional
          dimensions, and ``D`` means the number of dimensions.

    Example:

        >>> cov = torch.tensor([[2.0, 0.0],
        ...                     [0.0, 1.0]])
        >>> result = QF.pca_cov(cov)
        >>> result.components
        tensor([[1., 0.],
                [0., 1.]])
        >>> result.explained_variance
        tensor([2., 1.])

    .. note::
        The sign of each principal component is arbitrary: components are
        defined only up to sign, so an equivalent input may yield components
        multiplied by -1.

    .. seealso::
        - :func:`pca`: PCA computed directly from data.
        - :func:`nanpca`: NaN-aware principal component analysis.
    """
    batch_shape = cov.shape[:-2]
    _, s, v = torch.linalg.svd(cov.unsqueeze(0).flatten(end_dim=-3))
    return PcaResult(
        components=v.reshape(batch_shape + v.shape[1:]),
        explained_variance=s.reshape(batch_shape + s.shape[1:]),
    )
