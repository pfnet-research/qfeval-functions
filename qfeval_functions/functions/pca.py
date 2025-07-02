from dataclasses import dataclass

import torch

from .covar import covar


@dataclass
class PcaResult:
    components: torch.Tensor
    explained_variance: torch.Tensor


def pca(x: torch.Tensor) -> PcaResult:
    r"""Compute Principal Component Analysis (PCA) on the input data.

    This function performs PCA by computing the covariance matrix of the input
    data and then finding its principal components through singular value
    decomposition (SVD). The principal components are ordered by their
    explained variance in descending order.

    PCA is a dimensionality reduction technique that finds the directions of
    maximum variance in high-dimensional data and projects the data onto a
    lower-dimensional subspace formed by these principal components.

    Args:
        x (Tensor):
            Input data of shape ``(..., S, D)`` where ``S`` is the number of
            samples and ``D`` is the number of features/dimensions. The batch
            dimensions ``...`` are preserved.

    Returns:
        PcaResult:
            A dataclass containing:

            - **components** (*Tensor*): Principal component vectors of shape
              ``(..., D, D)``. Each row ``components[..., i, :]`` represents
              the i-th principal component direction.
            - **explained_variance** (*Tensor*): Explained variance ratios of
              shape ``(..., D)`` corresponding to each principal component,
              ordered from largest to smallest.

    Example:
        >>> # 2D dataset with 3 samples
        >>> data = torch.tensor([[[1.0, 2.0],
        ...                       [3.0, 4.0],
        ...                       [5.0, 6.0]]])
        >>> result = QF.pca(data)
        >>> result.components.shape
        torch.Size([1, 2, 2])
        >>> result.explained_variance.shape
        torch.Size([1, 2])

        >>> # First principal component has highest variance
        >>> result.explained_variance[0, 0] > result.explained_variance[0, 1]
        tensor(True)

        >>> # Transform data to principal component space
        >>> transformed = torch.matmul(data - data.mean(dim=-2, keepdim=True),
        ...                           result.components.transpose(-1, -2))

    .. seealso::
        - :func:`pca_cov`: Compute PCA from a precomputed covariance matrix.

    .. note::
        In financial applications, dimensions often represent different assets
        or features, while samples represent time periods or observations.
    """
    return pca_cov(covar(x[..., None], x[..., None, :], dim=-3))


def pca_cov(cov: torch.Tensor) -> PcaResult:
    r"""Compute Principal Component Analysis (PCA) from a covariance matrix.

    This function performs PCA by applying singular value decomposition (SVD)
    directly to a precomputed covariance matrix. This is useful when you
    already have the covariance matrix computed or when working with
    pre-processed data.

    The principal components are ordered by their explained variance in
    descending order, corresponding to the eigenvalues of the covariance matrix.

    Args:
        cov (Tensor):
            Covariance matrix of shape ``(..., D, D)`` where ``D`` is the
            number of features/dimensions. The matrix should be symmetric
            and positive semi-definite. The batch dimensions ``...`` are
            preserved.

    Returns:
        PcaResult:
            A dataclass containing:

            - **components** (*Tensor*): Principal component vectors of shape
              ``(..., D, D)``. Each row ``components[..., i, :]`` represents
              the i-th principal component direction (eigenvector).
            - **explained_variance** (*Tensor*): Explained variance ratios of
              shape ``(..., D)`` corresponding to each principal component
              (eigenvalues), ordered from largest to smallest.

    Example:
        >>> # Create a 2x2 covariance matrix
        >>> cov_matrix = torch.tensor([[[2.0, 1.0],
        ...                             [1.0, 2.0]]])
        >>> result = QF.pca_cov(cov_matrix)
        >>> result.components.shape
        torch.Size([1, 2, 2])
        >>> result.explained_variance.shape
        torch.Size([1, 2])

        >>> # Principal components should be orthonormal
        >>> components = result.components[0]
        >>> torch.allclose(torch.matmul(components, components.T), torch.eye(2), atol=1e-6)
        True

        >>> # Eigenvalues should be in descending order
        >>> result.explained_variance[0, 0] >= result.explained_variance[0, 1]
        tensor(True)

    .. seealso::
        - :func:`pca`: Compute PCA directly from input data.
        - :func:`covar`: Compute covariance matrix from data.

    .. note::
        This function is particularly useful in financial applications where
        covariance matrices are frequently pre-computed for risk analysis.
    """
    batch_shape = cov.shape[:-2]
    _, s, v = torch.linalg.svd(cov.unsqueeze(0).flatten(end_dim=-3))
    return PcaResult(
        components=v.reshape(batch_shape + v.shape[1:]),
        explained_variance=s.reshape(batch_shape + s.shape[1:]),
    )
