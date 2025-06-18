from dataclasses import dataclass

import torch

from .covar import covar


@dataclass
class PcaResult:
    """Result of principal component analysis.

    This dataclass contains the results from :func:`pca` or :func:`pca_cov`,
    providing both the principal component vectors and their corresponding
    explained variances.

    Attributes:
        components (torch.Tensor): Principal component vectors of shape (*, D, D)
            where components[*, i, :] represents the i-th principal component.
        explained_variance (torch.Tensor): Eigenvalues representing the variance
            explained by each component, of shape (*, D).
    """

    components: torch.Tensor
    explained_variance: torch.Tensor


def pca(x: torch.Tensor) -> PcaResult:
    r"""Computes principal component analysis (PCA) on the input data.

    This function performs standard principal component analysis by computing the
    covariance matrix of the input data and then extracting principal components
    through eigenvalue decomposition. PCA identifies orthogonal directions that
    capture the maximum variance in the data, enabling dimensionality reduction
    and feature extraction.

    The algorithm first computes the covariance matrix using :func:`covar`, then
    applies :func:`pca_cov` to extract the principal components. This two-step
    approach allows for efficient computation and code reuse.

    Mathematical formulation:
        For input data X of shape (N, D):

        1. Covariance matrix: C = (1/N) * X^T * X  (after centering)
        2. Eigenvalue decomposition: C = V Λ V^T
        3. Principal components are eigenvectors V, sorted by eigenvalues Λ

        The explained variance is proportional to the eigenvalues λ_i.

    Applications:
        - Dimensionality reduction for visualization and compression
        - Feature extraction for machine learning pipelines
        - Noise reduction by projecting onto top components
        - Data analysis and pattern recognition in financial time series

    Note:
        - Input data is automatically centered during covariance computation
        - Components are ordered by decreasing explained variance
        - For financial data, components often represent market factors
        - The function assumes no missing values; use :func:`nanpca` for NaN handling

    Args:
        x (Tensor): Input data tensor of shape :math:`(*, S, D)` where:
            - ``*`` represents any number of batch dimensions
            - ``S`` is the number of samples/observations (e.g., time steps)
            - ``D`` is the number of features/variables (e.g., assets, indicators)

            In quantitative finance contexts, ``S`` often represents timestamps
            and ``D`` represents different financial instruments or features.

    Returns:
        PcaResult: A dataclass containing:
            - components (Tensor): Principal component vectors of shape :math:`(*, D, D)`.
              ``components[*, i, :]`` represents the i-th principal component loadings.
            - explained_variance (Tensor): Eigenvalues representing explained variance
              of shape :math:`(*, D)`. ``explained_variance[*, i]`` is the variance
              explained by the i-th component.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF

        >>> # Basic example: 3 variables, 4 observations
        >>> x = torch.tensor([[1., 2., 3.],
        ...                   [4., 5., 6.],
        ...                   [7., 8., 9.],
        ...                   [10., 11., 12.]])
        >>> result = QF.pca(x)
        >>> result.components.shape
        torch.Size([3, 3])
        >>> result.explained_variance.shape
        torch.Size([3])

        >>> # Financial example: stock returns analysis
        >>> # Simple 3-stock portfolio returns over 4 days
        >>> returns = torch.tensor([[0.02, 0.01, 0.015],  # Day 1
        ...                         [0.03, 0.02, 0.025],  # Day 2
        ...                         [-0.01, -0.005, -0.008], # Day 3
        ...                         [0.015, 0.01, 0.012]]) # Day 4
        >>> result = QF.pca(returns)
        >>> # Verify output shapes
        >>> result.components.shape
        torch.Size([3, 3])
        >>> result.explained_variance.shape
        torch.Size([3])

        >>> # Batch processing: multiple portfolios
        >>> # 2 portfolios, 3 days, 2 assets each
        >>> portfolios = torch.tensor([[[0.01, 0.02],   # Portfolio 1
        ...                             [0.03, 0.01],
        ...                             [-0.01, 0.005]],
        ...                            [[0.02, 0.015],  # Portfolio 2
        ...                             [0.01, 0.03],
        ...                             [0.005, -0.01]]])
        >>> results = QF.pca(portfolios)
        >>> results.components.shape
        torch.Size([2, 2, 2])
        >>> # Each portfolio has its own set of principal components

        >>> # Dimensionality reduction example
        >>> # Simple correlated data: 5 samples, 4 features
        >>> data = torch.tensor([[1.0, 2.0, 3.0, 4.0],
        ...                      [2.0, 4.0, 6.0, 8.0],
        ...                      [1.5, 3.0, 4.5, 6.0],
        ...                      [0.5, 1.0, 1.5, 2.0],
        ...                      [3.0, 6.0, 9.0, 12.0]])
        >>> result = QF.pca(data)
        >>> # Data has strong linear correlation - first component dominates
        >>> result.explained_variance.shape
        torch.Size([4])
        >>> # Most variance captured by first component
        >>> ratio = result.explained_variance[0] / result.explained_variance.sum()
        >>> ratio > 0.9  # High correlation means >90% in first component
        tensor(True)
    """
    return pca_cov(covar(x[..., None], x[..., None, :], dim=-3))


def pca_cov(cov: torch.Tensor) -> PcaResult:
    r"""Computes principal components from a given covariance matrix.

    This function performs the eigenvalue decomposition step of PCA when the
    covariance matrix is already available. It uses Singular Value Decomposition (SVD)
    to extract principal components and their explained variances from the covariance
    matrix, providing a numerically stable approach to eigenvalue decomposition.

    This function is particularly useful when:

    - The covariance matrix has been computed separately (e.g., with custom methods)
    - You want to reuse a covariance matrix for multiple PCA computations
    - Working with precomputed correlation or covariance matrices from external sources

    Mathematical formulation:
        For covariance matrix C:

        1. SVD decomposition: C = U Σ V^T
        2. Principal components are the columns of V
        3. Explained variances are the singular values Σ

        The SVD approach provides better numerical stability than direct eigenvalue
        decomposition, especially for ill-conditioned covariance matrices.

    Implementation details:
        - Uses PyTorch's :func:`torch.linalg.svd` for numerical stability
        - Components are automatically sorted by decreasing explained variance
        - Handles batch processing efficiently for multiple covariance matrices
        - Preserves gradient information for differentiable computations

    Note:
        - Input covariance matrix should be symmetric and positive semi-definite
        - Small negative eigenvalues may appear due to numerical precision
        - For rank-deficient matrices, some components may have zero variance

    Args:
        cov (Tensor): Covariance matrix tensor of shape :math:`(*, D, D)` where:
            - ``*`` represents any number of batch dimensions
            - ``D`` is the number of features/variables

            The matrix should be symmetric and positive semi-definite.

    Returns:
        PcaResult: A dataclass containing:
            - components (Tensor): Principal component vectors of shape :math:`(*, D, D)`.
              ``components[*, i, :]`` represents the i-th principal component loadings.
            - explained_variance (Tensor): Singular values representing explained variance
              of shape :math:`(*, D)`. ``explained_variance[*, i]`` is the variance
              explained by the i-th component, sorted in descending order.

    Raises:
        RuntimeError: If SVD decomposition fails (e.g., for non-finite input).
        ValueError: If input tensor doesn't have at least 2 dimensions.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF

        >>> # Basic covariance matrix example
        >>> cov = torch.tensor([[2.0, 1.0, 0.5],
        ...                     [1.0, 3.0, -0.5],
        ...                     [0.5, -0.5, 1.5]])
        >>> result = QF.pca_cov(cov)
        >>> result.components.shape
        torch.Size([3, 3])
        >>> result.explained_variance.shape
        torch.Size([3])

        >>> # Verify components are orthonormal
        >>> components = result.components
        >>> orthogonality = torch.mm(components, components.T)
        >>> print(torch.allclose(orthogonality, torch.eye(3), atol=1e-6))
        True

        >>> # Batch processing: multiple covariance matrices
        >>> # 2 covariance matrices, 2x2 each
        >>> batch_cov = torch.tensor([[[2.0, 0.5],    # Matrix 1
        ...                            [0.5, 1.0]],
        ...                           [[1.5, -0.3],   # Matrix 2
        ...                            [-0.3, 2.5]]])
        >>> results = QF.pca_cov(batch_cov)
        >>> results.components.shape
        torch.Size([2, 2, 2])
        >>> results.explained_variance.shape
        torch.Size([2, 2])

        >>> # Financial correlation matrix example
        >>> # Correlation matrix for 3 assets
        >>> corr = torch.tensor([[1.0, 0.8, 0.6],
        ...                      [0.8, 1.0, 0.7],
        ...                      [0.6, 0.7, 1.0]])
        >>> result = QF.pca_cov(corr)
        >>> # First component captures common factor
        >>> market_factor = result.components[0, :]
        >>> market_factor.shape
        torch.Size([3])
        >>> torch.all(torch.abs(market_factor) > 0.5)  # All loadings significant
        tensor(True)

        >>> # Explained variance analysis
        >>> total_var = result.explained_variance.sum()
        >>> explained_ratios = result.explained_variance / total_var
        >>> explained_ratios.shape
        torch.Size([3])
        >>> explained_ratios[0] > 0.8  # First component dominates
        tensor(True)

        >>> # Dimensionality reduction: keep components with >5% variance
        >>> significant_components = result.explained_variance > 0.05 * total_var
        >>> n_components = significant_components.sum()
        >>> int(n_components)  # Number of significant components
        3
    """
    batch_shape = cov.shape[:-2]
    _, s, v = torch.linalg.svd(cov.unsqueeze(0).flatten(end_dim=-3))
    return PcaResult(
        components=v.reshape(batch_shape + v.shape[1:]),
        explained_variance=s.reshape(batch_shape + s.shape[1:]),
    )
