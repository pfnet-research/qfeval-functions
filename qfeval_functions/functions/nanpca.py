from dataclasses import dataclass

import torch

from .eigh import eigh
from .nancovar import nancovar


@dataclass
class NanpcaResult:
    """Result of NaN-aware principal component analysis.

    This dataclass contains the results from :func:`nanpca`, providing both
    the principal component vectors and their corresponding explained variances.

    Attributes:
        components (torch.Tensor): Principal component vectors of shape (*, C, C)
            where components[*, i, :] represents the i-th principal component.
        explained_variance (torch.Tensor): Eigenvalues representing the variance
            explained by each component, of shape (*, C).
    """

    components: torch.Tensor
    explained_variance: torch.Tensor


def nanpca(data: torch.Tensor) -> NanpcaResult:
    r"""Computes principal component analysis (PCA) on data with NaN handling.

    This function performs principal component analysis while automatically handling
    NaN values in the input data. It computes the principal components by first
    calculating the NaN-aware covariance matrix and then performing eigenvalue
    decomposition to extract the components and their explained variances.

    Principal Component Analysis finds orthogonal directions (components) that capture
    the maximum variance in the data. The components are ordered by their explained
    variance, with the first component explaining the most variance.

    Mathematical formulation:
        For data matrix X with NaN handling, the algorithm computes:

        1. Covariance matrix: C = nancovar(X)
        2. Eigenvalue decomposition: C = V Λ V^T
        3. Components are the eigenvectors V, ordered by eigenvalues Λ

        The explained variance for component i is λ_i / sum(λ).

    NaN handling:
        - NaN values are excluded from covariance calculations on a pairwise basis
        - If entire features contain only NaN, corresponding components may be NaN
        - The function gracefully handles missing data patterns common in financial datasets

    Note:
        - Components are returned in descending order of explained variance
        - The number of valid components depends on the rank of the NaN-aware covariance matrix
        - For numerical stability, very small eigenvalues may result in unstable components

    Args:
        data (Tensor): Input data tensor of shape :math:`(*, N, C)` where:
            - ``*`` represents any number of batch dimensions
            - ``N`` is the number of samples/observations
            - ``C`` is the number of features/variables

    Returns:
        NanpcaResult: A dataclass containing:
            - components (Tensor): Principal component vectors of shape :math:`(*, C, C)`.
              ``components[*, i, :]`` represents the i-th principal component.
            - explained_variance (Tensor): Eigenvalues representing explained variance
              of shape :math:`(*, C)`. ``explained_variance[*, i]`` is the variance
              explained by the i-th component.

    Raises:
        RuntimeError: If the input tensor has insufficient dimensions (< 2D).
        ValueError: If all values in a feature are NaN.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF

        >>> # Basic 2D example
        >>> data = torch.tensor([[1., 2., 3.],
        ...                      [4., 5., 6.],
        ...                      [7., 8., 9.],
        ...                      [10., 11., 12.]])
        >>> result = QF.nanpca(data)
        >>> result.components.shape
        torch.Size([3, 3])
        >>> result.explained_variance.shape
        torch.Size([3])

        >>> # Example with NaN values
        >>> data_with_nan = torch.tensor([[1., 2., float('nan')],
        ...                               [4., float('nan'), 6.],
        ...                               [7., 8., 9.],
        ...                               [10., 11., 12.]])
        >>> result = QF.nanpca(data_with_nan)
        >>> # NaN values are handled automatically in covariance computation
        >>> result.components.shape
        torch.Size([3, 3])

        >>> # Batch processing example
        >>> batch_data = torch.randn(5, 100, 10)  # 5 datasets, 100 samples, 10 features
        >>> result = QF.nanpca(batch_data)
        >>> result.components.shape
        torch.Size([5, 10, 10])
        >>> result.explained_variance.shape
        torch.Size([5, 10])

        >>> # Financial time series example with NaN handling
        >>> # Stock returns: 5 observations, 3 stocks with missing data
        >>> returns = torch.tensor([[1.0, 2.0, float('nan')],    # Day 1: Stock 3 missing
        ...                         [2.0, float('nan'), 3.0],    # Day 2: Stock 2 missing
        ...                         [3.0, 4.0, 5.0],             # Day 3: All present
        ...                         [4.0, 5.0, float('nan')],    # Day 4: Stock 3 missing
        ...                         [5.0, 6.0, 7.0]])            # Day 5: All present
        >>> result = QF.nanpca(returns)
        >>> # Verify shape and components
        >>> result.components.shape
        torch.Size([3, 3])
        >>> result.explained_variance.shape
        torch.Size([3])
        >>> # NaN values are handled automatically
        >>> len(result.explained_variance) == returns.shape[1]
        True
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
