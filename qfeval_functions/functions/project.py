import torch


def project(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Project tensor using a projection matrix.

    This function applies a linear projection to the input tensor :attr:`x`
    using the projection matrix :attr:`a`. The projection is computed via
    matrix multiplication, transforming data from the input dimension space
    to the output dimension space.

    The mathematical operation performed is:

    .. math::
        \text{output} = x \cdot a^T

    where the projection matrix :attr:`a` transforms vectors from input
    dimension :math:`I` to output dimension :math:`O`.

    Args:
        a (Tensor):
            Projection matrix of shape ``(..., O, I)`` where ``O`` is the
            number of output dimensions and ``I`` is the number of input
            dimensions. The batch dimensions ``...`` must be broadcastable
            with the batch dimensions of :attr:`x`.
        x (Tensor):
            Input tensor to be projected of shape ``(..., S, I)`` where
            ``S`` is the number of samples/sections and ``I`` is the number
            of input dimensions. The batch dimensions ``...`` must be
            broadcastable with the batch dimensions of :attr:`a`.

    Returns:
        Tensor:
            Projected tensor of shape ``(..., S, O)`` where the input
            dimensions have been transformed to output dimensions.

    Raises:
        ValueError:
            If the last dimension of :attr:`a` does not match the last
            dimension of :attr:`x` (i.e., ``a.shape[-1] != x.shape[-1]``).
        ValueError:
            If the batch dimensions of :attr:`a` and :attr:`x` are not
            broadcastable.

    Example:
        >>> # Simple 2D to 1D projection
        >>> projection_matrix = torch.tensor([[1.0, 0.5]])  # Shape: (1, 2)
        >>> data = torch.tensor([[1.0, 2.0],
        ...                      [3.0, 4.0]])               # Shape: (2, 2)
        >>> result = QF.project(projection_matrix, data)
        >>> result.shape
        torch.Size([2, 1])
        >>> result
        tensor([[2.0],
                [5.0]])

        >>> # Batch projection with multiple samples
        >>> batch_proj = torch.tensor([[[1.0, 0.0],
        ...                             [0.0, 1.0]],
        ...                            [[1.0, 1.0],
        ...                             [1.0, -1.0]]])      # Shape: (2, 2, 2)
        >>> batch_data = torch.tensor([[[1.0, 2.0],
        ...                             [3.0, 4.0]],
        ...                            [[1.0, 1.0],
        ...                             [2.0, 2.0]]])       # Shape: (2, 2, 2)
        >>> QF.project(batch_proj, batch_data).shape
        torch.Size([2, 2, 2])

        >>> # Dimensionality reduction example
        >>> pca_matrix = torch.tensor([[0.7071, 0.7071],   # First principal component
        ...                           [0.7071, -0.7071]])  # Second principal component
        >>> original_data = torch.tensor([[1.0, 1.0],
        ...                               [2.0, 2.0],
        ...                               [3.0, 1.0]])
        >>> projected = QF.project(pca_matrix, original_data)

    .. note::
        In quantitative finance applications, this function is commonly used
        for:

        - Factor model projections where dimensions represent assets and
          output dimensions represent risk factors
        - Principal component analysis (PCA) transformations
        - Portfolio optimization with constraint matrices
        - Cross-sectional analysis across different time sections
    """

    if a.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"The last dimension must match: f{a.shape} vs f{x.shape}"
        )

    # Calculate the result's batch shape.
    try:
        shape = (
            torch.zeros(a.shape[:-2] + (0,)) + torch.zeros(x.shape[:-2] + (0,))
        ).shape[:-1]
    except RuntimeError:
        raise ValueError(f"Incompatible batch shape: f{a.shape} vs f{x.shape}")

    x = x.expand(shape + (-1, -1)).reshape((-1,) + x.shape[-2:])
    a = a.expand(shape + (-1, -1)).reshape((-1,) + a.shape[-2:])
    result = torch.bmm(a, x.transpose(-1, -2)).transpose(-1, -2)
    return result.reshape(shape + result.shape[-2:])
