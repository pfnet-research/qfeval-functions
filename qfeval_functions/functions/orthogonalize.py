import torch


def orthogonalize(
    x: torch.Tensor, y: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Orthogonalizes x with respect to y along the specified dimension.

    Args:
        x (torch.Tensor): The tensor to be orthogonalized.
        y (torch.Tensor): The tensor with respect to which x will be
            orthogonalized.
        dim (int): The dimension along which the orthogonalization will be
            performed.  Default is -1.

    Returns:
        x (torch.Tensor): The orthogonalized tensor.
    """
    # Calculate the dot product of x and y along the specified dimension.
    dot_product = (x * y).sum(dim=dim, keepdim=True)

    # Compute the projection of x onto y.
    projection = dot_product * y / y.square().sum(dim=dim, keepdim=True)

    # Subtract the projection from x to obtain the orthogonalized tensor.
    return x - projection
