import torch


def project(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Projects the given tensor `x` using the given projection matrix `a`.

    Args:
        a (Tensor): A projection matrix.
        x (Tensor): A tensor to be projected.

    Returns
        y (Tensor): The projected tensor.

    Shape:
        - a: :math:`(*, O, I)`, where :math:`*` represents any number of
            dimensions (including None), :math:`O` is the number of output
            dimensions, and :math:`I` is the number of input dimensions.
        - x: :math:`(*, S, I)`, where :math:`*` represents any number of
            dimensions (including None), :math:`S` is the number of sections,
            and :math:`I` is the number of input dimensions.
        - return: :math:`(*, S, O)`, where :math:`*` represents any number of
            dimensions (including None), :math:`S` is the number of sections,
            and :math:`O` is the number of output dimensions.

    In qfeval, I/O dimensions and sections often represent symbols and
    timestamps respectively.
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
