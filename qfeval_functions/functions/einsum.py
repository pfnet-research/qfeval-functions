import torch


def einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    r"""Typed version of torch.einsum.

    As of PyTorch 1.8, torch.einsum is not typed, and it is inconvenient in
    static type analysis.
    """
    return torch.einsum(equation, *operands)  # type: ignore
