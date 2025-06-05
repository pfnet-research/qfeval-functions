import torch


def ffill(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Propagates last valid values forward."""
    if x.shape[dim] == 0:
        return x
    x = x.transpose(0, dim)
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    indices = torch.cummax(~torch.isnan(x), dim=0).indices
    x = x[indices, torch.arange(x.shape[1], device=x.device)[None]]
    return x.reshape(shape).transpose(0, dim)
