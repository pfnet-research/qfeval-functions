import torch
import torch.nn.functional as F


def _coordinate_ranks(values: torch.Tensor) -> torch.Tensor:
    """Coordinate-compress every row of a two-dimensional tensor."""
    sorted_values, ordering = values.sort(dim=1)
    is_new = torch.ones_like(sorted_values, dtype=torch.bool)
    is_new[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    sorted_ranks = is_new.cumsum(dim=1) - 1
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(1, ordering, sorted_ranks)
    return ranks


def _stable_partition_destination(is_zero: torch.Tensor) -> torch.Tensor:
    """Return destinations that stably partition each row by a bit."""
    zero_position = is_zero.to(torch.long).cumsum(dim=1) - 1
    one_position = (~is_zero).to(torch.long).cumsum(dim=1) - 1
    zero_count = is_zero.sum(dim=1, keepdim=True)
    return torch.where(is_zero, zero_position, zero_count + one_position)


def _apply_partition(
    values: torch.Tensor, destination: torch.Tensor
) -> torch.Tensor:
    """Apply a row-wise stable-partition permutation."""
    partitioned = torch.empty_like(values)
    partitioned.scatter_(1, destination, values)
    return partitioned


def _window_has_nan(x: torch.Tensor, span: int) -> torch.Tensor:
    """Mark complete trailing windows containing one or more NaNs."""
    nan_prefix = F.pad(
        x.isnan().to(torch.long).cumsum(dim=1),
        (1, 0),
    )
    return (nan_prefix[:, span:] - nan_prefix[:, :-span]) != 0
