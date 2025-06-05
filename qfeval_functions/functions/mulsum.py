import string
import typing

import numpy as np
import torch

from .einsum import einsum


def mulsum(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: typing.Union[int, typing.Tuple[int, ...]] = (),
    keepdim: bool = False,
    mean: bool = False,
    *,
    _ddof: int = 0,
) -> torch.Tensor:
    r"""Calculate `(x * y).sum(...)` in a memory-efficient way."""

    # 1. Align the number of dimensions (c.f., NumPy broadcasting).
    length = max(len(x.shape), len(y.shape))
    x = x.reshape((1,) * (length - len(x.shape)) + x.shape)
    y = y.reshape((1,) * (length - len(y.shape)) + y.shape)

    # 2. Figure out the final shape.
    # NOTE: x[None][:0] is a trick to call a torch function without wasting
    # cpu/memory resources.
    mul_shape = (x[None][:0] * y[None][:0]).shape[1:]

    # 3. Parse dim.
    # `mask` should be a tuple of ints, each of which should represent whether
    # the dimension is aggregated (1) or not (0).
    mask = torch.sum(torch.zeros((0,) * length), dim=dim, keepdim=True).shape

    # 4. Prepare einsum parameters and apply them to einsum.
    input_eq = string.ascii_lowercase[: len(x.shape)]
    result_eq = "".join(c for c, m in zip(input_eq, mask) if m == 0)
    result = einsum(f"{input_eq},{input_eq}->{result_eq}", x, y)

    # 5. (If keepdim is enabled,) Restore aggregated dimensions.
    if keepdim:
        result = result.reshape(
            tuple(1 if m == 1 else s for s, m in zip(mul_shape, mask))
        )

    # 6. (If mean is enabled,) Divide the result by the number of aggregated
    # elements.
    if mean:
        result = result / (
            int(np.prod([s if m == 1 else 1 for s, m in zip(mul_shape, mask)]))
            - _ddof
        )

    return result
