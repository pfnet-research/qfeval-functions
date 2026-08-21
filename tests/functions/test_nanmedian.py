import math
from typing import Tuple

import numpy as np
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def _random_with_nans(
    shape: Tuple[int, ...], seed: int, nan_ratio: float = 0.3
) -> torch.Tensor:
    """Return a float64 tensor with randomly placed NaN values."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    x[torch.rand(shape, generator=generator) < nan_ratio] = math.nan
    return x


def test_nanmedian_odd_counts_match_numpy() -> None:
    """With odd numbers of valid values, the result matches numpy."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(40, 9, generator=generator, dtype=torch.float64)
    # Remove an even number of values per row so that the number of
    # valid values stays odd.
    for i in range(x.shape[0]):
        k = 2 * int(torch.randint(0, 4, (1,), generator=generator).item())
        if k > 0:
            x[i, :k] = math.nan
    np.testing.assert_allclose(
        QF.nanmedian(x, dim=1).numpy(), np.nanmedian(x.numpy(), axis=1)
    )


def test_nanmedian_even_count_returns_lower_middle() -> None:
    """With an even count, the lower middle value is returned, unlike
    numpy.nanmedian, which averages the two middle values."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert QF.nanmedian(x).item() == 2.0
    assert np.nanmedian(x.numpy()) == 2.5  # numpy averages instead

    x = torch.tensor(
        [7.0, math.nan, 1.0, 5.0, 3.0, math.nan], dtype=torch.float64
    )
    assert QF.nanmedian(x).item() == 3.0
    assert np.nanmedian(x.numpy()) == 4.0  # numpy averages instead
    # The averaging behavior is available via nanquantile.
    assert QF.nanquantile(x, 0.5).item() == 4.0

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        QF.nanmedian(x, dim=1),
        torch.tensor([2.0, 20.0], dtype=torch.float64),
    )


def test_nanmedian_result_is_an_input_element() -> None:
    """The result is always one of the valid input values."""
    x = _random_with_nans((30, 8), seed=1, nan_ratio=0.4)
    x[0] = math.nan
    result = QF.nanmedian(x, dim=1)
    for i in range(x.shape[0]):
        valid = x[i][~x[i].isnan()]
        if valid.numel() == 0:
            assert math.isnan(result[i].item())
        else:
            assert bool((valid == result[i]).any())


def test_nanmedian_matches_nanquantile_lower() -> None:
    """nanmedian equals nanquantile(0.5, interpolation="lower")."""
    x = _random_with_nans((25, 7), seed=2)
    x[0] = math.nan
    torch.testing.assert_close(
        QF.nanmedian(x, dim=1),
        QF.nanquantile(x, 0.5, dim=1, interpolation="lower"),
        equal_nan=True,
    )
    torch.testing.assert_close(
        QF.nanmedian(x),
        QF.nanquantile(x, 0.5, interpolation="lower"),
        equal_nan=True,
    )
    x3 = _random_with_nans((4, 5, 6), seed=3)
    torch.testing.assert_close(
        QF.nanmedian(x3, dim=(0, 2), keepdim=True),
        QF.nanquantile(
            x3, 0.5, dim=(0, 2), keepdim=True, interpolation="lower"
        ),
        equal_nan=True,
    )


def test_nanmedian_all_nan_slice() -> None:
    """A slice consisting only of NaN values must yield NaN."""
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [math.nan, math.nan, math.nan]],
        dtype=torch.float64,
    )
    result = QF.nanmedian(x, dim=1)
    assert result[0].item() == 2.0
    assert math.isnan(result[1].item())

    all_nan = torch.tensor([math.nan, math.nan])
    assert math.isnan(QF.nanmedian(all_nan).item())

    x3 = torch.full((2, 3, 4), math.nan, dtype=torch.float64)
    x3[0] = 5.0
    result3 = QF.nanmedian(x3, dim=(1, 2))
    assert result3[0].item() == 5.0
    assert math.isnan(result3[1].item())


def test_nanmedian_dim_and_keepdim_shapes() -> None:
    """Check output shapes for every form of the dim argument."""
    x = torch.randn(2, 3, 4)
    assert QF.nanmedian(x).shape == ()
    assert QF.nanmedian(x, keepdim=True).shape == (1, 1, 1)
    assert QF.nanmedian(x, dim=0).shape == (3, 4)
    assert QF.nanmedian(x, dim=1, keepdim=True).shape == (2, 1, 4)
    assert QF.nanmedian(x, dim=-1).shape == (2, 3)
    assert QF.nanmedian(x, dim=-2, keepdim=True).shape == (2, 1, 4)
    assert QF.nanmedian(x, dim=(0, 2)).shape == (3,)
    assert QF.nanmedian(x, dim=(0, 2), keepdim=True).shape == (1, 3, 1)
    assert QF.nanmedian(x, dim=(-1, 0), keepdim=True).shape == (1, 3, 1)
    assert QF.nanmedian(x, dim=(0, 1, 2)).shape == ()
    assert QF.nanmedian(x, dim=(0, 1, 2), keepdim=True).shape == (1, 1, 1)


def test_nanmedian_negative_dim() -> None:
    """Negative dimensions must match their positive counterparts."""
    x = _random_with_nans((4, 5, 6), seed=4)
    torch.testing.assert_close(
        QF.nanmedian(x, dim=-1), QF.nanmedian(x, dim=2), equal_nan=True
    )
    torch.testing.assert_close(
        QF.nanmedian(x, dim=(-1, -3)),
        QF.nanmedian(x, dim=(0, 2)),
        equal_nan=True,
    )


def test_nanmedian_tuple_dim_equals_flattened() -> None:
    """Tuple-dim reduction equals reducing the flattened dimensions."""
    x = _random_with_nans((3, 4, 5), seed=5)
    torch.testing.assert_close(
        QF.nanmedian(x, dim=(1, 2)),
        QF.nanmedian(x.reshape(3, -1), dim=1),
        equal_nan=True,
    )
    # Non-adjacent dimensions require permuting before flattening.
    torch.testing.assert_close(
        QF.nanmedian(x, dim=(0, 2)),
        QF.nanmedian(x.permute(1, 0, 2).reshape(4, -1), dim=1),
        equal_nan=True,
    )


def test_nanmedian_dtype_preservation() -> None:
    """The result must preserve float32/float64 dtypes and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(3, 5, dtype=dtype)
        x[0, 0] = math.nan
        result = QF.nanmedian(x, dim=1)
        assert_basic_properties(result, x, expected_shape=torch.Size([3]))
        assert QF.nanmedian(x).dtype == dtype
        assert QF.nanmedian(x, dim=(0, 1)).dtype == dtype


def test_nanmedian_with_infinity() -> None:
    """Infinite values are ordered normally; no arithmetic is done."""
    x = torch.tensor([1.0, math.inf, 2.0])
    assert QF.nanmedian(x).item() == 2.0
    x = torch.tensor([-math.inf, 1.0, math.inf])
    assert QF.nanmedian(x).item() == 1.0
    # Even counts return the lower middle value, even if the upper
    # middle value is infinite.
    x = torch.tensor([1.0, 2.0, math.inf, math.inf])
    assert QF.nanmedian(x).item() == 2.0
    x = torch.tensor([-math.inf, -math.inf, 5.0, math.nan])
    assert QF.nanmedian(x).item() == -math.inf
