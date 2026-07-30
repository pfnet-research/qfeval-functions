import math

import numpy as np
import torch

import qfeval_functions.functions as QF

from .test_utils import assert_basic_properties


def test_nanprod_vs_numpy_random_data() -> None:
    """Compare nanprod with numpy.nanprod on data with scattered NaNs."""
    torch.manual_seed(1)
    x = torch.randn(5, 6, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.25] = math.nan
    # Ensure every row and column has at least one valid value so that
    # the all-NaN semantic difference from numpy does not apply here.
    x[:, 0] = 1.5
    x[0, :] = 0.5
    for dim in (None, 0, 1, (0, 1)):
        for keepdim in (False, True):
            actual = QF.nanprod(x, dim=dim, keepdim=keepdim)
            expected = np.nanprod(x.numpy(), axis=dim, keepdims=keepdim)
            np.testing.assert_allclose(actual.numpy(), expected)


def test_nanprod_all_nan_differs_from_numpy() -> None:
    """All-NaN slices yield NaN, unlike numpy.nanprod which yields 1."""
    x = torch.tensor([[2.0, 3.0], [math.nan, math.nan]], dtype=torch.float64)
    result = QF.nanprod(x, dim=1)
    torch.testing.assert_close(
        result[0], torch.tensor(6.0, dtype=torch.float64)
    )
    assert torch.isnan(result[1])
    # numpy returns the multiplicative identity instead.
    np.testing.assert_allclose(
        np.nanprod(x.numpy(), axis=1), np.array([6.0, 1.0])
    )
    # Full reduction over an all-NaN tensor.
    all_nan = torch.full((3,), math.nan, dtype=torch.float64)
    assert torch.isnan(QF.nanprod(all_nan))
    assert np.nanprod(all_nan.numpy()) == 1.0
    # keepdim keeps the NaN result with reduced shape.
    result_keepdim = QF.nanprod(x, dim=1, keepdim=True)
    assert result_keepdim.shape == (2, 1)
    assert torch.isnan(result_keepdim[1, 0])


def test_nanprod_no_nan_matches_torch_prod() -> None:
    """Verify nanprod matches torch.prod on NaN-free data."""
    torch.manual_seed(2)
    x = torch.randn(4, 5, dtype=torch.float64)
    for dim in (0, 1, -1):
        for keepdim in (False, True):
            torch.testing.assert_close(
                QF.nanprod(x, dim=dim, keepdim=keepdim),
                torch.prod(x, dim=dim, keepdim=keepdim),
            )
    torch.testing.assert_close(QF.nanprod(x), torch.prod(x))


def test_nanprod_skips_nan() -> None:
    """Test that NaN values are skipped entirely."""
    x = torch.tensor([2.0, math.nan, 3.0], dtype=torch.float64)
    torch.testing.assert_close(
        QF.nanprod(x), torch.tensor(6.0, dtype=torch.float64)
    )
    x2 = torch.tensor(
        [[1.0, math.nan, 3.0], [4.0, 5.0, math.nan]], dtype=torch.float64
    )
    torch.testing.assert_close(
        QF.nanprod(x2, dim=1), torch.tensor([3.0, 20.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        QF.nanprod(x2, dim=0),
        torch.tensor([4.0, 5.0, 3.0], dtype=torch.float64),
    )


def test_nanprod_zeros_and_signs() -> None:
    """Test zeros and sign correctness."""
    # Zero is a valid value and yields a zero product.
    x = torch.tensor([2.0, 0.0, math.nan, 3.0], dtype=torch.float64)
    torch.testing.assert_close(
        QF.nanprod(x), torch.tensor(0.0, dtype=torch.float64)
    )
    # An odd number of negative factors yields a negative product.
    x_odd = torch.tensor([-2.0, 3.0, math.nan, -4.0, -1.0])
    torch.testing.assert_close(QF.nanprod(x_odd), torch.tensor(-24.0))
    # An even number of negative factors yields a positive product.
    x_even = torch.tensor([-2.0, math.nan, -3.0])
    torch.testing.assert_close(QF.nanprod(x_even), torch.tensor(6.0))


def test_nanprod_with_infinity() -> None:
    """Test IEEE 754 behavior of infinities as valid values."""
    # 0 * inf within a slice is NaN under IEEE 754 arithmetic.
    x = torch.tensor([0.0, math.inf, 2.0], dtype=torch.float64)
    assert torch.isnan(QF.nanprod(x))
    # Infinity alone keeps its sign.
    x_pos = torch.tensor([2.0, math.inf, math.nan], dtype=torch.float64)
    assert QF.nanprod(x_pos) == math.inf
    x_neg = torch.tensor([-2.0, math.inf, math.nan], dtype=torch.float64)
    assert QF.nanprod(x_neg) == -math.inf


def test_nanprod_keepdim_shapes() -> None:
    """Test output shapes with and without keepdim."""
    x = torch.randn(2, 3, 4, dtype=torch.float64)
    assert QF.nanprod(x, dim=1).shape == (2, 4)
    assert QF.nanprod(x, dim=1, keepdim=True).shape == (2, 1, 4)
    assert QF.nanprod(x, dim=(0, 2)).shape == (3,)
    assert QF.nanprod(x, dim=(0, 2), keepdim=True).shape == (1, 3, 1)
    assert QF.nanprod(x).shape == ()
    assert QF.nanprod(x, keepdim=True).shape == (1, 1, 1)


def test_nanprod_tuple_negative_dims() -> None:
    """Test tuples containing negative dimension indices on a 3D tensor."""
    torch.manual_seed(3)
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    x[torch.rand(x.shape) < 0.2] = math.nan
    result = QF.nanprod(x, dim=(-1, -2))
    torch.testing.assert_close(
        result, QF.nanprod(x, dim=(1, 2)), equal_nan=True
    )
    # Should be equivalent to flattening those dimensions.
    torch.testing.assert_close(
        result, QF.nanprod(x.reshape(3, -1), dim=1), equal_nan=True
    )


def test_nanprod_scalar_output() -> None:
    """Test that dim=None without keepdim returns a scalar tensor."""
    x = torch.tensor([[2.0, math.nan], [3.0, 4.0]], dtype=torch.float64)
    result = QF.nanprod(x)
    assert result.shape == ()
    torch.testing.assert_close(result, torch.tensor(24.0, dtype=torch.float64))


def test_nanprod_single_element() -> None:
    """Test single-element tensors."""
    x = torch.tensor([7.0], dtype=torch.float64)
    torch.testing.assert_close(
        QF.nanprod(x, dim=0), torch.tensor(7.0, dtype=torch.float64)
    )
    # A single NaN has no valid values, so the result is NaN.
    x_nan = torch.tensor([math.nan], dtype=torch.float64)
    assert torch.isnan(QF.nanprod(x_nan, dim=0))


def test_nanprod_dtype_preservation() -> None:
    """Test that nanprod preserves dtype and device."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[2.0, math.nan, 3.0]], dtype=dtype)
        result = QF.nanprod(x, dim=1, keepdim=True)
        assert_basic_properties(result, x, expected_shape=torch.Size([1, 1]))
        torch.testing.assert_close(result, torch.tensor([[6.0]], dtype=dtype))
