import math

import numpy as np
import pytest
import torch

import qfeval_functions
import qfeval_functions.functions as QF


class SoftTopKBottomK(torch.nn.Module):
    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        max_iter: int = 200,
        topk_only: bool = False,
    ):
        assert epsilon > 0

        super().__init__()  # type:ignore[no-untyped-call]
        self.k = k
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.topk_only = topk_only

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return QF.soft_topk_bottomk(
            scores,
            self.k,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            topk_only=self.topk_only,
        )


class SoftTopK(SoftTopKBottomK):

    def __init__(
        self,
        k: int,
        epsilon: float = 0.1,
        max_iter: int = 200,
        topk_only: bool = False,
    ):
        super().__init__(k, epsilon, max_iter, topk_only=True)


def test_soft_topk() -> None:
    """Asserts consistency with `SoftTopK`."""
    qfeval_functions.random.seed()
    k = 5
    epsilon = 0.1

    # One dimensional cases.
    x = QF.randn(50)

    actual = QF.soft_topk(x, dim=0, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(x.reshape(1, -1))[0]
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-4)

    # Two dimensional cases.
    x = QF.randn(100, 50)

    actual = QF.soft_topk(x, dim=1, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(x)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), k, atol=1e-4)

    actual = QF.soft_topk(x, dim=0, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(x.t()).t()
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-4)

    # Three dimensional cases.
    x = QF.randn(100, 50, 20)

    actual = QF.soft_topk(x, dim=2, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(
        x.reshape(-1, x.shape[2])
    ).reshape(x.shape)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=2).numpy(), k, atol=1e-4)

    actual = QF.soft_topk(x, dim=1, k=k, epsilon=epsilon)
    input = x.transpose(1, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(1, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), k, atol=1e-4)

    actual = QF.soft_topk(x, dim=0, k=k, epsilon=epsilon)
    input = x.transpose(0, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(0, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-4)


def test_soft_bottom_topk() -> None:
    """Asserts consistency with `qfeval.extension.SoftBottomTopK`."""
    qfeval_functions.random.seed()

    k = 5
    epsilon = 0.1

    # One dimensional cases.
    x = QF.randn(50)

    actual = QF.soft_topk_bottomk(x, dim=0, k=k, epsilon=epsilon)
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(x.reshape(1, -1))[0]
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), 0, atol=1e-5)

    # Two dimensional cases.
    x = QF.randn(100, 50)

    actual = QF.soft_topk_bottomk(x, dim=1, k=k, epsilon=epsilon)
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(x)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), 0, atol=1e-5)

    actual = QF.soft_topk_bottomk(x, dim=0, k=k, epsilon=epsilon)
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(x.t()).t()
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), 0, atol=1e-5)

    # Three dimensional cases.
    x = QF.randn(100, 50, 20)

    actual = QF.soft_topk_bottomk(x, dim=2, k=k, epsilon=epsilon)
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(
        x.reshape(-1, x.shape[2])
    ).reshape(x.shape)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=2).numpy(), 0, atol=1e-5)

    actual = QF.soft_topk_bottomk(x, dim=1, k=k, epsilon=epsilon)
    input = x.transpose(1, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(1, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), 0, atol=1e-5)

    actual = QF.soft_topk_bottomk(x, dim=0, k=k, epsilon=epsilon)
    input = x.transpose(0, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopKBottomK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(0, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), 0, atol=1e-5)


def test_soft_topk_basic_functionality() -> None:
    """Test basic soft_topk functionality."""
    qfeval_functions.random.seed()

    # Simple 1D case
    x = torch.tensor([1.0, 5.0, 3.0, 8.0, 2.0], dtype=torch.float64)
    k = 2
    result = QF.soft_topk(x, k=k, dim=0, epsilon=0.1)

    # Result should be non-negative and sum to k
    assert torch.all(result >= 0)
    torch.testing.assert_close(
        result.sum(), torch.tensor(k, dtype=torch.float64), atol=1e-4, rtol=1e-4
    )

    # Larger values should have higher weights
    sorted_indices = torch.argsort(x, descending=True)
    assert result[sorted_indices[0]] > result[sorted_indices[-1]]


def test_soft_topk_bottomk_basic_functionality() -> None:
    """Test basic soft_topk_bottomk functionality."""
    qfeval_functions.random.seed()

    # Simple 1D case
    x = torch.tensor([1.0, 5.0, 3.0, 8.0, 2.0], dtype=torch.float64)
    k = 2
    result = QF.soft_topk_bottomk(x, k=k, dim=0, epsilon=0.1)

    # Result should sum to approximately 0
    torch.testing.assert_close(
        result.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )

    # Check that it's consistent with topk_only=False flag
    result_explicit = QF.soft_topk_bottomk(
        x, k=k, dim=0, epsilon=0.1, topk_only=False
    )
    torch.testing.assert_close(result, result_explicit)


@pytest.mark.random
def test_soft_topk_shape_preservation() -> None:
    """Test that soft_topk preserves tensor shape."""
    # 2D tensor
    x_2d = torch.randn(3, 5, dtype=torch.float64)
    k = 2

    result_dim0 = QF.soft_topk(x_2d, k=k, dim=0)
    assert result_dim0.shape == x_2d.shape

    result_dim1 = QF.soft_topk(x_2d, k=k, dim=1)
    assert result_dim1.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(2, 4, 6, dtype=torch.float64)
    result_3d = QF.soft_topk(x_3d, k=k, dim=2)
    assert result_3d.shape == x_3d.shape


@pytest.mark.random
def test_soft_topk_bottomk_shape_preservation() -> None:
    """Test that soft_topk_bottomk preserves tensor shape."""
    # 2D tensor - ensure k constraint: dim >= 2*k (for bottomk)
    x_2d = torch.randn(6, 8, dtype=torch.float64)  # 6 >= 2*2=4, 8 >= 2*2=4
    k = 2

    result_dim0 = QF.soft_topk_bottomk(x_2d, k=k, dim=0)
    assert result_dim0.shape == x_2d.shape

    result_dim1 = QF.soft_topk_bottomk(x_2d, k=k, dim=1)
    assert result_dim1.shape == x_2d.shape

    # 3D tensor
    x_3d = torch.randn(6, 8, 10, dtype=torch.float64)  # All dims >= 2*2=4
    result_3d = QF.soft_topk_bottomk(x_3d, k=k, dim=2)
    assert result_3d.shape == x_3d.shape


@pytest.mark.random
def test_soft_topk_sum_constraint() -> None:
    """Test that soft_topk results sum to k."""
    qfeval_functions.random.seed()

    for _ in range(5):
        x = torch.randn(20, dtype=torch.float64)
        k = 5
        result = QF.soft_topk(x, k=k, dim=0)
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(k, dtype=torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )

    # Multi-dimensional case
    x_2d = torch.randn(10, 15, dtype=torch.float64)
    k = 3
    result_2d = QF.soft_topk(x_2d, k=k, dim=1)
    expected_sums = torch.full((10,), k, dtype=torch.float64)
    torch.testing.assert_close(
        result_2d.sum(dim=1), expected_sums, atol=1e-4, rtol=1e-4
    )


@pytest.mark.random
def test_soft_topk_bottomk_sum_constraint() -> None:
    """Test that soft_topk_bottomk results sum to 0."""
    qfeval_functions.random.seed()

    for _ in range(5):
        x = torch.randn(20, dtype=torch.float64)
        k = 5
        result = QF.soft_topk_bottomk(x, k=k, dim=0)
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )

    # Multi-dimensional case
    x_2d = torch.randn(10, 15, dtype=torch.float64)
    k = 3
    result_2d = QF.soft_topk_bottomk(x_2d, k=k, dim=1)
    expected_sums = torch.zeros(10, dtype=torch.float64)
    torch.testing.assert_close(
        result_2d.sum(dim=1), expected_sums, atol=1e-5, rtol=1e-5
    )


@pytest.mark.random
def test_soft_topk_non_negativity() -> None:
    """Test that soft_topk produces non-negative values."""
    qfeval_functions.random.seed()

    for _ in range(10):
        x = torch.randn(25, dtype=torch.float64)
        k = 5
        result = QF.soft_topk(x, k=k, dim=0)
        assert torch.all(
            result >= 0
        ), "soft_topk should produce non-negative values"


@pytest.mark.random
def test_soft_topk_different_k_values() -> None:
    """Test soft_topk with different k values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)

    for k in [1, 3, 5, 8]:
        result = QF.soft_topk(x, k=k, dim=0)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(k, dtype=torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(result >= 0)


@pytest.mark.random
def test_soft_topk_bottomk_different_k_values() -> None:
    """Test soft_topk_bottomk with different k values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)

    # For bottomk: need dim >= 2*k, so max k = 5 for dim=10
    for k in [1, 2, 3, 4]:
        result = QF.soft_topk_bottomk(x, k=k, dim=0)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.random
def test_soft_topk_different_dimensions() -> None:
    """Test soft_topk along different dimensions."""
    qfeval_functions.random.seed()
    x = torch.randn(4, 6, 8, dtype=torch.float64)
    k = 3

    # Test each dimension
    for dim in range(x.ndim):
        result = QF.soft_topk(x, k=k, dim=dim)
        assert result.shape == x.shape
        assert torch.all(result >= 0)

        # Check sum constraint along the specified dimension
        sum_result = result.sum(dim=dim)
        expected_sum = torch.full_like(sum_result, k, dtype=torch.float64)
        torch.testing.assert_close(
            sum_result, expected_sum, atol=1e-4, rtol=1e-4
        )


@pytest.mark.random
def test_soft_topk_bottomk_different_dimensions() -> None:
    """Test soft_topk_bottomk along different dimensions."""
    qfeval_functions.random.seed()
    # Make sure all dimensions satisfy constraint: dim >= 2*k
    x = torch.randn(8, 10, 12, dtype=torch.float64)  # All dims >= 2*3=6
    k = 3

    # Test each dimension
    for dim in range(x.ndim):
        result = QF.soft_topk_bottomk(x, k=k, dim=dim)
        assert result.shape == x.shape

        # Check sum constraint along the specified dimension
        sum_result = result.sum(dim=dim)
        expected_sum = torch.zeros_like(sum_result, dtype=torch.float64)
        torch.testing.assert_close(
            sum_result, expected_sum, atol=1e-5, rtol=1e-5
        )


@pytest.mark.random
def test_soft_topk_negative_dimension() -> None:
    """Test soft_topk with negative dimension indices."""
    qfeval_functions.random.seed()
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    k = 2

    # Test negative dimension
    result_neg = QF.soft_topk(x, k=k, dim=-1)
    result_pos = QF.soft_topk(x, k=k, dim=2)

    torch.testing.assert_close(result_neg, result_pos)


@pytest.mark.random
def test_soft_topk_bottomk_negative_dimension() -> None:
    """Test soft_topk_bottomk with negative dimension indices."""
    qfeval_functions.random.seed()
    x = torch.randn(3, 4, 5, dtype=torch.float64)
    k = 2

    # Test negative dimension
    result_neg = QF.soft_topk_bottomk(x, k=k, dim=-1)
    result_pos = QF.soft_topk_bottomk(x, k=k, dim=2)

    torch.testing.assert_close(result_neg, result_pos)


@pytest.mark.random
def test_soft_topk_epsilon_parameter() -> None:
    """Test soft_topk with different epsilon values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test different epsilon values
    for epsilon in [0.01, 0.1, 1.0]:
        result = QF.soft_topk(x, k=k, dim=0, epsilon=epsilon)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(k, dtype=torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(result >= 0)

    # Smaller epsilon should produce sharper results
    result_small_eps = QF.soft_topk(x, k=k, dim=0, epsilon=0.01)
    result_large_eps = QF.soft_topk(x, k=k, dim=0, epsilon=1.0)

    # With smaller epsilon, the maximum should be larger (sharper)
    assert result_small_eps.max() > result_large_eps.max()


@pytest.mark.random
def test_soft_topk_bottomk_epsilon_parameter() -> None:
    """Test soft_topk_bottomk with different epsilon values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test different epsilon values
    for epsilon in [0.01, 0.1, 1.0]:
        result = QF.soft_topk_bottomk(x, k=k, dim=0, epsilon=epsilon)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.random
def test_soft_topk_max_iter_parameter() -> None:
    """Test soft_topk with different max_iter values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test different max_iter values
    for max_iter in [50, 200, 500]:
        result = QF.soft_topk(x, k=k, dim=0, max_iter=max_iter)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(k, dtype=torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(result >= 0)


@pytest.mark.random
def test_soft_topk_bottomk_max_iter_parameter() -> None:
    """Test soft_topk_bottomk with different max_iter values."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test different max_iter values
    for max_iter in [50, 200, 500]:
        result = QF.soft_topk_bottomk(x, k=k, dim=0, max_iter=max_iter)
        assert result.shape == x.shape
        torch.testing.assert_close(
            result.sum(),
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.random
def test_soft_topk_topk_only_flag() -> None:
    """Test that soft_topk_bottomk with topk_only=True matches soft_topk."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    result_soft_topk = QF.soft_topk(x, k=k, dim=0)
    result_topk_only = QF.soft_topk_bottomk(x, k=k, dim=0, topk_only=True)

    torch.testing.assert_close(result_soft_topk, result_topk_only)


@pytest.mark.random
def test_soft_topk_bottomk_dtype_preservation() -> None:
    """Test that soft_topk_bottomk preserves input dtype."""
    k = 3

    # Float32
    x_float32 = torch.randn(10, dtype=torch.float32)
    result_float32 = QF.soft_topk_bottomk(x_float32, k=k, dim=0)
    assert result_float32.dtype == torch.float32

    # Float64
    x_float64 = torch.randn(10, dtype=torch.float64)
    result_float64 = QF.soft_topk_bottomk(x_float64, k=k, dim=0)
    assert result_float64.dtype == torch.float64


@pytest.mark.random
def test_soft_topk_bottomk_device_preservation() -> None:
    """Test that soft_topk_bottomk preserves input device."""
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    result = QF.soft_topk_bottomk(x, k=k, dim=0)
    assert result.device == x.device


@pytest.mark.random
def test_soft_topk_error_handling() -> None:
    """Test soft_topk error handling for invalid inputs."""
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test negative epsilon
    with pytest.raises(AssertionError, match="epsilon must be greather than 0"):
        QF.soft_topk(x, k=k, dim=0, epsilon=-0.1)

    # Test zero epsilon
    with pytest.raises(AssertionError, match="epsilon must be greather than 0"):
        QF.soft_topk(x, k=k, dim=0, epsilon=0.0)

    # Test NaN input
    x_nan = torch.tensor([1.0, math.nan, 3.0], dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Input tensor has nan or inf elements"
    ):
        QF.soft_topk(x_nan, k=2, dim=0)

    # Test infinite input
    x_inf = torch.tensor([1.0, math.inf, 3.0], dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Input tensor has nan or inf elements"
    ):
        QF.soft_topk(x_inf, k=2, dim=0)


@pytest.mark.random
def test_soft_topk_bottomk_error_handling() -> None:
    """Test soft_topk_bottomk error handling for invalid inputs."""
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    # Test negative epsilon
    with pytest.raises(AssertionError, match="epsilon must be greather than 0"):
        QF.soft_topk_bottomk(x, k=k, dim=0, epsilon=-0.1)

    # Test zero epsilon
    with pytest.raises(AssertionError, match="epsilon must be greather than 0"):
        QF.soft_topk_bottomk(x, k=k, dim=0, epsilon=0.0)

    # Test NaN input
    x_nan = torch.tensor([1.0, math.nan, 3.0], dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Input tensor has nan or inf elements"
    ):
        QF.soft_topk_bottomk(x_nan, k=2, dim=0)

    # Test infinite input
    x_inf = torch.tensor([1.0, math.inf, 3.0], dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Input tensor has nan or inf elements"
    ):
        QF.soft_topk_bottomk(x_inf, k=2, dim=0)


def test_soft_topk_edge_cases() -> None:
    """Test soft_topk edge cases."""
    # k=1 case
    x = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
    result_k1 = QF.soft_topk(x, k=1, dim=0)
    torch.testing.assert_close(
        result_k1.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    # k equal to dimension size (all elements selected)
    x_small = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float64)
    result_all = QF.soft_topk(x_small, k=3, dim=0)
    torch.testing.assert_close(
        result_all.sum(),
        torch.tensor(3.0, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    # Very small tensor
    x_tiny = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result_tiny = QF.soft_topk(x_tiny, k=1, dim=0)
    assert result_tiny.shape == x_tiny.shape
    torch.testing.assert_close(
        result_tiny.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )


def test_soft_topk_bottomk_edge_cases() -> None:
    """Test soft_topk_bottomk edge cases."""
    # k=1 case
    x = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
    result_k1 = QF.soft_topk_bottomk(x, k=1, dim=0)
    torch.testing.assert_close(
        result_k1.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )

    # Very small tensor
    x_tiny = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result_tiny = QF.soft_topk_bottomk(x_tiny, k=1, dim=0)
    assert result_tiny.shape == x_tiny.shape
    torch.testing.assert_close(
        result_tiny.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


def test_soft_topk_constant_input() -> None:
    """Test soft_topk with constant input values."""
    # All elements equal
    x_const = torch.full((5,), 3.0, dtype=torch.float64)
    k = 2
    result = QF.soft_topk(x_const, k=k, dim=0)

    # Should still sum to k
    torch.testing.assert_close(
        result.sum(), torch.tensor(k, dtype=torch.float64), atol=1e-4, rtol=1e-4
    )
    # With constant input, all elements should have similar weights
    assert torch.all(result >= 0)
    assert result.std() < 0.5  # Should be relatively uniform


def test_soft_topk_bottomk_constant_input() -> None:
    """Test soft_topk_bottomk with constant input values."""
    # All elements equal
    x_const = torch.full((5,), 3.0, dtype=torch.float64)
    k = 2
    result = QF.soft_topk_bottomk(x_const, k=k, dim=0)

    # Should still sum to 0
    torch.testing.assert_close(
        result.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )
    # With constant input, result should be close to zero everywhere
    assert torch.all(torch.abs(result) < 0.1)


@pytest.mark.random
def test_soft_topk_gradient_compatibility() -> None:
    """Test that soft_topk works with gradient computation."""
    x = torch.randn(10, dtype=torch.float64, requires_grad=True)
    k = 3

    result = QF.soft_topk(x, k=k, dim=0)
    loss = result.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


@pytest.mark.random
def test_soft_topk_bottomk_gradient_compatibility() -> None:
    """Test that soft_topk_bottomk works with gradient computation."""
    x = torch.randn(10, dtype=torch.float64, requires_grad=True)
    k = 3

    result = QF.soft_topk_bottomk(x, k=k, dim=0)
    loss = result.abs().sum()  # Use abs since sum is 0
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_soft_topk_numerical_stability() -> None:
    """Test soft_topk numerical stability with various scales."""
    k = 3

    # Very large values
    x_large = torch.tensor([1e6, 2e6, 3e6, 4e6, 5e6], dtype=torch.float64)
    result_large = QF.soft_topk(x_large, k=k, dim=0)
    assert torch.isfinite(result_large).all()
    torch.testing.assert_close(
        result_large.sum(),
        torch.tensor(k, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    # Very small values
    x_small = torch.tensor([1e-6, 2e-6, 3e-6, 4e-6, 5e-6], dtype=torch.float64)
    result_small = QF.soft_topk(x_small, k=k, dim=0)
    assert torch.isfinite(result_small).all()
    torch.testing.assert_close(
        result_small.sum(),
        torch.tensor(k, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )


def test_soft_topk_bottomk_numerical_stability() -> None:
    """Test soft_topk_bottomk numerical stability with various scales."""
    k = 2  # Use k=2 so constraint is 7 >= 2*2=4 (satisfied)

    # Very large values - need at least 2*k=4 elements
    x_large = torch.tensor(
        [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6], dtype=torch.float64
    )
    result_large = QF.soft_topk_bottomk(x_large, k=k, dim=0)
    assert torch.isfinite(result_large).all()
    torch.testing.assert_close(
        result_large.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )

    # Very small values
    x_small = torch.tensor(
        [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6], dtype=torch.float64
    )
    result_small = QF.soft_topk_bottomk(x_small, k=k, dim=0)
    assert torch.isfinite(result_small).all()
    torch.testing.assert_close(
        result_small.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.random
def test_soft_topk_reproducibility() -> None:
    """Test that soft_topk produces consistent results."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    result1 = QF.soft_topk(x, k=k, dim=0)
    result2 = QF.soft_topk(x, k=k, dim=0)

    torch.testing.assert_close(result1, result2)


@pytest.mark.random
def test_soft_topk_bottomk_reproducibility() -> None:
    """Test that soft_topk_bottomk produces consistent results."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 3

    result1 = QF.soft_topk_bottomk(x, k=k, dim=0)
    result2 = QF.soft_topk_bottomk(x, k=k, dim=0)

    torch.testing.assert_close(result1, result2)


@pytest.mark.random
def test_soft_topk_batch_processing() -> None:
    """Test soft_topk with batch dimensions."""
    qfeval_functions.random.seed()
    batch_size = 5
    x_batch = torch.randn(batch_size, 10, dtype=torch.float64)
    k = 3

    result_batch = QF.soft_topk(x_batch, k=k, dim=1)
    assert result_batch.shape == x_batch.shape

    # Check constraints for each batch
    for i in range(batch_size):
        torch.testing.assert_close(
            result_batch[i].sum(),
            torch.tensor(k, dtype=torch.float64),
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.all(result_batch[i] >= 0)


@pytest.mark.random
def test_soft_topk_bottomk_batch_processing() -> None:
    """Test soft_topk_bottomk with batch dimensions."""
    qfeval_functions.random.seed()
    batch_size = 5
    x_batch = torch.randn(batch_size, 10, dtype=torch.float64)
    k = 3

    result_batch = QF.soft_topk_bottomk(x_batch, k=k, dim=1)
    assert result_batch.shape == x_batch.shape

    # Check constraints for each batch
    for i in range(batch_size):
        torch.testing.assert_close(
            result_batch[i].sum(),
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )


def test_soft_topk_comparison_with_hard_topk() -> None:
    """Test soft_topk behavior compared to hard top-k."""
    qfeval_functions.random.seed()
    x = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0], dtype=torch.float64)
    k = 2

    # Get hard top-k indices
    _, hard_indices = torch.topk(x, k, dim=0)

    # Get soft top-k
    soft_result = QF.soft_topk(
        x, k=k, dim=0, epsilon=0.01
    )  # Small epsilon for sharpness

    # Soft top-k should assign higher weights to hard top-k elements
    for idx in hard_indices:
        # Elements in hard top-k should have higher soft weights than others
        other_indices = [i for i in range(len(x)) if i not in hard_indices]
        for other_idx in other_indices:
            assert soft_result[idx] > soft_result[other_idx]


@pytest.mark.random
def test_soft_topk_performance() -> None:
    """Test soft_topk performance with larger tensors."""
    qfeval_functions.random.seed()
    x_large = torch.randn(500, 200, dtype=torch.float64)
    k = 10

    result = QF.soft_topk(x_large, k=k, dim=1)
    assert result.shape == x_large.shape

    # Check constraints
    expected_sums = torch.full((500,), k, dtype=torch.float64)
    torch.testing.assert_close(
        result.sum(dim=1), expected_sums, atol=1e-4, rtol=1e-4
    )
    assert torch.all(result >= 0)


@pytest.mark.random
def test_soft_topk_bottomk_performance() -> None:
    """Test soft_topk_bottomk performance with larger tensors."""
    qfeval_functions.random.seed()
    x_large = torch.randn(500, 200, dtype=torch.float64)
    k = 10

    result = QF.soft_topk_bottomk(x_large, k=k, dim=1)
    assert result.shape == x_large.shape

    # Check constraints
    expected_sums = torch.zeros(500, dtype=torch.float64)
    torch.testing.assert_close(
        result.sum(dim=1), expected_sums, atol=1e-5, rtol=1e-5
    )


@pytest.mark.random
def test_soft_topk_bottomk_memory_efficiency() -> None:
    """Test memory efficiency of soft_topk_bottomk."""
    # Test that soft_topk_bottomk doesn't create excessive intermediate tensors
    for i in range(3):
        x = torch.randn(100, 50, dtype=torch.float64)
        result = QF.soft_topk_bottomk(x, k=5, dim=1)
        assert result.shape == x.shape
        # Force deletion
        del x, result


@pytest.mark.random
def test_soft_topk_mathematical_properties() -> None:
    """Test mathematical properties of soft_topk."""
    qfeval_functions.random.seed()
    x = torch.randn(10, dtype=torch.float64)
    k = 4

    result = QF.soft_topk(x, k=k, dim=0)

    # Non-negativity
    assert torch.all(result >= 0)

    # Sum constraint
    torch.testing.assert_close(
        result.sum(), torch.tensor(k, dtype=torch.float64), atol=1e-4, rtol=1e-4
    )

    # Monotonicity: if x[i] > x[j], then result[i] >= result[j] (approximately)
    sorted_x, sorted_indices = torch.sort(x, descending=True)
    sorted_result = result[sorted_indices]

    # Check that result is approximately monotonic (allowing small violations due to smoothness)
    for i in range(len(sorted_result) - 1):
        assert (
            sorted_result[i] >= sorted_result[i + 1] - 0.1
        )  # Allow small tolerance


def test_soft_topk_special_values() -> None:
    """Test soft_topk with special float values (but not NaN/inf)."""
    # Test with zeros
    x_zeros = torch.zeros(5, dtype=torch.float64)
    k = 2
    result_zeros = QF.soft_topk(x_zeros, k=k, dim=0)
    torch.testing.assert_close(
        result_zeros.sum(),
        torch.tensor(k, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    # Test with negative values
    x_neg = torch.tensor([-3.0, -1.0, -5.0, -2.0], dtype=torch.float64)
    result_neg = QF.soft_topk(x_neg, k=2, dim=0)
    assert torch.all(result_neg >= 0)
    torch.testing.assert_close(
        result_neg.sum(),
        torch.tensor(2.0, dtype=torch.float64),
        atol=1e-4,
        rtol=1e-4,
    )

    # The least negative (largest) values should get higher weights
    assert result_neg[1] > result_neg[2]  # -1.0 > -5.0


def test_soft_topk_bottomk_special_values() -> None:
    """Test soft_topk_bottomk with special float values (but not NaN/inf)."""
    # Test with zeros
    x_zeros = torch.zeros(5, dtype=torch.float64)
    k = 2
    result_zeros = QF.soft_topk_bottomk(x_zeros, k=k, dim=0)
    torch.testing.assert_close(
        result_zeros.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )

    # Test with negative values
    x_neg = torch.tensor([-3.0, -1.0, -5.0, -2.0], dtype=torch.float64)
    result_neg = QF.soft_topk_bottomk(x_neg, k=2, dim=0)
    torch.testing.assert_close(
        result_neg.sum(),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )
