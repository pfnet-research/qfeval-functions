import numpy as np
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
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-6)

    # Two dimensional cases.
    x = QF.randn(100, 50)

    actual = QF.soft_topk(x, dim=1, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(x)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), k, atol=1e-6)

    actual = QF.soft_topk(x, dim=0, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(x.t()).t()
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-6)

    # Three dimensional cases.
    x = QF.randn(100, 50, 20)

    actual = QF.soft_topk(x, dim=2, k=k, epsilon=epsilon)
    expected = SoftTopK(k=k, epsilon=epsilon)(
        x.reshape(-1, x.shape[2])
    ).reshape(x.shape)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=2).numpy(), k, atol=1e-6)

    actual = QF.soft_topk(x, dim=1, k=k, epsilon=epsilon)
    input = x.transpose(1, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(1, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=1).numpy(), k, atol=1e-6)

    actual = QF.soft_topk(x, dim=0, k=k, epsilon=epsilon)
    input = x.transpose(0, 2)
    shape = input.shape
    input = input.reshape(-1, shape[2])
    expected = SoftTopK(k=k, epsilon=epsilon)(input).reshape(shape)
    expected = expected.transpose(0, 2)
    np.testing.assert_allclose(actual.numpy(), expected.numpy())
    np.testing.assert_allclose(actual.sum(dim=0).numpy(), k, atol=1e-6)


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
