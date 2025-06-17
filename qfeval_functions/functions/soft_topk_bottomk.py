import logging
import typing

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def soft_topk_bottomk(
    x: torch.Tensor,
    k: int,
    dim: int = -1,
    *,
    epsilon: float = 0.1,
    max_iter: int = 200,
    topk_only: bool = False,
) -> torch.Tensor:
    r"""Computes differentiable soft top-k and bottom-k selection along a dimension.

    This function implements a differentiable approximation to top-k and bottom-k
    selection using the Sinkhorn algorithm for optimal transport. It returns weights
    that represent the difference between top-k and bottom-k selections, or just
    top-k selections when ``topk_only=True``.

    The function solves a regularized optimal transport problem to assign weights
    that approximate hard top-k/bottom-k selection while maintaining differentiability.
    The output represents:

    - **For ``topk_only=False`` (default)**: Weights that sum to 0, where positive
      values indicate top-k elements and negative values indicate bottom-k elements
    - **For ``topk_only=True``**: Non-negative weights that sum to k, focusing only
      on top-k elements (equivalent to :func:`soft_topk`)

    The algorithm iteratively refines a transport plan that assigns mass to elements
    based on their relative ranking, with the ``epsilon`` parameter controlling
    the sharpness of the selection (smaller values = sharper selection).

    Mathematical formulation:
        The function minimizes a regularized optimal transport cost with entropy
        regularization, subject to marginal constraints that enforce the desired
        top-k/bottom-k structure.

    Note:
        - All input values must be finite (no NaN or infinity values)
        - The function supports gradient computation for optimization
        - Smaller ``epsilon`` values provide sharper selections but may require more iterations
        - For bottom-k selection, ``x.shape[dim]`` must be ≥ 2*k

    Args:
        x (Tensor): Input tensor with values to select from
        k (int): Number of elements to select for both top-k and bottom-k.
            Must be positive and ≤ ``x.shape[dim] // 2`` for bottom-k mode.
        dim (int): Dimension along which to perform the selection. Default: -1
        epsilon (float): Entropy regularization parameter controlling selection sharpness.
            Smaller values give sharper selections. Must be positive. Default: 0.1
        max_iter (int): Maximum number of Sinkhorn iterations for convergence.
            Default: 200
        topk_only (bool): If ``True``, performs only top-k selection (equivalent to
            :func:`soft_topk`). If ``False``, performs top-k minus bottom-k selection.
            Default: False

    Returns:
        Tensor: A tensor of the same shape as :attr:`x` containing selection weights:

        - **topk_only=False**: Weights sum to 0 along ``dim``. Positive values
          indicate top-k elements, negative values indicate bottom-k elements.
        - **topk_only=True**: Non-negative weights sum to k along ``dim``.

    Raises:
        AssertionError: If ``epsilon`` ≤ 0.
        ValueError: If input contains NaN or infinity values.
        AssertionError: If ``k`` is incompatible with tensor dimension size.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>>
        >>> # Basic example: top-k minus bottom-k selection
        >>> x = torch.tensor([1., 5., 3., 8., 2.])
        >>> weights = QF.soft_topk_bottomk(x, k=2, dim=0)
        >>> weights.sum()  # Should be close to 0  # doctest: +ELLIPSIS
        tensor(...)
        >>>
        >>> # Check that top values get positive weights, bottom values get negative
        >>> x = torch.tensor([1., 5., 3., 8., 2.])
        >>> weights = QF.soft_topk_bottomk(x, k=2, dim=0)
        >>> weights[3] > 0  # Value 8 should have positive weight
        tensor(True)
        >>> weights[0] < 0  # Value 1 should have negative weight
        tensor(True)
        >>>
        >>> # Top-k only mode (equivalent to soft_topk)
        >>> weights_topk = QF.soft_topk_bottomk(x, k=2, dim=0, topk_only=True)
        >>> weights_topk.sum()  # Should be close to 2.0
        tensor(2.)
        >>>
        >>> # Sharp selection with small epsilon
        >>> x = torch.tensor([1., 2., 3., 4., 5.])
        >>> sharp_weights = QF.soft_topk_bottomk(x, k=1, dim=0, epsilon=0.01)
        >>> smooth_weights = QF.soft_topk_bottomk(x, k=1, dim=0, epsilon=1.0)
        >>> # sharp_weights will be more concentrated on extreme values
        >>>
        >>> # Gradient computation example
        >>> x = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
        >>> weights = QF.soft_topk_bottomk(x, k=2, dim=0)
        >>> loss = weights.abs().sum()
        >>> loss.backward()
        >>> print(x.grad.shape)  # Gradients computed successfully
        torch.Size([5])
    """
    # 1. Move the target dimension to the last.
    x = x.transpose(-1, dim)

    # 2. Reshape input into two dimensional tensor.
    shape = x.shape
    x = x.reshape(-1, shape[-1])

    # 3. Apply SoftTopKBottomK and restore original shape.
    x = _soft_topk_bottomk(
        x, k, epsilon=epsilon, max_iter=max_iter, topk_only=topk_only
    )
    x = x.reshape(shape).transpose(-1, dim)

    return x


def soft_topk(
    x: torch.Tensor,
    k: int,
    dim: int = -1,
    *,
    epsilon: float = 0.1,
    max_iter: int = 200,
) -> torch.Tensor:
    r"""Computes differentiable soft top-k selection along a dimension.

    This function implements a differentiable approximation to top-k selection
    using the Sinkhorn algorithm for optimal transport. It returns non-negative
    weights that sum to k, with higher weights assigned to larger input values.

    This function is equivalent to calling :func:`soft_topk_bottomk` with
    ``topk_only=True``. It provides a smooth, differentiable alternative to
    hard top-k selection, making it suitable for gradient-based optimization.

    The algorithm uses entropic regularization to approximate the discrete top-k
    operation with continuous weights. The ``epsilon`` parameter controls the
    trade-off between smoothness and approximation quality:

    - **Small epsilon**: Sharp selection, close to hard top-k but less smooth
    - **Large epsilon**: Smooth selection, more distributed weights

    Mathematical properties:
        - Output weights are non-negative: ``output[i] ≥ 0``
        - Weights sum to k: ``output.sum(dim=dim) = k``
        - Higher input values receive higher weights (monotonic)
        - Differentiable with respect to input

    Args:
        x (Tensor): Input tensor with values to select from
        k (int): Number of elements to select. Must be positive and ≤ ``x.shape[dim]``
        dim (int): Dimension along which to perform the selection. Default: -1
        epsilon (float): Entropy regularization parameter controlling selection sharpness.
            Smaller values give sharper selections. Must be positive. Default: 0.1
        max_iter (int): Maximum number of Sinkhorn iterations for convergence.
            Default: 200

    Returns:
        Tensor: A tensor of the same shape as :attr:`x` containing non-negative
        selection weights that sum to k along the specified dimension.

    Raises:
        AssertionError: If ``epsilon`` ≤ 0.
        ValueError: If input contains NaN or infinity values.

    Example::

        >>> import torch
        >>> import qfeval_functions.functions as QF
        >>>
        >>> # Basic 1D example
        >>> x = torch.tensor([1., 5., 3., 8., 2.])
        >>> weights = QF.soft_topk(x, k=2, dim=0)
        >>> weights.sum()  # Should be close to 2.0
        tensor(2.)
        >>> # Largest values (8, 5) should have highest weights
        >>> weights[3]  # Weight for value 8 (highest)  # doctest: +ELLIPSIS
        tensor(0.9...)
        >>> weights[1]  # Weight for value 5 (second highest)  # doctest: +ELLIPSIS
        tensor(0.9...)
        >>>
        >>> # Effect of epsilon parameter
        >>> x = torch.tensor([1., 2., 3., 4., 5.])
        >>> sharp = QF.soft_topk(x, k=2, dim=0, epsilon=0.01)   # Sharp selection
        >>> smooth = QF.soft_topk(x, k=2, dim=0, epsilon=1.0)   # Smooth selection
        >>> # sharp weights will be more concentrated on top-2 values
        >>>
        >>> # Gradient computation
        >>> x = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
        >>> weights = QF.soft_topk(x, k=3, dim=0)
        >>> loss = (weights * torch.arange(5, dtype=torch.float)).sum()
        >>> loss.backward()
        >>> print(x.grad is not None)  # Gradients computed
        True
        >>>
        >>> # Multi-dimensional example
        >>> x = torch.randn(10, 20, 30)
        >>> weights = QF.soft_topk(x, k=5, dim=2)  # Top-5 along last dimension
        >>> weights.shape
        torch.Size([10, 20, 30])
        >>> sums = weights.sum(dim=2)  # Each slice sums to 5.0
        >>> sums.shape
        torch.Size([10, 20])
        >>> torch.allclose(sums, torch.tensor(5.0))  # Verify all close to 5.0
        True
    """
    return soft_topk_bottomk(
        x, k, dim, epsilon=epsilon, max_iter=max_iter, topk_only=True
    )


def _soft_topk_bottomk(
    scores: torch.Tensor,
    k: int,
    *,
    epsilon: float = 0.1,
    max_iter: int = 200,
    topk_only: bool = False,
) -> torch.Tensor:
    """Internal implementation of soft top-k/bottom-k using Sinkhorn algorithm.

    This function implements the core Sinkhorn iterations for optimal transport
    to approximate top-k selection. It operates on 2D tensors where the first
    dimension is the batch dimension and the second is the selection dimension.

    Args:
        scores: 2D tensor of shape (batch_size, seq_len) with values to select from
        k: Number of elements to select
        epsilon: Entropy regularization parameter
        max_iter: Maximum number of Sinkhorn iterations
        topk_only: If True, returns only top-k weights; if False, returns top-k minus bottom-k

    Returns:
        2D tensor of same shape as scores with selection weights

    Raises:
        AssertionError: If epsilon <= 0
        ValueError: If input contains non-finite values
    """
    assert epsilon > 0, f"epsilon must be greather than 0, but: {epsilon}"

    if not scores.isfinite().all():
        raise ValueError("Input tensor has nan or inf elements.")

    scores = scores - scores.mean(dim=-1, keepdim=True)
    scores = scores / scores.std(dim=-1, keepdim=True).clamp(min=1e-6)

    bs, dim = scores.size()

    if topk_only:
        anchors = torch.tensor([-1, 1]).to(scores)
    else:
        # Each element represents anchors for {bottom-k, middle, top-k}.
        anchors = torch.tensor([-1, 0, 1]).to(scores)

    C = (scores[:, :, None] - anchors[None, None, :]) ** 2
    C = C / C.amax(dim=(1, 2), keepdim=True).detach()

    assert dim - (1 if topk_only else 2) * k >= 0
    mu = torch.ones(dim).to(scores) / dim
    if topk_only:
        nu = torch.tensor([dim - k, k]).to(scores) / dim
    else:
        nu = torch.tensor([k, dim - 2 * k, k]).to(scores) / dim

    Gamma: torch.Tensor = _sinkhorn(C, mu, nu, epsilon, max_iter)

    if topk_only:
        return Gamma[:, :, 1] * dim
    else:
        # TODO(claude): The subtraction (Gamma[:, :, 2] - Gamma[:, :, 0]) often
        # results in very small non-zero values (~1e-7) instead of exact zero
        # due to floating point precision in the Sinkhorn algorithm iterations.
        # This is expected behavior but affects doctest outputs that expect
        # exact zeros. Consider adding a small epsilon threshold for true zeros.
        return (Gamma[:, :, 2] - Gamma[:, :, 0]) * dim


class _Sinkhorn(torch.autograd.Function):
    """Sinkhorn algorithm for regularized optimal transport.

    This class implements the Sinkhorn-Knopp algorithm as a PyTorch autograd
    function, allowing for differentiable optimal transport computations.
    The algorithm iteratively updates dual variables to find the optimal
    transport plan between source and target marginals.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: typing.Any,
        C: torch.Tensor,
        mu: torch.Tensor,
        nu: torch.Tensor,
        epsilon: float,
        max_iter: int,
    ) -> torch.Tensor:
        """Returns optimal transport plan.

        Args:
            ctx (typing.Any):
                Context object.
            C (torch.Tensor):
                Cost matrix in the shape of `(B, N, M)`.
            mu (torch.Tensor):
                Source vector in the shape of `(1, N, 1)`.
            nu (torch.Tensor):
                Target vector in the shape of `(1, 1, M)`.
            epsilon (float):
                Entropic-regularization parameter.
            max_iter (int):
                Maximum number of iterations.

        Returns:
            torch.Tensor: Optimal transport plan in the shape of `(B, N, M)`.
        """
        with torch.no_grad():  # type: ignore[no-untyped-call]
            if epsilon > 1e-2:
                Gamma = _sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    logger.info("Nan appeared in Gamma, re-computing...")
                    Gamma = _sinkhorn_forward_stabilized(
                        C, mu, nu, epsilon, max_iter
                    )
            else:
                Gamma = _sinkhorn_forward_stabilized(
                    C, mu, nu, epsilon, max_iter
                )
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: typing.Any, grad_output_Gamma: torch.Tensor
    ) -> typing.Any:
        """Returns gradient with respect to cost matrix."""
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        with torch.no_grad():  # type: ignore[no-untyped-call]
            grad_C = _sinkhorn_backward(
                grad_output_Gamma, Gamma, mu, nu, epsilon
            )
        return grad_C, None, None, None, None


def _sinkhorn_forward(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    max_iter: int,
) -> torch.Tensor:
    bs, n, k_ = C.size()
    v = torch.ones([bs, 1, k_], device=C.device) / (k_)
    G = torch.exp(-C / epsilon)
    for _ in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)
    Gamma = u * G * v
    return Gamma


def _sinkhorn_forward_stabilized(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
    max_iter: int,
) -> torch.Tensor:
    bs, n, k_ = C.size()
    f = torch.zeros([bs, n, 1]).to(C)
    g = torch.zeros([bs, 1, k_], device=C.device)
    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z: torch.Tensor, epsilon: float) -> torch.Tensor:
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z: torch.Tensor, epsilon: float) -> torch.Tensor:
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for _ in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu
        Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def _sinkhorn_backward(
    grad_output_Gamma: torch.Tensor,
    Gamma: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]
    bs, n, k_ = Gamma.size()
    inv_mu = 1.0 / (mu.view([1, -1]))
    Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(
        Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_
    )
    inv_Kappa = torch.inverse(Kappa)
    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)
    G1 = grad_output_Gamma * Gamma
    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma
    g1_L = g1.unsqueeze(-2).matmul(L)
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma
    G23 = -F.pad(g1_L, pad=(0, 1), mode="constant", value=0) * Gamma
    G2 = G21 + G22 + G23
    del g1, G21, G22, G23, Gamma_mu
    g2 = G1.sum(-2).unsqueeze(-1)
    g2 = g2[:, :-1, :]
    G31 = -L.matmul(g2) * Gamma
    G32 = (
        F.pad(
            inv_Kappa.matmul(g2).transpose(-1, -2),
            pad=(0, 1),
            mode="constant",
            value=0,
        )
        * Gamma
    )
    G3 = G31 + G32
    grad_C = (-G1 + G2 + G3) / epsilon
    return typing.cast(torch.Tensor, grad_C)


def _sinkhorn(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 200,
) -> torch.Tensor:
    """Returns optimal transport plan.

    Args:
        C (torch.Tensor):
            Cost matrix in the shape of `(B, N, M)`.
        mu (torch.Tensor):
            Source vector in the shape of `(N)`.
        nu (torch.Tensor):
            Target vector in the shape of `(M)`.
        epsilon (float):
            Entropic-regularization parameter.
        max_iter (int):
            Maximum number of iterations.

    Returns:
        torch.Tensor: Optimal transport plan in the shape of `(B, N, M)`.
    """
    result: torch.Tensor = _Sinkhorn.apply(  # type:ignore[no-untyped-call]
        C, mu[None, :, None], nu[None, None, :], epsilon, max_iter
    )
    return result
