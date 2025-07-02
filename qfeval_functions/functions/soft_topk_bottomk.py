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
    r"""Compute differentiable soft top-k and bottom-k selection along a dimension.

    This function implements a differentiable approximation to the top-k and
    bottom-k selection operators using optimal transport theory and the Sinkhorn
    algorithm. Unlike hard selection, this soft version provides smooth gradients
    and allows for end-to-end training in neural networks while maintaining
    interpretability similar to traditional ranking operations.

    The algorithm uses entropic regularization to solve the optimal transport
    problem between the input distribution and target distributions representing
    top-k, middle, and bottom-k elements. The result is a continuous relaxation
    of discrete ranking operations.

    Args:
        x (Tensor):
            Input tensor of any shape. The soft selection is applied along
            the specified dimension.
        k (int):
            Number of elements to select for top-k and bottom-k. Must satisfy
            ``2*k <= x.size(dim)`` when ``topk_only=False``, or 
            ``k <= x.size(dim)`` when ``topk_only=True``.
        dim (int, optional):
            Dimension along which to perform the soft selection. Default is -1
            (last dimension).
        epsilon (float, optional):
            Entropic regularization parameter controlling the smoothness of the
            approximation. Smaller values produce sharper selections but may
            cause numerical instability. Default is 0.1.
        max_iter (int, optional):
            Maximum number of Sinkhorn iterations for solving the optimal
            transport problem. Default is 200.
        topk_only (bool, optional):
            If True, only performs top-k selection (binary classification).
            If False, performs top-k, middle, and bottom-k selection 
            (ternary classification). Default is False.

    Returns:
        Tensor:
            Soft selection weights with the same shape as input. When
            ``topk_only=False``, values near +1 indicate top-k elements,
            values near 0 indicate middle elements, and values near -1
            indicate bottom-k elements. When ``topk_only=True``, values
            near 1 indicate top-k elements and values near 0 indicate
            the remaining elements.

    Example:
        >>> # Basic 2D example with ternary selection
        >>> x = torch.tensor([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
        >>> weights = soft_topk_bottomk(x, k=1, dim=1)
        >>> weights
        tensor([[-0.7624, -0.2123,  0.0000,  0.2123,  0.7624],
                [-0.7624, -0.2123,  0.0000,  0.2123,  0.7624]])

        >>> # Selection along different dimension
        >>> soft_topk_bottomk(x, k=1, dim=0)
        tensor([[-0.9999, -0.9999, -0.9999, -0.9999, -0.9999],
                [ 0.9999,  0.9999,  0.9999,  0.9999,  0.9999]])

        >>> # Sharper selection with smaller epsilon
        >>> soft_topk_bottomk(x, k=1, dim=1, epsilon=1e-3)
        tensor([[-0.9965, -0.0035,  0.0000,  0.0035,  0.9965],
                [-0.9965, -0.0035,  0.0000,  0.0035,  0.9965]])

        >>> # Binary top-k only selection
        >>> x = torch.tensor([1., 5., 2., 8., 3.])
        >>> weights = soft_topk_bottomk(x, k=2, topk_only=True)
        >>> weights  # Higher values for top-2 elements
        tensor([    0.0010,     0.9019,     0.0097,     0.9999,     0.0875])

        >>> # Multi-dimensional tensor
        >>> x = torch.randn(2, 3, 4)
        >>> weights = soft_topk_bottomk(x, k=1, dim=-1)
        >>> weights.shape
        torch.Size([2, 3, 4])

        >>> # Financial portfolio selection
        >>> returns = torch.tensor([0.12, 0.08, 0.15, 0.05, 0.18, 0.03])
        >>> selection_weights = soft_topk_bottomk(returns, k=2)
        >>> selection_weights  # Soft weights for top/bottom performers
        tensor([ 0.2661, -0.2857,  0.7336, -0.7488,  0.9478, -0.9130])

    .. seealso::
        - :func:`soft_topk`: Binary top-k selection only.
        - :func:`torch.topk`: Hard top-k selection (non-differentiable).
        - :func:`stable_sort`: Stable sorting with NaN handling.

    .. note::
        The entropic regularization parameter :attr:`epsilon` controls the
        trade-off between approximation accuracy and numerical stability.

    .. warning::
        Very small values of :attr:`epsilon` (< 1e-3) may cause numerical
        instability and NaN values. Use stabilized Sinkhorn iterations for
        high-precision requirements.

    .. warning::
        The function assumes finite input values. NaN or infinite values
        will raise a ValueError during computation.

    References:
        - Optimal Transport: https://optimaltransport.github.io/
        - Sinkhorn Algorithm: https://arxiv.org/abs/1306.0895
        - Differentiable Ranking: https://arxiv.org/abs/1802.08665
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
    r"""Compute differentiable soft top-k selection along a dimension.

    This function implements a differentiable approximation to the top-k
    selection operator, returning binary selection weights where values
    close to 1 indicate top-k elements and values close to 0 indicate
    the remaining elements. This is a specialized version of 
    :func:`soft_topk_bottomk` with ``topk_only=True``.

    The function uses optimal transport theory and the Sinkhorn algorithm
    to provide smooth gradients suitable for end-to-end training while
    maintaining interpretability similar to hard top-k selection.

    Args:
        x (Tensor):
            Input tensor of any shape. The soft top-k selection is applied
            along the specified dimension.
        k (int):
            Number of top elements to select. Must satisfy ``k <= x.size(dim)``.
        dim (int, optional):
            Dimension along which to perform the soft top-k selection.
            Default is -1 (last dimension).
        epsilon (float, optional):
            Entropic regularization parameter controlling the smoothness of the
            approximation. Smaller values produce sharper selections but may
            cause numerical instability. Default is 0.1.
        max_iter (int, optional):
            Maximum number of Sinkhorn iterations for solving the optimal
            transport problem. Default is 200.

    Returns:
        Tensor:
            Soft selection weights with the same shape as input. Values near 1
            indicate top-k elements, and values near 0 indicate the remaining
            elements.

    Example:
        >>> # Basic 2D example
        >>> x = torch.tensor([[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])
        >>> weights = soft_topk(x, k=1, dim=0)
        >>> weights
        tensor([[0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                [0.9999, 0.9999, 0.9999, 0.9999, 0.9999]])

        >>> # Top-2 selection along rows
        >>> soft_topk(x, k=2, dim=1)
        tensor([[0.0000, 0.0006, 0.0783, 0.9217, 0.9994],
                [0.0000, 0.0006, 0.0783, 0.9217, 0.9994]])

        >>> # Sharp selection with small epsilon
        >>> soft_topk(x, k=1, dim=1, epsilon=1e-3)
        tensor([[0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000, 0.5000, 0.5000]])

        >>> # Portfolio top performers selection
        >>> returns = torch.tensor([0.12, 0.08, 0.15, 0.05, 0.18])
        >>> top_performers = soft_topk(returns, k=2)
        >>> top_performers  # Highest weights for best returns
        tensor([    0.0965,     0.0003,     0.9044,     0.0000,     0.9988])

        >>> # Multi-batch processing
        >>> batch_scores = torch.randn(10, 20)
        >>> top_features = soft_topk(batch_scores, k=5, dim=1)
        >>> top_features.shape
        torch.Size([10, 20])

    .. seealso::
        - :func:`soft_topk_bottomk`: Ternary top-k, middle, and bottom-k selection.
        - :func:`torch.topk`: Hard top-k selection (non-differentiable).
        - :func:`stable_sort`: Stable sorting with NaN handling.


    References:
        - Optimal Transport: https://optimaltransport.github.io/
        - Differentiable Top-k: https://arxiv.org/abs/1802.08665
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
        return (Gamma[:, :, 2] - Gamma[:, :, 0]) * dim


class _Sinkhorn(torch.autograd.Function):
    """Sinkhorn algorithm for regularized optimal transport."""

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
