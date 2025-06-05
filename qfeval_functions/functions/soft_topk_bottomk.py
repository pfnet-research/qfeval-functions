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
    r"""Apply SoftTopKBottomK module along with given dimension.

    See `qfeval.extension.SoftTopKBottomK` for futher information.

    Examples:
        >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> soft_topk_bottomk(x, k=1, dim=1)
        tensor([[-1.2967e-01, -6.4996e-02,  1.8626e-08,  6.4996e-02,  1.2967e-01],
                [-1.2967e-01, -6.4996e-02,  3.7253e-08,  6.4996e-02,  1.2967e-01]])
        >>> soft_topk_bottomk(x, k=1, dim=0)
        tensor([[-0.3912, -0.3912, -0.3912, -0.3912, -0.3912],
                [ 0.3912,  0.3912,  0.3912,  0.3912,  0.3912]])
        >>> soft_topk_bottomk(x, k=1, dim=1, epsilon=1e-3)
        tensor([[-9.9999e-01, -1.1132e-05,  3.9829e-10,  5.9813e-03,  9.9402e-01],
                [-9.9998e-01, -1.1133e-05,  3.9801e-10,  5.9777e-03,  9.9402e-01]])
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
    r"""Apply soft top-k operator along with given dimension.

    See `qfeval.extension.SoftTopk` for futher information.

    Examples:
        >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> soft_topk(x, k=1, dim=1)
        tensor([[0.1406, 0.1666, 0.1962, 0.2297, 0.2669],
                [0.1406, 0.1666, 0.1962, 0.2297, 0.2669]])
        >>> soft_topk(x, k=1, dim=0)
        tensor([[0.3775, 0.3775, 0.3775, 0.3775, 0.3775],
                [0.6225, 0.6225, 0.6225, 0.6225, 0.6225]])
        >>> soft_topk(x, k=1, dim=1, epsilon=1e-3)
        tensor([[4.4156e-29, 2.1423e-20, 1.0394e-11, 5.0171e-03, 9.9498e-01],
                [4.4139e-29, 2.1414e-20, 1.0389e-11, 5.0151e-03, 9.9499e-01]])
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
