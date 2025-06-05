import torch

import qfeval_functions
import qfeval_functions.functions as QF


def test_orthonormalize() -> None:
    qfeval_functions.random.seed(1)
    a = QF.randn(17, 7, 31)
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True)
    b = QF.orthonormalize(a)
    eye = torch.eye(7)[None, :, :]
    a_actual = QF.einsum("bxi,byi->bxy", a, a)
    a_actual = a_actual - eye
    b_actual = QF.einsum("bxi,byi->bxy", b, b)
    b_actual = b_actual - eye
    # Inner products between original vectors should be non-zero.
    assert torch.all(torch.std(a_actual, dim=(1, 2)) > 0.01)
    # Inner products between orthonormal vectors should be zero.
    assert torch.all(torch.std(b_actual, dim=(1, 2)) < 1e-4)
    # The orthonormalized one should be similar to the original one.
    assert torch.all(QF.correl(a, b, dim=(1, 2)) > 0.9)
    # Assert that it uses the Gram-Schmidt process.
    for i in range(1, a.size(1)):
        c = QF.einsum("bi,bxi->bx", b[:, i], a[:, :i])
        assert torch.all(c.abs().mean(dim=1) < 1e-4)
