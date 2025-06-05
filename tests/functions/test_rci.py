import torch

from qfeval_functions.functions.rci import rci


def test_rci() -> None:
    x1 = torch.Tensor([500, 510, 515, 520, 530])
    y1 = torch.Tensor([torch.nan] * 4 + [100.0])
    assert torch.allclose(
        rci(x1, period=5),
        y1,
        equal_nan=True,
    )

    x2 = torch.Tensor([530, 520, 515, 510, 500])
    y2 = torch.Tensor([torch.nan] * 4 + [-100.0])
    assert torch.allclose(
        rci(x2, period=5),
        y2,
        equal_nan=True,
    )
    assert torch.allclose(
        rci(torch.stack((x1, x2)), period=5),
        torch.stack((y1, y2)),
        equal_nan=True,
    )

    x3 = torch.Tensor([float(i) for i in range(50)])
    y3 = torch.Tensor([torch.nan] * 8 + [100.0] * 42)
    assert torch.allclose(
        rci(torch.stack((x3, -x3)), period=9),
        torch.stack((y3, -y3)),
        equal_nan=True,
    )

    # rci(x4)[4] : (1- 6*(3^2+1^2+1^2+1^2+0)/5/(5^2-1))*100 =  40
    # rci(-x4)[4]: (1- 6*(1^2+3^2+1^2+1^2+4^2)/5/(5^2-1))*100 =-40
    x4 = torch.Tensor([4, 1, 2, 3, 5])
    y4 = torch.Tensor([torch.nan] * 4 + [40.0])

    assert torch.allclose(
        rci(torch.stack((x4, -x4)), period=5),
        torch.stack((y4, -y4)),
        equal_nan=True,
    )
