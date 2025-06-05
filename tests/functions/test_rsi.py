import numpy as np
import pandas as pd
import pytest
import torch

from qfeval_functions.functions.rsi import _ema_2dim_recursive
from qfeval_functions.functions.rsi import rsi


def _naive_ema_1dim(x: torch.Tensor, alpha: float) -> torch.Tensor:
    y = [x[0]]
    for i in range(len(x) - 1):
        y.append((1 - alpha) * y[i] + alpha * float(x[i + 1]))
    return torch.Tensor(y)


def test_ema() -> None:
    test_input = torch.rand(20, 30)
    df = pd.DataFrame(test_input.numpy())
    res = _ema_2dim_recursive(test_input, 0.1)
    naive_res = torch.stack([_naive_ema_1dim(v, 0.1) for v in test_input])

    np.testing.assert_allclose(
        res.numpy(),
        naive_res.numpy(),
        1e-4,
        1e-4,
    )
    np.testing.assert_allclose(
        res.numpy(),
        df.ewm(alpha=0.1, axis=1, adjust=False).mean().to_numpy(),
        1e-4,
        1e-4,
    )


@pytest.mark.parametrize("use_sma", [False, True])
def test_rsi_compare_to_talib(use_sma: bool) -> None:
    # fmt: off
    input1 = torch.Tensor(
        [0.6235, 0.1706, 0.3752, 0.7054, 0.8990, 0.6094, 0.0059, 0.3593, 0.1826,
         0.9151, 0.7466, 0.4087, 0.0056, 0.9460, 0.6626, 0.0086, 0.0736, 0.1902,
         0.7002, 0.9204, 0.0557, 0.0570, 0.2576, 0.0705, 0.3982, 0.8095, 0.3947,
         0.1699, 0.5444, 0.7492, 0.9748, 0.9154, 0.9813, 0.4870, 0.1490, 0.3727,
         0.4822, 0.4026, 0.2695, 0.9464])
    input2 = torch.zeros(40)
    # talib.RSI(np.float64(input1.numpy())) to get below result
    expect_out1 = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, 50.35751093,
         44.61314913, 45.28153382, 46.52713003, 51.70675654, 53.78877762,
         45.49614267, 45.50944472, 47.67913364, 45.84585291, 49.50802769,
         53.73696277, 49.25609654, 46.97039567, 51.04635441, 53.1660725,
         55.45432227, 54.69645998, 55.42427616, 49.05848796, 45.23257006,
         48.11649394, 49.51854405, 48.49300354, 46.74952339, 55.5080972])
    expect_out2 = np.array([torch.nan] * 14 + [0.] * 26)
    rtol_case1 = 1e-4
    atol_case1 = 1e-4
    if use_sma:  # If use_sma=True and use_sma=False result is not too far.
        rtol_case1 = 0.14
        atol_case1 = 6.2
    rtol_case2 = 1e-10  # case2 is always 0
    atol_case2 = 1e-10

    # fmt: on

    results = [rsi(input1, use_sma=use_sma).numpy(), rsi(input2).numpy()]

    np.testing.assert_allclose(
        results[0],
        expect_out1,
        rtol_case1,
        atol_case1,
    )

    np.testing.assert_allclose(
        results[1],
        expect_out2,
        rtol_case2,
        atol_case2,
    )
    results = rsi(torch.stack((input1, input2)), use_sma=use_sma).numpy()
    np.testing.assert_allclose(
        results[0],
        expect_out1,
        rtol_case1,
        atol_case1,
    )

    np.testing.assert_allclose(
        results[1],
        expect_out2,
        rtol_case2,
        atol_case2,
    )
