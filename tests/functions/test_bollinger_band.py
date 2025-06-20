import numpy as np
import torch

import qfeval_functions.functions as QF
import pytest


def test_constant() -> None:
    # If constant, bollinger_band width is always 0
    x = torch.tensor([1.0] * 40)
    window = 20
    upper, middle, lower = QF.bollinger_band(x, window, 1)
    expected = torch.Tensor(
        [np.nan] * (window - 1) + [1.0] * (len(x) - window + 1)
    )
    assert torch.allclose(upper, expected, equal_nan=True)
    assert torch.allclose(middle, expected, equal_nan=True)
    assert torch.allclose(lower, expected, equal_nan=True)


def test_simple() -> None:
    x = torch.tensor([1.0 if i % 2 else 0.0 for i in range(100)])
    upper, middle, lower = QF.bollinger_band(x, 2, 1)
    # avg: 0.5
    # sigma^2 = (0.5^2 + -0.5^2)/2 = 0.25
    # sigma = 0.5
    assert torch.allclose(
        upper,
        torch.tensor([np.nan] + [0.5 + 0.5] * 99),
        atol=1e-5,
        equal_nan=True,
    )
    assert torch.allclose(
        middle, torch.tensor([np.nan] + [0.5] * 99), atol=1e-4, equal_nan=True
    )
    assert torch.allclose(
        lower,
        torch.tensor([np.nan] + [0.5 - 0.5] * 99),
        atol=1e-5,
        equal_nan=True,
    )


def test_tensor() -> None:
    x = torch.tensor(
        [[1.0 if i % 2 else 0.0 for i in range(20)] for _ in range(3)]
    )
    upper, middle, lower = QF.bollinger_band(x, 2, 1)
    assert upper.shape == middle.shape == lower.shape == torch.Size([3, 20])

    x = torch.rand(10, 10, 10)
    upper, middle, lower = QF.bollinger_band(x, 2, 1)
    assert upper.shape == (10, 10, 10)
    assert middle.shape == (10, 10, 10)
    assert lower.shape == (10, 10, 10)


def test_random_value_with_talib() -> None:
    # fmt: off
    x = torch.Tensor(
        [0.42279603, 0.5755721, 0.58751445, 0.03773417, 0.73638503,
         0.25402939, 0.04536971, 0.55620341, 0.62274016, 0.48510387,
         0.52237645, 0.7157161, 0.49254371, 0.68423531, 0.63047875,
         0.00441984, 0.31827711, 0.10711729, 0.09892804, 0.64533061,
         0.04406418, 0.35187429, 0.81887892, 0.41364349, 0.88688021,
         0.86142079, 0.09044755, 0.0078753, 0.19997286, 0.46492434,
         0.61874523, 0.61873171, 0.03989606, 0.42136836, 0.92513961,
         0.90268511, 0.9307005, 0.65926031, 0.48582011, 0.23783928,
         0.43594974, 0.49574985, 0.12702874, 0.26668398, 0.2251975,
         0.3460219, 0.87498694, 0.21398366, 0.65796, 0.45692477])
    # talib.BBANDS(x_np) to get result.
    talib_upper = torch.Tensor(
        [np.nan, np.nan, np.nan, np.nan, 0.94947543,
         0.94765601, 0.9004624, 0.88339553, 0.95286007, 0.81982693,
         0.8574532, 0.74324493, 0.74537667, 0.77850597, 0.7847418,
         1.02931536, 0.91773002, 0.89281711, 0.6802418, 0.69380729,
         0.68666464, 0.69873873, 0.99511442, 0.98379277, 1.12826523,
         1.13362123, 1.24156949, 1.19296335, 1.17827843, 0.94366286,
         0.7370592, 0.8654295, 0.85228795, 0.85669764, 1.10701368,
         1.23981106, 1.36098571, 1.16911079, 1.13843058, 1.16428522,
         1.01591097, 0.73353865, 0.65199593, 0.58230662, 0.58275724,
         0.53992867, 0.89428671, 0.88368225, 0.98515369, 0.97682798])
    talib_middle = torch.Tensor(
        [np.nan, np.nan, np.nan, np.nan, 0.47200036,
         0.43824703, 0.33220655, 0.32594434, 0.44294554, 0.39268931,
         0.44635872, 0.580428, 0.56769606, 0.57999509, 0.60907006,
         0.50547874, 0.42599094, 0.34890566, 0.23184421, 0.23481458,
         0.24274345, 0.24946288, 0.39181521, 0.4547583, 0.50306822,
         0.66653954, 0.61425419, 0.45205347, 0.40931934, 0.32492817,
         0.27639306, 0.38204989, 0.38845404, 0.43273314, 0.52477619,
         0.58156417, 0.64395793, 0.76783078, 0.78072113, 0.64326106,
         0.54991399, 0.46292386, 0.35647754, 0.31265032, 0.31012196,
         0.29213639, 0.36798381, 0.3853748, 0.46363, 0.50997545])
    talib_lower = torch.Tensor(
        [np.nan, np.nan, np.nan, np.nan, -0.00547471,
         -0.07116196, -0.2360493, -0.23150685, -0.06696899, -0.03444831,
         0.03526424, 0.41761107, 0.39001545, 0.38148421, 0.43339833,
         -0.01835787, -0.06574813, -0.19500579, -0.21655339, -0.22417814,
         -0.20117775, -0.19981297, -0.211484, -0.07427617, -0.1221288,
         0.19945785, -0.01306111, -0.28885641, -0.35963974, -0.29380652,
         -0.18427309, -0.10132972, -0.07537987, 0.00876864, -0.05746129,
         -0.07668272, -0.07306985, 0.36655077, 0.42301168, 0.1222369,
         0.08391701, 0.19230907, 0.06095916, 0.04299402, 0.03748668,
         0.04434412, -0.15831909, -0.11293266, -0.05789369, 0.04312293])
    # fmt: on
    upper, middle, lower = QF.bollinger_band(
        torch.stack((x,)), 5, 2.0
    )  # talib default value
    assert torch.allclose(
        upper[0],
        talib_upper,
        atol=1e-4,
        equal_nan=True,
    )
    assert torch.allclose(
        middle[0],
        talib_middle,
        atol=1e-4,
        equal_nan=True,
    )
    assert torch.allclose(
        lower[0],
        talib_lower,
        atol=1e-4,
        equal_nan=True,
    )
