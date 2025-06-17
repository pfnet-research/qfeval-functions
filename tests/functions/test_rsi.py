import math

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


def test_rsi_ema_helper_function() -> None:
    """Test the EMA helper function used by RSI."""
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
        df.T.ewm(alpha=0.1, adjust=False).mean().T.to_numpy(),
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
    expect_out2 = np.array([math.nan] * 14 + [0.] * 26)
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
    results2 = rsi(torch.stack((input1, input2)), use_sma=use_sma).numpy()
    np.testing.assert_allclose(
        results2[0],
        expect_out1,
        rtol_case1,
        atol_case1,
    )

    np.testing.assert_allclose(
        results2[1],
        expect_out2,
        rtol_case2,
        atol_case2,
    )


def test_rsi_basic_functionality() -> None:
    """Test basic RSI functionality with known price movements."""
    # Simple upward trend should produce high RSI
    upward_trend = torch.tensor(
        [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ]
    )
    result = rsi(upward_trend, span=14)

    # First 14 values should be NaN
    assert torch.isnan(result[:14]).all()
    # RSI should be high (>50) for upward trend
    assert result[14].item() > 50


def test_rsi_downward_trend() -> None:
    """Test RSI with downward price trend."""
    # Simple downward trend should produce low RSI
    downward_trend = torch.tensor(
        [
            24.0,
            23.0,
            22.0,
            21.0,
            20.0,
            19.0,
            18.0,
            17.0,
            16.0,
            15.0,
            14.0,
            13.0,
            12.0,
            11.0,
            10.0,
        ]
    )
    result = rsi(downward_trend, span=14)

    # First 14 values should be NaN
    assert torch.isnan(result[:14]).all()
    # RSI should be low (<50) for downward trend
    assert result[14].item() < 50


def test_rsi_constant_values() -> None:
    """Test RSI with constant price values."""
    constant_prices = torch.tensor([10.0] * 20)
    result = rsi(constant_prices, span=14)

    # First 14 values should be NaN
    assert torch.isnan(result[:14]).all()
    # RSI should be 0 for constant prices (no gains or losses)
    for i in range(14, len(result)):
        assert result[i].item() == 0.0


def test_rsi_alternating_pattern() -> None:
    """Test RSI with alternating up/down pattern."""
    alternating = torch.tensor(
        [
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
            10.0,
            12.0,
        ]
    )
    result = rsi(alternating, span=14)

    # First 14 values should be NaN
    assert torch.isnan(result[:14]).all()
    # RSI should be around 50 for balanced gains/losses
    assert 40 < result[14].item() < 60
    assert 40 < result[15].item() < 60


def test_rsi_different_spans() -> None:
    """Test RSI with different span parameters."""
    prices = torch.tensor(
        [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
            29.0,
        ]
    )

    for span in [5, 10, 14, 20]:
        if span <= len(prices):
            result = rsi(prices, span=span)

            # First (span) values should be NaN
            assert torch.isnan(result[:span]).all()
            # Remaining values should be finite and in valid RSI range [0, 100]
            finite_values = result[span:]
            assert torch.all(torch.isfinite(finite_values))
            assert torch.all(finite_values >= 0)
            assert torch.all(finite_values <= 100)


def test_rsi_sma_vs_ema() -> None:
    """Test RSI with SMA vs EMA calculation methods."""
    prices = torch.randn(30) + 50  # Random walk around 50

    result_ema = rsi(prices, span=14, use_sma=False)
    result_sma = rsi(prices, span=14, use_sma=True)

    # Both should have same shape
    assert result_ema.shape == result_sma.shape

    # Both should have NaN for first 14 values
    assert torch.isnan(result_ema[:14]).all()
    assert torch.isnan(result_sma[:14]).all()

    # Finite values should be in valid RSI range
    assert torch.all(result_ema[14:] >= 0)
    assert torch.all(result_ema[14:] <= 100)
    assert torch.all(result_sma[14:] >= 0)
    assert torch.all(result_sma[14:] <= 100)


def test_rsi_2d_tensors() -> None:
    """Test RSI with 2D tensors (multiple price series)."""
    prices = torch.tensor(
        [
            [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ],
            [
                24.0,
                23.0,
                22.0,
                21.0,
                20.0,
                19.0,
                18.0,
                17.0,
                16.0,
                15.0,
                14.0,
                13.0,
                12.0,
                11.0,
                10.0,
            ],
        ]
    )

    result = rsi(prices, span=14, dim=1)

    assert result.shape == prices.shape

    # First 14 values should be NaN for both series
    assert torch.isnan(result[:, :14]).all()

    # First series (upward) should have high RSI
    assert result[0, 14].item() > 50
    # Second series (downward) should have low RSI
    assert result[1, 14].item() < 50


def test_rsi_dimension_handling() -> None:
    """Test RSI with different dimension specifications."""
    prices = torch.randn(5, 20) + 50

    result_dim1 = rsi(prices, span=14, dim=1)
    result_dim_neg1 = rsi(prices, span=14, dim=-1)

    # Should be identical
    np.testing.assert_array_equal(result_dim1.numpy(), result_dim_neg1.numpy())

    # Test dim=0 with appropriate tensor size
    prices_for_dim0 = torch.randn(20, 5) + 50  # Swap dimensions for dim=0 test
    result_dim0 = rsi(prices_for_dim0, span=5, dim=0)
    assert result_dim0.shape == prices_for_dim0.shape


def test_rsi_extreme_values() -> None:
    """Test RSI with extreme price values."""
    # Large values
    large_prices = torch.tensor(
        [
            1e6,
            1e6 + 1e5,
            1e6 + 2e5,
            1e6 + 3e5,
            1e6 + 4e5,
            1e6 + 5e5,
            1e6 + 6e5,
            1e6 + 7e5,
            1e6 + 8e5,
            1e6 + 9e5,
            1e6 + 1e6,
            1e6 + 1.1e6,
            1e6 + 1.2e6,
            1e6 + 1.3e6,
            1e6 + 1.4e6,
        ]
    )
    result_large = rsi(large_prices, span=14)

    # Should still be in valid range
    finite_values = result_large[torch.isfinite(result_large)]
    assert torch.all(finite_values >= 0)
    assert torch.all(finite_values <= 100)

    # Very small values
    small_prices = torch.tensor(
        [
            1e-6,
            1.1e-6,
            1.2e-6,
            1.3e-6,
            1.4e-6,
            1.5e-6,
            1.6e-6,
            1.7e-6,
            1.8e-6,
            1.9e-6,
            2e-6,
            2.1e-6,
            2.2e-6,
            2.3e-6,
            2.4e-6,
        ]
    )
    result_small = rsi(small_prices, span=14)

    finite_values = result_small[torch.isfinite(result_small)]
    assert torch.all(finite_values >= 0)
    assert torch.all(finite_values <= 100)


def test_rsi_with_nan_values() -> None:
    """Test RSI behavior with NaN input values."""
    prices_with_nan = torch.tensor(
        [
            10.0,
            11.0,
            math.nan,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ]
    )
    result = rsi(prices_with_nan, span=14)

    # RSI function may handle NaN by treating it as 0 or skipping it
    # The actual behavior should be validated rather than assumed
    assert result.shape == prices_with_nan.shape
    # First 14 values should be NaN regardless
    assert torch.isnan(result[:14]).all()


def test_rsi_single_large_move() -> None:
    """Test RSI with single large price movement."""
    # Mostly flat with one large upward move
    prices = torch.tensor(
        [
            10.0,
            10.1,
            10.0,
            10.1,
            10.0,
            10.1,
            10.0,
            50.0,
            10.1,
            10.0,
            10.1,
            10.0,
            10.1,
            10.0,
            10.1,
        ]
    )
    result = rsi(prices, span=14)

    # RSI should react to the large move
    assert torch.isnan(result[:14]).all()
    # The large gain should push RSI higher
    assert result[14].item() > 50


def test_rsi_boundary_conditions() -> None:
    """Test RSI at boundary conditions."""
    # Minimum length for RSI calculation
    min_prices = torch.tensor(
        [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ]
    )
    result = rsi(min_prices, span=14)

    assert len(result) == len(min_prices)
    assert torch.isnan(result[:14]).all()
    assert torch.isfinite(result[14])
    assert 0 <= result[14].item() <= 100


def test_rsi_mathematical_properties() -> None:
    """Test mathematical properties of RSI."""
    prices = torch.tensor(
        [
            10.0,
            12.0,
            11.0,
            13.0,
            12.0,
            14.0,
            13.0,
            15.0,
            14.0,
            16.0,
            15.0,
            17.0,
            16.0,
            18.0,
            17.0,
            19.0,
        ]
    )
    result = rsi(prices, span=14)

    # RSI should always be between 0 and 100
    finite_values = result[torch.isfinite(result)]
    assert torch.all(finite_values >= 0)
    assert torch.all(finite_values <= 100)

    # Test that pure gains produce high RSI
    pure_gains = torch.tensor(
        [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
        ]
    )
    result_gains = rsi(pure_gains, span=14)
    assert result_gains[14].item() > 90  # Should be very high RSI


def test_rsi_ema_implementation() -> None:
    """Test EMA implementation used in RSI calculation."""
    test_data = torch.randn(5, 20)
    alpha = 0.1

    result = _ema_2dim_recursive(test_data, alpha)

    assert result.shape == test_data.shape
    assert result.dtype == test_data.dtype

    # First values should equal input first values
    np.testing.assert_array_equal(result[:, 0].numpy(), test_data[:, 0].numpy())

    # Verify EMA formula for second value
    expected_second = alpha * test_data[:, 1] + (1 - alpha) * test_data[:, 0]
    np.testing.assert_allclose(
        result[:, 1].numpy(), expected_second.numpy(), rtol=1e-10
    )


def test_rsi_batch_processing() -> None:
    """Test RSI with batch processing across multiple series."""
    batch_size = 4
    seq_length = 20
    span = 14

    # Create batch of price series with different patterns
    prices = torch.randn(batch_size, seq_length) + 50
    prices[0] = torch.linspace(40, 60, seq_length)  # Upward trend
    prices[1] = torch.linspace(60, 40, seq_length)  # Downward trend
    prices[2] = torch.full((seq_length,), 50.0)  # Constant

    result = rsi(prices, span=span, dim=1)

    assert result.shape == (batch_size, seq_length)

    # Verify patterns
    assert result[0, span].item() > 50  # Upward trend -> high RSI
    assert result[1, span].item() < 50  # Downward trend -> low RSI
    assert result[2, span].item() == 0  # Constant -> zero RSI


def test_rsi_numerical_stability() -> None:
    """Test RSI numerical stability with edge cases."""
    # Very small price changes
    small_changes = torch.tensor(
        [
            100.0,
            100.0001,
            100.0002,
            100.0001,
            100.0003,
            100.0002,
            100.0004,
            100.0003,
            100.0005,
            100.0004,
            100.0006,
            100.0005,
            100.0007,
            100.0006,
            100.0008,
        ]
    )
    result_small = rsi(small_changes, span=14)

    finite_values = result_small[torch.isfinite(result_small)]
    assert torch.all(finite_values >= 0)
    assert torch.all(finite_values <= 100)

    # Test with zero gains and losses (should result in RSI = 0)
    zero_change = torch.full((20,), 50.0)
    result_zero = rsi(zero_change, span=14)

    finite_values = result_zero[torch.isfinite(result_zero)]
    assert torch.all(finite_values == 0)
