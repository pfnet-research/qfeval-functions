"""Common test utilities and helpers for qfeval-functions test suite.

This module provides reusable utility functions for testing PyTorch functions
across the qfeval-functions library. The utilities focus on common test patterns
like dtype preservation, device preservation, memory efficiency, and consistency.
"""

from typing import Any
from typing import Callable
from typing import Optional

import torch


def assert_basic_properties(
    result: torch.Tensor,
    input_tensor: torch.Tensor,
    expected_shape: Optional[torch.Size] = None,
    check_dtype: bool = True,
    check_device: bool = True,
) -> None:
    """Assert basic properties: dtype, device, and optionally shape preservation."""
    if expected_shape is None:
        expected_shape = input_tensor.shape

    assert result.shape == expected_shape

    if check_dtype:
        assert result.dtype == input_tensor.dtype

    if check_device:
        assert result.device == input_tensor.device


def generic_test_dtype_preservation(
    func: Callable, *args: Any, **kwargs: Any
) -> None:
    """Generic dtype preservation test."""
    test_dtypes = [torch.float32, torch.float64]

    for dtype in test_dtypes:
        if args:
            # Convert first argument to test dtype
            test_args = list(args)
            if isinstance(test_args[0], torch.Tensor):
                test_args[0] = test_args[0].to(dtype=dtype)
            result = func(*test_args, **kwargs)
        else:
            x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            result = func(x, **kwargs)

        if isinstance(result, torch.Tensor):
            assert result.dtype == dtype
        elif hasattr(result, "values"):  # NamedTuple with values
            assert result.values.dtype == dtype


def generic_test_device_preservation(
    func: Callable, *args: Any, **kwargs: Any
) -> None:
    """Generic device preservation test."""
    if args:
        test_args = list(args)
        input_device = (
            test_args[0].device
            if hasattr(test_args[0], "device")
            else torch.device("cpu")
        )
        result = func(*test_args, **kwargs)
    else:
        x = torch.tensor([1.0, 2.0, 3.0])
        input_device = x.device
        result = func(x, **kwargs)

    if isinstance(result, torch.Tensor):
        assert result.device == input_device
    elif hasattr(result, "values"):  # NamedTuple with values
        assert result.values.device == input_device


def generic_test_memory_efficiency(
    func: Callable, tensor_size: int = 100, iterations: int = 3, **kwargs: Any
) -> None:
    """Generic memory efficiency test."""
    for i in range(iterations):
        x = torch.randn(tensor_size)
        result = func(x, **kwargs)
        # Force deletion
        del x, result


def generic_test_single_element(func: Callable, **kwargs: Any) -> None:
    """Generic single element test."""
    x_single = torch.tensor([42.0])
    result = func(x_single, **kwargs)
    assert result.shape == x_single.shape


def generic_test_consistency(
    func: Callable, test_data: torch.Tensor, **kwargs: Any
) -> None:
    """Test that multiple calls produce same result."""
    result1 = func(test_data, **kwargs)
    result2 = func(test_data, **kwargs)
    torch.testing.assert_close(result1, result2, equal_nan=True)
