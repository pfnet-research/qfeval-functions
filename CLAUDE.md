# qfeval-functions Development Guide

PyTorch-based quantitative finance library for mathematical operations on financial data.

## Environment Setup

**CRITICAL**: Always use `uv` for all Python commands: `uv run python`, `uv run pytest`
- Python 3.9+ required
- Dependencies: PyTorch, NumPy, SciPy, Pandas, Pytest

## Essential Commands

```bash
make test           # Run all tests
make test-cov       # Run with coverage
make format         # Format code (black + isort)
make lint           # Run all linting
uv run pytest tests/functions/test_xyz.py -v  # Run specific test
```

## Testing Requirements

### Test Structure
- Naming: `test_{function}_{specific_case}()` (e.g., `test_fillna_edge_case_all_nans()`)
- Required test categories:
  1. Basic functionality
  2. Edge cases (empty/single-element tensors, all NaN/inf)
  3. Multi-dimensional (2D, 3D, 4D) and negative indexing
  4. Dtype preservation (float32/float64)
  5. Special values (NaN, ±inf)
  6. Comparison with reference implementations (NumPy/SciPy/Pandas)

### Test Utilities
Use `tests.functions.test_utils` for common assertions:
```python
from tests.functions.test_utils import assert_basic_properties
assert_basic_properties(result, input_tensor, expected_shape=input_tensor.shape)
```

### Known Limitations
- `apply_for_axis`: Only works with dimension-preserving functions
- Statistical tests may suppress `SmallSampleWarning` for sparse data

## Code Standards

- **Imports**: stdlib → third-party → local (use absolute imports)
- **Functions**: Preserve dtype/device, handle NaN properly, support multi-dimensional tensors
- **Docs**: Include type hints, edge cases, and examples in docstrings

## Technical Notes

- **apply_for_axis**: Requires dimension-preserving functions
- **Performance**: Use vectorized PyTorch operations, avoid Python loops
- **Pandas deprecations**: Some tests use workarounds for deprecated features

## Development Workflow

1. Run `make test` before changes
2. Run `make format` after implementation
3. Run `make lint` before committing
4. Add comprehensive tests for new functions
