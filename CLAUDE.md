# Claude Code Development Guide

This document provides essential information for Claude Code when working on the qfeval-functions project.

## Project Overview

This is a PyTorch-based quantitative finance functions library that provides mathematical operations for financial analysis. The library includes functions for:

- Statistical operations (moving averages, standard deviations, correlations)
- Mathematical transformations (eigenvalue decomposition, PCA, orthogonalization)
- Time series operations (cumulative functions, shifts, fills)
- Financial indicators (RSI, Bollinger bands, EMA)
- Special handling for NaN values and edge cases

## Development Environment

### Python Environment
- **Always use `uv` for Python execution**: `uv run python`, `uv run pytest`, etc.
- The project uses uv for dependency management and virtual environment handling
- Python version: 3.13+ (see pyproject.toml for exact requirements)

### Key Dependencies
- PyTorch: Core tensor operations
- NumPy: Numerical computations and testing
- SciPy: Statistical reference implementations
- Pandas: Data frame operations (limited usage)
- Pytest: Testing framework

## Common Development Tasks

### Testing
```bash
# Run all tests
make test

# Run specific test files
uv run pytest tests/functions/test_apply_for_axis.py -v

# Run tests with coverage
make test-cov

# Run only function tests
uv run pytest tests/functions/ -v
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Run all linting checks
make lint

# Individual linting tools
make lint-black    # Black formatting check
make lint-isort    # Import sorting check
make flake8        # Code style check
make mypy          # Type checking
```

### Project Structure
```
qfeval_functions/
├── functions/          # Core function implementations
│   ├── apply_for_axis.py
│   ├── eigh.py
│   ├── fillna.py
│   ├── mstd.py
│   └── ...
└── random/            # Random number generation utilities

tests/
├── functions/         # Function tests
│   ├── test_apply_for_axis.py
│   ├── test_eigh.py
│   └── ...
└── random/           # Random tests
```

## Testing Guidelines

### Test Categories
1. **Basic functionality tests**: Core operation verification
2. **Edge case tests**: NaN, infinity, empty tensors, single elements
3. **Dimension tests**: Multi-dimensional tensor handling
4. **Numerical precision tests**: Float64 precision, very small/large values
5. **Error handling tests**: Invalid inputs, boundary conditions

### Known Testing Considerations

#### Warning Suppression
Some tests suppress specific warnings that are expected behavior:

- **SmallSampleWarning**: In `test_nankurtosis.py` and `test_nanskew.py` when testing with sparse data containing many NaNs
- These warnings are expected and suppressed to focus on functionality verification

#### Function Compatibility
- **apply_for_axis function**: Only works with functions that maintain batch dimension sizes
- Functions that change tensor dimensions (like reductions without keepdim=True) will fail
- Always test dimension-preserving operations with this function

### Test Data Patterns
- Use `torch.randn()` for random test data
- Include NaN and infinity values for robustness testing
- Test with various dtypes (float32, float64, int32)
- Include multi-dimensional tensors (2D, 3D, 4D)

## Code Style and Conventions

### Function Implementation
- Follow existing patterns in the codebase
- Handle multi-dimensional tensors appropriately
- Preserve input tensor properties (dtype, device)
- Implement proper NaN handling where applicable

### Import Organization
- Standard library imports first
- Third-party imports (torch, numpy, etc.)
- Local imports last
- Use absolute imports for clarity

### Documentation
- Add comprehensive docstrings for new functions
- Include parameter and return type information
- Document edge cases and limitations
- Provide usage examples for complex functions

## Known Issues and Limitations

### Current Technical Debt
1. **apply_for_axis limitations**: Function assumes output tensor has same batch dimension as input
2. **Small sample warnings**: Some statistical tests require warning suppression for edge cases
3. **Pandas deprecations**: Some tests use deprecated pandas features with workarounds

### Performance Considerations
- Functions are optimized for PyTorch tensor operations
- Avoid Python loops where possible - use vectorized operations
- Consider memory usage for large tensor operations

## Development Workflow

1. **Before making changes**: Run `make test` to ensure baseline functionality
2. **After implementation**: Run `make format` to ensure code style compliance
3. **Before committing**: Run `make lint` to verify all quality checks pass
4. **For new functions**: Add comprehensive test coverage including edge cases

## Contact and Support

For questions about development practices or architectural decisions, consult the project maintainer or refer to existing code patterns in the repository.
