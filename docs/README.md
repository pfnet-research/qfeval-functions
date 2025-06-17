# Documentation

This directory contains the Sphinx documentation for qfeval-functions.

## Building Documentation

To build the HTML documentation:
```bash
make docs
```

To run doctests:
```bash
make docs-test
```

To clean build artifacts:
```bash
make docs-clean
```

## Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index
- `quickstart.rst` - Quick start guide
- `functions/` - Individual function documentation
  - `index.rst` - Function reference index
  - `_template.rst` - Template for new function pages
  - Individual `.rst` files for each function
- `_build/` - Build output (ignored by git)
- `_static/` - Static files (ignored by git)
- `_templates/` - Custom templates (ignored by git)

## Adding Function Documentation

1. Copy `functions/_template.rst` to `functions/function_name.rst`
2. Replace `function_name` with the actual function name
3. Add examples using `.. doctest::` blocks
4. Add the function to the appropriate section in `functions/index.rst`
5. Use `.. autofunction::` to include the function's docstring

## Doctests

Documentation examples are automatically tested via:
- Sphinx doctest builder for `.rst` files
- pytest doctest for Python module docstrings

All examples must be valid and executable.