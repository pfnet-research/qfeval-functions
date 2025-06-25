#!/usr/bin/env python3
"""Script to run doctests on docstring examples in the codebase."""

import doctest
import importlib
import os
import sys
from pathlib import Path
from typing import List
from typing import Tuple


def find_python_modules(base_path: Path) -> List[str]:
    """Find all Python modules in the given base path."""
    modules = []
    for py_file in base_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert file path to module name
        rel_path = py_file.relative_to(base_path.parent)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
        modules.append(module_name)

    return modules


def run_doctest_for_module(module_name: str) -> Tuple[int, int]:
    """Run doctest for a single module and return (failures, tests)."""
    try:
        module = importlib.import_module(module_name)
        result = doctest.testmod(module, verbose=False, report=False)
        return result.failed, result.attempted
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return 0, 0
    except Exception as e:
        print(f"Error running doctest for {module_name}: {e}")
        return 1, 1


def main() -> None:
    """Run doctests for all modules in the project."""
    project_root = Path(__file__).parent.parent
    qfeval_path = project_root / "qfeval_functions"

    if not qfeval_path.exists():
        print(f"Error: {qfeval_path} does not exist")
        sys.exit(1)

    # Add project root to Python path
    sys.path.insert(0, str(project_root))

    modules = find_python_modules(qfeval_path)

    total_failures = 0
    total_tests = 0

    print("Running doctests...")

    for module_name in sorted(modules):
        failures, tests = run_doctest_for_module(module_name)
        total_failures += failures
        total_tests += tests

        if tests > 0:
            status = "PASS" if failures == 0 else "FAIL"
            print(
                f"{module_name}: {tests} tests, {failures} failures [{status}]"
            )

    print(f"\nSummary: {total_tests} tests, {total_failures} failures")

    if total_failures > 0:
        print("Some doctests failed!")
        sys.exit(1)
    else:
        print("All doctests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
