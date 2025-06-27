from math import inf
from math import nan
from typing import Any

import torch

import qfeval_functions.functions as QF


def pytest_configure(config: Any) -> None:
    """Configure pytest with doctest setup."""
    # This will make the imports available in all doctests
    doctest_namespace = {
        "torch": torch,
        "QF": QF,
        "nan": nan,
        "inf": inf,
    }

    # Set the global namespace for doctests
    import pytest

    setattr(pytest, "doctest_namespace", doctest_namespace)


# Alternative approach: Use pytest's doctest_namespace fixture
def pytest_runtest_setup(item: Any) -> None:
    """Set up doctest namespace for each test."""
    if hasattr(item, "dtest"):
        # Add the imports to the doctest namespace
        item.dtest.globs.update(
            {
                "torch": torch,
                "QF": QF,
                "nan": nan,
                "inf": inf,
            }
        )
