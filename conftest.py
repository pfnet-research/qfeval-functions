import doctest
import re
from math import inf
from math import nan
from typing import Any

import torch

import qfeval_functions.functions as QF


class SpaceIgnoringOutputChecker(doctest.OutputChecker):
    """Custom doctest output checker that ignores spaces and removes trailing zeros."""

    def check_output(self, want: str, got: str, optionflags: int) -> bool:
        """Check output ignoring spaces and removing trailing zeros from decimals."""
        # First try the parent class check
        if super().check_output(want, got, optionflags):
            return True

        # Remove all whitespace
        want_normalized = re.sub(r"\s+", "", want)
        got_normalized = re.sub(r"\s+", "", got)

        # Remove trailing zeros from decimal numbers (e.g., 1.0000 -> 1., 2.5000 -> 2.5)
        # Pattern matches decimal numbers and removes trailing zeros (but keeps the decimal point)
        want_normalized = re.sub(
            r"(\d+\.\d*?)0+(?=\D|$)", r"\1", want_normalized
        )
        got_normalized = re.sub(r"(\d+\.\d*?)0+(?=\D|$)", r"\1", got_normalized)

        return want_normalized == got_normalized


def pytest_configure(config: Any) -> None:
    """Configure pytest with doctest setup."""
    # Set torch print options for consistent output formatting
    torch.set_printoptions(precision=4, sci_mode=False)

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

    # Patch doctest to use our custom output checker globally
    doctest.OutputChecker = SpaceIgnoringOutputChecker  # type: ignore[misc]


def pytest_runtest_setup(item: Any) -> None:
    """Set up doctest namespace for each test."""
    # Set torch print options for consistent output formatting
    torch.set_printoptions(precision=4, sci_mode=False)

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
