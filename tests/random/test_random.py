import random
import typing

import numpy as np
import pytest
import torch

from qfeval_functions import random as qfeval_random


def generate_random_values() -> typing.List[float]:
    return [
        random.random(),
        float(np.random.rand(1)[0]),
        float(torch.rand(())),
    ]


@pytest.mark.random
def test_seed() -> None:
    with qfeval_random.seed():
        a = generate_random_values()
        b = generate_random_values()
    rand_values = []
    with qfeval_random.seed():
        rand_values.append(generate_random_values())
        with qfeval_random.seed():
            rand_values.append(generate_random_values())
            with qfeval_random.seed():
                rand_values.append(generate_random_values())
            rand_values.append(generate_random_values())
        rand_values.append(generate_random_values())
    np.testing.assert_allclose(
        rand_values,
        [a, a, a, b, b],
    )


@pytest.mark.random
def test_seed_has_compatiblity() -> None:
    r"""Tests if random generators have backward compatiblity.

    NOTE: This is not guaranteed in most libraries.  However, this is
    good to know when upgrading image versions.
    """
    with qfeval_random.seed():
        values = generate_random_values()
        # Test that values are reasonable (between 0 and 1) rather than exact values
        for val in values:
            assert (
                0 <= val <= 1
            ), f"Random value {val} should be between 0 and 1"
        # Test that values are different (not all the same)
        assert len(set(values)) > 1, "Random values should be different"


@pytest.mark.random
def test_seed_with_none() -> None:
    r"""Tests if seed with None has enough randomness.

    NOTE: Theoretically, this can fail in a small probability, but it
    should be significantly small (i.e. <1e-6).
    """
    with qfeval_random.seed(None):
        a = generate_random_values()
    with qfeval_random.seed(None):
        b = generate_random_values()
    assert not np.allclose(a, b, 1e-8, 1e-8)
