import random
import typing

import numpy as np
import torch

from qfeval_functions import random as qfeval_random


class TestSeed:
    def test_seed(self) -> None:
        with qfeval_random.seed():
            a = self.generate_random_values()
            b = self.generate_random_values()
        rand_values = []
        with qfeval_random.seed():
            rand_values.append(self.generate_random_values())
            with qfeval_random.seed():
                rand_values.append(self.generate_random_values())
                with qfeval_random.seed():
                    rand_values.append(self.generate_random_values())
                rand_values.append(self.generate_random_values())
            rand_values.append(self.generate_random_values())
        np.testing.assert_allclose(
            rand_values,
            [a, a, a, b, b],
        )

    def test_seed_has_compatiblity(self) -> None:
        r"""Tests if random generators have backward compatiblity.

        NOTE: This is not guaranteed in most libraries.  However, this is
        good to know when upgrading image versions.
        """
        with qfeval_random.seed():
            np.testing.assert_allclose(
                self.generate_random_values(),
                [0.6394267984578837, 0.3745401188473625, 0.8822692632675171],
            )

    def test_seed_with_none(self) -> None:
        r"""Tests if seed with None has enough randomness.

        NOTE: Theoretically, this can fail in a small probability, but it
        should be significantly small (i.e. <1e-6).
        """
        with qfeval_random.seed(None):
            a = self.generate_random_values()
        with qfeval_random.seed(None):
            b = self.generate_random_values()
        assert not np.allclose(a, b, 1e-8, 1e-8)

    def generate_random_values(self) -> typing.List[float]:
        return [
            random.random(),
            float(np.random.rand(1)[0]),
            float(torch.rand(())),
        ]
