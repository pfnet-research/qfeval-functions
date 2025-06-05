import random
import types
import typing

import numpy as np
import torch
import typing_extensions


class Seed(object):
    qfeval_rng: np.random.Generator = np.random.Generator(
        np.random.MT19937(random.randrange(0, 1 << 64))
    )
    fast: bool = True

    def __init__(self, seed: typing.Optional[int], fast: bool) -> None:
        super().__init__()
        self.push(seed, fast)

    def __enter__(self) -> "Seed":
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> typing_extensions.Literal[False]:
        self.pop()
        return False

    def push(
        self,
        seed: typing.Optional[int],
        fast: bool,
    ) -> None:
        self.state = {
            "qfeval": Seed.qfeval_rng.bit_generator.state,
            "fast": Seed.fast,
            "random": random.getstate(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.random.get_rng_state_all(),
            "numpy": np.random.get_state(),
        }
        Seed.qfeval_rng.bit_generator.state = np.random.MT19937(seed).state
        Seed.fast = fast
        random.seed(seed)
        # PyTorch accepts only a 64-bit integer.  For better randomness, this
        # generates a seed via random package, which should be initialized by
        # system urandom.
        s = random.randrange(1 << 64) if seed is None else seed
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        # NumPy accepts only a 32-bit integer or a list of 32-bit integers.
        # For better randomness, this generates a seed via PyTorch, which
        # should be initialized by a 64-bit integer.
        numpy_seed = torch.randint(1 << 16, (16,)) if seed is None else seed
        np.random.seed(numpy_seed)  # type: ignore

    def pop(self) -> None:
        Seed.qfeval_rng.bit_generator.state = self.state["qfeval"]  # type: ignore
        Seed.fast = self.state["fast"]  # type: ignore
        random.setstate(
            self.state["random"]  # type:ignore[arg-type]
        )  # type:ignore[return-value,arg-type]
        torch.random.set_rng_state(self.state["torch"])  # type: ignore
        torch.cuda.random.set_rng_state_all(self.state["torch_cuda"])  # type: ignore
        np.random.set_state(self.state["numpy"])  # type: ignore


def seed(seed: typing.Optional[int] = 42, fast: bool = False) -> Seed:
    """Sets the seed for generating random numbers."""
    return Seed(seed, fast)


def rng() -> np.random.Generator:
    return Seed.qfeval_rng


def is_fast() -> bool:
    return Seed.fast
