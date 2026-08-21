#!/usr/bin/env python3
"""Benchmark the exact moving-quantile implementations.

Run from the repository root, for example:

    uv run python benchmarks/benchmark_mquantile.py --repeats 5

Each implementation is checked against the original all-window sort before
its timing is reported.
"""

import argparse
import math
import pathlib
import platform
import statistics
import sys
import time
from dataclasses import dataclass

import torch

# A script's directory, rather than the repository root, is `sys.path[0]`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qfeval_functions.functions as QF  # noqa: E402
from qfeval_functions.functions.mquantile import (  # noqa: E402
    MQuantileAlgorithm,
)

ALGORITHMS: tuple[MQuantileAlgorithm, ...] = (
    "sort",
    "wavelet",
    "auto",
)
DEFAULT_CASES = (
    (16_384, 64),
    (16_384, 256),
    (16_384, 1_024),
    (16_384, 4_096),
    (65_536, 64),
    (65_536, 256),
    (65_536, 1_024),
)


@dataclass(frozen=True)
class Result:
    n: int
    span: int
    algorithm: MQuantileAlgorithm
    samples: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) * 1_000


def _parse_case(value: str) -> tuple[int, int]:
    try:
        n_text, span_text = value.split(":", maxsplit=1)
        n, span = int(n_text), int(span_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must have the form N:SPAN"
        ) from error
    if n <= 0 or span <= 0:
        raise argparse.ArgumentTypeError("N and SPAN must be positive")
    return n, span


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_algorithm(
    x: torch.Tensor,
    span: int,
    q: float,
    algorithm: MQuantileAlgorithm,
    warmup: int,
    repeats: int,
) -> tuple[Result, torch.Tensor]:
    with torch.no_grad():
        result = QF.mquantile(x, span, q, dim=-1, algorithm=algorithm)
        _synchronize(x.device)
        for _ in range(warmup):
            result = QF.mquantile(x, span, q, dim=-1, algorithm=algorithm)
            _synchronize(x.device)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = QF.mquantile(x, span, q, dim=-1, algorithm=algorithm)
            _synchronize(x.device)
            samples.append(time.perf_counter() - start)

    return (
        Result(
            n=x.shape[-1],
            span=span,
            algorithm=algorithm,
            samples=tuple(samples),
        ),
        result,
    )


def _format_results(results: list[Result], args: argparse.Namespace) -> str:
    baseline = {
        (result.n, result.span): result.median_ms
        for result in results
        if result.algorithm == "sort"
    }
    lines = [
        "# mquantile benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- PyTorch: {torch.__version__}",
        f"- Platform: {platform.platform()}",
        f"- Device: {args.device}",
        f"- dtype / batch / q: {args.dtype} / {args.batch} / {args.q}",
        f"- PyTorch CPU threads: {torch.get_num_threads()}",
        f"- Warmup / measured repetitions: {args.warmup} / {args.repeats}",
        "",
        "| N | span | algorithm | median (ms) | vs sort |",
        "|---:|---:|:---|---:|---:|",
    ]
    for result in results:
        sort_ms = baseline[(result.n, result.span)]
        speedup = sort_ms / result.median_ms
        lines.append(
            f"| {result.n:,} | {result.span:,} | "
            f"{result.algorithm} | {result.median_ms:.3f} | "
            f"{speedup:.2f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        dest="cases",
        help="benchmark case as N:SPAN; may be repeated",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=ALGORITHMS,
        dest="algorithms",
        help="algorithm to include; may be repeated (sort is always included)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="set PyTorch CPU threads; 0 keeps the environment default",
    )
    args = parser.parse_args()

    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch/repeats must be positive and warmup non-negative")
    if not 0.0 <= args.q <= 1.0 or math.isnan(args.q):
        parser.error("q must be in [0, 1]")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cases = tuple(args.cases) if args.cases else DEFAULT_CASES
    requested = tuple(args.algorithms) if args.algorithms else ALGORITHMS
    algorithms = ("sort",) + tuple(
        algorithm for algorithm in requested if algorithm != "sort"
    )
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    results = []

    for n, span in cases:
        if span > n:
            parser.error(f"span ({span}) must not exceed N ({n})")
        x = torch.randn((args.batch, n), dtype=dtype, generator=generator).to(
            device
        )
        expected = None
        for algorithm in algorithms:
            result, actual = _time_algorithm(
                x,
                span,
                args.q,
                algorithm,
                args.warmup,
                args.repeats,
            )
            if expected is None:
                expected = actual
            else:
                torch.testing.assert_close(actual, expected, equal_nan=True)
            results.append(result)
            description = (
                f"N={n} span={span} {algorithm}: " f"{result.median_ms:.3f} ms"
            )
            print(description, flush=True)

    print()
    print(_format_results(results, args))


if __name__ == "__main__":
    main()
