#!/usr/bin/env python3
"""Benchmark moving rank and extremum-distance implementations.

Run from the repository root, for example:

    uv run python benchmarks/benchmark_moving_order.py --repeats 5

Every fast result is checked against the materialized-window implementation
before its timing is reported.
"""

import argparse
import pathlib
import statistics
import sys
import time
from collections.abc import Callable

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qfeval_functions.functions as QF  # noqa: E402
from qfeval_functions.functions.apply_for_axis import (  # noqa: E402
    apply_for_axis,
)
from qfeval_functions.functions.margmax import (  # noqa: E402
    _mextremum_distance_compare,
)
from qfeval_functions.functions.margmax import (  # noqa: E402
    _mextremum_distance_predecessor,
)
from qfeval_functions.functions.mrank import _mrank_compare  # noqa: E402
from qfeval_functions.functions.mrank import _mrank_wavelet  # noqa: E402

DEFAULT_CASES = (
    (16_384, 64),
    (16_384, 256),
    (16_384, 512),
    (16_384, 1_024),
    (65_536, 256),
    (65_536, 1_024),
)


def _parse_case(value: str) -> tuple[int, int]:
    try:
        n_text, span_text = value.split(":", maxsplit=1)
        n, span = int(n_text), int(span_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must have the form N:SPAN"
        ) from error
    if n <= 0 or span <= 0 or span > n:
        raise argparse.ArgumentTypeError("require N >= SPAN > 0")
    return n, span


def _time(
    function: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        result = function()
        for _ in range(warmup):
            result = function()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = function()
            samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1_000, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_parse_case)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch/repeats must be positive and warmup non-negative")
    torch.set_num_threads(args.threads)

    cases = tuple(args.case) if args.case else DEFAULT_CASES
    generator = torch.Generator().manual_seed(args.seed)
    print("| N | span | function | baseline ms | fast ms | auto ms | speedup |")
    print("|---:|---:|:---|---:|---:|---:|---:|")
    for n, span in cases:
        x = torch.randn((args.batch, n), generator=generator)
        functions = (
            (
                "mrank",
                lambda: apply_for_axis(
                    lambda values: _mrank_compare(values, span, True), x, 1
                ),
                lambda: apply_for_axis(
                    lambda values: _mrank_wavelet(values, span, True), x, 1
                ),
                lambda: QF.mrank(x, span, dim=1),
            ),
            (
                "margmax",
                lambda: apply_for_axis(
                    lambda values: _mextremum_distance_compare(
                        values, span, True
                    ),
                    x,
                    1,
                ),
                lambda: apply_for_axis(
                    lambda values: _mextremum_distance_predecessor(
                        values, span, True
                    ),
                    x,
                    1,
                ),
                lambda: QF.margmax(x, span, dim=1),
            ),
            (
                "margmin",
                lambda: apply_for_axis(
                    lambda values: _mextremum_distance_compare(
                        values, span, False
                    ),
                    x,
                    1,
                ),
                lambda: apply_for_axis(
                    lambda values: _mextremum_distance_predecessor(
                        values, span, False
                    ),
                    x,
                    1,
                ),
                lambda: QF.margmin(x, span, dim=1),
            ),
        )
        for name, baseline, fast, automatic in functions:
            baseline_ms, expected = _time(baseline, args.warmup, args.repeats)
            fast_ms, actual = _time(fast, args.warmup, args.repeats)
            torch.testing.assert_close(actual, expected, equal_nan=True)
            auto_ms, actual = _time(automatic, args.warmup, args.repeats)
            torch.testing.assert_close(actual, expected, equal_nan=True)
            print(
                f"| {n:,} | {span:,} | {name} | {baseline_ms:.3f} | "
                f"{fast_ms:.3f} | {auto_ms:.3f} | "
                f"{baseline_ms / auto_ms:.2f}x |",
                flush=True,
            )


if __name__ == "__main__":
    main()
