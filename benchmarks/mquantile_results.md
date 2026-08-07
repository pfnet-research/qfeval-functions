# `mquantile` algorithm benchmark

Measured on 2026-07-31. The benchmark compares the original all-window
sort with batched wavelet-matrix range selection and the automatic policy.
Every timed result is checked with
`torch.testing.assert_close(..., equal_nan=True)` against the sort result.

## Implementations

| Algorithm | Approach | Time | Extra workspace |
|:---|:---|:---|:---|
| `sort` | Sort every sliding window (original) | `O(N * span * log(span))` | `O(N * span)` |
| `wavelet` | Coordinate-compress and answer all batched range selections while building a wavelet matrix | `O(N * log(N) + N * log(U))` | `O(N)` |
| `auto` | Select the measured best implementation; use moving extrema for `q=0/1` | workload-dependent | workload-dependent |

`U` is the number of distinct values in a slice. All implementations use
tensor operations on the input device.

## Environment

- Apple M2 Max (12 cores), 64 GB RAM
- Python 3.12.12
- PyTorch 2.7.1
- CPU, 8 PyTorch threads
- `float32`, batch size 1, `q=0.5`
- One warmup and three measured repetitions; table values are medians

Commands:

```console
uv run python benchmarks/benchmark_mquantile.py \
  --warmup 1 --repeats 3
uv run python benchmarks/benchmark_mquantile.py \
  --case 262144:256 --case 65536:4096 \
  --warmup 1 --repeats 3
```

## Results

| N | span | algorithm | median (ms) | speedup vs sort |
|---:|---:|:---|---:|---:|
| 16,384 | 64 | `sort` | 3.542 | 1.00x |
| 16,384 | 64 | `wavelet` | 5.301 | 0.67x |
| 16,384 | 64 | `auto` | 3.432 | 1.03x |
| 16,384 | 256 | `sort` | 16.922 | 1.00x |
| 16,384 | 256 | `wavelet` | 5.173 | 3.27x |
| 16,384 | 256 | `auto` | 5.316 | 3.18x |
| 16,384 | 1,024 | `sort` | 80.540 | 1.00x |
| 16,384 | 1,024 | `wavelet` | 5.138 | 15.67x |
| 16,384 | 1,024 | `auto` | 5.072 | 15.88x |
| 16,384 | 4,096 | `sort` | 311.395 | 1.00x |
| 16,384 | 4,096 | `wavelet` | 4.759 | 65.43x |
| 16,384 | 4,096 | `auto` | 4.695 | 66.33x |
| 65,536 | 64 | `sort` | 13.816 | 1.00x |
| 65,536 | 64 | `wavelet` | 27.416 | 0.50x |
| 65,536 | 64 | `auto` | 14.308 | 0.97x |
| 65,536 | 256 | `sort` | 66.666 | 1.00x |
| 65,536 | 256 | `wavelet` | 28.254 | 2.36x |
| 65,536 | 256 | `auto` | 27.214 | 2.45x |
| 65,536 | 1,024 | `sort` | 333.934 | 1.00x |
| 65,536 | 1,024 | `wavelet` | 27.788 | 12.02x |
| 65,536 | 1,024 | `auto` | 27.723 | 12.05x |

For the endpoint case `N=65,536`, `span=1,024`, and `q=0`, the automatic
linear moving-minimum path took 0.691 ms versus 335.605 ms for sort
(485.88x).

## Comparison

- The compact vectorized window kernel remains best at `span=64`.
- Batched `wavelet` is fastest from `span=128` in crossover measurements.
  At `N=16,384`, it reaches 15.67x at `span=1,024` and 65.43x at
  `span=4,096`.
- `algorithm="auto"` keeps short workloads on vectorized window
  operations, switches to `wavelet` for large windows, and specializes
  endpoint quantiles with a linear moving-extremum algorithm.

These thresholds are hardware-dependent. The script exposes `--device`,
`--dtype`, `--batch`, `--threads`, and repeatable `--case`/`--algorithm`
options so they can be remeasured on a target system.
