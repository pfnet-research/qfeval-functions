# Moving rank and extremum-distance benchmark

Measured on 2026-07-31 with an Apple M2 Max, PyTorch 2.7.1, `float32`,
batch size 1, and eight CPU threads. Each fast and automatic result was
checked against the exact materialized-window result.

```console
uv run python benchmarks/benchmark_moving_order.py \
  --warmup 2 --repeats 7 --threads 8
```

The baseline below already includes the small-window micro-optimizations:
prefix-counted NaN masks for `mrank`, and a reversed `argmax`/`argmin`
reduction for `margmax`/`margmin`. Thus the speedups conservatively compare
the automatic policy with its optimized fallback, not with the slower
implementation that preceded this change.

| N | span | function | baseline (ms) | large-window (ms) | auto (ms) | auto speedup |
|---:|---:|:---|---:|---:|---:|---:|
| 16,384 | 64 | `mrank` | 0.635 | 4.149 | 0.590 | 1.08x |
| 16,384 | 64 | `margmax` | 0.485 | 1.847 | 0.502 | 0.97x |
| 16,384 | 64 | `margmin` | 0.392 | 1.925 | 0.509 | 0.77x |
| 16,384 | 256 | `mrank` | 2.090 | 3.977 | 2.134 | 0.98x |
| 16,384 | 256 | `margmax` | 1.150 | 1.868 | 1.013 | 1.13x |
| 16,384 | 256 | `margmin` | 1.107 | 1.839 | 1.231 | 0.90x |
| 16,384 | 512 | `mrank` | 5.440 | 4.116 | 4.268 | 1.27x |
| 16,384 | 512 | `margmax` | 2.157 | 1.808 | 1.772 | 1.22x |
| 16,384 | 512 | `margmin` | 1.984 | 1.892 | 1.776 | 1.12x |
| 16,384 | 1,024 | `mrank` | 8.446 | 5.892 | 4.339 | 1.95x |
| 16,384 | 1,024 | `margmax` | 4.101 | 1.934 | 1.982 | 2.07x |
| 16,384 | 1,024 | `margmin` | 4.251 | 1.954 | 1.922 | 2.21x |
| 65,536 | 256 | `mrank` | 9.029 | 27.387 | 8.743 | 1.03x |
| 65,536 | 256 | `margmax` | 4.719 | 8.639 | 4.800 | 0.98x |
| 65,536 | 256 | `margmin` | 5.170 | 8.661 | 4.447 | 1.16x |
| 65,536 | 1,024 | `mrank` | 32.602 | 26.696 | 26.738 | 1.22x |
| 65,536 | 1,024 | `margmax` | 15.212 | 8.474 | 8.445 | 1.80x |
| 65,536 | 1,024 | `margmin` | 14.916 | 8.428 | 8.477 | 1.76x |

The large-window paths reduce transient storage from `O(N * span)` to
linear in the input size. `mrank` uses wavelet-matrix range counts;
`margmax` and `margmin` combine a linear moving extremum with stable,
batched predecessor queries.
