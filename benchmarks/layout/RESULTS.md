# Treemap Layout — Performance Benchmark Results

**Date**: 2026-04-04
**Python**: 3.10.17 (local, algo section) / 3.13.12 (dgx-spark, E2E section)
**Platform**: linux
**Backends**: ref=yes (local only)  cpu=yes  gpu=yes (cudf, dgx-spark)
**Repeats per cell**: algo=500  e2e=200 (median + p95 reported)
**Canvas**: 1000.0×1000.0

## Algorithm benchmark (normalize + layout, no DataFrame overhead)

> Run locally (Python 3.10) where the reference package is installed.

| n_partitions | largest_size | ref median | ref p95 | cpu median | cpu p95 | speedup (ref/cpu) |
|-------------:|-------------:|-----------:|--------:|-----------:|--------:|------------------:|
|            2 |         1000 |      5.8µs |   7.5µs |      5.7µs |   6.1µs |             1.02× |
|            5 |         1000 |     14.4µs |  23.0µs |     14.3µs |  14.9µs |             1.00× |
|           10 |         1000 |     28.6µs |  35.6µs |     28.6µs |  34.3µs |             1.00× |
|           25 |         1000 |     75.3µs |  84.4µs |     74.5µs |  82.9µs |             1.01× |
|           50 |         1000 |    160.8µs | 175.8µs |    156.9µs | 240.1µs |             1.02× |
|          100 |         1000 |    364.8µs | 472.9µs |    363.3µs | 454.9µs |             1.00× |
|          250 |         1000 |     1.15ms |  1.40ms |     1.13ms |  1.38ms |             1.02× |
|          500 |         1000 |     2.75ms |  3.10ms |     2.70ms |  2.98ms |             1.02× |

> **speedup** > 1× means our CPU impl is faster than the reference.
> speedup < 1× means slower (overhead from numpy vs pure Python for tiny n).

## E2E benchmark: treemap() on pre-resident data (pandas CPU vs cuDF GPU)

> Run on dgx-spark (NVIDIA DGX, Rapids 26.02, Python 3.13).
> Data built ONCE before timing. CPU=pandas already in memory, GPU=cuDF already on device.
> `gpu/cpu` ratio < 1× means GPU is faster. Crossover at ~50k total nodes.

| n_partitions | nodes/partition | total_nodes | cpu median | cpu p95 | gpu median | gpu p95 | gpu/cpu |
|-------------:|----------------:|------------:|-----------:|--------:|-----------:|--------:|--------:|
|            2 |              10 |          20 |    244.0µs | 278.7µs |    406.6µs | 436.6µs |   1.67× |
|            2 |             100 |         200 |    244.8µs | 268.4µs |    410.2µs | 433.7µs |   1.68× |
|            2 |            1000 |        2000 |    251.6µs | 272.3µs |    410.2µs | 426.0µs |   1.63× |
|            5 |              10 |          50 |    249.9µs | 276.2µs |    414.6µs | 430.4µs |   1.66× |
|            5 |             100 |         500 |    254.9µs | 284.7µs |    417.7µs | 433.6µs |   1.64× |
|            5 |            1000 |        5000 |    270.8µs | 298.4µs |    415.6µs | 455.8µs |   1.53× |
|           10 |              10 |         100 |    261.2µs | 297.0µs |    427.3µs | 443.9µs |   1.64× |
|           10 |             100 |        1000 |    265.5µs | 283.8µs |    429.0µs | 451.0µs |   1.62× |
|           10 |            1000 |       10000 |    306.2µs | 322.6µs |    431.2µs | 449.8µs |   1.41× |
|           25 |              10 |         250 |    313.3µs | 335.9µs |    464.3µs | 484.6µs |   1.48× |
|           25 |             100 |        2500 |    323.7µs | 350.5µs |    458.1µs | 479.2µs |   1.42× |
|           25 |            1000 |       25000 |    427.5µs | 451.2µs |    473.0µs | 494.7µs |   1.11× |
|           50 |              10 |         500 |    372.2µs | 396.0µs |    523.7µs | 548.6µs |   1.41× |
|           50 |             100 |        5000 |    394.7µs | 417.9µs |    526.4µs | 541.6µs |   1.33× |
|           50 |            1000 |       50000 |    600.2µs | 623.2µs |    540.2µs | 570.0µs | **0.90×** |
|          100 |              10 |        1000 |    521.3µs | 552.6µs |    668.2µs | 688.0µs |   1.28× |
|          100 |             100 |       10000 |    562.6µs | 585.5µs |    678.6µs | 711.7µs |   1.21× |
|          100 |            1000 |      100000 |    949.3µs | 976.8µs |     1.02ms |  1.05ms |   1.07× |
|          250 |              10 |        2500 |     1.08ms |  1.10ms |     1.21ms |  1.24ms |   1.13× |
|          250 |             100 |       25000 |     1.18ms |  1.21ms |     1.23ms |  1.25ms |   1.04× |
|          250 |            1000 |      250000 |     2.13ms |  2.15ms |     1.79ms |  1.83ms | **0.84×** |
|          500 |              10 |        5000 |     2.22ms |  2.25ms |     2.33ms |  2.36ms |   1.05× |
|          500 |             100 |       50000 |     2.40ms |  2.43ms |     2.36ms |  2.41ms | **0.98×** |
|          500 |            1000 |      500000 |     4.34ms |  4.39ms |     3.36ms |  3.43ms | **0.77×** |

## Interpretation

- **CPU parity with reference:** built-in impl matches removed dep within noise (1.00–1.02×).
- **Vectorized coordinate transforms:** replaced 4× per-row dict `.map()` lookups with a single
  DataFrame merge for both normalize and global-positioning steps. CPU E2E baseline dropped from
  ~800µs to ~250µs vs the pre-vectorization implementation.
- **E2E GPU vs CPU:** cuDF groupby has a fixed per-call overhead (~400µs baseline on this machine).
  GPU becomes competitive around **50k total nodes** (n_parts × nodes_per_partition).
  At 250k nodes GPU is 1.19× faster; at 500k nodes 1.30× faster.
- **Layout algorithm itself is CPU-bound** (sequential recursion, not GPU-parallelisable).
  The GPU path transfers partition sizes to CPU before layout runs — layout time is identical.
- For typical usage (10–100 partitions, thousands of nodes), total treemap latency is
  sub-millisecond and negligible vs upstream community detection (hundreds of ms to seconds).
