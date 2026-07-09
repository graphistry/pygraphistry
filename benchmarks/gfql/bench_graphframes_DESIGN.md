# GFQL vs Spark GraphFrames — benchmark design

Harness: `bench_graphframes.py`. Compares GFQL (dataframe-native, single-node
columnar) against Spark GraphFrames (JVM, `local[*]`) on SNAP graphs.

## Tasks (same semantics on both systems)

- **filter** — a `WHERE` on a numeric column. We precompute node `degree` at
  cold-load and filter nodes with `degree >= p90(degree)`. GFQL:
  `g.gfql([n(filter_dict={'degree': ge(thr)})])`. GraphFrames:
  `gf.degrees.filter(degree >= thr).count()`. SNAP graphs carry no attributes,
  so degree is the natural threshold column; the degree precompute is charged
  to cold-load for *both* systems.
- **hop1 / hop2** — 1- and 2-hop undirected neighborhood from a fixed 50-node
  high-degree seed set. GFQL: `[n(is_in seeds), e_undirected(hops=k), n()]`.
  GraphFrames has no k-hop-neighborhood primitive (`bfs` = shortest path
  between predicates, `find` = fixed motif), so we expand via iterated
  undirected edge joins — still pure Spark.
- **pagerank** — full-graph PageRank. GFQL CPU/polars →
  `g.compute_igraph('pagerank')`; GPU → `g.compute_cugraph('pagerank')`.
  GraphFrames → `gf.pageRank(resetProbability=0.15, maxIter=N)` (damping 0.85).

## Why median-of-5 + 2 warmups

Warmups absorb one-time costs — JIT, lazy-frame plan compilation, Spark JVM
class-loading and executor spin-up, filesystem cache priming — so timed runs
measure steady-state compute, not startup. Median (not mean) of 5 is robust to
the occasional GC pause / stop-the-world spike on a shared box. Cold load is
timed separately, once, because it is a different question (ETL cost) from
warm query latency.

## Fairness caveats (documented, not hidden)

- **Spark JVM warmup**: even after 2 warmups, `local[*]` carries per-query
  scheduler/task-serialization overhead that dominates on small results —
  Spark is built for distributed throughput, not single-node latency.
- **Materialization**: Spark is lazy; every task ends in `.count()` (or
  `.vertices.count()`) to force honest end-to-end timing. GFQL likewise
  materializes via `len(_nodes)/len(_edges)`.
- **`local[*]` vs distributed**: this measures single-box multicore, GraphFrames'
  single-node configuration, not a distributed cluster. A real cluster would amortize overhead differently;
  we are explicitly benchmarking the single-node regime where GFQL lives.
- **GFQL PageRank routing**: polars has no native PageRank; the polars engine
  converts to pandas and calls igraph. That conversion is inside the timed
  region (honest), but it means "gfql-polars pagerank" ≈ igraph-on-CPU.
- **PageRank iterations**: GraphFrames uses fixed `maxIter=N`; igraph/cugraph
  iterate to a tolerance. Not iteration-for-iteration identical — compare
  wall-clock-to-usable-scores, not per-iteration cost.

## Parity (same answer on both)

Each task returns a `result_size` written to JSONL: filter → node count above
threshold, hop → neighborhood size, pagerank → vertex count. Filter and hop
sizes should match exactly across systems (identical set semantics); a mismatch
flags a bug (e.g. directedness or seed-set drift). The harness validates
successful rows after the sweep and exits nonzero if any dataset/task has a
`result_size` mismatch across successful systems. PageRank scores are compared by
rank correlation (Spearman) of the top-K vertices offline, not exact values,
since the algorithms differ in convergence criteria.

## Guardrails

Every (system, task) is wrapped: an error/OOM records
`{"status":"error","error":...}` and the run continues. Missing pyspark /
graphframes / GPU is skipped with a message — never aborts the matrix.
Results stream to JSONL (one line per system×task×dataset), flushed per row so
a mid-run crash still leaves partial data.
```
python bench_graphframes.py --dataset lj --systems gfql-polars,graphframes \
    --tasks filter,hop1,hop2,pagerank --warmups 2 --iters 5 --out lj.jsonl
```
```
python bench_graphframes.py --dataset orkut --dry-run   # print plan only
```
