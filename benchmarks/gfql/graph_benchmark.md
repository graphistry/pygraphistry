# Graph Benchmark q1-q9 (graph-benchmark)

This benchmark replays q1-q9 from `prrao87/graph-benchmark` against Graphistry using pandas/cuDF and GFQL filters.
It expects the benchmark repo to be checked out as a sibling (default: `/home/lmeyerov/Work/graph-benchmark`) and
its dataset generated with `generate_data.sh`.

## Setup

```sh
# In the sibling repo
cd /home/lmeyerov/Work/graph-benchmark
bash generate_data.sh 100000
```

## Run

```sh
cd /home/lmeyerov/Work/pygraphistry
python benchmarks/graph_benchmark_q1_q9.py --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark
```

Optional flags:

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --runs 5 \
  --warmup 1 \
  --output-json /tmp/graph_benchmark_q1_q9.json
```

Preindexed variant (relation/type split per query, still vectorized pandas):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --runs 5 --warmup 1
```

Include preindex build time in per-query medians (adds `preindex_ms` and `median_ms_with_preindex`):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --include-preindex \
  --runs 5 --warmup 1
```

Presorted variant (global sort by rel/src/dst and node_type/node_id):

```sh
python benchmarks/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode presorted \
  --runs 5 --warmup 1
```

## Notes

- q1-q7 use GFQL filters to match the graph-benchmark query intent, then pandas aggregates for counts/averages.
- q8-q9 count all length-2 paths (including multiplicity) with vectorized degree math over FOLLOWS edges.
- The dataset uses separate ID spaces per node type; the loader offsets them into a single ID space.

## Memgraph comparison on DGX

Use `run_graph_benchmark_memgraph_dgx.sh` on `dgx-spark` to run the same q1-q9 workload across GFQL CPU, GFQL GPU, and Memgraph:

```sh
ssh -o BatchMode=yes dgx-spark
cd /home/lmeyerov/Work/pygraphistry2
GRAPH_BENCHMARK_ROOT=$HOME/graph-benchmark \
RESULTS_DIR=plans/gfql-memgraph-benchmarks/results \
RUNS=5 WARMUP=1 RAPIDS_VERSION=26.02 \
benchmarks/gfql/run_graph_benchmark_memgraph_dgx.sh
```

The wrapper expects generated graph-benchmark parquet data under `$GRAPH_BENCHMARK_ROOT/data/output/`. It starts Memgraph as a host container with `/tmp` mounted for CSV staging, runs GFQL CPU/GPU in a RAPIDS container using `--network host`, loads Memgraph with `--load-method csv`, then renders `graph_benchmark_gfql_memgraph.md` from the three JSON files. The comparison table reports GFQL query-only and query-plus-preindex accounting separately, while each GFQL JSON includes `query_policies` for the effective per-query policy names. `GFQL_QUERY_VARIANT=standard` is the default and applies the simple benchmark policy: direct dataframe shortcuts for q3/q4/q5/q6/q7, plus scoped setup-time uniqueness-gated lazy Polars CPU, q6 interest-id/gender indexes, and typed cuDF GPU paths for large HAS_INTEREST edge sets when available. The CPU policy includes q5 location-first final semi-join, q6 setup-time interest-id and gender-indexed location join, and q7 target-country-first path pruning. Tiny runs stay on the base dataframe paths. `GFQL_QUERY_VARIANT=dataframe-shortcut` forces dataframe shortcuts and is retained for exploratory sweeps.



Benchgraph-like neighbors-with-data/filter probe:

```sh
python benchmarks/gfql/graph_benchmark_neighbors_probe.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --engine cpu-policy \
  --depth 2 \
  --strategy strategy-policy \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-neighbors-probe.json
```

The neighbors probe accepts `--engine pandas|cudf|polars|cpu-policy|both|all`, `--depth 2|3|4`, and `--strategy both|path-join|preaggregated|factorized-preaggregated|strategy-policy`. Default `both` keeps Polars optional and parity-checks path join vs preaggregation. `cpu-policy` is scoped to this Benchgraph-like workload: pandas below 1,000,000 FOLLOWS rows, Polars at or above that threshold. `strategy-policy` keeps path join for small/shallow workloads and uses count-carrying factorized preaggregation for large depth >= 3 runs across engines. DGX evidence: depth-2 full CPU policy selected Polars at about 42 ms; refreshed depth-3 full factorized preaggregation ran about 92 ms on Polars CPU-policy and 67 ms on cuDF; refreshed depth-4 ran about 104 ms on Polars CPU-policy and 90 ms on cuDF.


Pokec-style expansion/filter probe:

```sh
python benchmarks/gfql/graph_benchmark_expansion_probe.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --engine all \
  --depths 1,2,3,4 \
  --workload all \
  --strategy strategy-policy \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-expansion-probe.json
```

The expansion probe mirrors advertised Pokec `expansion_1..4` and `expansion_*_with_filter` shapes over graph-benchmark FOLLOWS. It reports exact-hop distinct endpoint counts from a deterministic top-outdegree seed. `strategy-policy` uses frontier deduplication, which tiny runs parity-check against path joins. DGX evidence on full data: Polars is best for expansion_1/2/3/4 at about 5 ms / 9 ms / 9 ms / 18 ms, and filtered variants at about 6 ms / 11 ms / 12 ms / 18 ms; cuDF stays below about 24 ms but does not beat Polars on this coverage slice.


Current benchmark evidence ledger: `plans/gfql-memgraph-benchmarks/results/current-benchmark-evidence.md` separates repeated medians, one-run triage, GFQL-internal probes, comparator rows, and pending/blocker rows. Update it with `RESULTS.md` after any new benchmark run.

Advertised workload coverage notes:

- Covered with DGX evidence: q1-q9, Benchgraph-like neighbors depth 2-4, Kuzu q1-q9/neighbors, Pokec-style expansion/filter depth 1-4, anchored repeated-alias 2-cycles, and bounded seeded directed triangles.
- Partial: short/long fixed-hop patterns are expressible through existing forward/reverse hop mechanics, but exact `pattern_long`/`LIMIT 1` rows are not separately benchmarked.
- Gaps: shortest/all-shortest path rows still need the most engine work; global/high-intersection cyclic stress remains optional coverage before deeper CSR/WCOJ work.
- Planner microbenchmarks (`starts_with`, `or_filter`, BFS-expand-from-source, indexed order/count variants) and LDBC-style analytical rows remain follow-up coverage after path batching and any needed exact pattern rows.



Repeated-alias cycle probe:

```sh
python benchmarks/gfql/graph_benchmark_pattern_probe.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --engine all \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-pattern-probe.json
```

The pattern probe covers anchored repeated-alias 2-cycles like `MATCH (n)-[e1]->(m)-[e2]->(n) RETURN m` and bounded seeded directed triangles (`seed -> a -> b -> seed`). DGX full 2-cycle medians show direct binary joins are already fast: Polars about 1.66 ms, pandas about 2.52 ms, cuDF about 3.07 ms. Full directed-triangle seed-count 30 selected 28 triangle-bearing seeds and still favored binary joins: cuDF about 6.34 ms, Polars about 40.52 ms, pandas about 41.57 ms. Cypher supports the repeated-alias 2-cycle for pandas/cuDF, but directed-triangle repeated-alias row projection is not yet supported. This demotes CSR/WCOJ for the tested bounded cyclic shapes; reserve it for global/high-intersection triangle or clique-like stress if needed.

Shortest-path/BFS probe:

```sh
python benchmarks/gfql/graph_benchmark_path_probe.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --engine pandas \
  --shortest-path-backend bfs \
  --max-hops 4 \
  --source-count 1 --target-count 3 \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-path-probe.json
```

The path probe benchmarks current supported Cypher `shortestPath` length rows over FOLLOWS and parity-checks against dataframe BFS distance oracles. It intentionally keeps supported single-pair shortestPath loop coverage, and also records rejected Cypher-level batched strategies; path-list projection and all-shortest rows remain unsupported/coverage gaps. DGX triage: original tiny 1x3 pairs ran about 100 ms pandas/BFS and 129 ms cuDF/cugraph versus about 3 ms for a dataframe BFS oracle. The reuse-adjacency refresh on full data (`runs=3,warmup=1`, max_hops=3) shows pandas/Cypher about 2.11s and cuDF/Cypher about 1.42s, adjacency rebuild about 1.0s, but reusable-adjacency BFS about 2 ms. Surface Cypher batching is rejected for now because tiny batched strategies were seconds-scale. Treat this as prioritization evidence for lower-level seed-pushed batched multi-pair lowering plus graph-state/CSR reuse or MS-BFS, not a claim-level engine result.

Polars q5-q7 CPU probe:

```sh
python benchmarks/gfql/graph_benchmark_polars_q5_q7.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --runs 7 --warmup 2 \
  --output-json /tmp/graph-benchmark-polars-q5-q7.json
```

Manual Memgraph-only run against an existing Bolt service:

```sh
uv run --with neo4j python benchmarks/gfql/graph_benchmark_memgraph_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --uri bolt://127.0.0.1:7687 \
  --runs 5 --warmup 1 \
  --load-method csv \
  --csv-dir /tmp/gfql_memgraph_import \
  --output-json /tmp/graph_benchmark_memgraph.json
```
