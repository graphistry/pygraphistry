# GFQL Benchmarks

Manual-only scripts for local performance checks. A subset runs in CI on GFQL changes.

Summary results go into `benchmarks/gfql/RESULTS.md` (raw outputs stay in `plans/`).

## Hop microbench

Run a small set of hop() scenarios across synthetic graphs.

```bash
uv run python benchmarks/gfql/hop_microbench.py --runs 5 --output /tmp/hop-microbench.md
```

## Frontier sweep

Sweep seed sizes on a fixed linear graph.

```bash
uv run python benchmarks/gfql/hop_frontier_sweep.py --runs 5 --nodes 100000 --edges 200000 --output /tmp/hop-frontier.md
```

Notes:
- Use `--engine cudf` for GPU runs when cuDF is available.
- Scripts print a table to stdout; `--output` writes Markdown results.

## Chain vs Yannakakis

Compare regular `chain()` against the Yannakakis same-path executor on synthetic graphs.

```bash
uv run python benchmarks/gfql/chain_vs_samepath.py --runs 7 --warmup 1 --output /tmp/chain-vs-samepath.md
```

By default, WHERE uses auto mode (value-mode + domain semijoin auto for non-adj clauses, edge semijoin auto for edge clauses).
To compare against baseline behavior, set `--non-adj-mode baseline`.
Use `--max-scenario-seconds 20` to fail fast on synthetic timeouts (best-effort).

To focus on dense multi-clause scenarios:

```bash
uv run python benchmarks/gfql/chain_vs_samepath.py \
  --graph-filter medium_dense,large_dense \
  --scenario-filter nonadj_multi \
  --runs 5 --warmup 1
```

Use `--seed` to make synthetic graph generation repeatable across runs.

To toggle non-adjacent WHERE experiments on synthetic scenarios:

```bash
uv run python benchmarks/gfql/chain_vs_samepath.py \
  --non-adj-mode value_prefilter \
  --non-adj-value-card-max 500 \
  --non-adj-order selectivity \
  --non-adj-bounds \
  --runs 7 --warmup 1
```

## Real-data GFQL

Run GFQL chain scenarios on demo datasets plus WHERE scenarios (df_executor), with separate sections and a per-section score.

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py --runs 7 --warmup 1 --output /tmp/realdata-gfql.md
```

To force baseline WHERE behavior for comparisons:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --non-adj-mode baseline \
  --runs 7 --warmup 1 --output /tmp/realdata-baseline.md
```

To test categorical domains for redteam:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py --datasets redteam50k --redteam-domain-categorical --runs 9 --warmup 2
```

To experiment with non-adjacent WHERE modes:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --datasets redteam50k \
  --non-adj-mode value_prefilter \
  --non-adj-value-card-max 500 \
  --non-adj-order selectivity \
  --non-adj-bounds \
  --runs 7 --warmup 1
```

Auto mode (value for low NDV, domain semijoin for the rest):

```bash
GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO=1 \
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --datasets redteam50k,transactions \
  --non-adj-mode auto \
  --non-adj-value-ops "==,!=" \
  --non-adj-value-card-max 10 \
  --runs 3 --warmup 1 --opt-max-call-ms 0
```

To experiment with aggregated inequality pruning for 2-hop non-adj clauses:

```bash
GRAPHISTRY_NON_ADJ_WHERE_INEQ_AGG=1 \
uv run python benchmarks/gfql/realdata_benchmarks.py --datasets redteam50k --runs 3 --warmup 1
```

Auto mode defaults to `==,!=` with a value-cardinality cap of 300 when no explicit value ops/card max are provided.

To add NDV probe columns (high/low cardinality) and extra WHERE scenarios:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --datasets redteam50k,transactions \
  --ndv-probes --ndv-probe-buckets 3 --ndv-log \
  --runs 3 --warmup 1
```

To enable OpenTelemetry spans for df_executor:

```bash
GRAPHISTRY_OTEL=1 \
GRAPHISTRY_OTEL_DETAIL=1 \
uv run --with opentelemetry-api --with opentelemetry-sdk \
  python benchmarks/gfql/realdata_benchmarks.py --datasets redteam50k --runs 3 --warmup 1
```

To export spans to OTLP (optional):

```bash
GRAPHISTRY_OTEL=1 \
GRAPHISTRY_OTEL_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
uv run --with opentelemetry-api --with opentelemetry-sdk --with opentelemetry-exporter-otlp \
  python benchmarks/gfql/realdata_benchmarks.py --datasets redteam50k --runs 3 --warmup 1
```

To limit datasets:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py --datasets redteam50k,transactions --runs 7 --warmup 1
```

To focus on a subset of scenarios:

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --datasets transactions,redteam50k \
  --skip-chain --where-filter ndv_ \
  --ndv-probes --ndv-probe-buckets 3 --ndv-log \
  --runs 3 --warmup 1 --max-scenario-seconds 5 --opt-max-call-ms 0
```

Available datasets: `redteam50k`, `transactions`, `facebook_combined`, `honeypot`, `twitter_demo`, `lesmiserables`, `twitter_congress`, `all`.

## Optional Kuzu comparisons

If the `kuzu` Python package is installed, you can run optional Kuzu comparisons (currently redteam-only):

```bash
uv run python benchmarks/gfql/realdata_benchmarks.py \
  --datasets redteam50k \
  --kuzu --kuzu-db-root /tmp/kuzu_bench \
  --runs 3 --warmup 1
```

Use `--kuzu-rebuild` to recreate the Kuzu database from CSVs when needed.

## Graph-benchmark q1-q9

Replay the q1-q9 queries from https://github.com/prrao87/graph-benchmark against Graphistry.
See `benchmarks/gfql/graph_benchmark.md` for setup details.

```bash
uv run python benchmarks/gfql/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-q1-q9.json
```

Preindexed variant (relation/type split per query):

```bash
uv run python benchmarks/gfql/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-q1-q9-preindexed.json
```

Include preindex build time in per-query medians (adds `preindex_ms` and `median_ms_with_preindex`):

```bash
uv run python benchmarks/gfql/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode preindexed \
  --include-preindex \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-q1-q9-preindexed-with-preindex.json
```

Presorted variant (global sort by rel/src/dst and node_type/node_id):

```bash
uv run python benchmarks/gfql/graph_benchmark_q1_q9.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --mode presorted \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-q1-q9-presorted.json
```

## Memgraph q1-q9 comparison

Run GFQL CPU, GFQL GPU, and Memgraph q1-q9 on `dgx-spark` with the RAPIDS Docker workflow:

```bash
ssh -o BatchMode=yes dgx-spark
cd /home/lmeyerov/Work/pygraphistry2
GRAPH_BENCHMARK_ROOT=$HOME/graph-benchmark \
RESULTS_DIR=plans/gfql-memgraph-benchmarks/results \
RUNS=5 WARMUP=1 RAPIDS_VERSION=26.02 \
benchmarks/gfql/run_graph_benchmark_memgraph_dgx.sh
```

If `$HOME/graph-benchmark` is missing, clone `https://github.com/prrao87/graph-benchmark` on `dgx-spark` and run `bash generate_data.sh 100000` first. The wrapper starts Memgraph, runs GFQL CPU/GPU inside a RAPIDS container, loads Memgraph via server-side CSV by default, runs the Bolt query benchmark, and writes a comparison markdown table under `RESULTS_DIR`. The table shows GFQL query-only and query-plus-preindex accounting separately, while each GFQL JSON includes `query_policies` for the effective per-query policy names. `GFQL_QUERY_VARIANT=standard` is the default and applies the simple benchmark policy: direct dataframe shortcuts for q3/q4/q5/q6/q7, plus scoped setup-time uniqueness-gated lazy Polars CPU, q6 interest-id/gender indexes, and typed cuDF GPU paths for large HAS_INTEREST edge sets when available. The CPU policy includes q5 location-first final semi-join, q6 setup-time interest-id and gender-indexed location join, and q7 target-country-first path pruning. Tiny runs stay on the base dataframe paths. `GFQL_QUERY_VARIANT=dataframe-shortcut` forces dataframe shortcuts and is retained for exploratory sweeps. See `docs/source/gfql/benchmark_memgraph.rst` and `benchmarks/gfql/graph_benchmark.md`.


Benchgraph-like neighbors-with-data/filter probe:

```bash
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
- Partial: `pattern_long`/`LIMIT 1` now has tiny exact-row coverage; full unbounded dataframe reference is guarded by `--pattern-long-max-reference-edges` and is not a claim-level full result.
- Gaps: shortest/all-shortest path rows still need the most engine work; global/high-intersection cyclic stress remains optional coverage before deeper CSR/WCOJ work.
- Planner microbenchmarks (`starts_with`, `or_filter`, BFS-expand-from-source, indexed order/count variants) now have dataframe-equivalent tiny/full coverage; exact Cypher/comparator planner rows and LDBC-style analytical rows remain follow-up coverage after path batching and any needed exact pattern rows.



Repeated-alias cycle probe:

```sh
python benchmarks/gfql/graph_benchmark_pattern_probe.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --engine all \
  --runs 5 --warmup 1 \
  --output-json /tmp/graph-benchmark-pattern-probe.json
```

The pattern probe covers anchored repeated-alias 2-cycles like `MATCH (n)-[e1]->(m)-[e2]->(n) RETURN m`, bounded seeded directed triangles (`seed -> a -> b -> seed`), and tiny exact-row `pattern_long` coverage (`n1 -> n2 -> n3 -> n4 <- n5`, deterministic `ORDER BY n5 LIMIT 1`). DGX full 2-cycle medians show direct binary joins are already fast: Polars about 1.66 ms, pandas about 2.52 ms, cuDF about 3.07 ms. Full directed-triangle seed-count 30 selected 28 triangle-bearing seeds and still favored binary joins: cuDF about 6.34 ms, Polars about 40.52 ms, pandas about 41.57 ms. Cypher supports the repeated-alias 2-cycle for pandas/cuDF, but directed-triangle repeated-alias row projection is not yet supported. Tiny pattern_long medians were pandas Cypher 30.66 ms, cuDF Cypher 57.09 ms, and Polars binary join 52.47 ms; full unbounded pattern_long reference is guarded because the dataframe reference is already expensive on tiny data. This demotes CSR/WCOJ for the tested bounded cyclic shapes; reserve it for global/high-intersection triangle or clique-like stress if needed.

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

```bash
python benchmarks/gfql/graph_benchmark_polars_q5_q7.py \
  --graph-benchmark-root /home/lmeyerov/Work/graph-benchmark \
  --runs 7 --warmup 2 \
  --output-json /tmp/graph-benchmark-polars-q5-q7.json
```


## WHERE opt matrix (comparative)

Run a focused matrix of WHERE scenarios across opt profiles (value mode, domain semijoin, auto, edge semijoin, etc).
Outputs are grouped by profile + scenario group, with defaults targeting dense multi-clause and real-data stress cases.

```bash
uv run python benchmarks/gfql/where_opt_matrix.py --runs 3 --warmup 1
```

To target only dense multi-clause synthetic cases:

```bash
uv run python benchmarks/gfql/where_opt_matrix.py \
  --groups synthetic_multi_clause \
  --profiles baseline,auto,vector \
  --runs 5 --warmup 1
```

## GFQL filter + PageRank CPU vs GPU

Benchmark a realistic GFQL search -> local Cypher `CALL ... .write()` PageRank -> GFQL search pipeline on large SNAP social graphs, comparing `pandas+igraph` against `cudf+cugraph`.
The graph is loaded once, then the main pipeline is benchmarked warm on the resident graph.

```bash
uv run python benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_cpu_gpu.py \
  --dataset twitter \
  --engine both \
  --degree-quantile 0.99 \
  --pagerank-quantile 0.99 \
  --warmup 1 --runs 3
```

DGX-sized GPlus example:

```bash
# GPU
python benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_cpu_gpu.py \
  --dataset gplus \
  --engine cudf \
  --degree-quantile 0.995 \
  --pagerank-quantile 0.9995 \
  --warmup 1 --runs 2 \
  --output-json plans/gfql-gpu-pagerank-benchmark/results/gplus_gpu_q995_pr9995.json

# CPU
python benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_cpu_gpu.py \
  --dataset gplus \
  --engine pandas \
  --degree-quantile 0.995 \
  --pagerank-quantile 0.9995 \
  --warmup 1 --runs 1 \
  --output-json plans/gfql-gpu-pagerank-benchmark/results/gplus_cpu_q995_pr9995.json
```

Selected DGX result (`gplus`, `degree_q=0.995`, `pagerank_q=0.9995`):
- Warm CPU pipeline: `83.61s`
- Warm GPU pipeline: `3.41s`
- Warm speedup: `17.53x`
- This rerun now measures the smoother local Cypher `GRAPH { MATCH ... }` search stages around local Cypher `CALL graphistry.{igraph,cugraph}.pagerank.write()`.
- Stage medians:
  - Search 1 via local Cypher `GRAPH { }`: `57.3711s` CPU vs `2.5435s` GPU (`21.18x`)
  - PageRank via local Cypher write: `22.1346s` CPU vs `0.4668s` GPU (`46.28x`)
  - Search 2 via local Cypher `GRAPH { }`: `10.48s` CPU vs `0.5696s` GPU (`20.28x`)
- Graph sizes:
  - Full graph: `107,614` nodes / `30,494,866` edges on both engines
  - After search 1 via local Cypher `GRAPH { }`: `73,010` nodes / `11,755,106` edges on both engines
  - Final graph after PageRank cutoff + search 2 via local Cypher `GRAPH { }`:
    - CPU (`igraph`): `41,147` nodes / `1,341,817` edges
    - GPU (`cugraph`): `42,002` nodes / `1,278,572` edges
- Note: the final graph differs modestly because `igraph` and `cugraph` produce slightly different PageRank score distributions, so the top-quantile cutoff lands on a different boundary.
- Raw notes: `plans/gfql-gpu-pagerank-benchmark/results/gplus_q995_pr9995_summary.md`
- Notebook walkthrough: `demos/gfql/benchmark_filter_pagerank_cpu_gpu.ipynb`

## Cached load/prep CPU vs GPU

Benchmark cached SNAP ingest/prep separately from the warm GFQL -> PageRank -> GFQL pipeline.
This measures only local cached file -> in-memory graph preparation:
- edge-list read (`pandas.read_csv` / `cudf.read_csv`)
- node materialization (degree table + seed flag)
- Graphistry bind (`nodes(...).edges(...)`)

```bash
uv run python benchmarks/gfql/filter_pagerank/load_prepare_cpu_gpu.py \
  --dataset gplus \
  --engine both \
  --degree-quantile 0.995 \
  --warmup 2 --runs 5
```

Selected DGX cached-load results:
- Twitter (`degree_q=0.99`):
  - CPU prepare: `0.2756s`
  - GPU prepare: `0.1013s`
  - total speedup: `2.72x`
  - stage medians: read `0.2156s` vs `0.0862s`, node prep `0.0620s` vs `0.0148s`, bind ~`0.0001s` on both
- GPlus (`degree_q=0.995`):
  - CPU prepare: `8.7160s`
  - GPU prepare: `3.9323s`
  - total speedup: `2.22x`
  - stage medians: read `6.9096s` vs `3.0395s`, node prep `1.8097s` vs `0.8613s`, bind ~`0.0001s` on both
- Raw outputs: `plans/gfql-gpu-pagerank-benchmark/results/twitter_load_prepare_infer.json`, `plans/gfql-gpu-pagerank-benchmark/results/gplus_load_prepare_infer.json`

Optimization note:
- An explicit integer-dtype probe was not adopted. It was slightly slower on Twitter and overflowed on GPlus in pandas, so the benchmark keeps parser inference for now.

## DGX configuration

Current measured environment for the selected GPlus run:

- Host: `dgx-spark`
- GPU: `NVIDIA GB10`
- Driver: `580.126.09`
- Container: `graphistry/test-gpu:latest`
- Python: `3.12.12`
- pandas: `2.3.3`
- cudf: `25.12.00`
- cugraph: `25.12.02`
- igraph: `1.0.0`

The benchmark reports both full pipeline timings and split stage timings so we can separate:
- GFQL/dataframe acceleration (`pandas` vs `cudf`)
- graph algorithm acceleration (`igraph` vs `cugraph`)


### Neo4j exploratory comparison

Tracked manual benchmark script for the Neo4j + GDS analog:

```bash
uv run --no-project --with neo4j \
  python benchmarks/gfql/filter_pagerank/filter_pagerank_pipeline_neo4j.py \
  --dataset twitter \
  --degree-quantile 0.99 \
  --pagerank-quantile 0.99 \
  --warmup 1 --runs 3 \
  --output-json plans/gfql-gpu-pagerank-benchmark/results/twitter_neo4j_tracked_q99_pr99.json
```

Notes:
- Manual/DGX-only benchmark: requires local Docker access plus the `neo4j` Python driver.
- Defaults to Neo4j Community `2026.02.2` with GDS, `16G` heap, `16G` pagecache, and `32G` transaction memory.
- Reuses raw DB/import scratch space under `plans/gfql-gpu-pagerank-benchmark/neo4j/`.

Exact Twitter 3-way comparison (`degree_q=0.99`, `pagerank_q=0.99`):
- Graphistry CPU (`pandas + igraph`): `2.36s` warm pipeline
  - stage medians: search1 `0.84s`, pagerank `1.18s`, search2 `0.33s`
- Graphistry GPU (`cudf + cugraph`): `0.25s` warm pipeline
  - stage medians: search1 `0.14s`, pagerank `0.02s`, search2 `0.09s`
- Neo4j (`Neo4j + GDS`): `13.51s` warm pipeline
  - stage medians: filter1 `5.74s`, pagerank `3.20s`, filter2 `3.51s`
- Relative to the exact same Twitter shape, Graphistry CPU is `5.72x` faster than Neo4j and Graphistry GPU is `54.48x` faster.
- Stage 1 shape matches across all three engines: `44,273` nodes / `873,810` edges.
- Final graph drift remains modest because the PageRank backends/cutoff boundaries differ:
  - Graphistry CPU: `42,217` nodes / `618,212` edges
  - Graphistry GPU: `42,372` nodes / `586,116` edges
  - Neo4j: `43,068` nodes / `667,484` edges

Larger GPlus analog (`degree_q=0.995`, `pagerank_q=0.9995`):
- imported and runnable in Neo4j after switching import IDs to `string`
- naive single-transaction seed expansion OOMed at `dbms.memory.transaction.total.max`
- batched seed/core expansion fixed the OOM
- even with batching, the full pipeline exceeded `3m07s` before the main transaction reached `Closing`

So the honest current comparison is:
- Neo4j is workable for the smaller Twitter analog, but already materially slower than both Graphistry CPU and GPU on the exact same shape.
- On the selected GPlus benchmark shape, Neo4j is already dramatically slower than Graphistry CPU (`83.61s`) and Graphistry GPU (`3.41s`) before teardown/cleanup is even done.
- Raw notes: `plans/gfql-gpu-pagerank-benchmark/results/neo4j_summary.md`

## GFQL physical-index harnesses (`index_*.py`)

Catalog for the seeded-traversal adjacency-index work (PR #1658; see
`docs/source/gfql/index_adjacency.rst` for the user guide + published numbers):

- `index_smoke.py` / `index_ddl_smoke.py` — correctness smokes: differential
  parity (index path == scan path) and DDL/wire/`index_policy`/`gfql_explain`
  round-trips. Container-runnable MIRRORS of the canonical pytest suite
  (`graphistry/tests/compute/gfql/index/test_index.py`).
- `index_perf.py` — microbenchmark: seeded 1-hop/2-hop index-vs-scan across
  engines and frontier sizes; the cost-gate calibration numbers come from here.
- `index_takeover_bench.py` + `index_vs_kuzu_prepared.py` — **canonical
  competitor pair** behind the published "9–36x vs Kuzu" seeded-traversal
  claims: guarded timings (path-taken assertions + result==scan oracle),
  prepared-statement fairness on the Kuzu side.
- `index_vs_dbs.py` — broader 3-way sweep (GFQL / Kuzu / Neo4j) over the same
  seeded shapes; superset of the pair above, heavier setup (dockerized Neo4j).
- `index_bulk_olap_bench.py` — bulk group-by/degree OLAP shapes across the 4
  engines (answers "does the index help bulk scans?" — it does not; scans win).
- `index_largegraph_bench.py` — scale probe on large real graphs
  (flat-in-N behavior of the seeded path up to ~80M edges).

Methodology for all: warm medians, one engine per process for large runs,
NO-CHEATING guards (a timing is void unless the intended path was taken AND
the result matches the scan oracle). Run on an idle box; GPU rows need
`--gpus all` and RAPIDS.

### Planner microbench probe

`graph_benchmark_planner_probe.py` covers dataframe-equivalent planner-style rows: `starts_with`, `or_filter`, `indexed_order_by`, `parallel_counting`, and `bfs_expand_from_source`. Run tiny first, then full under `dgx-guard/safe_run.sh`:

```bash
python benchmarks/gfql/graph_benchmark_planner_probe.py \
  --graph-benchmark-root /tmp/graph-benchmark-gfql-memgraph \
  --engine all \
  --runs 3 \
  --warmup 1
```

Full DGX medians on 100k persons / 2.42M FOLLOWS were: starts_with pandas/cuDF/Polars 5.97/0.59/1.04 ms; or_filter 1.94/0.45/0.88 ms; indexed_order_by 29.33/11.95/8.34 ms; parallel_counting 0.02/0.02/0.01 ms; bfs_expand_from_source 30.15/2.54/3.75 ms. These are internal policy rows, not comparator claims: cuDF wins larger scan/filter and two-hop expansion, while Polars wins order/count.


### LDBC-style analytical probe

`graph_benchmark_ldbc_probe.py` covers q5/q6/q7-adjacent analytical rows over graph-benchmark data: branch semijoin count, interest-to-city top-k, age+interest-to-state top-1, and country+interest+FOLLOWS top-k. It is dataframe-equivalent internal policy evidence, not an external comparator claim.

```bash
python benchmarks/gfql/graph_benchmark_ldbc_probe.py \
  --graph-benchmark-root /tmp/graph-benchmark-gfql-memgraph \
  --engine all \
  --runs 3 \
  --warmup 1 \
  --output-json /tmp/gfql-ldbc-probe.json
```

Tiny DGX medians were branch pandas/cuDF/Polars 1.05/2.28/1.13 ms; city top-k 1.28/3.06/1.20 ms; state top-1 1.70/3.76/2.92 ms; country-interest-follow top-k 1.57/3.32/5.46 ms. Full DGX medians were branch 5.21/2.97/4.68 ms; city top-k 6.39/3.70/6.32 ms; state top-1 4.02/3.86/5.46 ms; country-interest-follow top-k 35.56/8.50/13.62 ms. Current interpretation: cuDF wins these full analytical semijoin/groupby rows, but they remain below/near the rough 50 ms sizable threshold, so use this as scoped policy evidence for selective analytical workloads rather than a broad GPU-always rule.
