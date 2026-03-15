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
- Warm speedup: `24.51x`
- This rerun now measures the smoother local Cypher `MATCH ... RETURN GRAPH` search stages around local Cypher `CALL graphistry.{igraph,cugraph}.pagerank.write()`.
- Stage medians:
  - Search 1 via local Cypher `RETURN GRAPH`: `50.06s` CPU vs `2.36s` GPU (`21.18x`)
  - PageRank via local Cypher write: `21.92s` CPU vs `0.47s` GPU (`46.28x`)
  - Search 2 via local Cypher `RETURN GRAPH`: `11.63s` CPU vs `0.57s` GPU (`20.28x`)
- Graph sizes:
  - Full graph: `107,614` nodes / `30,494,866` edges on both engines
  - After search 1 via local Cypher `RETURN GRAPH`: `73,010` nodes / `11,755,106` edges on both engines
  - Final graph after PageRank cutoff + search 2 via local Cypher `RETURN GRAPH`:
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
