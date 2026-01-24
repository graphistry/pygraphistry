# Benchmarks

Manual-only scripts for local performance checks. Not wired into CI.

Summary results go into `benchmarks/RESULTS.md` (raw outputs stay in `plans/`).

## Hop microbench

Run a small set of hop() scenarios across synthetic graphs.

```bash
uv run python benchmarks/run_hop_microbench.py --runs 5 --output /tmp/hop-microbench.md
```

## Frontier sweep

Sweep seed sizes on a fixed linear graph.

```bash
uv run python benchmarks/run_hop_frontier_sweep.py --runs 5 --nodes 100000 --edges 200000 --output /tmp/hop-frontier.md
```

Notes:
- Use `--engine cudf` for GPU runs when cuDF is available.
- Scripts print a table to stdout; `--output` writes Markdown results.

## Chain vs Yannakakis

Compare regular `chain()` against the Yannakakis same-path executor on synthetic graphs.

```bash
uv run python benchmarks/run_chain_vs_samepath.py --runs 7 --warmup 1 --output /tmp/chain-vs-samepath.md
```

To toggle non-adjacent WHERE experiments on synthetic scenarios:

```bash
uv run python benchmarks/run_chain_vs_samepath.py \
  --non-adj-mode value_prefilter \
  --non-adj-value-card-max 500 \
  --non-adj-order selectivity \
  --non-adj-bounds \
  --runs 7 --warmup 1
```

## Real-data GFQL

Run GFQL chain scenarios on demo datasets plus WHERE scenarios (df_executor), with separate sections and a per-section score.

```bash
uv run python benchmarks/run_realdata_benchmarks.py --runs 7 --warmup 1 --output /tmp/realdata-gfql.md
```

To test categorical domains for redteam:

```bash
uv run python benchmarks/run_realdata_benchmarks.py --datasets redteam50k --redteam-domain-categorical --runs 9 --warmup 2
```

To experiment with non-adjacent WHERE modes:

```bash
uv run python benchmarks/run_realdata_benchmarks.py \
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
uv run python benchmarks/run_realdata_benchmarks.py \
  --datasets redteam50k,transactions \
  --non-adj-mode auto \
  --non-adj-value-ops "==,!=" \
  --non-adj-value-card-max 10 \
  --runs 3 --warmup 1 --opt-max-call-ms 0
```

Auto mode defaults to `==,!=` with a value-cardinality cap of 300 when no explicit value ops/card max are provided.

To add NDV probe columns (high/low cardinality) and extra WHERE scenarios:

```bash
uv run python benchmarks/run_realdata_benchmarks.py \
  --datasets redteam50k,transactions \
  --ndv-probes --ndv-probe-buckets 3 --ndv-log \
  --runs 3 --warmup 1
```

To enable OpenTelemetry spans for df_executor:

```bash
GRAPHISTRY_OTEL=1 \
GRAPHISTRY_OTEL_DETAIL=1 \
uv run --with opentelemetry-api --with opentelemetry-sdk \
  python benchmarks/run_realdata_benchmarks.py --datasets redteam50k --runs 3 --warmup 1
```

To export spans to OTLP (optional):

```bash
GRAPHISTRY_OTEL=1 \
GRAPHISTRY_OTEL_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
uv run --with opentelemetry-api --with opentelemetry-sdk --with opentelemetry-exporter-otlp \
  python benchmarks/run_realdata_benchmarks.py --datasets redteam50k --runs 3 --warmup 1
```

To limit datasets:

```bash
uv run python benchmarks/run_realdata_benchmarks.py --datasets redteam50k,transactions --runs 7 --warmup 1
```

To focus on a subset of scenarios:

```bash
uv run python benchmarks/run_realdata_benchmarks.py \
  --datasets transactions,redteam50k \
  --skip-chain --where-filter ndv_ \
  --ndv-probes --ndv-probe-buckets 3 --ndv-log \
  --runs 3 --warmup 1 --max-scenario-seconds 5 --opt-max-call-ms 0
```

Available datasets: `redteam50k`, `transactions`, `facebook_combined`, `honeypot`, `twitter_demo`, `lesmiserables`, `twitter_congress`, `all`.
