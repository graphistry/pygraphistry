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

## Real-data GFQL

Run GFQL chain scenarios on demo datasets (no WHERE predicates).

```bash
uv run python benchmarks/run_realdata_benchmarks.py --runs 7 --warmup 1 --output /tmp/realdata-gfql.md
```

To limit datasets:

```bash
uv run python benchmarks/run_realdata_benchmarks.py --datasets redteam50k,transactions --runs 7 --warmup 1
```

Available datasets: `redteam50k`, `transactions`, `facebook_combined`, `honeypot`, `twitter_demo`, `lesmiserables`, `twitter_congress`, `all`.
