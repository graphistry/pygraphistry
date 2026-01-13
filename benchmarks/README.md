# Benchmarks

Manual-only scripts for local performance checks. Not wired into CI.

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
