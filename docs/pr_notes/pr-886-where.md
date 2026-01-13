# PR 886 Notes: GFQL WHERE + hop performance

## GPU toggles / experiments
- `GRAPHISTRY_CUDF_SAME_PATH_MODE=auto|oracle|strict` controls same-path executor selection when `Engine.CUDF` is requested.
- `GRAPHISTRY_HOP_FAST_PATH=0` disables hop fast-path traversal for A/B comparisons.

## Commits worth toggling (GPU perf/debug)
- d05d9db9 perf(hop): domain-based fast path traversal
- 6cc23688 perf(hop): undirected single-pass expansion
- d1e11784 perf(df_executor): DF-native cuDF forward prune
- e85fa8e7 fix(filter_by_dict): allow bool filters on object columns

## Manual benchmarks (not in CI)
- `benchmarks/run_hop_microbench.py`
- `benchmarks/run_hop_frontier_sweep.py`
- Example: `uv run python benchmarks/run_hop_microbench.py --runs 5 --output /tmp/hop-microbench.md`
