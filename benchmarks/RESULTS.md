# Benchmark Results Log

Summary-only log for notable benchmark runs. Raw per-scenario outputs live in
`plans/` (gitignored) and should be referenced here.

| Date | Commit | Scripts | Summary | Notes |
|------|--------|---------|---------|-------|
| 2026-01-17 | f492135e (feat/where-clause-executor) | `run_chain_vs_samepath.py` (median-of-7, warmup-1); `run_realdata_benchmarks.py` (median-of-7, warmup-1) | Synthetic: yann/regular median ~0.51x (52/54 wins). Real data: expanded to 7 datasets, medians ~30–173ms. | Raw outputs: `plans/pr-886-where/benchmarks/phase-12-revert-8-11.md`, `plans/pr-886-where/benchmarks/phase-13-realdata.md` |
