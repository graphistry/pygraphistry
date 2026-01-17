# Benchmark Results Log

Summary-only log for notable benchmark runs. Raw per-scenario outputs live in
`plans/` (gitignored) and should be referenced here.

| Date | Commit | Scripts | Summary | Notes |
|------|--------|---------|---------|-------|
| 2026-01-17 | f492135e (feat/where-clause-executor) | `run_chain_vs_samepath.py` (median-of-7, warmup-1); `run_realdata_benchmarks.py` (median-of-7, warmup-1) | Synthetic: yann/regular median ~0.51x (52/54 wins). Real data: expanded to 7 datasets, medians ~30–173ms. | Raw outputs: `plans/pr-886-where/benchmarks/phase-12-revert-8-11.md`, `plans/pr-886-where/benchmarks/phase-13-realdata.md` |
| 2026-01-17 | 7080e356 (feat/where-clause-executor) | `run_realdata_benchmarks.py` (median-of-7, warmup-1) | Real data now includes WHERE (df_executor): redteam ~14s, transactions ~11s, others ~14–282ms. Chain-only medians ~31–175ms. | Raw outputs: `plans/pr-886-where/benchmarks/phase-14-realdata.md` |
| 2026-01-17 | 2e2e7e18 (feat/where-clause-executor) | `run_realdata_benchmarks.py` (median-of-7, warmup-1) | Added per-section scores. Chain score (median of medians) 72.78ms; WHERE score 247.07ms. | Raw outputs: `plans/pr-886-where/benchmarks/phase-14-realdata.md` |
| 2026-01-17 | 6bec468b (feat/where-clause-executor) | `run_realdata_benchmarks.py --datasets redteam50k --runs 9 --warmup 2` | Redteam-only rerun: chain score 157.83ms; WHERE score 13.12s. Low selectivity (WHERE keeps ~83.6% nodes / 74.3% edges). | Raw outputs: `plans/pr-886-where/benchmarks/phase-14-redteam-highruns.md`, `plans/pr-886-where/benchmarks/phase-14-redteam-selectivity.md` |
