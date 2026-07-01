# dgx GB10 safety harness

Run **every** GPU / large-graph benchmark on the shared unified-memory dgx box through
`safe_run.sh`. Background: on the GB10, **GPU memory *is* system RAM** (unified, 119 GB,
16 GB swap). A 1.8B-edge cudf/cugraph load once OOM-thrashed the box for ~9.5 h.

Proven (2026-07-01): `docker --memory` is **transparent** to cudf/unified allocs (does not
contain them). The working caps are **RMM allocation limit** (cudf + cugraph decline cleanly)
+ a **host watchdog** (for the pandas/host path) + a **preflight estimate** refusal.

- `sitecustomize.py` — auto-applies `GFQL_RMM_LIMIT_GB` to any Python in the container (no
  workload edits). Put its dir on `PYTHONPATH` (safe_run.sh does this).
- `preflight.py` — `peak_gb(edges)`/`is_safe(edges)`; refuses runs over budget.
- `safe_run.sh` — wraps `docker run`: preflight refuse + RMM inject + host watchdog force-kill
  + hard timeout. Example:

```bash
benchmarks/dgx/safe_run.sh --name idx --est-edges 80000000 --rmm-gb 80 --floor-gb 20 \
  --timeout 3600 --pythonpath /opt/pygraphistry -- \
  -v $PWD:/opt/pygraphistry -v ~/data:/data:ro --entrypoint python \
  graphistry/test-rapids-official:26.02-cuda12-gfql /opt/pygraphistry/benchmarks/gfql/index_takeover_bench.py
```
