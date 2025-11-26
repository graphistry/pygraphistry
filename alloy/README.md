# Alloy Checks for GFQL F/B/F + WHERE

Purpose: bounded, mechanized equivalence checks between the GFQL path-spec and the set-based forward/backward/forward algorithm with WHERE lowerings.

## Model
- Path semantics: bindings are sequences aligned to `seqSteps`; WHERE is per binding. Mirrors Python hop/chain construction.
- Set semantics: executor-style F/B/F over per-alias node/edge sets; WHERE lowered via per-alias summaries.
- Scopes: ≤8 Nodes, ≤8 Edges, ≤4 Steps, ≤4 Values. Null/NaN not modeled; hashing treated as prefilter and omitted.
- Lowerings: inequalities via min/max summaries; equality via exact sets (bitsets modeled as sets).

## Commands
- Default small checks (fast): `bash alloy/check_fbf_where.sh`
- Full scopes (core + scenarios): `FULL=1 bash alloy/check_fbf_where.sh`
- Add multi-chain full-scope: `FULL=1 MULTI=1 bash alloy/check_fbf_where.sh`

Env vars:
- `ALLOY_IMAGE` (default `ghcr.io/graphistry/alloy6:6.2.0`)
- `ALLOY_FALLBACK_IMAGE` (default `local/alloy6:latest`)
- `ALLOY_PUSH=1` to push built image to ghcr when falling back.

## CI behavior
- PR/push: small + scenario suite (faster).
- schedule/workflow_dispatch: full scopes + optional multi-chain (heavier).
- Job pre-pulls `ghcr.io/graphistry/alloy6:6.2.0`; falls back to local build and pushes when allowed.

## Notes / exclusions
- Null/NaN semantics excluded; verified in Python/cuDF tests.
- Hashing omitted; treat any hashing as sound prefilter, exactness rechecked in model.
- Model uses set semantics for outputs (nodes/edges appearing on some satisfying path).
