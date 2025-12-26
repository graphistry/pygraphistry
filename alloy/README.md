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

## Scope and Limitations

### What IS Formally Verified
- WHERE clause lowering to per-alias value summaries
- Equality (`==`, `!=`) via bitset filtering
- Inequality (`<`, `<=`, `>`, `>=`) via min/max summaries
- Multi-step chains with cross-alias comparisons
- Graph topologies: fan-out, fan-in, cycles, parallel edges, disconnected

### What is NOT Formally Verified
- **Hop ranges** (`min_hops`, `max_hops`): Approximated by unrolling to fixed-length chains
- **Output slicing** (`output_min_hops`, `output_max_hops`): Treated as post-filter
- **Hop labeling** (`label_node_hops`, `label_edge_hops`, `label_seeds`): Not modeled
- **Null/NaN semantics**: Verified in Python tests instead
- **Hashing**: Treated as prefilter and omitted (exactness rechecked in model)

### Test Coverage for Unverified Features
Hop ranges and output slicing are covered by Python parity tests:
- `tests/gfql/ref/test_enumerator_parity.py`: 11+ hop range scenarios
- `tests/gfql/ref/test_cudf_executor_inputs.py`: 8+ WHERE + hop range scenarios

These tests verify the cuDF executor matches the reference oracle implementation.

See issue #871 for the testing & verification roadmap.
