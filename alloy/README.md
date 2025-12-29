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
- `tests/gfql/ref/test_df_executor_inputs.py`: 50+ WHERE + hop range scenarios
- `tests/gfql/ref/test_df_executor_inputs.py::TestImpossibleConstraints`: 10 impossible/contradictory constraint tests

These tests verify the native executor matches the reference oracle implementation.

### Bugs Found That Inform Future Verification (PR #846)

The following bugs were found during executor development that formal verification could catch:

1. **Backward traversal join direction** (`_find_multihop_start_nodes`) - joined on wrong column
2. **Empty set short-circuit missing** (`_materialize_filtered`) - no early return for empty sets
3. **Wrong node source for non-adjacent WHERE** - used incomplete alias_frames instead of graph nodes
4. **Multi-hop path tracing through intermediates** - backward prune filtered wrong edges
5. **Reverse/undirected edge direction handling** - missing is_undirected checks

See issue #871 for recommended Alloy model extensions:
- P1: Add hop range modeling
- P1: Add backward reachability assertions
- P2: Add empty set propagation assertion
- P2: Add contradictory WHERE scenarios (attempted but model's value semantics are too nuanced; covered by Python tests)

See issue #871 for the full testing & verification roadmap.
