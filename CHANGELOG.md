# Changelog

All notable changes to the PyGraphistry are documented in this file. The PyGraphistry client and other Graphistry components are tracked in the main [Graphistry major release history documentation](https://graphistry.zendesk.com/hc/en-us/articles/360033184174-Enterprise-Release-List-Downloads).

The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and all PyGraphistry-specific breaking changes are explictly noted here.

## [Development]
<!-- Do Not Erase This Section - Used for tracking unreleased changes -->

### Documentation
- **GFQL engine-selection docs (pandas / polars / cuDF / polars-gpu)**: New :doc:`Choosing a GFQL Engine <gfql/engines>` page — a numbers-first, persona-tested guide to the four interchangeable engines. Adds the one-keyword `engine='polars'` speedup (up to ~38× over pandas on real graphs, no GPU), a motivating warm-median comparison table on real public graphs (LiveJournal 35M / Orkut 117M), a decision matrix (workload shape × size × hardware → engine, with the measured ~10K-edge CPU crossover, GPU-work-bound rule, polars-gpu memory-pressure caveat, and GPU-or-error contract), a cuDF-vs-polars-gpu disambiguation (eager-op vs fused-lazy; cuDF is not deprecated), an honest "when *not* to use Polars" section, the differential-parity guarantee, and a methodology + reproducer-script disclosure. Rewrote the top of `gfql/performance.rst` to lead with the engine comparison (de-marketed the prose), wired the new page into the GFQL toctree + recommended paths, and added Polars/polars-gpu to the engine examples in `gfql/quick.rst` and `gfql/about.rst` (previously only pandas/cuDF were documented). Driven by 4-persona doc user-testing (pandas DS, RAPIDS user, perf engineer, skeptical evaluator).

### Fixed
- **GFQL polars/polars-gpu seeded traversal no longer pays an O(E) NaN re-scan per hop on float-column graphs (identity-stable `_pl_nan_to_null` + clean-cache)**: every `hop()` runs `_coerce_input_formats` → `_pl_nan_to_null`, which scanned `is_nan().any()` over every float column on the (unchanged) resident edge frame **on every call** — so repeated seeded hops on a resident graph (the seeded-Search / native-hop pattern) grew O(E) with edge count on polars while pandas stayed flat. Measured: an indexed polars `g.hop` on 8M edges with one float column was **33.8 ms and growing**; it is now **0.20 ms and FLAT** (~140× faster; O(degree)), matching pandas. `_pl_nan_to_null` now probes an eager polars frame for real NaN once, returns a clean frame UNCHANGED (restoring the #1726 identity guard that #1731 reverted — so frame-identity caches like the #1658 index keep engaging), rewrites only the columns that genuinely carry NaN (values identical to the old unconditional `fill_nan`), and caches the id of frames verified clean so later calls skip the probe (recycle-safe via `weakref.finalize`; a rebound/new frame re-probes, so NaN→null semantics are unchanged). Regression: full 4-engine `test_index.py` parity + new `TestPlNanCleanCache` cases (NaN-present cleaned, clean-frame cached same-object, distinct frames independent).
- **GFQL seeded `gfql()`/Cypher chains now engage the resident #1658 adjacency index on ALL FOUR engines (incl. typed edges) instead of always scanning**: a seeded hop expressed as a `gfql([...])`/Cypher CHAIN (rather than a direct `g.hop(...)`) previously fell back to the O(E) scan even with a resident index, because both chain executors attach a synthetic per-edge id column to the edge frame (pandas/cuDF eager: `copy(deep=False)` + assign; polars native lazy: `with_row_index`) → a fresh edge-frame object → the index's `source_ref is df` identity guard missed. Both chain paths now re-point the resident edge adjacency indexes at the augmented frame via `GfqlIndexRegistry.rebind_edges` (provably safe: the shallow copy / `with_row_index` preserves the indexed src/dst columns by value; the structural fingerprint — row count + bound columns + engine — is unchanged by an added column). Additionally, a **simple scalar-equality `edge_match`** (typed edges, e.g. `-[:KNOWS]->` / `e_forward({"type": X})`) is now index-coverable on the wavefront path: `index_seeded_hop` applies the edge predicate to the CSR-matched rows each hop, parity-exact with the scan's `filter_edges_by_dict`. Predicate/membership-list `edge_match`, `edge_query`, and the direct-hop (non-wavefront) path deliberately stay on the scan path (no over-reach). Measured (dgx, 500k nodes / 4M edges / 4 edge types, warm median, index result == scan result): pandas typed 1-hop **2.1×** / untyped **2.7×**; cuDF typed **1.9×** / untyped **2.0×**; polars typed **1.3×** / untyped **5.4×**; polars-gpu typed engaged (1.0×) / untyped **12.4×**; 2-hop typed engaged on all four engines. 29 new differential + engagement regression tests (`test_index.py`), 4-engine, plus the existing 109-test index suite green.
- **GFQL Cypher `CASE` with mixed-dtype branches is now engine-consistent on cuDF (no more `GFQLTypeError`)**: pandas coerces the two `CASE` branches to a common type, but cuDF's `.where` raised `TypeError: cudf does not support mixed types`, surfaced as a hard `GFQLTypeError`. This bit `CASE WHEN path IS NULL THEN -1 ELSE length(path) END` over an UNREACHABLE `shortestPath` (int `-1` branch vs an object/null hops branch) — pandas returned `-1`, cuDF errored. The row-AST `CASE` evaluator now unifies the branch dtypes and retries, so cuDF returns the same value pandas does. Regression test `test_case_mixed_dtype.py`.
- **GFQL polars engine raises a clean typed error (not an opaque polars `InvalidOperationError`) for a string predicate on a non-string column**: `WHERE col STARTS WITH/CONTAINS/regex ...` on a Categorical/Enum/numeric column raised a clean `GFQLSchemaError` ("string predicate used on non-string column") on pandas/cuDF but leaked polars' internal `InvalidOperationError: expected String type, got: cat` on polars. `filter_by_dict_polars` now raises the same `GFQLSchemaError` (categorical treated as non-string, exactly as `filter_by_dict`), so all three engines agree. Regression test `test_polars_string_predicate_nonstring.py`; also confirms connected-join predicate pushdown is parity-correct on polars for string/numeric/eq/bool columns.
- **GFQL string predicates (`Contains`/`Startswith`/`Endswith`/`Match`/`Fullmatch`) are now value-safe on non-string columns**: applying a string predicate to a numeric/temporal/bool column raised an opaque `AttributeError: Can only use .str accessor with string values!` on pandas and cuDF. They now follow openCypher semantics — a string op over a non-string value is null → excluded (matching the established per-cell behavior on an object column holding non-strings) — and never stringify the column (which would diverge pandas↔cuDF, e.g. wrongly matching `5 CONTAINS '5'`). String, mixed-object, and all-null columns are unchanged. Regression tests in `test_str.py` across dtypes; cuDF parity confirmed.
- **GFQL predicate `from_json` no longer downgrades comparison predicates to numeric-only (fixes JSON round-trip of string/temporal `=`, `<`, `>`, etc.)**: the predicate deserialization registry mapped `EQ`/`NE`/`GT`/`GE`/`LT`/`LE`/`Between` to the numeric-only `predicates.numeric` classes, while Cypher lowering and the public predicate API build the richer `predicates.comparison` versions (which also accept strings and temporals). A JSON round-trip (remote GFQL, serialization, caching) therefore rebuilt a string/temporal equality or a temporal comparison as numeric-only, raising `GFQLTypeError "val must be numeric"`. `from_json` now registers the `comparison` versions; numeric-valued predicates are unaffected (comparison is a superset). Round-trip regression tests in `test_from_json.py`.
- **GFQL Cypher `ORDER BY ... DESC` now places NULLs FIRST (openCypher NULL-largest), fixing silent-wrong `DESC ... LIMIT k` top-k**: openCypher orders NULL as the largest value (`ASC` → nulls last, `DESC` → nulls first), but the general row pipeline hardcoded nulls-last for every key on all engines — so any `ORDER BY <col> DESC` over a column containing NULLs was mis-ordered and `DESC ... LIMIT k` silently dropped the NULL group that should rank first. Fixed on pandas/cuDF (per-key null-indicator single-pass in `row/pipeline.py`; stability-free, so cuDF's non-stable sort is safe) and polars (per-key `nulls_last` list in `order_by_polars`). Hand-derived-oracle regression suite `test_order_by_null_placement.py` across pandas/polars/cuDF. Pre-existing bug (not introduced by the OLAP split); the OLAP fast paths already ordered nulls correctly, so this brings the general/fallback path into agreement.

### Added
- **GFQL shortestPath native-graph cache across pairwise calls (#1705, perf)**: the igraph/cugraph graph
  built for a `shortestPath` scalar is now cached per graph and reused across the pairwise shortestPath ops of
  one query execution (and across executions on the same graph object), instead of rebuilt every call. The
  cache is keyed on the edge op's canonical pattern (so a different pattern never reuses another's graph) with
  directedness folded in, and is RESET whenever the bound edge frame is rebound (`g.edges(...)`/`g.nodes(...)`)
  so distances are never stale; results are byte-identical to the uncached path (igraph/cugraph/BFS parity).
  Also fixes the `min_hops=0` self-pair distance (a node reaches itself at length 0) for endpoints absent from
  the edge set.
- **GFQL Cypher row-pipeline single-alias predicate pushdown**: Planner-generated `rows(alias_prefilters=...)` hints pre-filter eligible node and single-hop edge alias frames before binding joins on pandas/cuDF while retaining the original post-join filters for exact semantics and engine-safe fallback.
- **GFQL viz-filter-pipeline acceptance suite + regression benchmark (viz-filter L3)**: `test_viz_pipeline_conformance.py` — curated full-panel pipelines (node+edge filters, exclusion-dominates composition, `(pred OR IS NULL)` keep-null leaves, both EXISTS prune-isolated flavors, searchAny composition, deterministic paging), graph-state prune shapes with exact node+edge pins, a 40-seed panel-state fuzzer with an independent plain-pandas second oracle, and a case/regex/unicode trick matrix (ß/İ full-case-mapping pins, metachar literal-vs-regex, null cells, Categorical) — all parity-or-NIE across pandas/cuDF/polars/polars-gpu. `benchmarks/gfql/viz_filter_pipeline.py` — six streamgl-viz panel scenarios (filters, keep-self GRAPH prune, EXISTS prune, node/edge search, combined) at 100K/1M/10M with native-frame-per-engine fairness, an NIE-tolerant matrix, and JSON receipts (first receipt: 100k × 4 engines, everything within the ~350ms interactive reference except pandas combined). Documented findings from the first runs: the same-path WHERE route dedupes parallel edges (diverges from the panel algebra's edge multiplicity — pinned + tracked), and edge-alias searchAny declines on polars (tracked).
- **GFQL Cypher `searchAny(entity, term[, opts])` cross-column search predicate + `g.search_nodes()`/`g.search_edges()` (viz-filter L2, native on all four engines)**: True where ANY of the entity's columns matches the term — the streamgl-viz inspector's table-search semantics as a composable WHERE predicate: OR across columns; case-insensitive substring by DEFAULT (case-folded, never regex — the common call avoids every engine regex limit); regex opt-in obeying the same per-engine decline rules as `=~`; dtype gate AS SEMANTICS (string columns always; integer columns iff the term is a numeric literal, per the inspector's `/^[0-9.-]+$/` gate; floats/dates/booleans reachable via the explicit `columns:` list). Options map `{caseSensitive, regex, columns}` is strict-validated (unknown keys error listing the valid ones); unbound aliases and missing explicit columns error clearly; null cells never match. Lowered like the pattern-predicate markers (a `search_any` row op + fresh marker column), so it composes through AND/OR/NOT and different node/edge terms coexist in one pipeline. Per-column matching reuses the parity-hardened `Contains` predicate on pandas/cuDF and a lowercase-fold/any_horizontal lowering on polars; oracle-pinned + 4-engine parity-or-NIE conformance cases. Python twins `g.search_nodes(term, columns=, case_sensitive=, regex=)` / `g.search_edges(...)` filter their own table and return a Plottable (polars-frame twins decline honestly for now — use the cypher op). Honest declines (NIE, use engine='pandas'): edge-alias searchAny on polars, and explicit columns beyond string/int/bool dtypes on polars AND cuDF (incl. floats: repr diverges across engines — dgx-probed).
- **GFQL Cypher `EXISTS { <pattern> }` pattern-existence subqueries (openCypher-standard), native on all four engines**: `WHERE EXISTS { (n)-[:R]->() }` and `WHERE NOT EXISTS { (n)--() }` now parse and run — the declarative prune-isolated building blocks for the streamgl-viz filter pipeline. An `EXISTS` body reuses the existing pattern-predicate lowering wholesale (`semi_apply_mark` / `anti_semi_apply` row ops), so pandas/cuDF worked immediately; the polars engine gains NATIVE lowerings for the semi-apply family (correlated key sets computed by the polars chain executor's named-flag columns; order-preserving `is_in` joins) plus `rows(binding_ops=...)` for the single-entity row table — previously all honest-NIE. Aliases introduced inside the braces are existentially scoped (`EXISTS { (n)--(m) }` allowed with `m` unbound outside — bare pattern predicates keep the conservative guard), inline property maps work, and the one supported inner `WHERE` form is endpoint inequality — `EXISTS { (n)--(m) WHERE m <> n }`, the drop-self-loop prune-isolated flavor (pandas/cuDF filter the correlated bindings; polars excludes self-loop edges, which is exactly the `m <> n` witness). Both prune flavors are oracle-pinned in the conformance matrix on a self-loop discriminator graph, 4-engine parity-or-NIE. Honest declines with clear errors: `EXISTS` in RETURN/WITH projections, general inner `WHERE`, multi-pattern bodies, full `MATCH..RETURN` subquery bodies, multi-alias correlation on polars.
- **GFQL polars execution config is Python-settable and live**: `set_cpu_streaming(bool)` and `set_gpu_executor('in-memory'|'streaming')` in `graphistry.compute.gfql.lazy` (plus the public `GPU_EXECUTORS` options and `GpuExecutor` type) set the CPU-streaming / GPU-executor knobs from Python. They resolve **Python override > environment variable > default**, read **live** per collect — previously these were env-only (`GFQL_POLARS_CPU_STREAMING` / `GFQL_POLARS_GPU_EXECUTOR`) and frozen at import, so neither a Python setting nor a post-import env change took effect. `None` resets a setter to env/default.
- **GFQL engine conversion honors the `validate`/`warn` convention**: `Engine.df_to_engine(df, engine, *, validate=, warn=)` threads the repo-wide `validate` (`'strict'`/`'strict-fast'`/`'autofix'`; `True`→strict, `False`→autofix) + `warn` protocol into the pandas→polars and pandas→cuDF converters. On a mixed-type object column that Arrow/polars/cuDF cannot represent, `strict` raises (`NotImplementedError` for polars, `ArrowConversionError` for cuDF) and `autofix` coerces the column to string and warns — the same convention as `plot()`/`upload()`. Each engine keeps its established default (polars `strict` = parity-or-raise; cuDF `autofix` = its shipped best-effort coercion, now `warn`-suppressible).
- **GFQL Cypher numeric functions + `toLower`/`toUpper`/`lower`/`upper` (openCypher/neo4j/GQL-standard)**: added the standard scalar functions `floor`, `ceil` (alias `ceiling`, per Cypher 25 GQL conformance and the GQL grammar's `CEIL|CEILING`), `round(x)` / `round(x, precision)`, and `toLower` / `toUpper` plus their GQL-conformance aliases `lower` / `upper` (ISO GQL §20.24 character-string functions; neo4j accepts both spellings) — the idiomatic case-insensitive compare `WHERE toLower(n.name) = 'bob'` — across the Cypher `WHERE`/`RETURN` expression surface. Evaluated natively on pandas/cuDF and polars (differential-parity tested). **`round` follows neo4j's documented tie-breaking** (standards-vetted against the neo4j manual): precision 0 (and the 1-arg form) rounds ties toward **positive infinity** (`round(-1.5)` → `-1.0`, `round(2.5)` → `3.0`), and precision > 0 rounds ties **away from zero** (HALF_UP: `round(-1.55, 1)` → `-1.6`) — not the numpy/polars half-to-even defaults, which give wrong answers on ties. The `round(x, precision, mode)` 3-arg form is not yet supported. Complements the already-supported `abs`/`sqrt`/`sign` and chained comparisons (`WHERE 1 < n.age < 65`). The `^` exponentiation operator is deferred: standards vetting settled it as **left-associative** (the openCypher TCK pins left, and neo4j's shipped Cypher 5/25 parser left-folds `Pow` — the manual's "right to left" row is a docs bug), but the current conformance corpus marks it reject-expected, so re-adding it is a coordinated corpus change. `LIKE`/`ILIKE`/`BETWEEN` remain intentionally unimplemented (verified absent from both Cypher and ISO GQL — GQL's only `LIKE` is the unrelated `CREATE GRAPH TYPE … LIKE g` DDL).
- **GFQL Cypher `=~` regex-match operator (openCypher/neo4j-standard)**: added the standard `=~` string predicate to the Cypher `WHERE`/expression grammar — `MATCH (n) WHERE n.name =~ '(?i)al.*' RETURN n`. Semantics match openCypher/neo4j: **Java-regex dialect, full-string/anchored match** (`n.name =~ 'AB'` matches only `'AB'`, not `'ABCDEF'`; use `.*`/`^..$` for partial), with inline flags (`(?i)`/`(?m)`/`(?s)`) honored; lowers to the existing `fullmatch` predicate. Works in the simple `WHERE prop =~ '...'` form (all engines via `filter_by_dict`) and composes through `AND`/`OR`/`NOT`/`RETURN` expressions via the shared expression engine (pandas/cuDF; polars supports the simple-WHERE form and declines the complex `OR`/`NOT` row-filter form with an honest `NotImplementedError` — the pre-existing polars `where_rows` limitation, not `=~`-specific). Also adds native polars `Match`/`Fullmatch` predicate lowering (previously `NotImplementedError`), so `=~` and the Python `match()`/`fullmatch()` predicates run natively on polars. Differential-parity tested against the pandas oracle. `LIKE`/`ILIKE` remain intentionally unimplemented (not in any graph standard — use `=~`/`CONTAINS`/`STARTS WITH`).
- **GFQL physical adjacency indexes — pay-as-you-go seeded-traversal acceleration**: Opt-in CSR adjacency / node-id indexes that turn seeded `hop()`/`gfql()` traversal from an O(E) full edge scan into an O(degree) `searchsorted` gather. Sidecar over edge **row positions** (never reorders `.edges`/`.nodes`), fingerprint-validated so a `.edges()` rebind safely invalidates a stale index (treated as absent — never a wrong answer). Three uniform surfaces driving one registry: **Python** (`g.create_index('edge_out_adj'|'edge_in_adj'|'node_id')`, `drop_index`, `show_indexes`, `gfql_index_edges`/`gfql_index_all`, `gfql_explain`), **Cypher DDL** (`CREATE/DROP GFQL INDEX FOR <kind>`, `SHOW GFQL INDEXES` — the mandatory `GFQL` token disambiguates from standard property `CREATE INDEX`), and the **JSON wire protocol** (`{"type":"CreateIndex",...}` ops + `index_policy` in the request envelope). Optimizer policy `gfql(..., index_policy='use'|'auto'|'force'|'off')` (default `use` = use-resident-never-build). Engine-polymorphic (numpy host arrays for pandas/polars, **cupy on-device for cudf**); the seeded fast path is hooked at every scan site (`compute/hop.py`, the lazy polars hop, and the polars single-hop chain fast path) and falls back to the scan/join path for any uncovered feature (edge/source/dest match, `target_wave_front`, `min_hops>1`, labeling). Differential-parity verified (indexed subgraph == scan subgraph) across pandas/cudf/polars/polars-gpu × forward/reverse/undirected × 1–3 hop × wavefront, hardened by an adversarial review that found and fixed several index-vs-scan divergences the original scenarios missed (`max_hops` honored; duplicate node ids, edge endpoints absent from the node table, int32/int64 seed-vs-key dtype mismatches, and stale `.edges()` rebinds all now match scan or fall back — never a wrong answer), with object-identity fingerprinting (recycle-proof) and 6 added differential regression tests. **Perf (dgx-spark, deg-8, warm median; every measured cell guarded so the index path was actually taken AND the index result equals the scan result):** seeded latency is **flat in graph size** — GFQL-pandas 1-hop 0.124/0.122 ms at 0.8M/8M nodes (6.4M/64M edges) while the O(E) scan grows 105 → 1045 ms — and **beats kuzu (CSR) and neo4j by 9–28×** on CPU at 0.8M nodes (1-hop: GFQL-pandas 0.123 ms vs kuzu 1.15 ms vs neo4j 1.45 ms; 1–2-hop: 0.150 ms vs 4.25 ms vs 2.54 ms, matched answer counts). Build cost (one O(E log E) sort) is amortized over queries; `index_policy='auto'` builds only when the planner predicts selectivity, invalid policy strings now fail fast, repeated `CREATE GFQL INDEX ...` rebuilds stale resident indexes after frame rebinds, `gfql_explain()` has a typed report contract, and the planner cost gate is tunable via Python helpers or `GFQL_INDEX_COST_GATE_FRAC*` environment variables. cuDF/polars-GPU are flat but floored by ~3 ms (cuDF) GPU kernel-launch overhead — selective traversal is an indexing problem, not a compute one (CPU wins it). No change to default (un-indexed) behavior.
- **GFQL native Polars engine — traversals (`engine='polars'`)**: Added a native, vectorized Polars execution engine for the core GFQL traversals `hop()` and `chain()`, dispatched at the engine boundary so the production pandas/cuDF paths are untouched. `Engine.POLARS` is opt-in (explicit `engine='polars'`); `engine='auto'` with Polars input still coerces to pandas as before. Covers forward/reverse/undirected single-hop traversal, directed multi-hop chains, node/edge filter dicts and predicates (lowered to Polars expressions), `edge_match`/`source_node_match`/`destination_node_match`, `target_wave_front`, and alias names; the BFS advances via semi/anti joins (no per-row Python work). Validated by differential parity against the pandas engine (hop + chain test suites plus a randomized fuzzer) and benchmarked vs pandas (`benchmarks/gfql/pandas_vs_polars.py`) — Polars wins at scale (up to ~2.5x on multi-edge chains at millions of edges; crossover ~50–100k rows). Variable-length/multi-hop edges, undirected edges in multi-edge chains, hop labels, and node `query=` raise `NotImplementedError` for now (use `engine='pandas'`).
- **GFQL native Polars engine — variable-length `min_hops>1` traversals (`engine='polars'`/`'polars-gpu'`)**: Forward/reverse lower-bound traversals (`e(min_hops=N, max_hops=M)`) now run natively on the Polars engine — no pandas bridge. The eager Polars hop runs pandas' min_hops algorithm vectorized: a NON-anti-joined BFS (the wavefront carries revisits so a cycle keeps bumping the reach depth until `max_hops`), a 3-case termination gate (`max_reached<min`→empty; goal-edges-labeled-≥min→a descending layered backward-tree walk; cyclic-revisit-only→unpruned ball), and — the subtle part proven out against the pandas oracle — the exact min_hops NODE-output rule: the wavefront is the **endpoints of the retained edges**, MINUS seed nodes never genuinely re-reached at ≥min_hops, with FULL node attributes only for nodes carrying a retained-path hop-label and NULL attributes for source-side endpoint-only nodes (so a subsequent node-attribute filter rejects them, matching pandas' `track_node_hops` labeled path exactly). This is wired up for `chain()`/`gfql()`. **NO CHEATING:** UNDIRECTED `min_hops>1` (needs connected-components + 2-core seed retention) and a *direct* `hop(min_hops>1)` (which would need pandas' separate un-labeled direct-hop node-output plus the `target_wave_front` threading the chain supplies — without them it silently diverges) both raise `NotImplementedError` for `engine='polars'` (use `chain()`/`gfql()`, or `engine='pandas'`). Validated by differential parity vs the pandas oracle: the 500-seed randomized chain fuzzer (`test_polars_chain_fuzz_parity`, hardened to compare null-aware node **attributes** and edge **multiplicity**, not just id/endpoint sets) is **500/500**, a min_hops+attribute-filter amplified fuzz and metamorphic invariants pass, and `engine='polars-gpu'` (cudf_polars) runs the full 500-seed fuzz **500/500** on-device.
- **GFQL native Polars engine — cypher row pipeline (`engine='polars'`)**: Extended the Polars engine to the Cypher `MATCH … RETURN` row surface, natively vectorized. **NO CHEATING:** the polars engine never silently falls back to the pandas engine — every query runs natively on polars or raises an honest `NotImplementedError` pointing at `engine='pandas'` (falling back to pandas would misrepresent pandas performance as polars; only a human may consent to a bridge). `chain_polars` splits boundary `call()` ops (mirroring the pandas `_handle_boundary_calls`) and runs each trailing row op per-op native or raises. **Native polars** (no pandas round-trip): frame ops (`rows`/`limit`/`skip`/`distinct`/`drop_cols`), `select`/`with_`/`return_` projection (a conservative cypher-expr-AST → `pl.Expr` lowering covering property access, arithmetic, comparison, boolean, literals), `order_by` (`.sort`), `group_by` (`count`/`sum`/`avg`/`min`/`max`), `unwind` (literal-list cross-join), the result projection for property/expr columns, and entity-text `RETURN n` rendering for int/string/bool nodes (`pl.concat_str`). **Honestly deferred** (raise `NotImplementedError`, no pandas fallback): multi-entity `rows(binding_ops=…)`, cross-entity same-path `WHERE` (`DFSamePathExecutor`), float/temporal/nested entity-text, and exotic expressions (CASE/list/map/temporal, `collect` aggregates) — these are the forward native-engineering targets. Validated by differential parity vs pandas including a TCK-style conformance lane (`test_engine_polars_cypher_conformance.py`: native-only curated corpus + seeded fuzzer + NULL/3-valued-logic graph + entity-text escaping, plus a `DEFERRED` list asserting deferred queries raise rather than silently bridge) and benchmarked (`benchmarks/gfql/cypher_row_pipeline.py`). **Perf (interleaved, 1M nodes, each engine on its native-frame graph, all fully native):** polars wins **5.6–38×** across the surface — `RETURN n` ~38×, `ORDER BY` ~17×, `WHERE`+`ORDER BY`+`LIMIT` ~14×, traversals 6–7.5×, projections/aggregations/`DISTINCT` 5.6–6.9×. cuDF/pandas paths untouched.
- **GFQL lazy Polars engine + GPU target (`engine='polars-gpu'`, cudf_polars)**: The Polars traversal engine now builds a single deferred `pl.LazyFrame` plan per single-hop and materializes `out_edges`+`out_nodes` in ONE `collect_all` on a chosen **execution target** (CPU or GPU). `engine='polars-gpu'` (`Engine.POLARS_GPU`, explicit opt-in only — AUTO never selects it) runs that same lazy plan on the RAPIDS cudf_polars backend (`pl.GPUEngine(raise_on_fail=True)` — NO-CHEATING: a GPU-incapable plan node **raises** rather than silently running on CPU and being reported as a GPU result; see Fixed). The collect-once design is what makes GPU pay off: a benchmark showed per-op eager GPU collect was a *regression* (repeated H2D), while collect-once is a **2.84× single-hop GPU win @1M** with CPU parity. Frames stay `pl.DataFrame` (handled like `POLARS` everywhere); the target is carried by a context var set at the chain/hop dispatch boundary, so `engine='polars'` (CPU) is byte-for-byte unchanged. Validated by differential parity `engine='polars-gpu' == engine='polars'` across the cypher conformance corpus + traversals (`test_engine_polars_gpu.py`, skips when no cudf_polars/GPU). Multi-hop and the chain forward/backward fusion (where the GPU win currently dilutes) are follow-up optimizations.
- **GFQL `gfql_explain` planner diagnostics**: `gfql_explain(...)` now surfaces *why* the seeded-hop planner chose the index or the scan, not just a `used_index` yes/no. Each per-hop step carries the frontier seed count (`frontier_n`), the number of distinct source keys (`n_keys`), a **free Σ-seed-degree fanout estimate** (`seed_deg_sum`/`est_result_rows`, read directly from the CSR `group_offsets` already on the adjacency index — no extra scan), the engine's cost-gate crossover fraction (`threshold_frac`), and a human-readable `decision_reason` (e.g. `"frontier below cost gate -> index"`, `"frontier N >= 0.02*n_keys (…) -> scan cheaper"`, `"query not index-coverable"`, `"policy=off"`). The report also lifts `est_seed_cardinality`, `est_result_rows`, `chosen_direction`, and the top-level `decision_reason` for quick inspection. Purely additive and diagnostic — the fanout estimate is computed **only** inside an `index_trace()`/`gfql_explain` context, so the hot traversal path pays nothing and traversal behavior is unchanged.

### Added
- **GFQL native Polars engine — more cypher row coverage (`toFloat`, `collect`/`collect(DISTINCT)`, `WHERE … IN`)**: three surfaces that previously raised `NotImplementedError` on `engine='polars'` now run natively, parity-validated vs the pandas oracle across all four engines (and honest-NIE where pandas can't be matched). **`toFloat(x)`** lowers int/uint/bool/float → `Float64` (NaN preserved — float64 has no separate null sentinel, unlike `toInteger`); a non-numeric String declines (NIE) because pandas `astype(float)` *raises* rather than null-on-failure. **`collect(x)` / `collect(DISTINCT x)`** aggregations complete the native `group_by` surface (every other agg was already native): drop nulls, preserve within-group first-occurrence order (`collect` keeps dups; `DISTINCT` dedups keep-first), all-null group → `[]`. **`where_rows`/`WHERE … IN [list]`** membership lowers to `is_in` (a null cell is excluded per openCypher 3VL). No change to any already-native path.
- **GFQL Polars-CPU streaming collect (opt-in, large traversals)**: `GFQL_POLARS_CPU_STREAMING=1` runs the polars-CPU lazy collects (`hop`/`chain`) on the polars **streaming** executor instead of the default in-memory collect. Benchmarked ~1.04–1.11× faster on big multi-hop traversals (10M nodes / 80M edges: 20.0→18.0 s) and parity-identical, but ~0.86× (slower) on small/interactive sizes (streaming overhead) — so it is **opt-in, default off** (no change to default behavior). Use for large batch traversals where CPU is the target.
- **GFQL Cypher `count(*)` short-circuit — O(1) instead of O(N) materialize**: a lone `RETURN count(*)` over a single node or edge pattern (`MATCH (n) RETURN count(*)`, `MATCH ()-[r]->() RETURN count(*)`) previously materialized the entire matched frame and ran a constant-key `group_by` just to count its rows. The lowering now emits a new `count_table` row op that reads the scanned table's height directly (or, when the pattern applies a filter, sums the boolean alias-mask column) in a single reduction — no full-frame copy, no group_by. Applies only to the provably-equivalent shapes: exactly one non-DISTINCT `count(*)`, no group keys / post-aggregate exprs / row-level `WHERE` / `UNWIND` / paging / multi-relationship binding, and either a pure node scan (`relationship_count == 0`) or a single relationship counted on its edge alias — every other shape (node-alias-over-relationship, multi-hop paths, `count(DISTINCT …)`, grouped counts) falls through to the general aggregate path unchanged. Engine-polymorphic across pandas/cuDF/polars/polars-gpu; differential parity verified on all four engines (`count_all_nodes`/`count_all_edges` cypher conformance cases + a `count_table` row-op subject case, chain and DAG surfaces). No change to any result value — only the execution path.
- **GFQL Cypher aggregate fast paths — single-hop grouped aggregate + two-hop connected count**: grouped-aggregate `RETURN`s over the common OLAP shapes now execute on dedicated fast paths instead of the general bindings-materialize-then-group route: single-hop `MATCH (n) RETURN <key>, count(n)/sum(n.x)/avg(n.x) [ORDER BY … LIMIT …]` and two-hop connected `MATCH (a)-[]->(b)-[]->(c) RETURN count(*)`/`count(a)`. Byte-identical to the general path on pandas/cuDF/polars/polars-gpu (differential parity vs the hand-derived openCypher oracle), with openCypher ordering honored — NULL sorts as the largest value, so a grouped `ORDER BY <key> ASC LIMIT k` places the NULL-key bucket last (this fixed a polars nulls-first divergence). The string-Cypher compile cache keys on the resolved node dtypes so a plan compiled under one engine's dtype view is never reused for another on the same graph; the cache stores data-independent plans only (no cross-call result staleness under in-place graph mutation). Unsupported grouped shapes (e.g. `min`/`max` over a grouped bindings key) raise an honest validation error rather than silently taking a wrong path.
- **GFQL `engine='polars'`/`'polars-gpu'` can now run off-engine analytic `call()`s (umap/hypergraph/compute_cugraph/…) — `call_mode='auto'` (default)**: a GFQL `call()` that invokes a Plottable-method analytic with **no native polars implementation** (and never will — `umap`, `hypergraph`, `compute_cugraph`, `compute_igraph`, `layout_*`, `collapse`, `get_topological_levels`, …) previously raised `NotImplementedError` under a polars engine, forcing users off polars for the whole pipeline. It now runs as a **mode-gated, warned modality switch**: `call_mode='auto'` (default) bridges the graph off-engine (`polars`→pandas, `polars-gpu`→cuDF **on-device**), runs the analytic, and coerces the result back to polars **losslessly via Arrow**, warning once per (process, function). `call_mode='strict'` keeps the honest `NotImplementedError` decline (for benchmark integrity / a hard memory ceiling). `polars-gpu` is **GPU-or-error**: it bridges to cuDF and declines rather than silently dropping a GPU analytic to host pandas. This is **deliberately narrower than CHAIN traversal/filter/row ops**, which stay parity-or-NIE (a bridge there would hide a missing impl and cheat a benchmark) — the split is mechanical (`is_row_pipeline_call`), and the chain and DAG surfaces bridge consistently. Configurable via `graphistry.compute.gfql.lazy.set_call_mode('auto'|'strict')` (Python) or `GFQL_POLARS_CALL_MODE` (env), resolving Python override > env > default('auto'), read live. Verified on dgx: `compute_cugraph` PageRank is byte-parity with the pandas oracle under both `polars` and `polars-gpu`.

### Changed
- **GFQL Cypher parser: internal cleanup — WHERE consumed directly from `MatchClause`**: the grammar bundles a trailing `WHERE` onto its `MATCH` clause; the transformer previously split it back out into a synthetic standalone item and re-attached it, so the legacy clause-assembler ran unchanged (a temporary seam that kept the LALR switch byte-identical). The assembler now consumes `MatchClause.where` directly (primary MATCH keeps its WHERE on the clause; a post-WITH re-entry MATCH's WHERE goes to `reentry_wheres`), deleting the split/re-attach round-trip and the now-unreachable standalone-WHERE handling in both `query_body` and `graph_constructor`. Pure internal refactor, **no behavior change**: verified byte-identical ASTs vs the prior parser across a 1,989-query repo corpus, and the full cypher suite passes.
- **GFQL Cypher parser: Earley → LALR(1) (~70× faster parse, same AST)**: `parse_cypher` now uses a single LALR(1) parser (contextual lexer) instead of Earley (dynamic lexer) — ~0.25ms vs ~17ms per query on representative queries. The WHERE grammar was unified to one `where_clause: "WHERE" expr` rule (the former `where_predicates | expr` dual rule was a genuine reduce/reduce ambiguity that forced Earley), and the structured flat-AND predicate form is recovered by a post-parse lift that re-parses the WHERE body with a LALR sub-parser (`start="where_predicate_chain"`). The full WHERE language is preserved — OR/XOR/NOT, parentheses, arithmetic, IN, pattern predicates — nothing is restricted to a subset. The grammar was then **purified to (near-)unambiguous**: WHERE binds to its preceding clause *in the grammar* (bundled into `match_clause`; the WITH..WHERE attachment ambiguity is eliminated, not tie-broken), and name-rooted dot chains derive only via `qualified_name` (the `property_access` redundancy is gone) — dropping the redundant `label_predicate_expr` rule (`(n:Admin)` parses via `grouped_expr`), and excluding a top-level `IN` from list-literal elements (`[x IN xs ...` is comprehension syntax; allowing a bare `IN` list element made it overlap). Net: the LALR conflict profile goes from 8 shift/reduce to **ZERO conflicts** — the grammar is now **provably unambiguous LALR(1)** and builds under Lark's `strict=True` (a build-time proof, machine-checked in CI as zero conflicts, plus a strict-mode build test where the optional `interegular` dep is present). Every input has a single derivation. **Machine-checked invariants** (`test_grammar_invariants.py`): (1) **zero LALR conflicts** (dependency-free, always-on in CI) plus a `strict=True` build test (skipped where the optional `interegular` dep is absent) — a grammar edit that introduces any ambiguity fails CI; (2) semantic ambiguity is ZERO — every corpus query has exactly one Earley derivation and it transforms to a single AST, no exceptions; (3) a rule-coverage gate (`test_every_grammar_rule_is_exercised_by_the_corpus`) forces every new grammar rule through the invariants; (4) differential tests run the production pipeline under both parsers (byte-identical ASTs, identical rejections). **Full-repo differentials** (1,850+ scraped queries, LALR-vs-Earley and old-vs-new production): **zero AST divergences**, and a small set of deliberate language fixes — accept-by-accident shapes with ill-defined semantics, now honest syntax errors and pinned as tests (`DELIBERATE_LANGUAGE_FIXES`): `RETURN DISTINCT` with no items (Earley re-lexed DISTINCT as a variable name), double WHERE (the old positional attachment kept *both* predicates in different AST fields), double post-WITH WHERE, WHERE after UNWIND, WHERE before MATCH in a graph constructor, and the invalid "list of IN-booleans" (`[x IN xs, y]`; use `[(x IN xs), y]` for a genuine bool list). The lift stays an internal optimization: only a flat `AND` chain over present columns lifts to `filter_dict`; parens / OR / XOR / NOT and any absent-column case stay on `where_rows` (a correctness boundary, since `where_rows` treats an absent property as null while `filter_dict` would raise). Full cypher suite (1,681 tests) passes. Downstream execution (`filter_dict` vs `where_rows` routing) is unchanged. A pre-existing `<>`-over-null 3VL divergence between the two execution paths (surfaced by #1653's metamorphic test) is tracked separately in #1683.

### Added
- **GFQL native Polars bindings-row tables (`rows(binding_ops)`) — traversal Cypher on polars (#1709)**: the Cypher multi-alias lowering's `rows(binding_ops=...)` op (one row per matched path) now runs natively on `engine='polars'` for **fixed-length connected patterns** — unblocking traversal-shaped Cypher that previously NIE'd: multi-alias property projections (`MATCH (a)-[e]->(b) RETURN a.x, e.w, b.y`), top-k in-degree (`RETURN b.id, count(a) ... ORDER BY ... LIMIT`, graph-benchmark q1/q2), and fixed multi-hop counts (`MATCH (a)-->(b)-->(c) RETURN count(*)`, q8/q9), with forward/reverse/undirected edges, node/edge filters, and edge-alias payload columns — **plus bounded directed variable-length segments** (`-[*i..k]->`, typed `-[:TYPE*i..k]->`, exactly-k; iterative pair joins with Cypher path multiplicity and zero-hop rows), covering the q3 shape (`MATCH (a:Person)-[:FOLLOWS*1..2]->(b:Person) ... RETURN avg(b.age)`). This rung covers the q1–q4 and q8–q9 binding-table shapes exercised by its parity tests; connected q5–q7 planning/execution is layered in follow-on PRs. Also adds native `with_(extend=True)` (emitted by the bindings-path aggregate lowering) and an honest decline for `group_by(key_prefixes=...)` (whole-row bindings grouping — was a latent silent-wrong-key trap). **Honestly deferred** (`NotImplementedError`, no pandas bridge): unbounded `[*]`, undirected/aliased variable-length, shortestPath scalar bindings, node/edge `query=`/endpoint-match params, cartesian (`MATCH (a),(b)`) mode, seeded re-entry contexts. Differential parity vs the pandas oracle (+20 tests incl. path multiplicity and undirected self-loops; conformance corpus extended). The Polars-GPU engine uses the same lazy target-collected plan; exact-head DGX validation is required before making a GPU performance claim.
- **GFQL Cypher connected-pattern (q5–q7) planning + execution on all engines**: completes the `rows(binding_ops)` follow-on promised above — multi-alias *connected* `MATCH` patterns that filter and join across shared endpoints (`MATCH (a)-[]->(b)<-[]-(c) WHERE … RETURN …`, star/two-star shapes) now plan and execute natively on pandas, cuDF, `engine='polars'`, and `engine='polars-gpu'` (previously an honest `NotImplementedError` on polars). The connected-join planner pushes eligible single-alias endpoint predicates into per-frame `filter_dict` prefilters when the column dtype makes the pushdown exact, and keeps them as a post-join `where_rows` residual otherwise (nullable/decimal/object/predicate/`AllOf`/`toLower` shapes), so results stay byte-identical to the full-join semantics on every engine. The two-star executor adds grouped-count and bindings-row fast paths with a per-execution result cache (no cross-call staleness under in-place graph mutation) and openCypher-correct ordering — NULL sorts as the largest value (`ORDER BY … ASC` puts nulls last), and `count(a)`/`count(DISTINCT a)`/`count(*)` follow Cypher multiplicity. Differential parity vs the hand-derived openCypher oracle across all four engines (connected-join conformance corpus).

### Fixed
- **GFQL Cypher `GRAPH { }` residual predicates now fail safely or apply as graph masks**: `GRAPH { MATCH ... WHERE ... }` no longer silently drops predicates that the graph-state path cannot apply. Safe one-node/one-edge residual filters, including disjunctions and `searchAny(...)`, are applied before graph-state matching; unsupported pattern-predicate, multi-alias, and Polars graph-residual cases now raise clear validation errors instead of returning an over-broad subgraph.
- **GFQL Cypher `=~` / scalar-fn cross-engine hardening (review wave, dgx-verified)**: (1) composed `=~` (`WHERE … =~ … OR …`, `RETURN`-expression position) now works on `engine='cudf'` — the series evaluator used raw `.str.fullmatch`, which cuDF lacks, and the resulting `AttributeError` was masked as "unsupported predicate op" (it now routes through the `Fullmatch` predicate's engine workarounds, and honest `NotImplementedError` declines pass through instead of being re-labeled); (2) the cuDF fullmatch emulation anchors alternations as a whole (`^(ab|cd)$` — bare `^ab|cd$` silently matched `'abXXX'`); (3) the cuDF case-insensitive `(?i)` lowercase-folding workaround now declines the fold-unsound shapes — uppercase escape classes (`.lower()` turns `\D` into `\d`, silently inverting the predicate), case-crossing character ranges (`(?i)[A-z]` silently narrowed; `[X-b]` folded to an invalid range), and non-ASCII patterns — while lowercase escapes (`\d`, `\.`) keep folding as before; lookaround, backreferences, and named-group refs decline up front (libcudf rejects them at kernel-compile time); (4) polars `Match`/`Fullmatch` lowering applies the same Rust-regex guard as `Contains` (lookaround/backrefs decline instead of a non-NIE `ComputeError` at collect); (5) `toLower`/`toUpper` on a non-string column decline with a Cypher-standard type error instead of broadcasting the stringified Series repr (pandas/cuDF) or raising a non-NIE `SchemaError` (polars-gpu); (6) polars `floor`/`ceil`/`round` cast to `Float64` so integer columns return Float per Cypher semantics, matching the pandas engine; (7) invalid regex patterns on the composed path raise a clear "invalid regex pattern" error instead of "unsupported predicate op".
- **GFQL Cypher `round()` hardening (review wave, dgx-verified)**: (1) the `polars` extra's floor is now `polars>=1.29` — `Expr.round(mode=)` shipped in py-1.29.0 (pola-rs/polars#22248), not 1.5 as previously pinned, so 1.5–1.28 installs crashed with a raw `TypeError` on `round(x, p>0)` under `engine='polars'`; (2) `round(x, p>308)` is the identity on both engines (a float64 has no digits there) instead of pandas raising through an unclear decline while polars returned identity — parity restored, `10.0**p` overflow guarded; (3) polars `round(x, p>0)` normalizes `-0.0` to `+0.0` like the pandas kernel (`round(-0.04, 1)` was `0.0` on pandas vs `-0.0` on polars — invisible to value equality, pinned by a sign-bit test); (4) documented the precision>0 decimal-string deviation vs neo4j (`round(2.675, 2)` = `2.67` binary-double here vs `2.68` BigDecimal there) and added deterministic tie/hazard matrix cases so ties actually reach cuDF/polars-gpu (the fixture's normal-distribution floats never tied).
- **GFQL `materialize_nodes()` on an edges-only graph under `engine='polars'`**: crashed with `AttributeError: 'Series' object has no attribute 'drop_duplicates'` — the node-id derivation used pandas-only `.drop_duplicates()` / `.reset_index()` / `.rename(columns=)` on a polars Series/DataFrame. Now polars-aware (`.unique(maintain_order=True)`, `.select().rename({...})`), matching the pandas oracle. Surfaced by any `let()` DAG on an edges-only graph under Polars (the DAG pre-materializes nodes). pandas/cuDF paths unchanged.
- **CI: `compute_networkx` HITS test was scipy-version flaky**: `test_compute_networkx_hits_outputs_hubs_and_authorities` used a pure directed 3-cycle (`a→b→c→a`), whose adjacency is a permutation matrix (all singular values equal). networkx HITS computes scores via scipy `svds(k=1)`, which then returns an arbitrary vector from that fully-degenerate singular space; when its components sum to ~0, HITS's `h /= h.sum()` normalization blows up to `inf`/`nan`. Which vector `svds` returns is scipy/LAPACK-version dependent, so the test passed on the pinned scipy but failed on the `nx-upper-scipy` CI profile's newer scipy — a latent flake, not a code regression. Switched to a graph with a well-defined hub/authority structure (`a→b`, `a→c`, `b→c`), whose dominant singular vector is unique (Perron-Frobenius) and therefore version-stable. Test-only change.
- **GFQL `engine='polars-gpu'` LazyFrame-input coercion now collects on the GPU executor**: `Engine.df_to_engine(lazyframe, POLARS_GPU)` materialized a `polars.LazyFrame` input with a bare `.collect()` on the CPU default executor — ignoring `gpu_executor()` and not distinguishing `POLARS` from `POLARS_GPU`. It now routes through the target-aware lazy collect, so a LazyFrame coerced under `polars-gpu` collects on the cudf-polars GPU executor (and `polars` honors `cpu_streaming()`). No change for already-materialized frame inputs (the common path).
- **GFQL Polars engine `contains` honors `regex=`/`flags=`**: the native polars `filter_by_dict` lowering of the `Contains` predicate always used `str.contains(..., literal=False)`, so a **literal** request (`contains(pat, regex=False)`) was still regex-interpreted — a pattern with a metacharacter over-matched (e.g. `contains('a.c', regex=False)` matched `'abc'`), diverging from pandas/cuDF. It also dropped `flags=`. Now `regex=False` lowers to a literal match (`literal=True`; case-insensitive literal folds both sides, matching pandas' result), and regex mode maps `case=`/`flags=` (IGNORECASE/MULTILINE/DOTALL/VERBOSE) to a Rust-regex inline flag prefix (`(?ims…)`). +1 engine-parametrized differential-parity test.
- **GFQL adjacency-index cost gate is engine-aware (never slower than scan)**: the seeded-hop planner falls back to a full scan once the frontier covers too large a fraction of the source keys (past that, scanning all edges once beats many index probes). That crossover fraction is *engine-dependent* — a vectorized-scan engine (polars/cuDF/GPU) has a far faster scan, so its crossover is much smaller than pandas'. The gate previously used a single `0.5·n_keys` threshold, so on polars a **resident index under `index_policy='use'` ran ~2× slower than the plain scan** for mid-size frontiers (measured ~frac 0.02–0.5; correct result, just slow). The gate now uses a per-engine crossover (pandas ~0.5, vectorized engines ~0.02; GPU provisional pending large-graph measurement), so a resident index never loses to the un-indexed path on any engine. Small-frontier wins (the point of the index) are unchanged. +1 engine-parametrized regression test.
- **GFQL `ne()` / `<>` on NULL now follows openCypher/SQL 3-valued logic (pandas)**: `n({"col": ne(x)})` and cypher `WHERE n.col <> x` over a NULL/NA cell used to KEEP the null row on the pandas engine (`NaN != x` → True), diverging from cuDF and the polars engine (both drop it) — and even from pandas' own `WHERE NOT n.col = x` path. Per openCypher/SQL three-valued logic, `null <> x` is `null` (an unknown value cannot be proven unequal to `x`), so a null cell is **not** a match and the row is excluded — consistent with `eq`/`gt`/`lt`/`IN` (which already dropped nulls). Fixed the `NE` predicate to mask out nulls; this corrects both the `filter_dict` predicate path and the single-entity cypher `<>` WHERE path on pandas. cuDF/polars/polars-gpu were already conformant. Verified across all four engines (`ne`, `<>`, `NOT =`, `NOT IN` all drop the null). Note: this is a behavior change for `ne()` on nullable columns under the default pandas engine. (Broader openCypher null-semantics alignment + docs tracked in #1664.)
- **GFQL membership filter (`n({col: [..]})` / `IN`) on NULL follows openCypher 3VL (cuDF)**: a list/membership filter over a NULL cell — e.g. `n({"kind": ["x", "z", None]})` — used to KEEP the null row on the **cuDF** engine (cuDF `isin` matched a null cell against a `None` list element), diverging from pandas and polars (both exclude it). Per openCypher/SQL 3VL, `null IN [...]` is `null` → not a member → excluded. Fixed in `filter_by_dict` (`& notna()` on the membership branch); a no-op for pandas/polars, a fix for cuDF. (Part of the #1664 openCypher-conformance sweep.)
- **GFQL `engine='polars-gpu'` silent CPU fallback removed (NO-CHEATING)**: the GPU collect used `pl.GPUEngine(raise_on_fail=False)`, so any plan node the cudf_polars backend can't execute would silently run **on CPU** and still be reported as a `polars-gpu` result — making `engine='polars-gpu'` indistinguishable from `engine='polars'` whenever the plan isn't fully GPU-capable (a benchmark showing near-identical `polars`/`polars-gpu` timings is exactly this tell). Flipped to `raise_on_fail=True` and translate the cudf_polars failure into a clear `NotImplementedError` pointing at `engine='polars'` for native CPU. `polars-gpu` is now **GPU-or-error**: any timing it produces is real on-device work, never CPU mislabeled as GPU. Verified on dgx-spark (LiveJournal 35M): the seeded-frontier `hop`/2-hop chain plan executes fully on GPU without raising (nvidia-smi 92% util during the loop), so existing GPU timings are unchanged — only the honesty guarantee is added. +1 regression test (the GPU-collect error path is translated, not swallowed).
- **GFQL Polars chain dtype-mismatched join keys**: a chain (`MATCH (a)-[]->(b)…`) crashed under `engine='polars'` with `SchemaError: datatypes of join keys don't match` when an edge endpoint column's dtype differed from the node-id dtype across the int↔float boundary — e.g. a null in a `source`/`destination` column promotes it to `float64` while integer node ids stay `int64`. The hop already aligned join keys; the chain fast paths + combine did not. Now aligns endpoint join keys to the node-id dtype for the traversal and restores the original endpoint dtype on the output (matching pandas). No-op when dtypes already match.
- **GFQL `chain()` OTel span placement**: the `gfql.chain` OpenTelemetry span (`@otel_traced`) had landed on the internal `_try_chain_fast_path` probe instead of the public `chain()` — so `chain()` lost its span and the span recorded the wrong function/attributes. Moved the decorator back onto `chain()`.
- **GFQL Polars engine adversarial-review correctness fixes (#1648)**: a multi-dimension adversarial review of the native polars engine found three reachable divergences from the pandas oracle, now fixed (NO-CHEATING — match pandas or decline honestly): (1) **duplicate alias** — a chain reusing an alias (`[n(name='a'), e(), n(name='a')]`) returned a malformed colliding-join schema (`a`/`a_right`) instead of raising; now raises the same `GFQLValidationError` E201 as pandas. (2) **integer-literal division** — `5/2` lowered to polars true division (`2.5`) but Cypher folds it to integer division (`2`), a silent wrong order when embedded non-monotonically (`ORDER BY n.val % (10/4)`); now declines (`NotImplementedError`). Column `/` int (Float on both engines) is unaffected. (3) **chain seed dtype** — an internal `start_nodes` seed whose id-column dtype diverged from the node-id dtype (e.g. an empty crossfilter selection defaulting to float64 vs int64 node ids) crashed the combine join with `SchemaError`; now aligns the seed join key (mirroring the hop). Also removed stale "pandas bridge" docstrings (the bridge was removed earlier), DRY-consolidated the cross-type/NaN dtype classifiers into `engine_polars/dtypes.py`, and documented a narrow `filter_by_dict` genuine-NaN residual (unreachable on the `from_pandas` ingestion path). +3 regression tests.
- **GFQL Polars engine ISO temporal comparison**: comparing Cypher temporal values (`time({...}) > time({...})`, `date({...}) < date({...})`) gave a wrong answer under `engine='polars'` — the cypher→gfql lowering renders the constructors to ISO strings (`'10:00+01:00'`), and the polars engine compared them **lexicographically** (wrong across timezones/precision; pandas parses them temporally). The lowering now detects an ISO date/datetime/time string-literal operand in a comparison and raises an honest `NotImplementedError` rather than returning a silently-wrong result (native temporal-typed comparison is a tracked follow-up). Surfaced by the TCK run (`expr-temporal7`). **With this, the native polars engine has ZERO wrong-answers across the full Cypher TCK** — every scenario either matches pandas or honestly declines.
- **GFQL Polars engine NaN comparison semantics**: comparisons over a NaN computed inside polars (e.g. `0.0/0.0 > 1`) returned polars' answer, which treats NaN as the *largest* value (`NaN > 1` → True) — but IEEE-754 / Python / pandas / Neo4j-Cypher compare any NaN as **false** (and `!=` as **true**). The expr lowering now masks float comparisons to the IEEE answer (`& ~is_nan` for `< > <= >= =`, `| is_nan` for `<> !=`), gated by conservative float-operand inference so int/string/bool comparisons are untouched (no `is_nan()` on non-float). Note: input NaN from `pandas`→`polars` is already converted to null (`nan_to_null`), so this only affects NaN produced by in-query float math. Surfaced by the TCK run (`expr-comparison2-5`).
- **GFQL Polars engine numeric-vs-string comparison**: comparing a numeric value to a string (e.g. `n.val > 'a'`, `0.0/0.0 > 'a'`) crashed under `engine='polars'` with `ComputeError: cannot compare string with numeric type` (pandas/cypher return a value/null). The lowering now detects a numeric↔string comparison (in both the expression path and the folded filter-predicate path) and raises an honest `NotImplementedError` instead of crashing. Surfaced by the TCK run (`expr-comparison2-5-4`).
- **GFQL Polars engine label match on the `labels` List column**: a label match (`MATCH (n:Label)`) that targets the reserved `labels` List column (e.g. a label with no one-hot `label__X` column — typed-schema unknown labels, OPTIONAL MATCH to a non-existent label) crashed under `engine='polars'` with `InvalidOperationError: cannot cast List type (inner: 'String', to: 'String')` — `filter_by_dict` lowered it to a scalar `==` that tried to cast the List to String. Now uses `list.contains` for List-dtype columns (correct Cypher label-membership: `Label ∈ n.labels`; empty for a non-existent label, matching pandas). Surfaced by the TCK run (`match7-28`, `firstparty-typed-schema1-3`).
- **GFQL Polars engine OPTIONAL MATCH null-fill decline**: a multi-clause `OPTIONAL MATCH` needing null-row fill (some seed rows unmatched) raised a misleading `GFQLValidationError` ("null-row alignment could not recover matched seed identities") under `engine='polars'` — the alignment machinery (matched-id meta, `.iloc` slicing, per-segment concat) is pandas-centric and the polars OPTIONAL MATCH doesn't populate the `_cypher_entity_projection_meta["ids"]` it needs. Now raises an honest `NotImplementedError` (use `engine='pandas'`). Surfaced by the TCK run (`match7-7`, `expr-graph4-4`).
- **GFQL Polars engine OPTIONAL MATCH absent-entity rendering**: an OPTIONAL MATCH miss returning a whole entity (`OPTIONAL MATCH (n) RETURN n` with no match) rendered as `'()'` under `engine='polars'` instead of `null` — the native entity-text expression did not nullify absent rows (whose alias marker column is null). Now mirrors the pandas renderer's `_nullify_missing_alias_rows`; a real property-less node still renders `()`. Surfaced by the TCK run (`match7-1`).
- **GFQL Cypher `WITH`-scalar `MATCH` re-entry on the Polars engine**: a bounded `MATCH ... WITH <scalar> ... MATCH ...` query (carrying a scalar across MATCH clauses) crashed under `engine='polars'` with `AttributeError: 'DataFrame' object has no attribute 'iloc'` — the engine-agnostic re-entry broadcast (`cypher/reentry/execution.py`) used pandas `.iloc`/`.assign`/`.drop(columns=)` on a Polars frame. Made the scalar-row extraction and constant-column broadcast engine-aware (`prefix_rows.row(i, named=True)` + `with_columns(pl.lit(...))` for Polars). The re-entry now completes; any downstream RETURN op the Polars engine doesn't yet render natively raises an honest `NotImplementedError` instead of the crash.
- **GFQL Polars engine predicate pandas-bridge removed (NO-CHEATING) + wider native coverage**: `filter_by_dict` on `engine='polars'` previously evaluated any predicate it couldn't lower natively by converting the column to pandas (`.to_pandas()`), running the predicate's pandas callable, and carrying the mask back — a silent polars→pandas bridge that misrepresented pandas semantics as polars. Removed it: an unsupported predicate now raises `NotImplementedError` (use `engine='pandas'`). To keep common queries native, expanded `predicate_to_expr` lowering to cover `AllOf` (conjunction — e.g. `WHERE n.val > 20 AND n.val < 90` folds to `AllOf[GT, LT]`, lowered recursively), `IsNull`/`IsNA`→`is_null()`, `NotNull`/`NotNA`→`is_not_null()`, and case-insensitive `STARTS WITH`/`ENDS WITH` (anchored `(?i)` regex on the escaped literal). Surfaced from the source-mined optimization review (`pygraphistry4` opportunity #6).
- **GFQL Cypher `UNION DISTINCT` on the Polars engine**: `RETURN … UNION RETURN …` (distinct) crashed under `engine='polars'`/`'polars-gpu'` with `AttributeError: 'DataFrame' object has no attribute 'drop_duplicates'` — the union de-dup in `gfql_unified._execute_compiled_query` called the pandas-only `drop_duplicates` on a Polars frame. Routed through a new engine-aware `Engine.df_unique` (Polars `unique(maintain_order=True)`; pandas/cuDF `drop_duplicates(keep='first')`), matching the existing `row/frame_ops.distinct` convention. Surfaced by the cross-repo TCK conformance run (`tck-gfql`, `TEST_POLARS=1`).
- **GFQL Cypher 3-valued boolean over null literals on the Polars engine**: `null AND null`, `null OR null`, and `NOT null` crashed under `engine='polars'` with `InvalidOperationError: bitand operation not supported for dtype null` / `dtype Null not supported in 'not' operation` — a bare `null` literal lowered to a Null-dtype Polars expression, on which `&`/`|`/`~` are undefined. The boolean lowering now casts AND/OR/NOT operands to `pl.Boolean` first, so Cypher Kleene 3-valued logic (`true AND null = null`, `false OR null = null`, `NOT null = null`) evaluates instead of raising; casting a real Boolean column is a no-op. Surfaced by the TCK run (`expr-boolean1/2/4`).
- **GFQL Polars engine temporal arithmetic decline**: `a.time + duration({minutes: 6})` (in `RETURN`, `ORDER BY`, or `WHERE`) silently became **string concatenation** under `engine='polars'` — Cypher `duration({...})` translates to an ISO duration string literal (`'PT6M'`), and the expr lowering applied `+` to two strings, so e.g. an `ORDER BY` sorted lexicographically on the concatenated text (wrong order). The lowering now raises `NotImplementedError` when `+`/`-` has an ISO-duration string-literal operand (`^-?P(?=[0-9T])`, which doesn't misfire on ordinary strings); the pandas engine handles temporal arithmetic. Surfaced by the TCK run (`with-orderby2`).
- **GFQL Polars engine temporal-constructor property decline**: a standalone property projection over a column holding Cypher temporal-constructor text (`date({year: 1910, month: 5, day: 6})`, `datetime({...})`, … — how Cypher/TCK store temporal values) leaked the raw constructor string under `engine='polars'` instead of the ISO form (`'1910-05-06'`) the pandas projection produces via `_normalize_temporal_constructor_series`. That normalizer is not yet ported natively, so both projection paths (`engine_polars.projection` final result projection and `row_pipeline.select_polars` `WITH`/`RETURN`) now detect temporal-constructor String columns and raise `NotImplementedError` (use `engine='pandas'`) rather than emit a wrong rendering. The detection scans String columns only (numeric/bool projections pay nothing). Surfaced by the TCK run (`with-orderby1-33`+). Whole-entity `RETURN a` over a temporal property is unaffected (it flattens and renders via `render_entity_text`).
- **GFQL Polars engine heterogeneous-column handling (via the `validate`/`warn` convention)**: converting a pandas frame with a mixed-type object column (e.g. `int` and `str` together — legal for dynamically-typed Cypher properties in pandas, but unrepresentable in polars/Arrow) to `engine='polars'` surfaced a cryptic `pyarrow.lib.ArrowInvalid: Could not convert 'xx' with type str: tried to convert to int64` from deep inside polars construction. `Engine.df_to_engine` now handles it per the repo-wide `validate`/`warn` convention: **strict** (the compute-path default) raises a clear `NotImplementedError` naming the offending column(s) and pointing at `engine='pandas'` (no silent coercion by default — that would change comparison semantics), while **`validate='autofix'`** coerces the column(s) to string and warns (matching the cuDF converter and the `plot()`/`upload()` boundary). Surfaced by the TCK run (`expr-comparison2`, `match-where5`, `with-where5`).
- **GFQL cuDF→polars conversion is dtype-lossless (via Arrow)**: converting a cuDF frame to `engine='polars'`/`'polars-gpu'` routed cuDF → pandas → polars, which double-converts **and is lossy** — a cuDF nullable `Int64`/`boolean` degraded through pandas to `float64`+NaN / `object`, so the polars frame started with the wrong dtype before the query ran. Now converts cuDF → **Arrow** → polars (cuDF's native interchange, near-zero-copy on the polars side), preserving dtypes and nulls. For `engine='polars-gpu'` this also removes a device→host→device round trip for the host frame. Verified on dgx (cuDF): `Int64`/`boolean`/`String` + nulls preserved for both polars and polars-gpu targets.
- **GFQL `ne()`/`<>` and `IN`/membership on NULL follow openCypher/SQL 3-valued logic (pandas + cuDF)**: a `ne()` / `<>` filter or a list-membership (`IN`) filter over a NULL cell now EXCLUDES the row — per 3VL, `null <> x` and `null IN [...]` evaluate to `null` (not a match) — consistent with `eq`/`gt`/`lt` (which already dropped nulls) and with the cuDF/polars engines. Previously the pandas `NE` predicate kept null rows (`NaN != x` → True) and cuDF matched a null cell against a `None`/`NaN` list element. Behavior change for `ne()` / `NOT IN` on nullable columns under the default pandas engine. (Broader openCypher null-semantics tracked in #1664.)

### Infrastructure

- **dgx GB10 benchmark safety harness**: `benchmarks/dgx/{safe_run.sh, preflight.py, sitecustomize.py, local_run.sh}` — an RMM device-allocation cap + host-memory watchdog + preflight refusal + hard timeout for running GPU / large-graph benchmarks on the unified-memory GB10 box without OOM-wedging it. Developer/benchmark tooling only; no runtime or library behavior change.
- **Docs CI: polars in the doc-test image (pinned)**: the Sphinx doc-example runner now installs `polars` (pinned to cooldown-safe versions) so `engine='polars'` documentation examples execute — and are SKIPPED, not failed, where polars is unavailable (e.g. the minimal-deps job, Python 3.14). Doc-build / CI only; no runtime or library change.
- **openCypher TCK conformance: `EXISTS { }` existential-subquery scenarios (`expr-existentialsubquery1-1`, `1-3`) now pass**, unblocking the `tck-gfql` CI job. No pygraphistry code change — pygraphistry already renders these correctly (a whole-entity `RETURN n` over a simple-pattern `EXISTS { }` yields the expected entity-text rows). The `success_wrong_rows` was purely a property-map display-whitespace convention (`(:A {prop: 1})` vs the TCK oracle's `(:A {prop:1})`; both valid Cypher). Reconciled harness-side in [tck-gfql#193](https://github.com/graphistry/tck-gfql/pull/193): entity-text whitespace normalization + promoting the two scenarios to supported (and deflaking the pre-existing scipy-`svds` `firstparty-networkx-hits-1` fixture). Conformance/CI only; no runtime or library behavior change.

## [0.57.0 - 2026-06-28]

### Changed
- **GFQL Cypher parse memoization (perf)**: `parse_cypher` now memoizes its result (LRU over the deterministic lark parse+transform → immutable frozen AST). Repeated identical Cypher queries skip the ~15 ms parse — the dominant per-call cost of small queries (~50% of a Cypher call at 100k rows) — making end-to-end query latency ~1.3–1.7× faster at small/interactive sizes across pandas/polars/cuDF. Safe to share the cached AST: every Cypher AST node is `@dataclass(frozen=True)` and `compile_cypher_query` does not mutate the parsed tree; validation errors still raise and are not cached.
- **GFQL structured whole-entity returns (#1650)**: Terminal Cypher `RETURN a` (whole node/edge) now emits **structured flattened columns** (`a.id`, `a.val`, `a.kind`, ...) instead of a single Cypher display string (`({id: 51, val: 51, kind: 'a'})`). The per-field columns already exist before projection, so this is "stop collapsing" rather than "rebuild": measured ~2–6.4× faster on pandas and ~2.7–4.3× on cuDF for whole-entity returns (the win grows with row count, since the old text render is O(rows) and the flat form is ~free), and the result is directly usable without re-parsing a string and survives JSON/CSV/Parquet/Arrow serialization and `plot()`. The human-readable Cypher display string remains available on demand via the `render_entity_text(result, alias)` presentation helper. OPTIONAL-MATCH / `WITH`-reentry / grouping paths that synthesize null/absent entities or still consume a single-column entity value are unchanged. Behavior change: callers that previously read the rendered display string from a terminal `RETURN a` column now receive flattened `a.*` columns. Edge case: a whole entity with NO fields to flatten — an entity with no id binding, no properties, and no type/label (in practice only an edge whose graph has no edge-id binding) — has no `{alias}.{field}` columns to emit, so it falls back to the single Cypher-display-text column under the bare alias (value is correct, e.g. `[]`); nodes always carry their id field and always flatten.

### Performance
- **GFQL temporal-detection dtype gate (#1650)**: `order_detect_temporal_mode` now short-circuits for numeric/bool/complex columns, which can never hold temporal *text*, instead of running an `astype(str)` + multi-regex `fullmatch` scan on every comparison. Eliminates spurious row-wise stringification in `where_rows`/comparison paths whose output never contains entity-text. Byte-identical results; measured `where_rows` speedups ~3.1× (pandas) and ~4.4–13.3× (cuDF, scaling with row count). Does not address whole-entity `RETURN a` text rendering, which is tracked separately.
- **GFQL generic single-hop fast path (perf, pandas + cuDF)**: a single `MATCH (n)` (node-only) or `MATCH (a {f})-[e]->(b)` (1-hop) — the dominant tabular/crossfilter + basic-graph-query shapes — now skip the forward/backward/combine BFS machinery in the generic engine: node-only returns the filtered node table; 1-hop returns the edges whose endpoints pass the node filters + those endpoint nodes. Same VALUES + node/edge sets as before (345-case adversarial golden: only dtype differs; hop/chain suites; gated to pandas/cuDF — dask/spark keep the full path). **~100× faster on pandas** (node filter 204→2 ms @10M; graph query similarly near-raw); cuDF stays on the resident frame (a couple semi-joins instead of the BFS + ~31 drop_duplicates), capturing the GPU semijoin win. **Minor behavior change:** the 1-hop now PRESERVES node-attribute dtypes (int stays int) instead of the full machinery's spurious int→float merge upcast — making pandas/cuDF consistent with the polars engine. The fast path is automatically skipped when a GFQL `policy` is installed, so the `prechain`/`postchain`/`postload` hooks (which can observe intermediate state and deny execution) always fire on the full path — keeping the optimization observationally transparent. Differential-verified equivalent to the full BFS path across 440 random well-formed graph×query cases plus targeted edge cases: it drops edges to nodes absent from the node table and dedups duplicate node ids (matching the full path's join semantics), and a NaN node id never validates a NaN edge endpoint.
- **GFQL generic node-only MATCH fast path (perf, all engines)**: a single `MATCH (n)` (no edge hop) — the dominant tabular/crossfilter shape (`MATCH (n) WHERE/RETURN …`, histograms, filters, table search) — now returns the filtered node table directly + empty edges, skipping the forward/backward/combine BFS machinery in the generic engine (pandas + cuDF). Byte-identical (345-case adversarial golden + hop/chain suites). **~100× faster on pandas at scale** (node filter 204→2 ms @10M, 0.36 ms @1M); cuDF ~0.8 ms @1M / 2.3 ms @10M (op stays on the resident frame). The 1-hop shape is left to the per-engine path (the generic node-merge upcasts int→float, so a join-free generic 1-hop would change dtypes).
- **GFQL polars unconstrained 1-hop fast path (perf)**: a single `MATCH (a)-[e]->(b)` where both nodes are unconstrained (no filter/name/query) and the edge has no match/name/query — the basic graph-query shape and the viz edge-crossfilter MATCH (`MATCH ()-[e]->() WHERE e.x RETURN e`, WHERE/RETURN run later) — now returns ALL edges + their endpoint nodes directly (direction-independent; isolated nodes excluded), skipping forward/backward/combine. Byte-identical (full polars conformance + row-pipeline parity + adversarial graphs: dup/self-loop/cycle/isolated). **~9× faster polars `[n,e,n]`: 95.6→10.3 ms @1M, 855→99 ms @10M.**
- **GFQL polars single-hop fast path (perf)**: a single `MATCH (a)-[e]->(b)` where both nodes have no name/query and the edge has no match/name/query — the basic graph query AND the "filter then expand" viz crossfilter (`MATCH (a {f})-[e]->(b)`, src/dst/both filters) — returns the edges whose endpoints pass the node filters + those endpoint nodes directly (isolated/dead-end excluded), skipping forward/backward/combine. (Unconstrained: all edges, any direction; filtered: forward/reverse — filtered-undirected falls through.) Byte-identical (full polars conformance + row-pipeline parity + adversarial src/dst/both/reverse on dup/self-loop/cycle/isolated). **~9× faster polars `[n,e,n]` (95.6→10.3 ms @1M, 855→99 ms @10M); filtered graph query similarly near-raw.**
- **GFQL polars node-only MATCH fast path (perf)**: a single `MATCH (n)` traversal (no edge hop) — the dominant tabular/crossfilter shape (`MATCH (n) WHERE/RETURN …`, histograms, filters, search) — now returns the filtered node table directly and skips the entire forward/backward/combine + `collect_all` (a ~2.5 ms fixed cost that dominated small/interactive queries). Byte-identical (full polars conformance + row-pipeline parity). **Moves the polars>pandas crossover BELOW 100K** for the real product workloads: e.g. categorical histogram 0.68→1.70× @100K and 1.38→7.62× @1M; node filter 2.44→13.85× @1M; timeline 2.55→8.12× @1M (vs pandas).
- **GFQL polars-GPU in-memory executor (perf+stability)**: the GPU target now collects with cudf-polars' `pl.GPUEngine(executor="in-memory")` instead of the default streaming `engine="gpu"`. GFQL results fit in device memory (the in-memory engine's regime), where it is both faster (semijoin 1.33×, antijoin 2.58×, unique 1.49× @10M) and far more STABLE — the default streaming executor spiked bimodally to ~1 s on the same 10M semijoin (median ~360 ms), while in-memory holds ~30 ms with max ~30. Parity preserved (polars-gpu == polars, 39 tests). NOTE: gfql chains are not GPU-compute-bound (host orchestration + the eager single-hop fast paths dominate), so this is a correctness/stability fix for GPU-collect paths, not a chain-GPU speedup.
- **GFQL polars chain combine collect-once (perf)**: the native polars `chain()` now builds the whole forward/backward COMBINE (combine_nodes/edges + endpoint materialization + alias names) as ONE deferred `pl.LazyFrame` plan over the already-materialized hop frames and collects ONCE, instead of ~a dozen eager ops that each internally `lazy().op().collect()`. Stable order columns restore the eager `g._nodes`/`g._edges` row order (lazy joins don't preserve it) so trailing `LIMIT`/`SKIP` is unaffected — byte-identical (full polars conformance + row-pipeline parity). NO recompute (inputs are materialized; unlike whole-chain fusion). ~5% faster polars 1-hop chain (95.6→90.3 ms @1M, 897→855 ms @10M); GPU-target neutral.
- **GFQL hop/chain redundant-dedup removal (perf)**: dropped the explicit `.unique()` dedup pass that fed only an `.isin()` membership test in the generic traversal — `_filter_edges_by_endpoint`, the undirected combine masks, and the per-hop wavefront filter (`compute/hop.py`, `compute/chain.py`). `isin(s) == isin(s.unique())` by set membership, so this is byte-identical (verified across 345 adversarial graph×query cases: dup/parallel edges, self-loops, isolated/dead-end nodes, cycles, undirected, multi-hop, fixed-point, min/max hops, names, filters, seeds). Each removed `.unique()` is one fewer kernel launch on GPU, where launch latency — not compute — dominates small/mid traversals: cuDF 1-hop chain ~126→103 ms @1M edges (~18% faster), pandas unaffected within noise.
- **GFQL row-expression parse memoization**: `parse_expr()` (the GFQL row-expression parser) now memoizes its result per expression string. Complements the Cypher-query parse memo (PR #1652); profiling showed that once whole-query parsing is cached, the residual fixed per-call compile cost is dominated by `parse_expr`, which is invoked ~4× per query compile (each RETURN/WHERE/WITH expression) and rebuilt a Lark transformer on every call. `parse_expr` is a pure function of the expression text (no params/schema) returning a tree of frozen (immutable) dataclasses, so identical expressions — re-parsed on every compile and recurring across queries (e.g. `a.val > 50`) — are served from an `lru_cache(maxsize=1024)`. Measured (dgx-spark): stacked on the query-parse memo, the fixed per-call cost of a repeated string Cypher query drops a further ~4.5 ms → ~1.8 ms (≈ parity with the equivalent native chain). The non-str/empty guard stays outside the cache; only successful parses are cached (invalid input re-raises every call).

### Fixed
- **GFQL single-hop fast path ignored `prune_to_endpoints`**: the generic single-hop fast path accepted edges with `prune_to_endpoints=True` (a public `e()/e_forward()/e_reverse()` kwarg) but returned BOTH endpoints, whereas the flag keeps only the arrival side (destinations for forward, sources for reverse). A query like `[n(), e_forward(prune_to_endpoints=True), n()]` silently returned the wrong node/edge set. The fast path now declines this shape and falls through to the full path, which honors the flag.
- **GFQL whole-entity `RETURN a, a.val` emitted a duplicate column**: flattening a whole entity `a` into `a.id, a.val, …` (#1650) shares the `{alias}.{field}` namespace with an explicit property projection, so `RETURN a, a.val` produced two `a.val` columns — a duplicate-named column that breaks column selection and silently drops data on `to_dict`/serialization. The duplicate (always identical data, since dotted backtick aliases are rejected) is now collapsed to a single column, keeping first occurrence.
- **GFQL chain on edges-only graphs (no node binding)**: A chain/Cypher query over a graph with edges but no node-id binding (`g._node is None`) that took the full traversal path (any fast-path-ineligible shape — multi-hop, named/queried/`prune_to_endpoints` edges — or any query with a policy attached) rebuilt the result from the *unbound* input graph, dropping the materialized node-id binding. The endpoint-reconciliation concat then synthesized a spurious `None`-named node column: a corrupt result on older pandas, and a hard `NotImplementedError` (void-block NA fill) on newer pandas (e.g. Python 3.14). The result now carries the materialized node-id binding and a single, correct node column, matching the fast-path output; the original edge binding (e.g. `None`) is still restored.
- **GFQL chain fast path bypassed policy hooks**: the degenerate-shape chain fast path (node-only `MATCH (n)`, single-hop) short-circuited before the `prechain`/`postchain`/`postload` policy hook dispatch, so a policy-bearing query that hit the fast path never fired those hooks (or per-op policy inspection). The fast path is now skipped whenever a policy is attached — `prechain` is a pre-compute gate that must observe/block every load — so policy-bearing queries take the full path; the no-policy path keeps the optimization.

## [0.56.1 - 2026-05-27]

### Added
- **GFQL schema effects (#1485)**: Added an internal typed schema-effect model for graph-growing GFQL calls so bound experimental `GraphSchema` snapshots are updated after successful degree, PageRank-style node-property writes, and edge-property write calls. Later local validation can see properties added by those calls without exposing a public `SchemaEffect` API or changing remote GFQL transport.
- **NetworkX Python compute API (#1619)**: Added `g.compute_networkx(...)` for the curated NetworkX algorithm subset already exposed through GFQL local Cypher, including node, edge, and `k_core` graph-returning outputs, plus updated NetworkX notebook/API docs.
- **GFQL NetworkX CALL parity (#1058)**: Expanded the local Cypher `graphistry.nx.*` CALL surface with explicit NetworkX dispatch for `degree_centrality`, `closeness_centrality`, `eigenvector_centrality`, `katz_centrality`, `connected_components`, `strongly_connected_components`, `core_number`, and multi-output `hits`, including row and `.write()` coverage.
- **NetworkX/SciPy optional dependency policy (#1618)**: Declared supported `networkx>=2.5,<4` and optional `scipy>=1.5,<2` ranges for NetworkX-backed GFQL CALL procedures, with runtime version guards and a focused lower/current-upper CI matrix.
- **GFQL schema Arrow boundary APIs (#1339)**: Added experimental public schema↔Arrow import/export helpers, graph-level Arrow declaration payloads, and opt-in `schema_validate='strict'|'autofix'` enforcement for `plot()`, `upload()`, `to_arrow()`, and `validate_arrow_schema()` when a `GraphSchema` is bound.
- **GFQL public schema declarations (#1337)**: Added experimental `graphistry.schema` exports for `NodeType`, `EdgeType`, `GraphSchema`, and `EdgeTopology`, plus top-level `graphistry` re-exports. `NodeType` and `EdgeType` accept Arrow-first `pyarrow.Schema` declarations, preserve dtype/nullability through GFQL `RowSchema`, and export back to Arrow with label/type columns via `to_arrow()`. `graphistry.bind(..., schema=schema)` / `g.bind(schema=schema)` attach public schema declarations to plotters, and Cypher preflight validation consumes the adapted internal `GraphSchemaCatalog` for declared labels, properties, relationship types, and source/destination topology checks. `GraphSchema(strict=False)` makes schema-bound `g.gfql_validate(...)` permissive by default while explicit call-level `strict=True` still forces strict validation.

### Changed
- **`get_degrees` runtime + memory refactor**: Restructured `get_degrees` to outer-merge the in/out per-node aggregates (small × small) before a single merge into the wide nodes frame, replacing the prior two-consecutive-wide-merges path composed from `get_indegrees`/`get_outdegrees`. Removed the `get_outdegrees` swap-and-rebuild-`Plottable` trick. Behavior changes: `get_outdegrees` now returns nodes in natural `materialize_nodes` order; edges with a null endpoint now contribute to the other side's degree (previously silently dropped).
- **GFQL schema physical-column concordance (#1640)**: `GraphSchema` now rejects incompatible logical types for the same physical node or edge property column across declared labels/types, including Arrow-imported declarations. Same-type nullability differences remain type-local, and merged table Arrow schemas mark the aggregate field nullable when needed.
- **GFQL call executor implementation shrink (#1058)**: DRYed private call execution, postcall graph-stat selection, and policy exception enrichment while preserving validated `call()` execution, postcall-on-error behavior, and policy-denial precedence.
- **AI feature test/runtime performance (#1058)**: Reused normalized `SentenceTransformer` model instances within each Python process during `encode_textual()` calls, reducing repeated model construction in `test-full-ai` and user workflows that encode with the same model repeatedly. Added `test-full-ai` duration reporting for continued CI profiling.
- **GFQL Cypher result postprocess shrink (#1058)**: Collapsed private result-projection alias/metadata helpers while preserving prefixed alias whole-row rendering, reentry entity metadata, and pandas/cuDF projection behavior.
- **GFQL hop implementation shrink (#1058)**: Removed stale hop-local debug scaffolding while preserving public `hop()` traversal, hop-label, and pandas/cuDF behavior.
- **GFQL call support implementation shrink (#1058)**: DRYed private call safelist schema-effect helpers and option-column collectors while preserving validated `call()` behavior, schema-effect keys, and diagnostics.
- **GFQL temporal folding implementation shrink (#1058)**: Reused the shared expression AST rebuild helper for temporal constructor folding, preserving recursive folding behavior while covering property-access children such as `duration({days: 1}).days`.
- **GFQL validate entrypoint implementation shrink (#1058)**: DRYed legacy `graphistry.compute.gfql.validate` issue construction, filter-key checks, schema-filter diagnostics, and report formatting while preserving deprecated public validation helpers and anchored error diagnostics.
- **GFQL Cypher reentry execution shrink (#1058)**: DRYed private reentry execution graph-state construction and removed stale internal scalar/free-form helper parameters while preserving whole-row/scalar/free-form reentry behavior, optional null-fill shape, and diagnostics.
- **GFQL Cypher reentry lowering support shrink (#1058)**: DRYed private reentry lowering-support helpers, removed stale old-shape defensive handling, and reused canonical expression/post-processing rebuild paths while preserving reentry behavior and diagnostic source spans.
- **GFQL call validation safelist + encode parity (#1058, #1253)**: DRYed repeated private safelist entry definitions for call validators while adding `encode_edge_size`, `encode_edge_weight`, and point/edge opacity, label, and title encode helpers with GFQL `call()` validation, apply-encodings/schema key contracts, and anchored validator coverage.
- **GFQL CALL procedure implementation shrink (#1058)**: DRYed local Cypher CALL output-column derivation, computed-output renaming, and NetworkX backend error/dependency plumbing while preserving the no-SciPy `graphistry.nx.pagerank` path. Public CALL procedure behavior and diagnostics are preserved.

## [0.56.0 - 2026-05-23]

### Migration / Compatibility Notes

Expected impact is low for users on documented APIs: current `api=3` authentication, valid GFQL/Cypher queries, public `g.gfql(...)` / `g.gfql_remote(...)`, documented `Chain` constructors, and documented predicate APIs continue to work. The rows below cover legacy auth settings, deprecated/private helper imports, deprecated compiler-inspection APIs, and Cypher queries that previously relied on unresolved or ambiguous names.

#### Authentication
`api=3` has been the documented/current authentication path; existing `api=3` username/password, token, personal-key, and SSO flows do not need migration.

| Before | Now | Notes |
|---|---|---|
| `graphistry.register(api=1, key=...)` | `graphistry.register(api=3, personal_key_id=..., personal_key_secret=...)` | Legacy api=1 key upload used removed `/api/check` and `/etl` paths. |
| `graphistry.register(key=...)` | `graphistry.register(personal_key_id=..., personal_key_secret=...)` | `key=` was the legacy api=1 key value. It is deprecated and ignored. |
| `graphistry.pygraphistry.PyGraphistry.api_key(<legacy_api1_key>)` | `graphistry.register(personal_key_id=..., personal_key_secret=...)` | `api_key()` was the legacy one-value api=1 key setter, not the api=3 personal-key login helper. It is deprecated and returns `None`; two-value personal-key calls such as `api_key(key_id, key_secret)` were not supported. |
| `GRAPHISTRY_API_KEY=...` | Pass `personal_key_id=` and `personal_key_secret=` to `graphistry.register(...)` | `GRAPHISTRY_API_KEY` is no longer loaded into the client session. |
| `graphistry.register(api=2, ...)` | `graphistry.register(api=3, username=..., password=...)`, `graphistry.register(api=3, token=...)`, or `graphistry.register(api=3, personal_key_id=..., personal_key_secret=...)` | API version 3 is the only supported upload API. Keep the credential style that matches your deployment. |
| `GRAPHISTRY_API_VERSION=1`, `GRAPHISTRY_API_VERSION=2`, or `client.api_version(1/2)` | Remove the setting, set `GRAPHISTRY_API_VERSION=3`, or call `client.api_version(3)` | Non-3 API versions now fail before upload. |

#### GFQL / Cypher Validation
Valid GFQL/Cypher queries should keep working. The examples below are invalid or ambiguous query shapes that now fail validation instead of reaching later fallback execution.

| Before | Now | Notes |
|---|---|---|
| `MATCH (a) RETURN ghost` | `MATCH (a) WITH a.name AS ghost RETURN ghost` | `ghost` was never bound. Project the value before returning it. |
| `MATCH (a) WHERE ghost.foo = 1 RETURN a` | `MATCH (a) WHERE a.foo = 1 RETURN a` | `ghost` was never bound. `WHERE` clauses must reference bound aliases. |
| `RETURN [ghost] AS xs` | `WITH 1 AS ghost RETURN [ghost] AS xs` or `RETURN [1] AS xs` | `ghost` was never bound. List and aggregate expressions do not create variables. |
| `MATCH (a) MATCH ()-[a]->() RETURN a` | `MATCH (a) MATCH ()-[r]->() RETURN a, r` | Node, edge, path, and scalar aliases must not reuse the same name for different entity kinds. |
| Querying `:Person` or `n.name` when the bound graph schema lacks the label/property | Add the label/property columns to the graph data/schema, or query labels/properties that exist | Legacy loose name-resolution flags do not bypass schema/catalog checks. |
| Running a query and discovering validation errors during execution | Run `g.gfql_validate(...)` or `g.gfql(..., validate=True)` first | Validation reports structured errors before running the query. |

#### Cypher Compiler APIs
Public Cypher execution remains `g.gfql(..., language="cypher")`. This table is for callers importing deprecated compiler inspection internals.

| Before | Now | Notes |
|---|---|---|
| `compile_cypher("MATCH ...")` for execution | `g.gfql("MATCH ...", language="cypher")` | `compile_cypher(...)` exposes deprecated compiler-internal shapes. |
| `compile_cypher("MATCH ...")` to materialize a GFQL chain | `cypher_to_gfql("MATCH ...")` or `gfql_from_cypher(...)` | Use the translation helper instead of a compiler-internal object. |
| Deep imports of `compile_cypher_query` or `CompiledCypher*` | Public execution through `g.gfql(...)`; public translation through `cypher_to_gfql(...)` / `gfql_from_cypher(...)` | Compiler-internal objects are not the public API. |

#### Private GFQL / Plotter Helpers
This table is for private imports/calls. Documented public GFQL and plotter APIs remain available.

| Before | Now | Notes |
|---|---|---|
| `graphistry.compute.gfql.row_ordering` | `graphistry.compute.gfql.row.ordering` | Private module ownership moved under the row package. |
| `graphistry.compute.gfql.order_expr_utils` | `graphistry.compute.gfql.row.ordering` | Row ordering helpers now live in the row package owner module. |
| `graphistry.compute.gfql.row_pipeline_mixin` | `graphistry.compute.gfql.row.pipeline` | Private row-pipeline compatibility shim removed. |
| `graphistry.compute.gfql.row_pipeline_dispatch` | `graphistry.compute.gfql.row.pipeline` | Private row-pipeline compatibility shim removed. |
| `graphistry.compute.gfql.cypher.reentry.runtime` for compile-time helpers | `graphistry.compute.gfql.cypher.reentry.compiletime` | Compile-time reentry ownership moved out of the old runtime-named module. |
| Runtime data-frame reentry helper imports from `graphistry.compute.gfql.cypher.reentry.runtime` | `graphistry.compute.gfql.cypher.reentry.execution` | Runtime reentry stitching helpers live in `execution`. |
| `_plot_dispatch_arrow(...)` | `_plot_dispatch(...)` | Arrow upload dispatch is the current private dispatch path. Public callers should use `plot()` / `upload()`. |
| `_table_to_pandas(table)` | `from graphistry.Engine import df_to_engine, Engine; df_to_engine(table, Engine.PANDAS)` | Engine conversion is the owner for table-to-pandas coercion. |

Documented public APIs remain available: `g.gfql(...)`, `g.gfql_remote(...)`, documented `Chain` constructors, and documented predicate APIs.

### Added
- **GFQL layout-chain predicate (#1254)**: Added public `is_layout_chain()` and `is_layout_kind()` helpers plus canonical layout/radial registry constants so downstream tooling can detect safelisted layout calls from GFQL `Chain` objects, chain lists, wire dictionaries, and legitimate pre-parse strings without maintaining brittle string-match lists. Also exposed `circle_layout`, `tree_layout`, `mercator_layout`, and `modularity_weighted_layout` as GFQL `call()` operations.
- **Schema artifacts for tooling contracts (#1326)**: Added committed structural JSON Schema artifacts for encodings, React settings, and URL params, plus a stdlib exporter/checker and CI drift guard for downstream LLM/tooling contract generation.
- **Viz settings public contracts (#1234)**: Added `graphistry.viz_settings` as a public facade for canonical URL-parameter and React-facing visualization setting key constants, `Literal` key aliases, typed settings payloads, and frontend contract metadata for downstream type-checking.
- **GFQL encode call parity (#1241)**: Added GFQL `call()` support for `encode_edge_icon` and `encode_axis`, plus deeper encode validator diagnostics for palette and categorical mapping payloads before remote or local execution.
- **Changed-line coverage hygiene (#1533)**: Added a PR-only changed-line coverage gate that combines CPU coverage from the existing minimal and GFQL core test jobs, reports covered/missing executable package lines touched by a PR, and enforces an initial changed-line threshold without blocking on historical uncovered code.
- **Coverage audit profiles (#1517)**: Added a coverage.py-backed audit helper with an initial GFQL dead-code triage profile that emits markdown/JSON zero-hit and low-hit reports for parser, lowering, row-pipeline, temporal, AST, unified, and chain files. CI now collects coverage inside the existing `test-gfql-core (3.12)` run via `pytest-cov`, uploads the pandas CPU audit artifact with per-file coverage lock-ins, and the RAPIDS Docker wrapper can run the same profile against periodic DGX RAPIDS 25.02 and 26.02 cuDF lock-ins.
- **GFQL policy / Cypher compiler hooks (#1454)**: Added experimental exact-key `precompile` and `postcompile` policy hooks for local Cypher string-query compilation. `postcompile` reports success or failure using the existing policy `success`, `error`, and `error_type` fields plus a stable `CompileSummary` with scalar compiler metadata.

### Changed
- **GFQL predicate/AST implementation shrink (#1058)**: DRYed repeated predicate comparison dispatch, regex string predicate scaffolding, and AST filter-dict serialization while preserving public predicate APIs, wire JSON, and compiler-plan behavior.
- **CI Spark/Neo4j dependency hardening (#1322)**: Pinned the Spark smoke-test installer to a committed hashed lockfile with a CI drift check, and moved the Neo4j connector test image from EOL Neo4j 4.1 to pinned Neo4j 5.26 LTS.
- **GFQL axis/ring diagnostics (#1245)**: Strengthened `encode_axis` and ring-layout axis validators with anchored row-indexed diagnostics for documented radial/linear axis payload mistakes, while preserving extension-subtype compatibility.
- **GFQL chain tag warning cleanup (#877)**: Avoided pandas object-dtype `fillna()` downcast warnings when merging named chain tag columns without warning filters or behavior changes.
- **GFQL chain input docs (#1255)**: Clarified that native GFQL chains must be passed as materialized Python list/dict/`Chain` objects, while `g.gfql(str)` remains the Cypher string-query entrypoint rather than a parser for stringified Python or JSON chain literals.
- **Compute hop min_hops label semantics (#878)**: `hop(..., min_hops=..., label_node_hops=...)` now labels nodes by the shortest retained path after `min_hops` pruning, instead of keeping a first-seen label from a shorter branch that was filtered out. The cuDF label-fill path now reliably uses merge-based mapping on RAPIDS 25.02+ instead of falling through to `cudf.Series.map(...)`.
- **GFQL temporal implementation shrink (#1563)**: DRYed temporal value timezone/scalar helpers, shared duration token parsing and day-time duration formatting ownership, and separated temporal AST traversal from constructor-folding decisions while preserving `temporal_text` compatibility and runtime behavior.
- **GFQL Cypher parser implementation shrink (#1599)**: DRYed duplicated private parser rule handlers and dataclass reconstruction paths while preserving parsed AST shape, source text/span fidelity, and compiler-plan surfaces.
- **GFQL Cypher binder implementation shrink (#1601)**: DRYed private binder projection, pattern-alias, scalar-binding, and stale strict-schema helper plumbing while preserving structured diagnostics, source/span fidelity, and compiler-plan surfaces.
- **GFQL Cypher projection-planning implementation shrink (#1607)**: DRYed private projection reference parsing, duplicate whole-row projection handling, and optional-projection shape gating while preserving structured diagnostics, source/span fidelity, and compiler-plan surfaces.
- **GFQL Cypher reentry compiletime implementation shrink (#1058)**: DRYed private reentry compile-time extras threading, carried-alias plan assembly, outer-query context stripping, and ORDER BY rewrite helpers while preserving structured diagnostics, source/span fidelity, and compiler-plan surfaces.
- **Legacy API-key compatibility deprecation (#1539)**: Deprecated the removed api=1 `api_key()` compatibility surface so explicit key use warns and is ignored, stale `GRAPHISTRY_API_KEY` environment values are not loaded, api=1/2 fail locally before upload, and remote integration docs now point to api=3 JWT or personal key ID/secret authentication.
- **GFQL Cypher reentry cleanup (#1555)**: DRYed duplicated carried-endpoint reentry flattening tests and direct reentry carrier backend tests, and trimmed stale private reentry helper export/comment scaffolding while preserving runtime behavior.
- **GFQL same-path native cleanup (#1549)**: Removed stale same-path physical-route test scaffolding and duplicated shortestPath parity cases that are already covered by the parametrized parity suite. Also removed unused private same-path executor delegation shims while preserving runtime behavior.
- **Compute predicate AST cleanup (#1550)**: DRYed private startswith/endswith string predicate helper logic, removed a duplicate predicate JSON registry entry, and consolidated duplicated predicate serialization and pandas/cuDF string-boundary tests while preserving public predicate JSON/API behavior.
- **GFQL chain API compatibility audit (#1541)**: Removed a private chain `rows()` binding-op injection helper surface and helper-level tests, keeping behavior covered through public `g.gfql([...])` execution, and pruned stale in-code Chain/GFQL examples from private implementation docs. Runtime behavior is unchanged.
- **GFQL coverage baseline cleanup (#1540)**: Simplified GFQL CPU/RAPIDS coverage baselines to the enforced per-file coverage floors, removing stale statement/covered-line metadata that can drift after private helper deletion tranches. Also removed dead Cypher lowering imports and an unused private lowering parameter.
- **GFQL private compatibility shim cleanup (#1520)**: Removed obsolete private GFQL compatibility import shims for row dispatch, row pipeline, row ordering, and order-expression helpers after auditing that repo callers now use the canonical owner modules. Also removed tests that only asserted retired private/import compatibility surfaces; runtime behavior is unchanged.
- **GFQL / Cypher parser/lowering delegate deletion (#1521)**: Removed stale private parser/lowering compatibility delegates and split-guard baselines: an unreachable parser raw-WHERE branch, a no-op variable-length relationship rejection shim, and old lowering-level reentry helper re-export bindings. Runtime semantics are unchanged; reentry helper imports now use their owning submodules directly.
- **GFQL temporal cleanup tranche (#1524)**: Removed stale post-split temporal helper imports, deduplicated duration timedelta helper ownership, retired an unused split-module day-time duration parser, and consolidated one-case temporal lowering tests while preserving `graphistry.compute.gfql.temporal_text` compatibility access.
- **GFQL / Cypher residual unplanned-chain fallback deletion (#1486)**: Removed the final approved runtime fallback branch for compiled Cypher queries that reach non-union execution without a logical plan. Post-#1503 inventory found no remaining approved-chain defer-code hits, and formerly approved `multiple_match_stages` / `optional_match_reentry` deferrals now fail fast instead of silently chain-executing.
- **GFQL / Cypher lowering DRY audit (#1509)**: Removed an unused private projection-validation delegate from `graphistry.compute.gfql.cypher.lowering` after auditing the post-reentry helper surface; behavior is unchanged because projection validation remains owned by `cypher.projection_planning`.
- **GFQL temporal text helper split (#1498)**: Split temporal constructor parsing, value parsing, duration parsing/folding, truncation folding, and AST rewrite helpers into `graphistry.compute.gfql.temporal` modules while keeping `graphistry.compute.gfql.temporal_text` as a small compatibility shim. No temporal semantics changed.
- **GFQL / Cypher path-alias carry multiple MATCH native planning (#1495)**: shortestPath queries that carry an unused named path alias into a follow-on `MATCH` now compile with a logical plan and dispatch through the physical same-path route instead of the final `multiple_match_stages` residual chain fallback. Path-value references such as `length(path)` after a follow-on `MATCH` now fail fast rather than relying on unsupported residual-chain path materialization.
- **GFQL / Cypher modularization audit cleanup (#1058)**: Removed an unused private aggregate-alias helper from Cypher lowering after auditing the large GFQL engine files for narrow safe deletion opportunities.
- **GFQL / Cypher shortestPath endpoint-binding multiple MATCH native planning (#1489)**: shortestPath queries whose earlier `MATCH` clauses only bind endpoint nodes now receive a logical plan and enter native physical dispatch instead of the residual `multiple_match_stages` chain fallback. Path-alias-carrying follow-on `MATCH` shapes remain explicitly classified under `multiple_match_stages`.
- **GFQL / Cypher M1 differential scaffold deletion (#1472)**: Removed the obsolete legacy-vs-candidate differential test scaffold after preserving its unique independent `OPTIONAL MATCH` row behavior as direct lowering coverage. No runtime/compiler behavior changed.
- **Tests / GFQL Cypher fixture cleanup (#1073)**: DRYed duplicate pandas/cuDF test graph factory twins in `graphistry/tests/compute/gfql/cypher/test_lowering.py` by sharing pandas DataFrame fixture helpers while preserving existing factory call sites and test semantics.
- **GFQL / Cypher non-top-level OPTIONAL MATCH native planning (#1478)**: `MATCH ... OPTIONAL MATCH ...` aggregate/scalar shapes such as `MATCH (a) OPTIONAL MATCH (a)-->(b) RETURN count(b) AS c` now compile with a logical plan and dispatch through the physical-planner route instead of the approved unplanned-chain fallback bucket. Removed `non_top_level_optional_match` from the residual fallback allowlist and added runtime cutover coverage.
- **GFQL / Cypher anonymous MATCH native planning (#1480)**: Anonymous `MATCH ()` scalar/count projections such as `MATCH () RETURN count(*) * 10 AS c` now compile with a logical plan and dispatch through the physical-planner route instead of using the approved unplanned-chain fallback bucket. Removed `anonymous_match` from the residual fallback allowlist and added runtime cutover coverage.
- **GFQL / Cypher ordinary sequential multiple MATCH native planning (#1477)**: Ordinary non-path-alias sequential `MATCH` clauses now receive a chained logical `PatternMatch` route and enter physical same-path dispatch instead of the residual unplanned chain fallback. Remaining multiple-`MATCH` path-alias/shortestPath shapes stay explicitly classified under the reviewed `multiple_match_stages` defer bucket for follow-up splitting.
- **GFQL / Cypher residual unplanned-chain fallback hardening (#1468, #1478, #1479)**: Cypher logical-planner deferrals now carry structured `logical_plan_defer_code` metadata, and runtime chain fallback is limited to explicitly approved residual buckets for multiple-`MATCH`/shortestPath lanes. Unclassified or unknown `logical_plan is None` compiled programs now fail fast before chain execution; scalar projection alias `MATCH` and non-top-level `OPTIONAL MATCH` shapes now route through the logical planner instead of approved chain fallbacks.
- **GFQL / Cypher pattern predicate existence semantics (#1449)**: Direct-Cypher `WHERE (pattern)` predicates now lower through correlated semi-apply markers instead of rewriting single positive predicates into appended `MATCH` clauses, preventing existence checks from multiplying result rows. Added pandas/cuDF coverage for the residual `expr-pattern1-10`, `expr-pattern1-13`, and `expr-pattern1-18` undirected pattern-predicate wrong-row cases.
- **GFQL / Cypher reentry failfast scaffolding cleanup (#1421)**: Removed the obsolete `graphistry.compute.gfql.cypher.reentry.runtime` compatibility re-export shim after compile-time reentry ownership moved to `reentry.compiletime`, moved tests off the old private `gfql_unified._compiled_query_reentry_state` access path, and lifted the stale closed-#1256 aggregate failfast so chained reentry secondary-property carries now flow through downstream aggregating `WITH` stages with positive row assertions.
- **GFQL / Cypher pre-strict binder compatibility guard deletion (#1420)**: Retired the legacy loose `FrontendBinder.bind(strict_name_resolution=False)` graph traversal path and unresolved-name fallbacks now that #1357 made strict binder semantics canonical. Cypher compile prepass and graph-constructor binding now pass `strict_name_resolution=True` explicitly, and binder tests now pin that the legacy false flag no longer admits unresolved `collect(...)`, single-alias list literal, or missing-schema inputs while preserving strict source-order traversal through `WITH → UNWIND → MATCH`.
- **GFQL / Cypher strict-name-resolution runtime default flip + permissive baseline retirement (#1357)**: `compile_cypher_query()` now binds the post-normalize Cypher AST with `strict_name_resolution=True` in `graphistry/compute/gfql/cypher/lowering.py`, making runtime alias/name enforcement align with strict validator behavior by default. Updated strict rollout baselines in `graphistry/tests/compute/gfql/cypher/test_binder_strict_compile_baseline.py` to pin strict-default admits/rejects (including structured E204 unresolved-identifier checks) and refreshed parity framing in `graphistry/tests/compute/gfql/cypher/test_validator_runtime_strict_parity.py`.
- **GFQL / Cypher bounded reentry terminal `WITH ... UNWIND [alias] AS x` identity admit (#1349)**: In bounded reentry compile-time lowering (`graphistry/compute/gfql/cypher/reentry/compiletime.py`), terminal reentry tails now admit the narrow identity-unwind shape `UNWIND [x] AS y` by rewriting downstream expressions from `y` to `x` and removing the unwind at compile time. This preserves vectorized execution while unlocking the previously fail-fast `... MATCH ... WITH c, bid UNWIND [c] AS c2 RETURN ...` lane; broader post-reentry UNWIND shapes remain explicitly unsupported.
- **GFQL / Cypher joined-row projection multi-whole-row admit (#1393)**: Multi-alias whole-row projection stages that run on bindings-row joins are now admitted for safe same-kind alias sets (node/node or edge/edge) instead of hard failing with the one-source-alias boundary. Stage active-alias fallback is constrained to that specific boundary, and joined connected multi-pattern whole-row projection coverage was added in `test_lowering.py`.
- **GFQL / Cypher ORDER BY on stringified-list columns uses Cypher list-orderability (#1359, meta #1353 item #1)**: When a list-valued property is stored as a string column (e.g. round-tripped through CSV / Arrow string columns), `ORDER BY` previously fell back to lex string sort, which mishandles negative numbers because `"-"` < `"2"` in ASCII (e.g. `"[1, -20]"` sorted before `"[1, 2]"`). Added `order_detect_stringified_list_series` + `parse_stringified_list_series` in `graphistry/compute/gfql/row/ordering.py`, and routed the row pipeline through `build_list_sort_columns` after `ast.literal_eval`-parsing the string entries when the column is fully list-shaped (`^\[.*\]$` per-row). Python-list-typed columns continue through the existing list-aware path unchanged. Includes pygraphistry-side regression coverage on both Python-list and stringified-list inputs (`test_string_cypher_order_by_python_list_column_uses_list_orderability`, `test_string_cypher_order_by_stringified_list_column_uses_list_orderability`). The matching TCK port-level fixture/runner fixes that flip the 14 `with-orderBy` wrong-row scenarios to `success_matches_expected` are tracked in [tck-gfql #36](https://github.com/graphistry/tck-gfql/issues/36).

### Fixed
- **Arrow conversion autofix null preservation (#867)**: `validate='autofix'` mixed-type pandas-to-Arrow repair now uses pandas' nullable string dtype when available, preserving missing values as Arrow nulls while keeping the existing `validate='strict'` failure path and `warn` behavior. Older pandas versions fall back to standard string coercion.
- **GFQL same-path cuDF multi-hop WHERE parity (#872)**: Fixed the RAPIDS 25.02 cuDF hop-label fill path used by exact/min-hop same-path execution so multi-hop `WHERE` pruning can complete without falling into `cudf.Series.map(...)` UDF compilation. Added pandas/cuDF parity coverage for linear, diamond, and cycle multi-hop `WHERE` shapes.
- **Compute node materialization for empty edges-only datasets**: `materialize_nodes()` now returns a bound empty node table for empty edge tables instead of leaving `_nodes` unset, and repeated degree materialization reuses bound empty node tables instead of dropping degree columns. This keeps downstream degree and GFQL degree paths from seeing edges-only plottables with missing node tables.
- **SSO / site-wide login (#1228)**: `ArrowUploader.sso_get_token` no longer raises when the server's JWT response omits or nulls the `active_organization` field. Caller-supplied `org_name` is preserved when the server response is silent, mismatched server-bound organizations raise an actionable error, and first-login site-wide flows can fall back to the JWT username as the personal organization. This is backwards compatible with both pre-graphistry/graphistry#3002 servers that omit `active_organization` and newer servers that emit it.
- **GFQL / Direct-Cypher WITH-scope ORDER BY validation (#1530)**: Raw Cypher string execution now rejects `ORDER BY` aliases that were already dropped by earlier `WITH` projections, matching strict binder scope semantics and unblocking the parked tck-gfql `with-orderby1-46` expected-error lane. Added direct `g.gfql()` and binder regressions for stale aliases plus valid projected/current-alias `ORDER BY` shapes.
- **GFQL / Cypher connected OPTIONAL MATCH cuDF start-node seeding (#1495)**: Connected OPTIONAL MATCH native dispatch now converts already-bound optional-arm seed rows to the requested dataframe engine before `isin()` filtering, avoiding a pandas/cuDF `Series` boundary failure in RAPIDS 25.02 full GFQL validation.
- **GFQL / Cypher bound node identity vs. `id` property (#1490)**: Direct-Cypher lowering now uses an internal node-identity token that resolves to the graph's bound node column at execution time, so custom node-id bindings no longer collide with a user data property named `id`. Added regression coverage for repeated node aliases, distinct/grouped node identity, and projecting `id` as ordinary node data.
- **GFQL / Cypher IC4 unlabeled `HAS_TAG` endpoint disambiguation (#1496)**: Connected direct-Cypher row bindings now disambiguate unlabeled `HAS_<Label>` destination aliases when graph node ids collide across labels and the implied destination label column exists. This keeps LDBC IC4/new-topics-style `...-[:HAS_TAG]->(tag)` bindings on tag rows before the CASE-derived `sum(valid)` / `sum(inValid)` post-aggregation `WHERE` filter is applied. Added pandas/cuDF regression coverage for the official IC4 query shape with cross-label numeric id collisions.
- **GFQL / Cypher IS7 optional-arm materialization bound (#1488)**: Connected `MATCH ... OPTIONAL MATCH` execution now seeds an OPTIONAL arm from accumulated base rows when the arm starts on an already-bound shared node alias, so IS7/message-replies-style `OPTIONAL MATCH (m)-[:HAS_CREATOR]->...` routes bound the optional traversal before `rows(binding_ops=...)` materializes the arm. Added instrumentation regression coverage proving the optional arm starts from only the selected base message instead of all message nodes.
- **GFQL / Cypher primitive literal boolean spans (#1470)**: Primitive literal parser nodes now carry source text/span metadata through boolean expression wrapping, preserving exact `true`/`false`/`null`/number operand slices while structured predicates and property maps still receive raw Python values. Removed the stale primitive-literal text fallback in the boolean atom wrapper and added focused parser coverage.
- **GFQL / row-pipeline map/list AST expression fallback shrink (#1469)**: Row-pipeline string expressions now evaluate map literals with vector-valued entries through the AST evaluator instead of rejecting the shape, which also enables list literals containing those maps. Cypher aggregate expressions inside map literals remain a fail-fast unsupported shape instead of executing with wrong rows. Added pandas/cuDF coverage for nested map/list values with null preservation.
- **GFQL / Cypher string subscript expected-error drift (#1450, #1353)**: Direct-Cypher integer subscripts on scalar strings now raise the row-pipeline list-like-base error instead of falling through to Python string indexing, aligning `WITH '1' AS list, 0 AS idx RETURN list[idx]` with the expected-error contract. Added focused positive list-index and negative string-index regression coverage in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **Tests / GFQL Cypher cuDF assertion hardening (#1439)**: Swept `graphistry/tests/compute/gfql/cypher/test_lowering.py` for remaining direct cuDF assertion conversions through `.to_pandas().to_dict(orient="records")` and routed them through the existing Arrow-first `_to_pandas_df` helper. This extends the #1415/#1437 IC6 assertion hardening pattern across the lowering suite and avoids RAPIDS 25.02 host-materialization crash paths in test assertions without changing runtime semantics.
- **GFQL / Cypher empty optional-match post-aggregate list-comprehension projection (#1367, #1353)**: Empty MATCH aggregate synthesis now evaluates post-aggregate projections before seeding the result row, so `size([x IN collect(r) WHERE x <> null]) AS cn` returns `0` instead of leaking the hidden `__cypher_postagg__` temporary column.
- **GFQL / Cypher mixed wrong-row tail cleanup (#1369, #1353)**: Exact zero-hop variable-length relationships (`*0..0`) now preserve the seed node as the valid zero-length path instead of returning no rows, and empty-graph post-aggregate expressions now project the final expression/column name after synthesizing the empty aggregate row (for example `MATCH (a) RETURN count(a) > 0` returns `false` under the requested output name). Follow-up test amplification also fixed sibling post-aggregate expressions that reused the same internal temp name across return items. Added pandas/cuDF-focused coverage in `graphistry/tests/compute/gfql/cypher/test_lowering.py`; sibling TCK probing moves `match5-8` and `return2-10` from wrong-row to matches-expected, while the remaining #1369 keys require separate pattern-predicate/varlen-chain, map/fixture, or WITH fixture/port follow-up.
- **Tests / IC6 tag-cooccurrence cuDF assertion routes through Arrow-safe helper (#1415, #880)**: `test_issue_1396_issue_1415_tag_cooccurrence_join_aggregation_counts_on_cudf` in `graphistry/tests/compute/gfql/cypher/test_lowering.py` previously called `result._nodes.to_pandas().to_dict(orient="records")` directly, which crashes the RAPIDS 25.02-cuda12.8 ARM image (segfault in `numba_cuda` CUDA driver context init during `cudf.core.column.numerical.to_pandas`). Switched to the existing `_to_pandas_df` helper (line 851), which routes cuDF results through `to_arrow().to_pandas()` and is the blessed pattern for cuDF-result assertions in this file. Validated on DGX RAPIDS 25.02 + 26.02 via `docker/test-rapids-official-local.sh`.
- **GFQL / Cypher IC3 carried-row reentry joined aggregation (#1413, #880)**: Cypher lowering now admits the IC3/cross-country-style `WITH person, collect(city) AS cities MATCH ...` bounded-reentry shape with multiple post-reentry `WITH` stages, carries the collected city list alongside the `person` whole-row alias, and evaluates whole-row node membership such as `NOT friendCity IN cities` against collected entity lists. Added adversarial regressions for `collect(city)` and `collect(DISTINCT city)` with searched CASE aggregation and post-aggregate filtering in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher joined-row aggregation CASE chained-comparison lowering (#1413, #880)**: Cypher lowering now rewrites chained comparisons inside searched `CASE WHEN ... THEN` conditions before row-expression validation, so LDBC IC4/new-topics-style joined-row aggregation queries using `$startDate <= post.creationDate < $endDate` no longer fail the local GFQL subset gate. The rewrite is constrained to unquoted CASE conditions, preserves unrelated CASE comparison bodies, and adds adversarial regression coverage for multiple chained CASE arms and multiple joined-row aggregation CASE flags in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher temporal historical named-zone canonicalization + comparison parity (`#1406`, `#1353`)**: Direct-Cypher datetime canonicalization now applies Neo4j/TCK-compatible historical timezone offsets for pre-standard-time `Europe/Stockholm` named-zone literals in `graphistry/compute/gfql/temporal_text.py` (notably `1818-07-21` -> `+00:53:28`). This closes the residual wrong-row case `expr-temporal2-6-5` and keeps equality/comparison behavior consistent when one side is zone-derived and the other is explicit offset text. Added focused regression coverage in `graphistry/tests/compute/gfql/cypher/test_temporal_text.py` and `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher structural list/map equality now preserves null-unknown semantics (#1405, #1353)**: Direct-Cypher and row-pipeline comparison evaluation in `graphistry/compute/gfql/row/pipeline.py` now uses recursive tri-valued structural equality for list/map families under `=`, `!=`, and `<>`, so nested null comparisons return `null` instead of collapsing to Python `true/false` (for example `[[1], [2]] = [[1], [null]]`, `{k: null} = {k: null}`). Added regression coverage in `graphistry/tests/compute/test_gfql.py`, plus expanded ORDER BY nested non-primitive (raw + stringified map/list) pandas/cuDF parity amplification in `graphistry/tests/compute/gfql/test_row_pipeline_ops.py`, including RAPIDS 25.02/26.02 dgx validation.
- **GFQL / Cypher tri-valued list/map/null expression semantics tranche C (#1407, #1353)**: Row-expression equality/membership now preserves openCypher three-valued null semantics for nested list/map comparisons in `graphistry/compute/gfql/row/pipeline.py` by using recursive value equality with null-unknown propagation and routing `IN` through the same tri-valued comparator. This fixes direct-Cypher wrong-row outcomes where nested null comparisons were previously collapsed to booleans (`expr-comparison1-6-5`, `expr-comparison1-7-{12..16}`, `expr-list5-{21,29,31,34}`). Optional single-MATCH empty-row projection synthesis now infers deterministic `IS NULL` / `IS NOT NULL` outputs for optional-alias operands instead of nulling all projected columns (`expr-null1-3`, `expr-null2-3`) via `projection_planning.py` + `lowering.py` empty-result-row wiring. Added focused regressions in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / row-pipeline ORDER BY stringified-list engine-boundary hardening + row-test layering cleanup (#1373)**: Hardened cuDF caller invariants in `graphistry/compute/gfql/row/pipeline.py` so stringified-list and dynamic-list-subscript ORDER BY lanes must materialize cuDF series in cuDF mode (fail-fast on CPU-series boundary violations instead of silently tolerating them). For the explicit RAPIDS `<=25.02` host-bridge compatibility lane, ORDER BY now converts back to cuDF before returning row tables. Also moved list/subscript ORDER BY coverage toward row-pipeline tests by adding stringified subscript ordering coverage in `graphistry/tests/compute/gfql/row/test_ordering.py`, adding cuDF bridge-path return-to-cuDF coverage in `graphistry/tests/compute/gfql/test_row_pipeline_ops.py`, and removing redundant row-heavy cypher integration duplicates from `graphistry/tests/compute/gfql/cypher/test_lowering.py` (keeping thin cypher smoke coverage).
- **GFQL / Cypher sequential non-optional MATCH clause merge for reply-author row-shaping joins (#1395)**: Cypher lowering now admits sequential non-optional `MATCH` clauses with relationship patterns by merging compatible clauses into a single connected-match lowering surface when node-only seed binding is not applicable (`graphistry/compute/gfql/cypher/lowering.py`). This unblocks IC8/IS7-style reply-author row-shaping queries that previously failed with `Only node-only pre-binding MATCH clauses are supported...` while preserving existing node-seed behavior. Added regression coverage for sequential-MATCH IC8 and IS7 shapes in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher recursive ancestor row-join duplicate carried-id reentry fallback (#1394)**: Whole-row bounded reentry execution now handles duplicate carried reentry ids with scalar carries by falling back to per-prefix-row suffix execution + union in `graphistry/compute/gfql_unified.py` when the structured duplicate-carried-row guard fires from `graphistry/compute/gfql/cypher/reentry/execution.py`. This unlocks repeated-primary chained reentry row-binding shapes that previously failed on `unique carried node rows`, while preserving the low-level invariant guard and adding a success regression lock in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher IC11 reentry-prefix alias scope (`WHERE NOT(person=friend)`) for employment/company row-join lane (#1391, #880)**: In `graphistry/compute/gfql/cypher/lowering.py`, `_lower_projection_chain` now preserves pre-projection binding aliases specifically for MATCH-row `WHERE` validation and bindings-row routing, so reentry prefix scope checks no longer incorrectly drop the seed alias before evaluating `NOT(person=friend)`. This admits the IC11/job-referral-shaped prefix (`MATCH ...[:KNOWS*1..2]... WHERE NOT(person=friend) WITH DISTINCT friend`) through native lowering while keeping downstream projection scope trimming unchanged. Added CPU + cuDF regression coverage in `graphistry/tests/compute/gfql/cypher/test_lowering.py` for the full employment/company/country trailing join/order shape.
- **GFQL / row-pipeline ORDER BY stringified-list cuDF GPU-native path + RAPIDS 25.02 compatibility guard (#1376, meta #992)**: `parse_stringified_list_series()` in `graphistry/compute/gfql/row/ordering.py` now returns engine-native `SeriesT` (cuDF stays cuDF), so the default ORDER BY list-key path remains GPU-native on modern RAPIDS/cuDF. Added a narrow version-gated fallback in `graphistry/compute/gfql/row/pipeline.py` for cuDF `<=25.02` list-sort lanes that are known to crash in upstream internals, bridging through Arrow-host materialization only for that compatibility path. Also narrowed generic exception handling in ordering index fallback and kept cuDF regression coverage for stringified-list ORDER BY behavior.
- **GFQL / Cypher precedence integer-division semantics in runtime row expressions (#1366, #1353)**: Runtime row-expression lowering now applies openCypher integer-division rewrite for integer-literal arithmetic in non-aggregate expressions, so precedence scenarios using `/` match truncating integer semantics instead of Python true-division (`expr-precedence2-1-{2,5,7,8,9,10,11,12,14,17}`). Added targeted regression coverage in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher shortestPath `length(path)` dtype + integer division semantics on cuDF/pandas parity (#1354)**: `length(path)` projections sourced from `__cypher_shortest_path_hops__*` now normalize through engine-aware numeric coercion in `graphistry/compute/gfql/row/pipeline.py` so cuDF no longer surfaces string/object-typed distances for reachable shortest-path rows. The same row-expression path now preserves Cypher integer-division behavior for integer scalar `/` expressions (truncate toward zero instead of always widening to float), fixing the direct-cypher tck wrong-row case `expr-mathematical8-1` (`1` vs prior `1.0`). Follow-up hardening removed broad module-import exception swallowing in shortestPath numeric-coercion guards so `ImportError`/`ModuleNotFoundError` propagate instead of being silently ignored. Added/updated regression coverage in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher duration toString + equality preserve openCypher components (#1361, #1353 item #2)**: `_normalize_duration_map` (`graphistry/compute/gfql/temporal_text.py`) now keeps the openCypher CIP `Duration` component groups (months / days / seconds-and-nanoseconds) separate through canonicalization instead of collapsing days into total nanoseconds. `duration({years: 12, months: 5, days: -14, hours: 16})` canonicalizes to `P12Y5M-14DT16H` (was `P12Y5M-13DT-8H`); `duration({days: 1, milliseconds: -1})` canonicalizes to `P1DT-0.001S` (was `PT23H59M59.999S`). Component-wise canonicalization also makes `=` between two durations with equal totals but different `(days, seconds)` component shapes return `false` per spec, instead of normalizing both sides to the same total. Closes 3 of 14 wrong-row classifications for `expressions/temporal` under direct-Cypher tck contract (`expr-temporal6-6-2`, `expr-temporal6-6-8`, `expr-temporal7-6-8`); the other 11 are tck-gfql port issues tracked separately as tck-gfql#36 / tck-gfql#38.
- **GFQL / Cypher reentry whole-row classifier alias-kind coverage (#1358)**: The bounded-reentry whole-row classifier (`_is_whole_row_with_item` / `_all_match_node_aliases` in `graphistry/compute/gfql/cypher/lowering.py`) only inspected `NodePattern.variable`, so prefix `WITH` carries of `RelationshipPattern.variable` (e.g. `WITH a, r`) or `MatchClause.pattern_aliases` (e.g. `MATCH path = ... WITH path, b`) silently fell into untested code paths in `_rewrite_multi_whole_row_prefix` when the #1341 single-MATCH flattener didn't admit the query. Added `_all_match_alias_kinds` covering all three alias kinds and a pre-flight check in `_compile_bounded_reentry_query` that raises a clean `GFQLValidationError` citing the unsupported alias kind ("relationship variable" or "named path alias") instead of producing undefined behavior. Added regression tests for both repros in `test_lowering.py`.

### Internal
- **GFQL row pipeline shrink (#1058)**: DRYed private row projection and correlated apply setup in `graphistry.compute.gfql.row.pipeline`, removed an unused row helper, and reused existing row alias lookup helpers while preserving pandas/cuDF row semantics.
- **CI HuggingFace offline cache hardening (#853)**: Keyed the feature-utils HuggingFace model cache from the generated AI lockfile's `sentence-transformers`, `transformers`, `tokenizers`, and `huggingface-hub` versions, removed the inert UMAP-lane HF warmup, and made `test-full-ai` verify cached sentence-transformer models under both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before running tests.
- **ReadTheDocs uv lockfile install (#1074)**: Switched the ReadTheDocs dependency install path from floating `pip install .[docs,pygraphviz]` to a committed Python 3.12 RTD lockfile installed with pinned uv and hashes, with a CI freshness guard and maintainer instructions for lockfile updates, preserving the existing Sphinx build commands, formats, and system packages.
- **GFQL compute predicate / AST helper cleanup (#1577)**: DRYed private numeric, string, and temporal predicate helper implementations plus duplicated directional edge-constructor delegation while preserving public predicate classes, factory names, JSON type names, edge aliases, and hop/chain behavior.
- **GFQL Cypher frontend/lowering residue cleanup (#1575)**: Shared private shortestPath alias-spec helpers across the AST normalizer and lowering paths, DRYed duplicated variable-length alias expression traversal, and consolidated Cypher CALL procedure payload construction while preserving compiler-plan surfaces, source-span diagnostics, and runtime behavior.
- **GFQL IR metadata compatibility seam restoration (#1566)**: Restored the `graphistry.compute.gfql.ir.metadata` helper module, `graphistry.compute.gfql.ir` re-exports, and metadata contract tests so the BoundIR/LogicalPlan type/nullability seam remains available for schema/type-system callers and downstream rewrite passes after the verifier cleanup.
- **GFQL IR metadata/verifier cleanup (#1566)**: Deleted the private `ir.metadata` helper layer, inlined the remaining nullable reads at IR consumers, and trimmed verifier helper scaffolding while preserving logical-plan verifier behavior.
- **GFQL physical dispatch cleanup (#1565)**: Removed stale runtime coupling to physical-executor wrapper classes so dispatch keys off `PhysicalPlan.route`, while preserving the existing planner wrapper/metadata contract and same-path, row-pipeline, procedure-call, and wavefront route behavior.
- **GFQL expression/source-text helper cleanup (#1564)**: Centralized duplicated Cypher ExprNode source-text rendering and expression-node rebuild helpers across the AST normalizer and lowering paths while preserving private compatibility aliases and runtime behavior.
- **GFQL row pipeline cleanup tranche (#1557)**: Centralized private node entity/label text formatting in the row entity-props helpers, removed unused private ordering-family helpers, and DRYed duplicated row-pipeline pandas/cuDF regression fixtures while preserving runtime behavior.
- **GFQL / Cypher binder fixture cleanup (#1556)**: Deleted stale strict-runtime binder baseline scaffolding, consolidated cross-kind rebind coverage, and trimmed binder expr-tree tests to public binder-consumption behavior while preserving validator/runtime parity coverage.
- **GFQL policy/validation/rollout test cleanup (#1554)**: Consolidated duplicated policy shortcut and rollout gate test scaffolding while preserving public deprecated validation APIs and GFQL policy/rollout behavior.
- **GFQL CALL/layout cleanup tranche (#1548)**: Removed duplicate CALL validation/layout tests, stale skipped ASTCall enrichment stubs, and a private Cypher CALL graph-execution wrapper while preserving CALL/layout runtime behavior.
- **GFQL planner/IR cleanup tranche (#1542)**: Removed the private logical-planner `IdGen` helper, removed a redundant physical-planner child traversal delegator, deleted duplicate physical-planner skeleton tests, and consolidated logical-planner unit coverage while preserving runtime planner behavior.
- **GFQL / Cypher parser/lowering test dedupe tranche (#1538)**: Consolidated duplicated parser/lowering regression tests, removed stale private parser-helper ownership assertions, and trimmed obsolete regression-comment boilerplate while preserving public parser/lowering behavior coverage.
- **Non-GFQL private helper cleanup (#1539)**: Removed stale private PlotterBase JSON/ETL1 dataset helpers tied to the retired api=1 `/etl` payload path, preserved older `_table_to_pandas()` and `_plot_dispatch_arrow()` helper entry points as deprecation-warning shims over the current dataframe/Arrow upload path, and pruned dead feature-utils scaffolding without changing active public APIs.
- **GFQL / Cypher result postprocess helper cleanup (#1522)**: Removed duplicated private result-projection formatting helpers from `graphistry.compute.gfql.cypher.result_postprocess` by reusing the shared row-entity formatting helpers already owned by `graphistry.compute.gfql.row.entity_props`. Runtime behavior is unchanged.
- **Compute hop safe merge cleanup (#1166)**: Replaced the remaining direct hop output/min-hop `.merge()` calls with the engine-aware `safe_merge()` helper, aligning edge hydration with the existing node hydration path without changing hop semantics.
- **GFQL / row pipeline cleanup tranche (#1523)**: Removed private RowPipelineMixin frame-op call-through wrappers now owned by `graphistry.compute.gfql.row.frame_ops` and DRYed duplicated row-pipeline/direct-Cypher tests while preserving the public row-pipeline dispatch surface and GFQL behavior.
- **GFQL / Cypher projection-planning delegate cleanup (#1514)**: Retired private projection-planning compatibility delegates from `graphistry.compute.gfql.cypher.lowering`, keeping validation and projection helper ownership in `graphistry.compute.gfql.cypher.projection_planning` while preserving compiler behavior.
- **GFQL / AST helper cleanup (#1512)**: Removed the unused internal `assert_record_match()` helper and consolidated duplicated edge `from_json()` field extraction and validation logic in `graphistry.compute.ast` behind private helpers while preserving the existing `ASTEdge*` classes, aliases, constructors, and wire payload behavior.
- **GFQL / Cypher parser helper cleanup (#1510)**: Removed stale parser transformer hooks and DRYed MATCH/OPTIONAL MATCH AST construction while preserving parser behavior.
- **GFQL / row pipeline frame-operation modularization (#1499)**: Split active row-table creation, empty-frame construction, `rows`, `drop_cols`, `skip`, `limit`, and `distinct` helpers from `graphistry/compute/gfql/row/pipeline.py` into the focused `graphistry.compute.gfql.row.frame_ops` module while preserving the existing `RowPipelineMixin` method surface and behavior.
- **GFQL / Cypher reentry helper ownership cleanup (#1497)**: Moved bounded-reentry helper ownership out of `graphistry.compute.gfql.cypher.lowering` into `graphistry.compute.gfql.cypher.reentry.lowering_support`, so reentry compile-time modules no longer import private helper families from the monolithic lowering module. No compiler/runtime semantics changed.
- **GFQL / Cypher reentry compiletime lowering-symbol shim deletion (#1471)**: Removed the broad `globals().update(vars(lowering))` compatibility shim from `graphistry.compute.gfql.cypher.reentry.compiletime`, replacing it with explicit imports for the bounded-reentry compile-time dependencies. Added a split-guard test so future reentry compiletime changes cannot reintroduce a broad lowering symbol-table rebind.
- **GFQL / Cypher final compat-executor reachability shrink (#1466)**: Audited `_execute_compiled_query_compat_non_union()` reachability after the native physical-dispatch lanes landed, planned projection-level `WITH/RETURN DISTINCT` shapes natively through the logical route, and removed the stale compat-executor wrapper so any residual unplanned feature-gap guard dispatches directly through the canonical chain executor. Updated runtime cutover coverage to assert physical planner route use without depending on the deleted private wrapper.
- **GFQL / Cypher simple top-level OPTIONAL MATCH native route (#1460)**: Simple input-free top-level `OPTIONAL MATCH` queries now receive a logical `PatternMatch(optional=True)` plan and dispatch through the native same-path physical route instead of the generic compatibility executor. Bounded optional reentry is handled by the optional-reentry native route. Added planner, lowering-route, and runtime cutover regressions covering matched rows and unmatched null-extension while asserting the compat executor is not used.
- **GFQL / Cypher optional reentry native route (#1461)**: Optional `MATCH` clauses after a bounded `WITH` reentry now carry a narrow logical/physical route marker and dispatch through native physical planning instead of the generic `_execute_compiled_query_compat_non_union()` fallback. Null-extension also preserves projected carried scalar values for unmatched prefix rows. Top-level `OPTIONAL MATCH` routing is handled separately. Added runtime cutover coverage for matched carried rows and null-extension behavior.
- **GFQL / Cypher D5 compat fallback inventory + CALL route cutover (#1455)**: Added a `procedure_call` physical route and `ProcedureCallExecutorWrapper` so direct-Cypher `CALL graphistry...` queries dispatch through native physical planning instead of the generic `_execute_compiled_query_compat_non_union()` fallback. The remaining generic fallback is documented and guarded for deferred `logical_plan is None` shapes such as top-level OPTIONAL MATCH, while stale `procedure_call` use inside the compat executor now fails fast.
- **GFQL / Cypher temporal comparison cuDF coverage audit (#1446)**: Broadened the Direct-Cypher temporal comparison regression matrix so the Temporal7 date, localtime, time, localdatetime, and datetime truth-table cases plus mixed literal/constructor equality cases run against actual cuDF-backed graph inputs as well as pandas. Added pandas/cuDF temporal row-property `ORDER BY` coverage for date, time, and datetime node columns to exercise the DataFrame-backed temporal ordering path directly.
- **GFQL / Cypher native physical-dispatch coverage before compat-executor deletion (#1441)**: Same-path and row-pipeline physical-plan wrappers now dispatch through a native chain execution helper instead of falling back to `_execute_compiled_query_compat_non_union`, while CALL-backed and unplanned compiled-query fallbacks remain on the compatibility wrapper. Added runtime cutover guards proving natural same-path and row-pipeline Cypher queries bypass the compat executor, narrowed cuDF row-projection host fallback to projected object-valued Cypher columns that RAPIDS cannot represent natively so RAPIDS 25.02 avoids its `DataFrame.to_pandas()` crash lane without bridging the full row table, and added a RAPIDS 25.02 radial-layout trig bridge for the GFQL layout-call GPU tests.
- **GFQL / Cypher temporal comparison TCK regressions (#1374, #1353)**: Added direct-Cypher regression coverage for the corrected Temporal7 date, localtime, time, localdatetime, and datetime comparison truth tables (`expr-temporal7-{1..5}-{1,2}`), plus mixed literal/constructor equality checks that exercise the shared temporal comparison normalization path.
- **GFQL / Cypher connected OPTIONAL MATCH row-join fallback cleanup (#1418)**: Removed the obsolete connected OPTIONAL MATCH branch from the generic compatibility executor in `graphistry/compute/gfql_unified.py`; connected OPTIONAL MATCH left-join plans now dispatch directly from the physical-plan path before generic same-path/row-pipeline compatibility dispatch. Added adversarial runtime cutover coverage in `graphistry/tests/compute/gfql/test_runtime_physical_cutover.py`.
- **GFQL / Cypher connected MATCH row-join fallback cleanup (#1418)**: Removed the obsolete connected MATCH row-join branch from the generic compatibility executor in `graphistry/compute/gfql_unified.py`; connected MATCH joins now route directly from the physical-plan executor before generic same-path/row-pipeline compatibility dispatch. Added runtime cutover coverage in `graphistry/tests/compute/gfql/test_runtime_physical_cutover.py`.
- **GFQL benchmark residual triage library for #880 coordinator lanes**: Added typed, reusable `benchmarks/gfql/benchmark_residual_triage.py` to parse `pyg-bench` run artifacts (`probe-results.json` plus optional run metadata), filter `backend=gfql` + `status=partial` lanes by issue reference, collapse to latest lane-key snapshots, bucket residuals by workaround semantics, and render split-ready markdown evidence for child-issue drafting. Added focused tests in `graphistry/tests/benchmarks/gfql/test_benchmark_residual_triage.py` covering load/filter/latest behavior, bucket classification, report rendering, and runs-dir resolution.
- **GFQL / Cypher joined bindings-row aggregation on non-active whole-row aliases (#1392, #880)**: Aggregate-stage lowering now permits bindings-row grouping on non-active whole-row aliases (for example `WITH DISTINCT tag, post WITH post, count(tag) ...`) when the alias is present in the current bindings scope. Group-key preservation now carries alias-prefixed columns for the actual grouped whole-row aliases instead of only the active alias, so downstream stages can still resolve grouped alias properties such as `post.id` and `post.creationDate`.
- **GFQL / Cypher strict binder post-WITH UNWIND traversal-order fix (#1371, #1357)**: In `FrontendBinder._bind_graph_sequence`, strict mode (`strict_name_resolution=True`) now binds graph clauses in source order (`span.start_pos`) so WITH-projected aliases are visible to later UNWIND/MATCH clauses in the same query text. This closes the strict false-positive where post-WITH UNWIND references were validated against pre-WITH scope. Added strict regression coverage and removed obsolete divergence baselines for this gap in binder/validator parity tests; #1420 later retired the separate loose traversal branch so this source-order path is now canonical.
- **GFQL / Cypher strict binder comprehension-local scope for quantifiers/list-comprehensions (#1371, #1357)**: Strict unresolved-identifier validation in `graphistry/compute/gfql/frontends/cypher/binder.py` now recognizes comprehension-local binders and scopes (`all/any/none/single(x IN ...)` and `[x IN ... WHERE ... | ...]`) so local tokens no longer false-positive as unresolved in strict mode. The unresolved-name guard remains in force outside the comprehension span (leaked locals still reject). Added strict binder coverage for admit+reject cases and updated strict parity/baseline tests to reflect P2 closure.
- **GFQL / Cypher strict binder CALL/YIELD scope through row-sequence binding (#1371, #1357)**: `FrontendBinder._bind_query` now binds `ast.call` before row-sequence projection binding for row-only query shapes, so `CALL ... YIELD ... RETURN ...` resolves yielded aliases under `strict_name_resolution=True` instead of false-positive unresolved-identifier failures. Added strict binder regressions for direct and aliased YIELD projections, and removed stale baseline notes/tests that still marked this lane as unresolved.
- **GFQL / Cypher strict binder namespaced builtin function handling (`duration.*`, `date.*`, `datetime.*`, `localdatetime.*`, `time.*`) (#1371, #1357)**: Strict unresolved-name/property-reference scans in `graphistry/compute/gfql/frontends/cypher/binder.py` now recognize known builtin namespace call forms (`namespace.fn(...)`) and avoid false unresolved-alias failures on temporal/duration builtin invocations under `strict_name_resolution=True`. Added strict binder coverage and updated validator/runtime strict parity + strict compile baseline fixtures to reflect P4 closure.
- **GFQL / Cypher bounded reentry ORDER BY+LIMIT preservation across MATCH re-entry (#1342)**: Relaxed bounded reentry admission so `WITH ... ORDER BY ... LIMIT <literal>` (no `SKIP`) can flow into trailing `MATCH` re-entry for multi-row prefixes when the trailing query does not define its own `ORDER BY`. Added regression coverage for single-column, multi-column, and DESC ordering with LIMIT across re-entry, plus cuDF parity, and narrowed fail-fast coverage to unsupported unbounded/SKIP shapes.
- **GFQL / Cypher OPTIONAL-prefix reentry no-match semantics (#1356)**: Fixed bounded reentry identity recovery when an `OPTIONAL MATCH ... WITH ... MATCH ...` prefix produces a null carry row with no whole-row projection metadata. Reentry now treats that lane as an empty seed set (no crash, empty result), while preserving matched-prefix behavior and guarding scope to secondary-carried-alias plans. Added regression coverage for both no-match and matched fixtures in `test_lowering.py`, plus a follow-up fix to keep empty-seed dtype/scope stable for downstream TCK lanes.
- **GFQL / Cypher bounded re-entry ORDER BY safety now resolves parameterized LIMIT values (#1349)**: Re-entry prefix order safety now treats `LIMIT $n` as bounded when query params provide an integer (including `0`), instead of rejecting all `ParameterRef` limits as unknown. This preserves fail-fast behavior for unresolved/non-integer limits and `SKIP`-based shapes while unlocking vectorized `MATCH ... WITH ... ORDER BY ... LIMIT $n MATCH ...` execution on both pandas and cuDF backends. Added targeted regressions for pandas + cuDF success and non-integer-parameter rejection in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher row-carrier plan metadata hardening (#989)**: Reentry compile-time planning now records per-alias property dependencies in `ReentryPlan.aliases[*].carried_properties` (including secondary-alias demotion and free-form bridge carries), so non-source row-carry semantics are explicit in the plan contract instead of inferred from rewrites. Added compile-contract coverage for both carried-alias and free-form lanes.
- **GFQL / Cypher bounded-reentry runtime extraction (#987 Step 3)**: Moved bounded-reentry data-frame execution helpers (`_compiled_query_reentry_state`, `_compiled_query_scalar_reentry_state`, `_compiled_query_freeform_reentry_state`, `_freeform_broadcast_row_to_nodes`, `_union_scalar_reentry_results`, `_apply_optional_reentry_null_fill`, `_aligned_reentry_rows`, `_reentry_carry_payload`, `_ordered_reentry_start_nodes`, `_reentry_validation_error`, the two suggestion constants) out of `graphistry/compute/gfql_unified.py` into a new `graphistry/compute/gfql/cypher/reentry/execution.py` module so the bounded-reentry contract assembled at compile time (`ReentryPlan`) and the matching data-frame stitching live next to each other. `_entity_projection_meta_entry` moved to `graphistry/compute/gfql/cypher/result_postprocess.py` next to `WholeRowProjectionMeta` since it is shared between the connected-OPTIONAL-MATCH and bounded-reentry paths. Pure-move refactor — no semantic change; `gfql_unified.py` shrinks by ~440 LOC and now re-exports the moved private names via aliased imports so existing tests reaching into `graphistry.compute.gfql_unified._compiled_query_reentry_state` continue to work.
- **GFQL / Cypher reentry compile-time module naming cleanup (#1333)**: Renamed bounded-reentry compile-time helper ownership from `graphistry/compute/gfql/cypher/reentry/runtime.py` to `graphistry/compute/gfql/cypher/reentry/compiletime.py` to better align naming with actual responsibilities (`execution.py` remains runtime data-frame behavior). Kept `reentry/runtime.py` as a compatibility re-export shim so existing imports continue to work unchanged.

### Documentation
- **GFQL component-labeling examples + README clarity (#1324)**: Added concise WCC/SCC labeling examples for `compute_cugraph`, `compute_igraph('clusters')`, and local Cypher `CALL graphistry.cugraph.*` write/row modes in GFQL docs, clarified that component IDs are partition labels (not stable semantic IDs), and tightened the main README GFQL intro sentence for readability.
- **GFQL / Cypher docs — variable-length boundary refresh (#973)**: Updated direct-Cypher capability docs (`docs/source/gfql/cypher.rst`, `docs/source/gfql/spec/cypher_mapping.md`) to reflect current support for connected variable-length patterns and bounded/exact variable-length `WHERE` pattern predicates, while preserving explicit fail-fast notes for remaining path/list-carrier and advanced row-shaping gaps.

### Changed
- **GFQL / Cypher binder cross-kind alias rebind guard (#1357)**: `FrontendBinder` now rejects MATCH patterns that re-use an existing alias as a different entity kind across all three cross-kind pairs (node↔edge, node↔scalar, edge↔scalar — where path aliases bind as `entity_kind="scalar"` alongside other scalar carriers like UNWIND output, WITH-projected expressions, and CALL/YIELD aliases) with `GFQLValidationError(E204)` carrying structured `existing_kind` / `new_kind` / `new_role` context. Previously a downstream re-entry compile-time guard at `graphistry/compute/gfql/cypher/reentry/compiletime.py` caught the scalar→node case for one specific WITH-prefix shape with the message `"Cypher MATCH after WITH scalar-only prefix aliases cannot be reused as node variables"`; the new binder-layer guard fires earlier and uniformly across all four cross-kind transitions on both MATCH and re-entry MATCH paths. Updated two `test_cycle_policy.py` assertions and one `test_lowering.py` assertion to match the new error surface — intent (reject scalar→node rebind) is preserved. Follow-up #1357/#1420 work completed the strict-name-resolution rollout and retired the loose binder compatibility branch.
- **GFQL / Cypher carried-endpoint rebind via single-MATCH flatten (#1341)**: Admit the LDBC SNB IC1 ``MATCH (p:Person {id: $pid}), (friend:Person) WHERE NOT p = friend WITH p, friend MATCH path = shortestPath((p)-[:KNOWS*1..3]-(friend)) RETURN ...`` rebind shape that the prior secondary-alias-rebind guard rejected. New AST pre-pass flattener at `graphistry/compute/gfql/cypher/reentry/flatten.py` detects the narrow case (single prefix MATCH + single pure-bare-carry WITH whose carry-set equals the prefix-bound aliases across nodes ∪ relationships ∪ paths + single trailing MATCH whose patterns reference only carried node aliases AND contain ≥1 RelationshipPattern) and merges patterns into a single MATCH so the existing two-endpoint `shortestPath` / shared-alias paths handle the rebind directly. The blanket reject at `lowering.py` is preserved as the residual fallback for shapes the flatten doesn't admit. Repurposed the prior failfast test into positive admit coverage and added IC1 + cuDF parity + admit-side oracle + AST-introspection lock-in (28 unit tests covering each disqualification branch).
- **GFQL / Cypher lowering — bounded/exact variable-length `WHERE` pattern predicates (#973)**: Removed the pre-normalization compiler gate that rejected bounded/exact variable-length `WHERE` pattern predicates and now lower these shapes through the existing WHERE-pattern rewrite and row-filter paths. Converted the old fail-fast test into positive execution coverage and added boolean-wrapper amplification (`OR`/`XOR`/`NOT`) for bounded variable-length `WHERE` predicates in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.
- **GFQL / Cypher relationship-row aggregates over MATCH multiplicity (#1343)**: First-stage aggregate lowering now routes multiplicity-sensitive aggregate shapes (`collect`, non-distinct `count`, `sum`/`avg`) on relationship-pattern MATCH queries through bindings-row execution so relationship-row multiplicity is preserved before aggregation. This unblocks `collect(...)` and grouped/global `count`/`sum` relationship-row shapes previously rejected by the stale "repeated MATCH rows" guard on the collapsed-alias path.
- **Compute / DataFrame join helper ownership moved from GFQL-local staging to shared dataframe namespace (#1380)**: Connected-join and same-path semijoin helper families now live under `graphistry/compute/dataframe/join.py` (exported via `graphistry/compute/dataframe/__init__.py`) rather than GFQL-local `dfops`/`same_path` helper ownership. Runtime call-sites were repointed (including `gfql_unified` connected join and same-path consumers) while preserving pandas/cuDF behavior.

### Tests
- **GFQL coverage uplift (#1534)**: Added audit-included temporal AST value coverage for timezone resolution, JSON dispatch, native datetime/date/time factory round-trips, datetime-text time parsing, and invalid temporal literal rejection. Raised the pandas GFQL coverage audit floor for `graphistry/compute/ast_temporal.py` from 44.12% to 95.09%.
- **GFQL / native chain reply-author row-shaping locks (#1412, #880)**: Added native GFQL `rows()` + explicit `rows(binding_ops=...)` regression coverage for the SNB IC8 `recent-replies` and IS7 `message-replies` reply-author projection shapes, locking the pygraphistry-side behavior needed to retire adapter-local reply-author joins in benchmark coverage.
- **GFQL / Cypher two-MATCH reentry varlen regression hardening (#1001)**: Strengthened reentry varlen acceptance assertions from shape-only checks to exact expected rows, and added forward/reverse split-vs-connected query equivalence regressions to guard against wrong-row drift in the `match5-25/26` query family.
- **GFQL / Cypher reentry ordered-top-k amplification (#1342, #880 partial)**: Added lowering regressions for MATCH-after-WITH re-entry with single-column and multi-column ordered top-k prefixes, carried-scalar top-k alignment, `LIMIT 0` empty-prefix behavior, `SKIP` failfast retention, plus cuDF parity coverage for the multi-row top-k lane.
- **GFQL / Cypher tag-cooccurrence join+aggregation cardinality amplification (#1396, #1415, #880 residual lane)**: Added focused IC6-shape regression coverage for `collect(distinct friend) -> UNWIND -> connected comma MATCH -> WITH tag.name, count(post)` with non-trivial grouped counts (`Alpha=2`, `Beta=1`) plus cuDF parity guard, so the residual tag-cooccurrence join-aggregation lane is pinned without adapter-side workaround assumptions.
- **DataFrame join helper branch coverage expansion + RAPIDS matrix validation (#1380)**: Expanded `graphistry/tests/compute/dataframe/test_join.py` to cover non-overwrite and empty-join schema branches (`joined_hidden_scalar_columns`, `joined_alias_columns`, `connected_inner_join_rows`), prefix/no-label projection behavior (`project_node_attrs`), inequality branch coverage (`ineq_eval_pairs`), and semijoin no-shared/delegation paths (`semijoin_eval_pairs`). Kept GFQL reentry/collision semantics in integration suites. Validated on DGX GPU for both RAPIDS `25.02` and `26.02` using `docker/test-rapids-official-local.sh` with GFQL profile + dataframe join tests.

### Internal
- **GFQL / Cypher row-carrier follow-through cleanup (#989, post-#1260 split)**: Retired transitional lowering-level bounded-reentry delegator shims (`_map_terminal_reentry_query`, `_drop_bare_alias_items_from_stage`, `_rewrite_multi_whole_row_prefix`, `_compile_bounded_reentry_query`) that only forwarded into `graphistry/compute/gfql/cypher/reentry/runtime.py`. Lowering now calls runtime-owned reentry helpers directly at use sites, and the split-guard tests were trimmed to keep only projection-planning delegator assertions.

## [0.55.1 - 2026-05-05]

### Tests
- **GFQL / Cypher reentry contract coverage (#989 follow-through)**: Added compile-shape regression `test_compile_cypher_records_freeform_reentry_plan_contract` to lock free-form intermediate MATCH as `ReentryPlan(free_form=True, scalar_only=False)` and prevent regressions to scalar-only fallback tagging.

### Infrastructure
- **CI / Spark lane speedup + gating**: `test-spark` now restores `~/.cache/uv` with a Spark-specific cache key, skips unconditional JDK apt install when `java` is already present, installs base `graphistry` (`-e .`) instead of the full `.[test]` extra for this smoke lane, and only runs when Spark-relevant paths (or infra/manual/scheduled triggers) are touched. This trims setup overhead and avoids running Spark setup on unrelated PRs.
- **CI / docs preflight guard**: Added `bin/check_docs_latex_unicode.sh` and a fast `docs-latex-unicode-guard` CI job to fail early on non-BMP Unicode in docs-fed text sources before the slower Dockerized `test-docs` LaTeX build.
- **Release process / deploy gate reminder**: Documented that tag-triggered PyPI publishes can pause in `waiting` on environment approval, and explicitly call out approving `Review deployments` for `pypi-release` before expecting the final PyPI job to complete.

### Added
- **GFQL/Cypher validate-only preflight API (#1320)**: Added `g.gfql_validate(...)` on `ComputeMixin` as a public no-execution validation entrypoint for GFQL chains/JSON-style queries, Let/DAG queries, and Cypher strings. The API returns structured diagnostics (`ok`, `diagnostics`, query/language metadata) instead of executing query operators. Cypher preflight runs parser+compiler checks and supports optional strict binder/schema mode (`strict=True`) using the bound graph schema catalog; chain/JSON preflight reuses existing `validate_chain_schema()` semantics (including `collect_all=True`), and Let/DAG preflight now includes best-effort schema checks for direct chain-like bindings.

### Changed
- **GFQL execution prevalidation semantics (#1320)**: `g.gfql(..., validate=True)` now runs local preflight validation before execution. `g.gfql_remote(..., validate=True)` now validates query payloads before implicit upload/network dispatch, so invalid queries fail locally prior to upload when possible. String query inputs are now treated consistently as Cypher during preflight (`g.gfql_validate("...")` and `g.gfql("...", validate=True)`), so users get Cypher parser/compiler diagnostics instead of shape-guessing type errors. `g.gfql_validate(...)` now raises structured GFQL exceptions on invalid queries (instead of returning `ok=False`), and collect-all mode surfaces full diagnostics via exception context for LM/retry workflows.

### Internal
- **GFQL / Cypher reentry follow-through cleanup (#989, post-#1260 extraction)**: In `graphistry/compute/gfql/cypher/reentry/runtime.py`, free-form intermediate MATCH plan construction now routes through the whole-row/free-form `ReentryPlan` contract instead of scalar-only fallback tagging. This makes the dedicated runtime `plan.free_form` lane reachable again and removes incidental scalar-only-path dependence for free-form reentry dispatch.
- **GFQL native types T4 — Arrow/type bridge contracts and coercion semantics (#1312, #1262, #1046)**: Added `graphistry/compute/gfql/ir/arrow_bridge.py` with stable schema-level interchange helpers `to_arrow()` and `from_arrow()` for `RowSchema` + schema-confidence metadata. The bridge records per-field logical-type metadata (`gfql.logical_type`) and confidence (`gfql.schema_confidence`) for deterministic round-trips, supports strict vs widening coercion (`coercion='strict'|'widen'`) at export/import boundaries, preserves scalar nullability exactly, and defines structural-type fallback behavior (`NodeRef`/`EdgeRef`/`PathType` as widened string bridge fields in widen mode). Added focused regression coverage in `graphistry/tests/compute/gfql/test_ir_arrow_bridge.py` for round-trip fidelity, nullability behavior, confidence handling, and strict/widen coercion boundaries.
- **GFQL type-system T3.b — nullable-helper consolidation follow-through (#1309, #1304 audit continuation under #1262/#1046)**: Migrated the two deferred nullable call sites onto `ir.metadata` helpers: binder UNION branch merge now derives `nullable` via `bound_variable_is_nullable(existing|branch_var)` instead of direct field OR, and lowering bound-nullable alias derivation now uses `bound_variable_is_nullable(variable)` instead of the inline `nullable or bool(null_extended_from)` flatten. Converted the IC1 differential scaffold trust-check to the same helper contract and added focused binder regression coverage (`RETURN null AS x UNION RETURN 2 AS x`) locking nullable-bit preservation across UNION branch merge.
- **GFQL type-system T5 — rollout-gate canary for the binder execution path (#1311, #1262 T5)**: Added `graphistry/compute/gfql/rollout.py` with a stable env-driven canary contract (`STRICT_SCHEMA_ENV = "GRAPHISTRY_GFQL_STRICT_SCHEMA"`, `env_bool`, `strict_schema_env_default`, `resolve_strict_schema`) and re-exports from `graphistry.compute.gfql`. Wired the resolver into `binder._strict_schema_mode` so the strict/permissive precedence on the *execution* compile path is now: explicit caller param > catalog `metadata["strict"]` > env-default > loose. Default off — existing loose-mode callers see no behavior change. **Scope:** T5 is complementary to the explicit preflight surface added in #1320/#1321 (`g.gfql_validate(strict=True)`, `g.gfql(..., validate=True)`) which hardcode strict at preflight. T5's env-gate exclusively governs the *execution* compile path's binder default — i.e., what `g.gfql(query)` does when the caller doesn't opt into explicit preflight. This is the canary surface for org-wide rollout. Added focused regression coverage in `graphistry/tests/compute/gfql/test_rollout.py` (39 tests across env-bool truth set, env-default semantics, resolver precedence, and package re-export pin) and `test_rollout_binder_integration.py` (12 tests pinning the binder seam: env-off preserves loose behavior, env-on rejects unknown labels, explicit/catalog tiers still win, valid queries continue to pass under env=strict). Both rollout test files were folded into the existing `test-gfql-core` and `test-pandas-compat-gfql` CI lanes — default-loose path runs with the main pytest invocation, env-on path runs as a follow-on `GRAPHISTRY_GFQL_STRICT_SCHEMA=true` step in the same job (so the canary receipt rides existing matrix axes rather than a new standalone lane). Operator-guidance doc landed at `docs/source/gfql/strict_mode.rst` (added to user-guide toctree) leading with the #1320 preflight API as the explicit operator entrypoint and framing the env-gate as the execution-path canary. With T3.b (#1309), T4 (#1313), and T5 (this) all landed, meta-issue #1262 close criteria are now satisfied.

## [0.55.0 - 2026-05-05]

### Infrastructure
- **CI / supply-chain hardening (PR-G 7a)**: Deprecated all third-party GitHub Actions in favor of native primitives — `styfle/cancel-workflow-action` replaced by a workflow-level `concurrency:` block in `ci-gpu.yml`; `dorny/paths-filter` replaced by an inline `git diff --name-only` shell filter in `ci.yml` (preserves all 6 outputs: `infra`, `python`, `gfql`, `cypher_frontend_ci`, `benchmarks`, `docs`). Configured zizmor `unpinned-uses` rule with a provider safelist (`actions/*`, `github/*`, `pypa/*` permitted on floating refs; everything else must hash-pin) and lowered the workflow-security gate from `--min-severity high` to `medium` so policy violations actually fail CI. Closed 5 PR-B follow-up gaps by adding `persist-credentials: false` to LFS / fetch-depth checkouts in `ci.yml` (×3), `codeql-analysis.yml`, and `publish-pypi.yml`. Also fixed a silent fail-closed bug surfaced during multi-wave review: when `git merge-base` cannot resolve (force-pushed branch with orphaned `event.before`, or PR base rebased mid-flight), the `changes` job now conservatively emits `true` for every output and downstream jobs run, instead of silently emitting `false` and skipping all gated tests (#1221, #1215, #1130).
- **CI / docs**: `test-readme` no longer runs `actions/setup-python` with an EOL Python 3.8 pin. The job now runs markdown lint directly via its Docker image, removing an unnecessary setup step and avoiding intermittent Python toolcache fetch timeouts.
- **CI / build lane**: `test-build` now runs on Python 3.14 with `build-py3.14.lock` instead of a fixed Python 3.8 runner, reducing reliance on EOL interpreter setup while preserving explicit 3.8 compatibility test lanes elsewhere in CI.
- **CI / token hardening**: CI workflows now declare explicit least-privilege default token scope (`permissions: contents: read`) and set `persist-credentials: false` on all checkout steps in `ci.yml` and `ci-gpu.yml`; GPU cancel job keeps a scoped `actions: write` override for run cancellation (#1130).
- **CI / GPU lockdown**: Temporarily disabled `ci-gpu.yml` GPU execution path (including `gpu_public` jobs) while runner availability and PR-D security hardening are addressed; attempted GPU-triggered runs now fail fast with re-enable guidance referencing issue #1130.
- **CI / release permissions**: `publish-pypi.yml` now explicitly declares `contents: read` alongside `id-token: write` for the publish job, keeping release workflow permissions least-privilege and resilient after repository default token permissions were tightened under issue #1130.
- **CI / workflow scanners**: Added `workflow-security.yml` with two checks for workflow changes: `actionlint` (with custom runner-label config in `.github/actionlint.yaml`) and a `zizmor` high-severity gate using `.github/zizmor.yml`. This phase explicitly defers SHA-pin enforcement (`unpinned-uses`) to future unscheduled work.
- **CI / trusted release gate**: `publish-pypi.yml` now binds publishing to environment `pypi-release` (for required-reviewer approvals), verifies pushed publish tags point to commits in `master` history, and restricts manual `workflow_dispatch` publishes to `master`.
- **Release docs / metadata**: Updated publish instructions to pull `master` in fast-forward-only mode (`git pull --ff-only origin master`), require a clean working tree before tagging (`git status --short` should be empty), push only the intended tag ref (`git push origin refs/tags/X.Y.Z`) instead of `--tags`/ambiguous ref pushes, clarified manual publish dispatch as maintainer-only recovery on `master`, added guidance to avoid rerunning already-published versions, and normalized legacy `pypi.python.org` links in `README.md` to `pypi.org`.
- **CI / OIDC context tightening**: `publish-pypi.yml` now verifies repository/workflow identity via `GITHUB_REPOSITORY` + `GITHUB_WORKFLOW_REF` and enforces release-tag format checks before publish. `DEVELOP.md` now documents the required PyPI Trusted Publisher binding (`repository`, `workflow`, `environment`, and trusted refs) so external OIDC policy stays aligned with workflow constraints.
- **CI / release provenance baseline**: `publish-pypi.yml` now adds CycloneDX SBOM generation (`evidence/sbom-cyclonedx.json`), GitHub build-provenance attestation for built distributions (`dist/*.whl`, `dist/*.tar.gz`), and release-evidence artifact upload for auditability. PyPI publish attestations are now enabled for both TestPyPI and PyPI paths.
- **CI / publish mode controls**: `publish-pypi.yml` now uses explicit `release_mode` (`evidence`, `test`, `release`) and split jobs (`build-evidence`, `publish-testpypi`, `publish-pypi`) so dry-run evidence generation does not trigger `pypi-release` approval prompts. Only `release` mode (or tag push) reaches PyPI publish and environment approval.
- **CI / TestPyPI dispatch compatibility**: `publish-pypi.yml` now creates a synthetic runner-local tag (`0.0.dev<run_id>`) for `workflow_dispatch` `release_mode=test` runs so Versioneer emits a PEP 440 uploadable version instead of a local-version (`+g<sha>`) string rejected by TestPyPI.
- **CI / release cooldown enforcement**: `publish-pypi.yml` now installs release build dependencies via pinned `uv` (`uv==0.11.3`) instead of direct pip dependency resolution (`-e .[build]`, `cyclonedx-bom`), so the existing `UV_EXCLUDE_NEWER: "6 days"` policy applies consistently on Python 3.8 publish runners.
- **CI / pip cooldown enforcement cleanup**: direct workflow dependency installs for GPU, DGL, Spark, UMAP/AI torch, and `workflow-security` `zizmor` now run through `uv pip` (with pinned `uv` bootstrap where needed) so `UV_EXCLUDE_NEWER: "6 days"` is the single enforced cooldown path; removed ambiguous `PIP_EXCLUDE_NEWER` workflow env usage.

### Added
- **CI / Polars**: Added `test-polars` CI job (Python 3.9–3.14) with a dedicated `test-polars` lockfile profile; `polars` is now a named `setup.py` extra so the test matrix installs and exercises `test_polars.py` on every PR (#1133).
- **Polars support**: `polars.DataFrame` and `polars.LazyFrame` now work in `plot()`, `materialize_nodes()`, `get_degrees()`, `get_indegrees()`, `get_outdegrees()`, and `hypergraph()`. Polars is an optional dependency — no behavior change when not installed. Upload path uses efficient Arrow conversion (`to_arrow()` with schema-metadata stripping and memoization); compute/hypergraph paths coerce to pandas at entry. `LazyFrame` is materialized via `.collect()` at each boundary. Adds `test_polars.py` with 17 tests; skips gracefully when polars is absent (#1133).
- **Validation / settings + axis contracts (#1240, #1239, #1251)**: Added reusable public validation/settings contracts under `graphistry.validate` including `URL_PARAM_NAMES`/`REACT_SETTING_NAMES` (and set forms), `normalize_url_params()` / `normalize_react_settings()`, axis URL default contracts (`RADIAL_AXIS_URL_DEFAULTS`, `LINEAR_AXIS_URL_DEFAULTS`, `axis_url_defaults()`), and typed axis payload contracts (`AxisRow`, `AxisBounds`, `RingContinuousAxis`, `RingCategoricalAxis`, plus centralized allowed-field constants). GFQL call validation now deep-validates `ring_continuous_layout.axis` and `ring_categorical_layout.axis` against these contracts, and `gfql_remote(persist=True)` metadata hydration now round-trips and reapplies URL params/defaults correctly after server persistence.
- **Dataset metadata hydration helper (#1252)**: Added `graphistry.from_dataset_id(...)` / `GraphistryClient.from_dataset_id(...)` to fetch `/api/v2/upload/datasets/{id}`, tolerate wrapped (`{"data": ...}`) and flat payloads, normalize legacy encoding payloads (`node_encodings`/`edge_encodings`) into plottable metadata shape, hydrate bindings/encodings/style/url params, and materialize a dataset URL on the returned `Plottable`.
- **Declarative React-shape encoding dispatcher (#1250)**: Added `g.apply_encodings(...)` / `graphistry.apply_encodings(...)` for React-style payloads (`encodePointColor`, `encodeEdgeColor`, `encodePointSize`, `encodePointIcons`, `encodeEdgeIcons`, `encodeAxis`) with strict/autofix validation behavior and parity-focused tests.

### Fixed
- **GFQL / cuDF row pipeline**: Hardened list+scalar concat execution to preserve cuDF-native row tables across fallback paths (including null/empty list edges), and reduced eager host-conversion in row-type probes to avoid unnecessary `to_pandas()` pressure in constrained GPU-sharing environments. This keeps `vals + score` / `score + vals` semantics aligned across pandas and cuDF while improving vectorized-path stability on RAPIDS 25.02/26.02.

### Tests
- **GFQL / cuDF coverage amplification**: Added explicit cuDF regression coverage for list+scalar concat in both splat directions with empty-list and null-scalar/null-list edge cases. Validated amplified GFQL suites on `dgx-spark`: RAPIDS 26.02 full row-pipeline + parser/lowering passes; RAPIDS 25.02 passes targeted amplified subsets, with remaining instability isolated to existing cuDF host-conversion paths under shared-memory pressure.

### Internal
- **GFQL / Cypher compiler (S4 compatibility-surface reduction, #1305, #1260)**: Reduced `CompiledCypher*` accessor surface by retiring planner/metadata compatibility properties no longer required by active compile/runtime paths. Removed from `CompiledCypherQuery`: `connected_optional_match`, `connected_match_join`, `query_graph`, `scope_stack`, and derived `logical_plan_route`; removed derived `logical_plan_route` from `CompiledGraphBinding` and `CompiledCypherGraphQuery`. Runtime/compile wiring now reads canonical nested payloads from `execution_extras` directly (`gfql_unified.py`, `reentry/runtime.py`, lowering attach helpers), preserving execution behavior while shrinking deprecated inspection surface. Added regression locking in `test_lowering.py` via canonical `execution_extras`/logical-plan assertions.
- **GFQL type-system T2 — binder strict schema validation hooks (#1302)**: Added binder-time strict schema checks in `graphistry/compute/gfql/frontends/cypher/binder.py` against `GraphSchemaCatalog` for missing labels/properties across MATCH patterns, WHERE predicates (including expr-tree path), and projection/unwind/call expressions. Diagnostics are deterministic and include alias/field/stage context with sorted availability hints. Strict-mode gating preserves loose/default behavior unless strict mode is enabled. Added focused regression coverage in `graphistry/tests/compute/gfql/cypher/test_binder.py` for admitted/rejected shapes and targeted strict/loose intersection amplification.
- **GFQL type-system T3 — type/nullability metadata propagation contract (#1300, #1262 T3)**: Added `graphistry/compute/gfql/ir/metadata.py` exposing a stable helper surface over the type/nullability metadata BoundIR/LogicalPlan already carry: `is_nullable`, `with_nullable`, `widen_to_nullable`, `column_logical_type`, `column_is_nullable`, `merge_types`, `bound_variable_type`, and `bound_variable_is_nullable`. Helpers treat `ScalarType.nullable` as the only LogicalType-level nullable bit today and pass `NodeRef`/`EdgeRef`/`PathType`/`ListType` through unchanged; `bound_variable_is_nullable` exposes `BoundVariable.nullable` directly so downstream passes asking "is this variable nullable" do not silently drop the binder-recorded bit on structural variables. Extended the LogicalPlan verifier with invariant 6 — type propagation continuity across a unary op's `input` slot: shared columns must agree on kind family (NodeRef/EdgeRef/ScalarType/PathType/ListType) and ScalarType nullability is monotone-widening, with `Filter`, `PatternMatch`, `SemiApply`, and `AntiSemiApply` carved out as the row-dropping operators that may legitimately narrow nullability. The check is skipped when either schema has no columns so existing planner-emitted plans that initialise `output_schema=RowSchema()` remain valid until the schema-population slice lands. Added `graphistry/tests/compute/gfql/test_ir_type_propagation.py` (61 tests) covering helper contract surface and verifier pass/fail cases (kind mismatch, nullability narrowing on Project/Distinct/OrderBy, Filter and PatternMatch carve-outs, dropped columns, structural-type pass-through, list-element recursion) plus a seam-amplification class that pins T3's helper + invariant-6 contract on plan shapes the post-#1303 lowering split (`projection_planning.py`, `cypher/reentry/runtime.py`) emits. Consolidated three existing inline patterns onto helpers within `ir/`: the two `isinstance(typ, ScalarType) and ... typ.nullable` checks inside `verifier.py` (invariant 5 optional-arm at line 188; invariant 6 nullability monotonicity at line 295) onto `is_nullable`, plus the BoundVariable read in `query_graph.py:224` (`is_required = not var.nullable`) onto `bound_variable_is_nullable`. Two further consolidation candidates were intentionally deferred under the cross-PR coordination guardrails: `binder.py:151` (`existing.nullable or branch_var.nullable` union merge — sits in T2 #1302's hot zone) and `lowering.py:571` (`variable.nullable or bool(variable.null_extended_from)` — sits in #1295/#1303's recently-landed hot zone). Those will land in a follow-on **T3.b slice tracked in #1309** once the sibling lanes settle. Helpers are also re-exported from `graphistry.compute.gfql.ir` (the package surface) so external adopters don't need to reach into the deep `ir.metadata` path.
- **GFQL / Cypher reentry cleanup — retire dual-contract scalar reentry extras path (#1294, #1260 S1)**: Removed legacy `CompiledCypherExecutionExtras` fields `scalar_reentry_alias` / `scalar_reentry_columns` and migrated runtime/compile branching to the canonical `reentry_plan` contract (`scalar_only` / `free_form` / whole-row lanes). `_execute_compiled_query_with_reentry` now dispatches scalar-only reentry via `ReentryPlan.scalar_columns` instead of legacy fields, lowering extras threading no longer forwards legacy scalars through `_execution_extras_with`, and whole-row reentry dispatch now requires a `ReentryPlan` (retiring the residual projection-contract fallback path). Added regression coverage locking absence of legacy extras fields and scalar-column plan parity in `test_lowering.py`.
- **GFQL type-system T1 — schema contract + canonical naming alignment (#1296)**: Stabilized the IR schema-catalog contract in `graphistry/compute/gfql/ir/compilation.py` while preserving existing `GFQLSchema is GraphSchemaCatalog` alias behavior. Added `GraphSchemaCatalog.from_schema_parts(...)` to normalize iterable inputs into frozen-set contract shapes and copy metadata safely, plus canonical accessors (`node_id`, `edge_source`, `edge_destination`) and membership helpers (`has_node_column`, `has_edge_column`) to provide a minimal stable interface for follow-on T2/T3 schema-validation and type-propagation lanes under #1262/#1259. Added focused contract tests in `graphistry/tests/compute/gfql/test_ir_compilation.py` and validated adjacent M0B conformance coverage remains green.
- **GFQL / Cypher monolith shrinkdown — extract bounded reentry glue from `lowering.py` (#1295, #1260 S2)**: Moved 17 reentry helpers (16 public + carry-internal `_literal_limit_value`, ~470 LOC) out of `graphistry/compute/gfql/cypher/lowering.py` into a new `graphistry/compute/gfql/cypher/reentry/` subpackage with four focused modules: `naming.py` (hidden-column / carry-name conventions), `scope.py` (alias-scope traversal for hidden reentry references), `carry.py` (prefix carry-column / order helpers, including `_literal_limit_value`), and `rewrite.py` (AST/query rewriters that retarget reentry expressions onto carried columns). The 16 public names are re-exported from `lowering.py` so existing callers (`from graphistry.compute.gfql.cypher.lowering import _reentry_hidden_column_name`) keep working unchanged; the moved modules pull lowering-internal helpers (`_unsupported`, `_unsupported_at_span`, `_render_expr_node`, `_rebuild_expr_node`, `_rewrite_expr_identifiers`, `_rewrite_where_clause_and_resync`, `_first_pattern_node_alias`, `_parse_row_expr`) lazily inside function bodies to avoid circular imports at module load. `carry.py` declares `ResultProjectionPlan` as a `TYPE_CHECKING`-only forward-string back-edge to lowering. `lowering.py` shrinks materially after the move; the orchestrator (`_compile_bounded_reentry_query`) and `_map_terminal_reentry_query` stay in `lowering.py` and are deferred to a follow-on monolith-shrinkdown slice.
- **GFQL / Cypher lowering — admit free-form carried-property bridge WHERE+RETURN shape (#1275)**: Extended bounded reentry lowering to admit intermediate free-form `MATCH` shapes where trailing predicates and projections reference carried alias properties (for example `WHERE c.id = x.id` and `RETURN x.id`) by wiring `_rewrite_multi_whole_row_prefix(...)` into the free-form lane, threading `multi_alias_carries` into `ReentryPlan.aliases` + expression rewrite, and removing the prior conservative carried-property failfast. Also fixed a structured-WHERE rewrite gap in reentry lowering: reentry `WhereClause` instances with `expr_tree=None` (predicate-only form) are now rewritten/resynced instead of being skipped/dropped, preventing runtime alias-validation failures on carried-property bridge predicates. Added/converted positive execution tests (including cuDF parity guard) for bridge WHERE and bridge WHERE+RETURN variants in `test_lowering.py` while preserving adjacent #1263 regression coverage.
- **GFQL / Cypher runtime — admit multi-prefix-row free-form intermediate MATCH (#1285, #1263 follow-up)**: Replaces the single-row-only failfast in `_compiled_query_freeform_reentry_state` with a per-row union loop in `_execute_compiled_query_with_reentry`, mirroring the scalar-only multi-row pattern from #1047. New `_freeform_broadcast_row_to_nodes` helper isolates per-row hidden-carry broadcast; the outer dispatcher loops over prefix rows, runs the suffix as a global MATCH per row, and unions results via the existing engine-polymorphic `_union_scalar_reentry_results`. Cartesian semantics fall out naturally — every prefix row × every trailing-MATCH row appears in the output. Out of scope (preserved as failfasts): `optional_reentry + multi-row free-form` (mirrors the scalar lane's existing guard at #1047). Carried-property bridge admit landed separately in #1287. Tests retargeted: the prior `..._failfast_rejects_..._on_multi_row_prefix` is now `..._executes_..._on_multi_row_prefix` (positive). New tests cover Cartesian semantics (2 prefix × 2 trailing = 4 rows; each pair appears twice), cuDF parity, and the OPTIONAL MATCH failfast. Pre-existing simple (single-row) free-form admits from #1279 stay green.
- **GFQL / Cypher lowering — #1273 acceptance slice admits scalar multi-alias WITH shape A**: First-stage bounded reentry now admits `MATCH ... WITH a.id AS a_id, b.id AS b_id ... MATCH ...` multi-alias scalar/property projections (shape A) while keeping whole-row multi-alias carries (for example `WITH a, b`) outside the admitted lane. The admission gate is explicitly narrowed to scalar/property projections only, so known TCK xfail contract shape `WITH n, x` (`with-where3-3`) remains fail-fast and unchanged pending broader #1273 closure. Tests were converted to positive execution assertions for the admitted shape, and remaining rejected shapes continue to point explicitly to #1273.
- **GFQL / Cypher row-boolean residual topology amplification (#1219)**: Added two cross-alias OR stress tests in `test_lowering.py` for previously under-covered topology classes: (1) disconnected cartesian-product `MATCH (a:A), (b:B)` with `WHERE a.x = 1 OR b.y = 2`, and (2) broader two-hop fanout `MATCH (a:A)-->(m:M)-->(b:B)` with the same disjunction. These lock row-wise disjunction semantics over wider join/fanout shapes on top of the existing 2x2 cross-alias union test.
- **GFQL / predicate pushdown safety — quote-aware null-safety classification hardening (#1219)**: `is_null_rejecting()` now masks quoted/backticked segments before scanning for boolean operators and null-safe forms (`IS NULL`, `IS NOT NULL`, `COALESCE`, `NULLIF`). This closes false-safe substring matches like `n.note = 'x is not null'` / `'coalesce('` that could previously be misclassified as null-safe for OPTIONAL-arm pushdown. Added regression coverage in `test_pushdown_safety.py` and pass-level lock in `test_predicate_pushdown_pass.py` to ensure quoted literal text does not trigger null-safe admission.
- **GFQL / predicate pushdown safety — whitespace-delimited OR detection hardening (#1219)**: `is_null_rejecting()` now normalizes unquoted whitespace and detects boolean operators via word-boundary token checks instead of exact `" and "` / `" or "` substrings. This closes a false-safe gap where newline/tab-delimited OR compounds (for example `n IS NULL\\nOR n.age > 5`) could bypass conservative OR handling and be incorrectly pushed into OPTIONAL arms. Added regression coverage in `test_pushdown_safety.py` plus a pass-level lock in `test_predicate_pushdown_pass.py`.
- **GFQL / Cypher row-boolean policy-boundary audit (#1219)**: Added `test_where_row_boolean_policy_boundary.py` to explicitly map current support boundaries before any future reject-subset gate decision. The suite locks parser/binder routing for structured/tree/mixed WHERE forms (including OR/NOT/XOR and pattern+row mixes), locks top-down execution behavior for representative row-boolean shapes, and pins known unsupported pattern-existence forms (`NOT((pattern))`, `exists { ... }`) as explicit validation failures. This codifies current behavior and provides a concrete evidence baseline for future policy changes.
- **GFQL / Cypher parser — quote-aware pattern-existence pre-scan (#1219)**: `_check_unsupported_syntax_patterns()` now masks quoted/backticked segments before applying pattern-existence unsupported-form regex checks. This fixes false-positive `unsupported-cypher-query` rejections for row-only queries where string literals contain lexemes like `exists { ... }` or `not((a)-...)`, while preserving rejection of true pattern-existence expressions. Added parser and policy-boundary regression coverage.
- **GFQL / Cypher parser — comment-aware pattern-existence pre-scan amplification (#1219)**: Extended the same unsupported-form pre-scan shielding to ignore `// ...` and `/* ... */` comment bodies. This closes an additional false-positive lane where comment text containing `exists { ... }` / `not((a)-...)` previously triggered `unsupported-cypher-query` on otherwise valid queries. Added narrow parser lexical-matrix tests and broader row-boolean boundary runtime checks with comment-wrapped lexemes.
- **GFQL / Cypher parser — pattern-existence boundary tightening for labeled/typed `NOT((pattern))` forms (#1219)**: Broadened unsupported-form pre-scan detection for `NOT((...))` pattern existence so node labels/properties and richer relationship tokens no longer bypass fail-fast rejection (for example `NOT((n:Admin)-[:R]->())`). Added parser, lowering, and policy-boundary regressions locking rejection of these variants while preserving valid row-boolean execution paths.
- **GFQL / Cypher lowering — runtime carry forwarding through trailing WITH narrowing (#1272, #999 partial)**: Fixed the IC5 sub-A collect/unwind reentry residual where carried scalars could degrade to null/NaN across a trailing `WITH d, bid` stage. Hidden reentry-property references are now excluded from one-source alias guard counting while still forcing bindings-row lowering for stages that reference hidden carry columns, so carry fields survive narrowing and remain readable in subsequent RETURN/ORDER BY. Updated regression coverage by converting the two #1272 failfast locks into positive execution assertions for both plain trailing-WITH and ORDER BY/LIMIT variants.
- **GFQL / Cypher lowering — residual multi-source row-projection failfast now points to issue #1273**: Remaining unsupported one-MATCH-source row-lowering residuals now raise the same `E108` boundary with an explicit `#1273` pointer so rejected shapes route directly to the tracked closure lane. Updated rejection-lock tests assert the issue pointer is present in emitted validation messages.
- **GFQL / Cypher lowering — admit simple free-form intermediate MATCH after WITH (#1263, #999 partial)**: Lifts the failfast at the `first_alias != reentry_alias` site of `_compile_bounded_reentry_query` for the conservative case: trailing MATCH whose first alias is not in the prefix WITH's carried whole-row set, with no carried-alias property reference in the trailing scope. Three-commit shape: (1) IR adds `ReentryPlan.free_form: bool = False` to mark the new shape (`graphistry/compute/gfql/cypher/reentry_plan.py`); (2) compile sets `reentry_alias = first_alias` and `non_source_alias_names = whole_row_carried` so the trailing MATCH compiles as a regular fresh-MATCH and the existing property-carry rewriter (#1248) treats every carried alias as non-source (`graphistry/compute/gfql/cypher/lowering.py`); (3) runtime adds `_compiled_query_freeform_reentry_state` in `gfql_unified.py` — for a single-prefix-row, broadcasts every `__cypher_reentry_*` hidden column from the prefix row onto every base node and returns `start_nodes=None` so the suffix runs as a global MATCH and the row pipeline carries the broadcast values through whichever alias the trailing MATCH binds. Compile-time failfast added for the carried-property variant (`free_form` AND any `<carried>.<prop>` ref in trailing scope) — the existing demote (#1071) + property-rewriter (#1248) chain double-wraps the synthesized hidden column name, and closing that case requires a rewrite-order refactor that lands as its own focused follow-up. Runtime-side multi-prefix-row free-form raises a clear failfast pointing at the multi-row follow-up. Tests retargeted: `..._failfast_rejects_intermediate_reentry_match_with_no_carried_source` → `..._failfast_rejects_intermediate_reentry_match_with_carried_property_in_trailing_where`; `..._failfast_rejects_simple_freeform_intermediate_reentry_match` → `..._executes_simple_freeform_intermediate_reentry_match`. Adds cuDF parity test and a multi-prefix-row failfast regression. Three TCK xfail-contract scenarios admitted to `MATCHES_EXPECTED` (`expr-typeconversion2-7`, `expr-typeconversion3-5`, `expr-typeconversion4-7` — all `MATCH (single-node) WITH * MATCH (n) RETURN <typecast>(n.<prop>)` shape) (#1263, #999, #989).
- **GFQL / Cypher lowering — scoped failfast for free-form intermediate MATCH after WITH (#1263, #999 partial)**: Replaced the generic `Cypher MATCH after WITH currently requires the trailing MATCH to start from the same carried node alias` error at `_compile_bounded_reentry_query` with a `#1263`-scoped error that names the LDBC SNB IC3 endpoint and points at the design doc for closure. The scoped wording calls out the gap shape — trailing MATCH whose first alias is not in the prefix `WITH`'s carried set — so users can identify whether their query falls into this lane or one of the other open IC3 sub-cases. Two regression-lock tests added: `test_string_cypher_failfast_rejects_simple_freeform_intermediate_reentry_match` (single-alias prefix; isolates the free-form gate without the slice 4.3a/b bare-ref interaction) and `test_string_cypher_failfast_rejects_intermediate_reentry_match_with_no_carried_source` retargeted from the prior #1256 wording. Closure of the gap (admitting the trailing MATCH as a fresh seed pattern that cross-joins with the carried row table at runtime) remains follow-up under #1263.
- **AI skills / workflow**: Updated the local `plan` skill template to reflect current pygraphistry validation commands, and added a new `review` skill with phased multi-wave review flow, adversarial confirmation, and explicit GPU validation guidance (local `docker/test-gpu-local.sh` or `dgx-spark` evidence path when no local GPU is available).
- **GFQL / Cypher lowering — secondary whole-row carry survives chained reentry boundaries (#1256, #989, #999 partial)**: Two extensions to `_demote_secondary_whole_row_aliases` in `graphistry/compute/gfql/cypher/lowering.py` so multi-alias carry survives a chained reentry compile. (1) **Forwarding-item drop** — bare-identifier projection items in downstream `WITH` stages whose name is a secondary alias are dropped at compile time (the same intent as slice 4.3c, integrated into the post-#1071 active rewrite path). Without this, a forwarding pattern like `WITH a, x, friend, ...` triggered the bare-ref failfast even though the carry already lives as a hidden scalar on the reentry-source's row table. (2) **Hidden-column forwarding** — for every `(secondary_alias, prop)` reference collected from trailing clauses, the synthesized `__cypher_reentry_<S>_<X>__` hidden alias is appended as a bare passthrough item to every downstream `WITH` stage so each recursive `compile_cypher_query` call sees it as a scalar carry. Without this, the inner compile failed alias resolution on the rewritten hidden identifier (`Unknown Cypher alias '__cypher_reentry_x_id__' in RETURN clause`) once a chained reentry stage narrowed the projection scope. New positive tests cover (a) chained reentry where the trailing MATCH continues to use the same primary alias, (b) chained reentry where the source is rebound between boundaries (`MATCH (a)-[:R]->(friend) ... MATCH (friend)-[:S]->(c)`), and (c) multi-alias `DISTINCT` forwarding through a single boundary. New failfast test pins the remaining gap — a trailing MATCH that does not start from any carried alias (free-form intermediate MATCH, the LDBC SNB IC3 prefix shape) — to the existing scoped error so future closure of that lane is regression-locked. Closes the chained-reentry portion of #1256; the free-form intermediate-MATCH case remains open follow-up.
- **GFQL / Cypher lowering — `ReentryPlan` IR + multi-alias whole-row carry slices (#989, #1026, #999 partial)**: Introduces an explicit `ReentryPlan` + `CarriedAlias` dataclass (`graphistry/compute/gfql/cypher/reentry_plan.py`) as the compile-time contract between a prefix `WITH` stage and the trailing `MATCH`, replacing the implicit handshake spread across tuple returns from `_bounded_reentry_carry_columns`, the `scalar_reentry_alias` / `scalar_reentry_columns` fields on `CompiledCypherExecutionExtras`, and runtime contract re-extraction in `_compiled_query_reentry_contract` (#987 step 1). Plan is exposed via `compiled_query.reentry_plan` and threaded through `_map_terminal_reentry_query` + `_attach_graph_context`. Builds three additional admit slices on top of the #1071 lift: (1) **slice 4.3a** lifts the residual single-whole-row gate at the compile site (`lowering.py`) and runtime contract (`gfql_unified.py`) so `WITH a, x` admits whenever only the trailing-MATCH source alias is referenced downstream — `ReentryPlan.aliases` records all whole-row aliases as `CarriedAlias` entries, with downstream non-source-alias bare references emitting an actionable failfast pointing to #989; (2) **slice 4.3b** adds a compile-time prefix rewrite (`_rewrite_multi_whole_row_prefix`) that turns `WITH a, x` into `WITH a, x.id AS __carry_x__id__` for every property of `x` referenced in trailing clauses (collected via `_collect_non_source_alias_property_refs` walking `WhereClause.expr_tree`, `ProjectionStage.where`, `ReturnItem`, `OrderItem`), and AST-rewrites trailing `<non_source>.<prop>` references to property access on the reentry-alias's hidden column — closes the multi-alias case of `#1026` regression-lock (`MATCH (a), (x) WITH a, x OPTIONAL MATCH (a)-->(b) RETURN x.id, b.id` now executes correctly with left-outer-join semantics, previously raised `GFQLValidationError`); (3) **slice 4.3c** drops bare carried-alias items at compile time when downstream `WITH a, x, y, collect(...)` re-projects them, so the bare-ref failfast does not false-positive on forwarding patterns. Adds 5 new positive/structure tests + 2 failfast-scope tests, retargets one prior `pytest.raises(GFQLValidationError)` regression-lock to a positive row assertion, and adds the multi-alias `OPTIONAL MATCH` regression test `test_issue_1026_multi_alias_with_optional_match_carries_secondary_property`. Cross-reentry-boundary forwarding (carry survival across `MATCH (a)-[:KNOWS]-(friend)`, the IC3 LDBC SNB shape) remains follow-up work tracked as #999 / slice 4.3d (#989, #1026, #999, #987).
- **GFQL / Cypher lowering — multi-alias carry through `WITH` before MATCH re-entry (#1071)**: Lifted the residual constraint that `WITH` before a MATCH re-entry must project exactly one whole-row node alias. The local Cypher compiler now accepts patterns like `MATCH (p)-[:KNOWS]-(friend) WITH p, friend MATCH (friend)-[:IS_LOCATED_IN]->(c) RETURN p.firstName, c.name` (LDBC SNB IC1 shape). Implementation pre-rewrites the prefix `WITH` in `_compile_bounded_reentry_query`: the trailing-MATCH primary alias is preserved as the sole whole-row carry; secondary aliases are demoted to scalar property carries via synthesized `S.X AS __cypher_reentry_<S>_<X>__` items, with downstream `S.X` references rewritten to bare hidden identifiers that compose with the existing single-alias carried-scalar machinery (#1047 / #1068). Returning a secondary alias as a whole-row entity (`RETURN s`) and re-binding a secondary alias as a node variable in the trailing MATCH remain unsupported with precise errors. Two prior failfast tests retargeted to assert correct multi-alias behavior; new tests cover IC1-shape, three-alias carry, and the secondary-whole-row-return rejection.
- **GFQL / Cypher lowering — OR/XOR around pattern predicates (#1236, #1031 slice 4)**: Local Cypher `WHERE` boolean trees now support disjunctive compositions that mix pattern-existence leaves with row expressions (for example `pattern OR expr`, `pattern XOR expr`). Lowering rewrites pattern leaves into correlated `semi_apply_mark(...)` row pre-filters that emit boolean marker columns, then evaluates the rewritten boolean expression in `where_rows(...)`. Added direct row-pipeline/safelist coverage for `semi_apply_mark` plus parser/lowering runtime regression tests for OR/XOR-around-pattern query shapes.
- **GFQL / Cypher residual cleanup (#1226, #1219)**: Retired `graphistry/compute/gfql/expr_split.py` by migrating top-level-AND splitting to `predicate_pushdown.py` (`split_top_level_and_conjuncts`) and updating conformance coverage to target the migrated splitter path. Fixed primitive literal atom fallback text in parser boolean-tree construction so `atom_text` now preserves Cypher keyword casing (`true`/`false`/`null`) instead of Python casing. Tightened optional-arm pushdown null-safety guardrails by treating OR-compound predicates as conservatively null-rejecting (disjunct-level alias analysis remains out of scope), with new regression coverage for mixed-alias `IS NOT NULL OR ...` forms.
- **GFQL / Cypher lowering — WHERE NOT (pattern) anti-semi execution (#1031 slice 2 phase 2b)**: `WherePatternPredicate(negated=True)` no longer hard-fails at lowering. The compiler now emits a row pre-filter call (`anti_semi_apply`) for negated pattern predicates in both general MATCH lowering and connected OPTIONAL MATCH clause lowering, with per-predicate validation (must include relationship, must not introduce new aliases, must share bound aliases). Added `anti_semi_apply` row-pipeline runtime operation + call-safelist entry, plus row-table base-graph context preservation needed for correlated bindings execution. New tests cover compile shape (`row_pre_filters` emission) and runtime behavior for `MATCH (n) WHERE NOT (n)-[:R]->()` (including mixed row-expression + NOT-pattern filtering), plus connected `OPTIONAL MATCH ... WHERE NOT (pattern)` filtering/null-fill semantics. Full `cypher/test_lowering.py` suite passes (756 passed / 66 skipped) and touched-module mypy is clean.
- **GFQL / Cypher parser + ast_normalizer — NOT-pattern AST plumbing (#1031 slice 2 phase 2a)**: Top-level `WHERE NOT (pattern)` shapes (e.g. `WHERE NOT (n)-[:R]->()`) now parse cleanly and lift into `WhereClause.predicates` as `WherePatternPredicate(negated=True)` entries instead of tripping the legacy "cannot yet be mixed with generic row expressions" E108.  `_split_top_level_and_pattern_leaves` adds a top-level `not(pattern_atom)` case that strips the NOT and emits the inner pattern as a negated leaf; `_build_where_with_pattern_lift` accepts both positive and negated leaves and emits one `WherePatternPredicate` per leaf with the matching `negated` flag.  ast_normalizer's `_rewrite_where_pattern_predicates_to_matches` partitions into positive (rewrites to appended MatchClause as before) vs negated (passes through to lowering).  Lowering now distinguishes the two cases: positive `WherePatternPredicate` still raises "must be rewritten before lowering" (defensive — slice 3 already rewrites all positives in ast_normalizer); negated raises a scoped "Cypher WHERE NOT (pattern) anti-semi-join lowering is not yet supported" pointing the way for the engine half (path-C row-pipeline anti-join, see `plans/1031-slices-2-3-4/findings/slice-2-scope.md`).  Adds `test_parse_lifts_top_level_not_pattern_to_negated_predicate` and `test_string_cypher_failfast_rejects_negated_pattern_until_slice2_lowering`.  De Morgan compositions, OR-around-pattern, and double-NOT remain rejected at the lift step (slice 4 / future).  Phase 2a only — runtime (anti-semi-join lowering) ships in a follow-up sub-PR (#1031).
- **GFQL / Cypher row-boolean residual matrix + guardrails (#1219 hardening)**: Locks compositional row-boolean WHERE shapes that #1217's Earley swap admitted but its initial test surface didn't cover.  Adds 11 native tests: nullable NOT/OR over a 4-row 3VL fixture (`NULL OR T = T`); N-ary OR (3 branches) + duplicate-branch companion isolating rightmost-drop associativity bugs; De Morgan equivalences (`NOT (A OR B)` ≡ `NOT A AND NOT B`; `NOT (A AND B)` ≡ `NOT A OR NOT B`) parametrized to assert both per-form expected rows AND the form-equivalence; double negation; XOR symmetric difference + XOR with NULL preserving 3VL; mixed-string-numeric AND inside OR exercising `_StringAllowingComparisonMixin` GT path; unit test locking `boolean_expr_to_text(BooleanExpr(op="pattern", ...))` round-trip for the (currently unreachable) defensive branch.  Three docstring guardrails: `expr_split.split_top_level_and` documents AND-only intent + the `_combine_conjuncts` AND-recombine mechanism that makes a hypothetical `split_top_level_or` silently incorrect; `predicate_pushdown._split_conjuncts` mirrored guard naming the failure mode; `_boolean_expr_text.boolean_expr_to_text` explicit `op == "pattern"` branch with both unreachability paths documented.  No production-code behavior change.  Closes the residual-frontier portion of #1219; deeper compositional shapes beyond current fixtures remain tracked under that issue (#1219, #1227).
- **GFQL / Cypher row-boolean residual matrix — cuDF parity follow-up (#1219 task 3)**: Parameterized the #1227 residual row-boolean test block across pandas and cuDF engines (`engine in [None, "cudf"]`) with runtime-gated cuDF execution and backend-stable row extraction for assertions. This extends the residual matrix coverage to cuDF paths without changing row-boolean semantics (#1276).
- **GFQL / Cypher parser + ast_normalizer — multi-positive WHERE pattern predicates (#1031 slice 3)**: AND-joined positive WHERE pattern predicates (`WHERE (n)-[:R]->() AND (n)-[:T]->()`) now lift into structured `WhereClause.predicates` as N `WherePatternPredicate` entries.  The ast_normalizer packs them into a single appended `MatchClause` whose `patterns: Tuple[Tuple[PatternElement, ...], ...]` carries one tuple per predicate (multi-pattern cartesian within MATCH), preserving the lowering invariant that only the FINAL match is connected — pre-binding seeds remain node-only.  Per-predicate validation (must include a relationship; cannot introduce new aliases) runs independently before the lift.  Removes the legacy `len(pattern_leaves) > 1` gate in `parser.py::_build_where_with_pattern_lift` and the corresponding gate in `ast_normalizer._rewrite_where_pattern_predicates_to_matches`.  Refactors `pattern_atom` to split the greedy `WHERE_PATTERN` lexer token (which gobbles `pattern AND pattern AND ...` chains as a single match) back into individual pattern-item texts via `_WHERE_PATTERN_ITEM_RE.finditer` and emit one `BooleanExpr(op="pattern")` per item, joined by an AND-tree via `_rebuild_and_tree`.  Adds `test_gfql_executes_multi_positive_where_pattern_predicates_as_intersected_seed` and updates the legacy rejection test to assert the new lift + compile shape.  Closes #1031 slice 3 (#1031).
- **GFQL / Cypher lowering**: Connected `MATCH + OPTIONAL MATCH` compilation now supports row-boolean `WHERE` expressions (`OR`/`NOT`/`XOR` and mixed row predicates) by carrying non-lowerable expressions into post-binding `where_rows(...)` filters for base and optional arms, preserving null-extension behavior while expanding supported disjunction shapes (#1219, #1224).
- **GFQL / Cypher parser**: switched the Cypher parser from Lark's LALR(1) backend to Earley.  *Earley's broader unification incidentally lifts the implicit LALR rejection on row-side OR/NOT/XOR among row predicates.  Coverage validated empirically across the risky shapes available to current fixtures: simple homogeneous AND/OR/NOT (correct rows); cross-alias OR with predicate-pushdown candidates (correct union — pushdown leaves the OR intact past the join); OPTIONAL MATCH + WHERE OR (the pre-existing OPTIONAL-MATCH-projection validator gates the projection shape regardless of WHERE — including the OR variant); type-coerced OR against a mixed-type Series (the call executor wraps pandas's `TypeError` as `GFQLTypeError(E303)` via the generic unsupported-row-expression path).  No silent wrong-rows surfaced in the shapes exercised; deeper compositional shapes (NOT inside OR with nullable arms, N-ary OR associativity, mixed-string-numeric AND inside OR, De Morgan compositions) are tracked under #1219.*  This eliminates four LALR-induced workarounds: the 3 dedicated pattern-shape `where_clause` grammar alternatives (now collapsed into a single `WHERE_PATTERN -> pattern_atom` leaf in `?primary`), `_canonicalize_where_single_pattern_and_expr` (regex source-rewrite that reordered `expr AND pattern AND expr` to `pattern AND <rest>` so LALR could match), `_mixed_where_pattern_expr_error` (pre-flight rejector replaced with a structural lift in `generic_where_clause`), and the `parse_cypher` `except LarkError` retry block.  `BooleanExpr.op` literal extended with `"pattern"` plus a new `BooleanExpr.pattern` payload field; the `pattern_atom` transformer wraps `WHERE_PATTERN` tokens as boolean-tree leaves; `_split_top_level_and_pattern_leaves` + `_rebuild_and_tree` + `_build_where_with_pattern_lift` extract pattern leaves from `expr_tree` into `WhereClause.predicates` as `WherePatternPredicate` entries before lowering.  Strict-improvement consequences (Earley accepts what LALR rejected): `WHERE expr OR expr` now parses as a structured `or` tree; `WHERE expr AND (expr OR expr)` parses as `and(left, or(...))`; `WHERE n:Label AND n.prop = X` routes through structured `where_predicates`; mixed label+property+string-comparison shapes work via the paired `_StringAllowingComparisonMixin` fix in `comparison.py`.  Slice 2/3/4 of #1200 territory (NOT-pattern, multi-positive-pattern, OR/XOR-around-pattern) emit explicit `unsupported` errors at the lift step.  1551 GFQL tests pass; the matching `tck-gfql` branch (`issue-1031-grammar-mixed-where-pattern-expr`) carries 8 paired contract refinements: `match-where1-10` lifted to UNEXPECTED_SUCCESS; `match-where5-{1,2,3}` + `expr-comparison2-1` + `with-where5-3` migrated to a new TYPE_ERROR_KEYS bucket (string-comparison-on-mixed-Series wraps as `GFQLTypeError(E303)`); `match-where5-4` upgraded to MATCHES_EXPECTED (Earley + 3-valued OR makes `WHERE i.var > 'te' OR i.var IS NOT NULL` semantically correct); `with2-1` filed under WRONG_ROW_KEYS (WITH-pipelined join now parses + executes but rows differ from oracle — separate gap); `with-where5-3` demoted from PROMOTION_ROW_KEYS (#1031, #1217).
- **GFQL / Comparison predicates**: extended `EQ`'s pre-existing string-accepting `_normalize_value` + `_validate_fields` overrides to its sibling comparison ops `NE` / `GT` / `LT` / `GE` / `LE` via a new `_StringAllowingComparisonMixin`.  Each `__call__` now handles `isinstance(self.val, str)` (pandas Series supports lexicographic `>`/`<`/`!=` on strings natively).  The strict raw-string rejection on the base `ComparisonPredicate` stays (still applies to `Between` and any direct-IR constructors where datetime-vs-string ambiguity matters).  Closes a latent gap surfaced by #1031's Earley swap, which unifies label + comparison routing and exposed that `WHERE prop OP 'string' AND <label>` patterns previously sidestepped `ComparisonPredicate` via raw-expr text (#1031, #1217).
- **GFQL / Cypher — `WhereClause.expr` field removal (closes #1213, completes #1200)**: Final slice of the umbrella. Drops the legacy raw-text `WhereClause.expr: Optional[ExpressionText]` field now that all production readers (binder, lowering, ast_normalizer) consume `expr_tree`-derived text via `boolean_expr_to_text()`. Sub-PR D removed the `expr=` argument from every `WhereClause(...)` writer site (3 in `parser.py` that synthesized `BooleanExpr` atoms in #1214, plus the structured-path constructions, plus `ast_normalizer.py:509` forwarder; gates switched to `expr_tree is not None`). Sub-PR E removed the field declaration from `cypher/ast.py` and rewrote the `WhereClause` docstring to the surviving routing contract: structured (predicates only — including `WherePatternPredicate`-only via `where_pattern_only_clause`), tree (expr_tree only), or mixed (both — for `WHERE pattern AND expr`). Test surface updated: `test_where_clause_expr_tree_invariant.py` rewritten from the now-vacuous `(expr is None) == (expr_tree is None)` symmetry assertion to a routing-shape contract test (9 query forms across the 3 shapes); 12 raw-text reader assertions across `test_parser.py`, `test_boolean_expr.py`, `test_ast_normalizer.py`, and `test_where_bool_conformance.py` migrated to `boolean_expr_to_text(where.expr_tree)`; the slice-1 backward-compat test asserting both fields was deleted as obsolete. `ExpressionText` itself stays — other AST nodes still use it (`ReturnItem`, `WithItem`, `OrderByItem`, page values, etc.). **API note**: `WhereClause` is exported from `graphistry.compute.gfql.cypher`; external code that constructs it directly (rare — it's parser output, not user-facing) must drop the `expr=` kwarg (#1200, #1213).
- **GFQL / Cypher lowering + ast_normalizer — `ExpressionText`-passthrough `WhereClause.expr` reader migration**: Second read-side slice of #1213. Migrated all `ExpressionText`-passthrough readers of `WhereClause.expr` to consume `WhereClause.expr_tree`-derived values via two new helpers in `cypher/lowering.py`: `_where_clause_expr_text(where)` synthesizes a fresh `ExpressionText` from `where.expr_tree` (using `where.span` to preserve master's error-position semantics — `where.span == where.expr.span` for parser-produced WhereClauses); `_rewrite_where_clause_and_resync(where, rewrite, field)` applies a text rewrite via the existing callback shape and resynchronizes `expr_tree` to a single-atom `BooleanExpr` carrying the rewritten text. **Side-effect fix**: closes the latent text/tree staleness bug that existed on master post-#1214 where `replace(where, expr=rewrite(where.expr, ...))` left `expr_tree` pointing at the pre-rewrite text. Sites migrated: `_extract_relationship_type_where` callers (2: connected-OPTIONAL-MATCH lowering at L8176, dynamic row-where extraction at L6322); `rewrite_expr` callers in reentry-where rewriting (2 sites at L7530-7538 and L7550-7553); `pre_join_filters.append` at L7998; `ast_normalizer.py::_rewrite_where` shortest-path rewrite. The last remaining read in `ast_normalizer.py:520` is a constructor passthrough (`expr=query.where.expr` inside a `WhereClause(...)` re-construction) — left for sub-PR D to drop along with the other writer-side construction-site updates. Behavior-preserving: full GFQL suite (1538/1538) green; targeted mypy clean. Sets up sub-PRs D + E (drop `expr=` from constructions; remove the field) for the colleague who landed #1214 (#1200, #1213).
- **GFQL / Cypher binder + lowering — text-only `WhereClause.expr` reader migration**: First read-side slice of #1213. Migrated all text-only readers of `WhereClause.expr.text` to consume `boolean_expr_to_text(WhereClause.expr_tree)` instead, leveraging the parser invariant `(expr is None) == (expr_tree is None)` from #1214. Sites: `frontends/cypher/binder.py::_where_predicates` (dropped the now-dead `elif where.expr is not None` fallback — unreachable per invariant), `cypher/lowering.py::_reject_unsupported_where_expr_forms`, `cypher/lowering.py::_reject_unsupported_variable_length_where_pattern_predicates`, and three `_check_expr` callers in lowering. Behavior-preserving: per-call output identical to master (verified via 1538 GFQL tests). Removed the now-redundant `test_where_predicates_with_expr_tree_none_falls_back_to_expr_text` test which exercised the dead fallback via a hand-built `WhereClause(expr=..., expr_tree=None)` fixture that violates the parser invariant. Span-access readers (entangled with `ExpressionText` passthrough) and `rewrite_expr` callers stay on `.expr` until #1213 sub-PR C (#1200, #1213).
- **GFQL / Cypher parser**: `generic_where_clause` now walks Lark's parsed `BooleanExpr` (always populated post-#1214 via single-atom synthesis for non-BooleanExpr operands) to lift bare label predicates and AND-spines of bare label predicates into structured `WhereClause.predicates`, replacing the previous text-level `split_top_level_and` + regex loop. Closes #1194 in the spirit of the issue: the grammar already declares the structured rule we want, and Lark's parsed tree is the single source of truth — no top-level AND splitting on text inside the WHERE body. `_BARE_LABEL_PREDICATE_RE.fullmatch` remains load-bearing on each atom as the false-positive guard from #1125 (unchanged). Adds two helpers: `_match_bare_label_atom` (atom-text → `(alias, labels)`) and `_lift_label_only_and_spine` (BooleanExpr AND-spine walker). Backward-compatible: identical I/O contract for all WHERE shapes (single label, multi-AND label chains, mixed label+property, OR/XOR/NOT, parenthesized boolean trees, string-literal false-positive guards). 4 new focused unit tests for the helpers complement the existing parser/binder/conformance/lowering coverage. Composes cleanly with #1214's expr_tree invariant — the walker now sees a uniform `BooleanExpr` for every routing case (#1194, #1213).
- **GFQL / Cypher WHERE expr_tree invariant**: Lifted `_boolean_expr_to_text` and `_flatten_top_level_ands` from `frontends/cypher/binder.py` to a new shared module `graphistry/compute/gfql/cypher/_boolean_expr_text.py` so subsequent slices can reconstruct WHERE-body text from `WhereClause.expr_tree` without depending on binder internals. Established the parser invariant `(WhereClause.expr is None) == (WhereClause.expr_tree is None)` by synthesizing single-atom `BooleanExpr` at three previously-asymmetric construction sites (`_mixed_where_clause`, `where_clause` non-structured branch, `generic_where_clause` single-atom fallback). Behavior-preserving: binder primary path now emits one `BoundPredicate` per top-level AND conjunct uniformly. Adds `test_where_clause_expr_tree_invariant.py` with 11 tests across structured, mixed, atom, and boolean-tree shapes (#1200, #1213).
- **GFQL / Cypher binder**: ``_where_predicates`` now walks ``WhereClause.expr_tree`` when populated and emits one ``BoundPredicate`` per top-level AND conjunct — flattened via ``_flatten_top_level_ands`` and stringified via ``_boolean_expr_to_text``.  Downstream passes (predicate pushdown) see pre-split single-conjunct strings, eliminating one source of text re-parsing.  Backward-compatible: when ``expr_tree`` is ``None`` (mixed-clause routes, structured ``where_predicates``, queries with no boolean operators), falls back to the existing ``where.expr.text`` path.  Inherits the slice-1 literal-atom fidelity caveat (``str(True) == "True"``).  20 tests in ``test_binder_expr_tree.py`` covering helper invariants, hand-built ``WhereClause`` fixtures (backward-compat, OR root, AND chain, no-expr), and end-to-end through ``parse_cypher`` + ``FrontendBinder`` (#1200, #1207).
- **GFQL / Cypher parser**: Parser now captures Lark's already-parsed boolean-expression tree (`and_op` / `or_op` / `xor_op` / `not_op`) as a structural `BooleanExpr` value on `WhereClause.expr_tree`, alongside the existing raw-text `.expr`.  Adds `grouped_expr` passthrough so parenthesized subexpressions preserve nested boolean structure rather than collapsing to atom leaves.  Backward-compatible: `.expr` stays populated; no binder/pushdown/`expr_split.py` changes this slice.  Downstream migration (binder walks tree to emit one `BoundPredicate` per top-level AND conjunct; pushdown drops text-split fallback; `expr_split.py` retired) is tracked in the remaining slices of #1200.  Slice 1 is strictly additive (#1200, #1202).
- **GFQL / Cypher WHERE boolean-shape conformance matrix**: Added `graphistry/tests/compute/gfql/cypher/test_where_bool_conformance.py` — 31 tests locking WHERE boolean-shape contracts across all 7 expression shapes (plain AND, `AND (OR)`, `(OR) AND`, NOT-prefix, XOR, mixed label+property, quoted/escaped AND) × 3 pipeline layers (parser routing to `.predicates` vs `.expr`, binder MATCH-clause predicate counts, pushdown `split_top_level_and` + `is_null_rejecting` per shape). Guards against silent drift in #1194 (grammar disambiguation) and #1200 (IR boolean-tree) migrations (#1201).
- **GFQL / IR refactor**: Consolidated the `("input", "left", "right", "subquery")` child-slot tuple that was duplicated across 5 sites (IR verifier, physical planner, two rewrite passes, one test helper) into a single `CHILD_SLOTS` constant and `iter_children` helper in `graphistry/compute/gfql/ir/logical_plan.py`. Adds 8 regression tests including: identity-preservation for `UnnestApply` and `PredicatePushdownPass` when no descendant is rewritten (the `rewritten_child is not child` guard is load-bearing for tier-2 fixed-point convergence), asymmetric identity preservation across `Join` when only one branch is rewritten, and a reflective `typing.get_type_hints` check that any future `LogicalPlan` subclass adding a new child slot must update `CHILD_SLOTS` (#1196, #1199).

### Fixed
- **GFQL / row pipeline — relationship alias in row expressions (#1072)**: Bare edge-alias tokens in row expressions (`select([("rel", "workAt")])`, `WHERE rel IS NOT NULL`, `RETURN rel`) no longer raise `unsupported token in row expression: '<alias>'`. `_gfql_resolve_token` in `graphistry/compute/gfql/row/pipeline.py` extends the bare-alias branch with edge-alias resolution: prefer `{alias}.{_edge}` when present, otherwise render `[:TYPE {prop: value, ...}]` via a new shared vectorized renderer in `graphistry/compute/gfql/row/entity_props.py` reused by `cypher/result_postprocess.py::_format_edge_entities` and the row-pipeline path. Edge-alias classification is gated on `_gfql_rows_edge_aliases` metadata propagated through row-table adapters so node-alias frames missing a node-id column are not misrendered as edges. Renderer normalizes string escaping (`\\`, `'`), float formatting (strips trailing `.0+`), and treats blank/whitespace `type` as absent (`[{...}]` instead of `[: {...}]`). Re-entry `WHERE` secondary-alias property predicates (`MATCH ... WITH a, b ... MATCH ... WHERE b.x = ...`) are now rewritten when the predicate has no `expr_tree` by synthesizing WHERE expression text from structured predicates and clearing consumed predicates in `cypher/lowering.py` to avoid stale dual representation. Carries upstream prerequisites #1071 (multi-alias carry through `WITH` before MATCH re-entry) and #1236 (OR/XOR-around-pattern WHERE lowering) needed for current TCK contract behavior. Adds regressions in `test_row_pipeline_ops.py` covering string escaping, float normalization, null-only properties, missing/empty `type`, multi-edge rows, `select`/`return` parity, node-alias misclassification guard, and bare-alias `id` parity; adds `test_lowering.py` regressions for re-entry WHERE rewrites on secondary aliases and OR/XOR-around carried+trailing alias property paths (#1072).
- **GFQL / comparison predicates**: Mixed-type scalar comparisons in Cypher `WHERE` execution (`>`, `<`, `>=`, `<=`) now preserve null-safe filter semantics across pandas and cuDF backends instead of failing whole-series evaluation on incomparable rows. Comparable rows keep backend-native ordering; incomparable/null rows evaluate non-matching (`False`) (#1219, #1223).
- **GFQL / predicate pushdown**: Fixed a silent `\b` regex bug in `_refs_for_segment` (`graphistry/compute/gfql/passes/predicate_pushdown.py`). The rf-string `rf"\\b..."` produced a literal backslash-b sequence, not a word-boundary assertion, so per-conjunct alias detection never matched and always fell back to the full original reference set — widening reference sets for every split conjunct and preventing some safe-pushable conjuncts from being recognized. Tests passed because the fallback is a superset. Also consolidated two duplicate "split WHERE body on top-level AND" implementations (one in `parser.py`, one in `predicate_pushdown.py`) into a shared helper `graphistry/compute/gfql/expr_split.py::split_top_level_and` with strictly quote/bracket/paren/backtick-aware splitting. Adds 20 direct unit tests for the shared helper and one regression test locking the `\b` fix (#1195, #1198).
- **GFQL / Cypher binder**: Replaced fragile regex-based WHERE label narrowing fallback in `_apply_where_label_narrowing` with AST-derived narrowing. `generic_where_clause` now lifts AND-joined bare label predicates (`WHERE n:Admin AND n:Active`) to structured `WhereClause.predicates` using the existing quote/bracket/paren/backtick-aware `_split_top_level_and_terms` helper; string-literal false-matches (e.g. `WHERE n.name = 'n:Admin'` incorrectly narrowing alias `n`) are closed by `fullmatch` anchoring. Removes `_WHERE_LABEL_RE` and `_WHERE_NON_CONJUNCTIVE_RE` from `binder.py`. Adds 10 targeted tests covering single/double/triple AND, multi-alias, multi-label-per-alias, lowercase `and`, XOR/OR/NOT conservative non-narrowing, mixed label+property all-or-nothing, and string-literal false-positive guards (#1125, #1193).
- **DataFrame engine coercion**: Unified all DataFrame-to-engine conversion behind `df_to_engine()` with explicit dispatch for Arrow, Spark, dask, dask_cudf, cuDF, Polars, and pandas; unknown types now raise `ValueError` instead of silently calling `.to_pandas()`. `_coerce_input_formats(g, engine)` replaces `_coerce_to_pandas(g)` as the engine-aware coercion entry point in `chain()`, `hop()`, and `materialize_nodes()`, preserving GPU (cuDF) output when input is cuDF. `to_pandas()` now handles all input types via the same dispatch. Adds `test_engine_coercion.py` with 50+ tests (#1148).
- **GFQL / cuDF engine guard**: `DFSamePathExecutor.run()` now checks cuDF availability before calling `_forward()` so `mode=auto` silently falls back to pandas and `mode=strict` raises a clear error, rather than propagating `ModuleNotFoundError` from inside hop/coerce internals (#1148).
- **RAPIDS 25.x / cuDF API**: Replaced deprecated `cudf.DataFrame.from_pandas()` with `cudf.from_pandas()` in `_cudf_from_pandas_best_effort()` (6 call sites) for RAPIDS 25.x+ compatibility (#1148).
- **GFQL / IR verifier**: `ListType.element_type` is now validated recursively by invariant 4 (schema consistency), catching invalid nested types at arbitrary depth. Converts the `xfail` placeholder test to a passing regression; adds a positive test for valid deep nesting (#1153).
- **GFQL / IR query graph**: `extract_query_graph` now preserves all optional-arm metadata when intermediate aliases are dropped by a final `RETURN` projection. Chained `OPTIONAL MATCH` arms retain their arm IDs and nullable-alias sets even when their outputs are absent from `RETURN`; `BoundQueryPart.metadata["arm_id"]` is used as the authoritative arm identity and unconditionally preferred over (potentially stale) `semantic_table` entries. Each nullable alias is now guaranteed to belong to exactly one arm even when sequential optional parts share the same output alias (#1156).
- **DataFrame input types**: `pa.Table` (Apache Arrow) and `pyspark.sql.DataFrame` (Spark) now work in `materialize_nodes()`, `get_degrees()`, `get_indegrees()`, `get_outdegrees()`, and `hypergraph()` without crashing. Both are coerced to pandas at each entry boundary; pandas/cuDF paths are unaffected. Mixed inputs (e.g. Arrow edges + pandas nodes) are handled correctly. Adds `test_df_types.py` with 22 tests covering Arrow compute, Arrow hypergraph, mixed-type boundaries, and Spark paths; adds a `test-spark` parallel CI job (Python 3.14, pyspark 4.x) (#1132).
- **GFQL / remote Cypher string path**: `gfql_remote("...")` no longer compiles strings through deprecated `compile_cypher()` compatibility API. Remote string compilation now uses the non-deprecated internal parser+lowering path (`parse_cypher` + `compile_cypher_query`) while preserving wire JSON behavior (`gfql_query`, `gfql_operations`, Let/Ref/CALL params, and union rejection). Added focused regressions in `graphistry/tests/compute/test_chain_remote_v2.py` to ensure the deprecated helper is not called from remote execution (#1185).

### Fixed
- **Plugins / cuDF**: `compute_igraph`, `layout_igraph`, and `layout_graphviz` now handle cuDF-backed graphs. Input DataFrames are converted to pandas via `ensure_pandas()` (with `nullable=True` to preserve nullable integer dtypes), and output DataFrames are restored to the detected original engine via `restore_engine()` (both in `graphistry.compute.engine_coercion`). `from_igraph` and `g_with_pgv_layout` merges with existing node/edge tables are also converted to pandas to avoid mixed-type merge errors. Previously, passing a cuDF graph to these functions would raise a `TypeError` inside the native library. The Cypher CALL path also benefits since it delegates to `compute_igraph` which now handles engine restoration internally.
- **Plugins / igraph + engine coercion (#1246)**: `from_igraph` renames duplicate `source`/`target` column labels emitted by `ig.get_edge_dataframe()` when those names also exist as edge attributes, fixing `"column 'source' is not unique"` crashes during endpoint remapping. `ensure_pandas` broadens its `to_pandas(nullable=True)` fallback to catch `NotImplementedError` (raised by newer cuDF on dtypes like `datetime64[ms]`) in addition to `TypeError`.

### Deprecated
- **GFQL / Cypher public API**: `compile_cypher()`, `compile_cypher_query()`, `CompiledCypherQuery`, `CompiledCypherUnionQuery`, and `CompiledCypherProcedureCall` are deprecated and **scheduled for removal in a future release**. All emit `DeprecationWarning` at use. Migrate to `g.gfql(..., language="cypher")` for execution or `cypher_to_gfql()` / `gfql_from_cypher()` for chain translation. Tracked in #1169.

### Added
- **Collections**: New `g.collections(...)` API for defining subsets via GFQL expressions with priority-based visual encodings. Includes helper constructors `graphistry.collection_set(...)` and `graphistry.collection_intersection(...)`, support for `showCollections`, `collectionsGlobalNodeColor`, and `collectionsGlobalEdgeColor` URL params, and automatic JSON encoding. Accepts GFQL AST, Chain objects, or wire-protocol dicts (#874).
- **Docs / Collections**: Added collections usage guide in visualization/layout/settings, tutorial notebook (`demos/more_examples/graphistry_features/collections.ipynb`), and cross-references in 10-minute guides, cheatsheet, and GFQL docs (#875).
- **GFQL / two-tier pass execution**: Extended `PassManager` to support two explicit pass tiers: Tier 1 structural passes run once in configured order; Tier 2 rewrite rules run in a fixed-point loop until a full sweep makes no changes or `max_iterations` is exceeded. `PassResult` gains a `changed: bool` field (default `True` for backward compatibility) used by the convergence check. Added `UnnestApply` as the first Tier 1 structural pass — rewrites non-correlated `Apply` nodes (empty `correlation_vars`) to `Join(join_type="cross")`, exposing them to downstream join-ordering passes; correlated Apply nodes are preserved. `PredicatePushdownPass` is wired as the first Tier 2 rewrite rule and now sets `changed=pushed > 0` for correct convergence. `DEFAULT_LOGICAL_PASSES` and `DEFAULT_TIER2_PASSES` are populated accordingly and wired into `gfql()` execution. 19 new unit tests across `test_pass_manager.py` and `test_unnest_apply.py` (#1189).
- **GFQL / pass framework skeleton**: Added `graphistry/compute/gfql/passes/` with `LogicalPass`, `PassResult`, and deterministic `PassManager.run()` sequencing. The pass manager now invokes IR `verify()` after each pass and fails fast on invalid pass output. Wired a new logical-pass pipeline hook into `gfql()` execution between logical-plan and physical-planner stages using a default no-op pass configuration to preserve runtime behavior. Added focused tests for pass ordering, verifier-failure propagation, and runtime pipeline hook invocation (`test_pass_manager.py`, `test_runtime_physical_cutover.py`) (#1180).
- **GFQL / predicate pushdown safety**: Added `graphistry/compute/gfql/ir/pushdown_safety.py` with three reusable utilities for `PredicatePushdownPass`: `is_null_rejecting(pred, null_extended_aliases)` — conservative syntactic heuristic returning True when a predicate references a null-extended alias (OPTIONAL MATCH) and does not use a null-safe form (IS NULL, IS NOT NULL, COALESCE, NULLIF); `is_null_safe` — inverse; `with_barrier_blocks_pushdown(scope_stack, pred_refs)` — returns True when a WITH-clause `ScopeFrame` prevents backward predicate movement for the given reference set. All three exported from `ir/__init__.py`. 41 unit tests (#1181).
- **GFQL / predicate pushdown rewrite**: Added `PredicatePushdownPass` implementation in `graphistry/compute/gfql/passes/predicate_pushdown.py` and wired it into logical planning route execution. The pass rewrites `Filter(input=PatternMatch(...))` by pushing safe predicates into `PatternMatch.predicates`, keeps residual filters for partial-push cases, and blocks null-rejecting pushdown into optional arms using existing safety helpers. Added focused pass tests and a lowering-route integration assertion (`test_predicate_pushdown_pass.py`, `cypher/test_lowering.py`) (#1187).
- **GFQL / scope-aware pushdown barriers**: Threaded binder scope metadata (`BoundIR.scope_stack`) into runtime logical-pass context via `CompiledCypherExecutionExtras` and `PlanContext`, and updated `PredicatePushdownPass` to enforce `with_barrier_blocks_pushdown()` using real scope data before moving conjuncts into `PatternMatch.predicates`. Added targeted regressions for blocked vs allowed WITH-boundary movement and compile-route scope metadata threading (`test_predicate_pushdown_pass.py`, `cypher/test_lowering.py`) (#1190).
- **GFQL / remote wire migration (M3 follow-up)**: `chain_remote.py` remote Cypher string path no longer imports or dispatches on `CompiledCypher*` classes for wire serialization. It now validates/serializes structural compiled-query shapes (`chain`, `graph_bindings`, `procedure_call`, `use_ref`) so remote wire payload generation is decoupled from compiler IR class identity while preserving existing Let and CALL wire formats. Added parity tests in `test_chain_remote_v2.py` for structural fake compiled-query inputs (including Let bindings/result serialization and structural union rejection) (#1168).
- **GFQL / M3 compatibility-deletion gate (PR3 slice)**: Cypher public compatibility surfaces now mark `compile_cypher()` and legacy `CompiledCypher*` exports as deprecated while retaining runtime compatibility. `graphistry.compute.gfql.cypher` now serves deprecated `CompiledCypherProcedureCall`, `CompiledCypherQuery`, `CompiledCypherUnionQuery`, and `compile_cypher_query` via lazy compatibility accessors with explicit deprecation warnings; `compile_cypher()` now emits a deprecation warning and migration guidance (`g.gfql(..., language="cypher")`, `cypher_to_gfql(...)`). API docs now describe `compile_cypher()` as deprecated/internal-shape oriented. Deferred boundaries remain unchanged: remote-wire migration stays tracked in #1168 and hard removal/versioned API cleanup stays tracked in #1169 (#1174).
- **GFQL / M3 runtime cutover to PhysicalPlanner**: `gfql()` compiled-query execution in `graphistry/compute/gfql_unified.py` now dispatches through `PhysicalPlanner.plan(...)` for planned logical routes, then executes via physical operator wrappers (`same_path`, `wavefront`, `row_pipeline`). Added bounded compatibility shims for currently-required lanes (`CALL`-backed compiled queries and connected OPTIONAL wavefront payloads), and explicit validation failure when a planned wavefront route lacks executable join payload metadata (no silent unmatched fallback). Added focused runtime cutover tests in `graphistry/tests/compute/gfql/test_runtime_physical_cutover.py` covering planner invocation, optional-wavefront parity, compatibility-shim continuity, and explicit wavefront mismatch failure behavior (#1173).
- **GFQL / M3 physical planner skeleton**: Added `graphistry/compute/gfql/physical_planner.py` with backend-neutral `PhysicalPlanner.plan(logical_plan, ctx) -> PhysicalPlan` route mapping and explicit wrapper contracts for current executor lanes: `same_path` (`execute_same_path_chain`), `wavefront` (`_apply_connected_match_join`), and `row_pipeline` (`execute_row_pipeline_call`). `PhysicalPlan` now carries stable route/operator metadata, and unsupported logical operator shapes fail with explicit planner errors (no silent fallback). Added focused planner route/wrapper tests in `graphistry/tests/compute/gfql/test_physical_planner.py` (#1164).
- **GFQL / M3 B3 compatibility matrix**: Added `docs/source/gfql/spec/m3_b3_compat_matrix.md` — per-surface closure plan for `CompiledCypherQuery` / `CompiledCypherUnionQuery` / `CompiledCypherGraphQuery` / `compile_cypher_query` before M3 deletion gate. Covers 6 surfaces (compiler internals, public API, runtime execution, remote wire, tests, docs) with sequenced closure strategy. Two deferred items tracked for follow-up issues (remote wire migration, public API deprecation removal) (#1165).
- **GFQL / Cypher compiler (M3 route coverage start)**: Expanded logical-plan route coverage for previously broad-deferred reentry/row-sequence flows under issue `#1154`. `MATCH ... WITH ... MATCH ...` reentry and row-sequence shapes (including `WITH`-only / `UNWIND` forms) now route as `planned` when supported by `LogicalPlanner`; planner now emits `PatternMatch` for covered multi-alias and edge-alias MATCH skeleton shapes. Route-boundary tests were added for planned reentry/row-sequence paths and for the still-deferred optional-reentry boundary (`OPTIONAL MATCH`) in this slice (#1155).
- **GFQL / M2 closure**: Added in-repo M2 exit-gate closure artifact at `docs/source/gfql/spec/m2_exit_gate_closure.md` with: PR1-PR5 evidence matrix, M2 exit checklist, RF4 implemented-vs-deferred construct mapping, merged-lane CI receipt summaries, and targeted local conformance receipts for planner/querygraph/verifier/cycle-policy plus scope/alias rejection slices. Includes deferred follow-up tracking issues `#1153` (verifier deep `ListType` recursion checks) and `#1154` (M3 route coverage for currently deferred reentry/row-sequence flows) for explicit deferred-item accounting under `#1137`/`#1139`.
- **GFQL / IR query graph**: Added `extract_query_graph(bound_ir: BoundIR) -> QueryGraph` in `graphistry/compute/gfql/ir/query_graph.py`. Walks `BoundQueryPart` query_parts, splits on WITH/RETURN scope boundaries, groups aliases within each scope into `ConnectedComponent` clusters via union-find on shared alias sets, collects boundary aliases (WITH outputs, handling renames like `WITH a AS b`) with their `LogicalType` from the scope-stack frame, and builds `OptionalArm` entries by grouping `null_extended_from` provenance sets — with `join_aliases` inferred from required (non-nullable) inputs to `optional_match` parts. Cycle-policy regression tests lock the existing lowering.py contracts for connected comma patterns, disconnected comma errors, and cross-WITH scalar alias reuse. `extract_query_graph` is exported from `ir/__init__.py`. 25 focused tests (#1135).
- **GFQL / IR compiler**: Added `graphistry/compute/gfql/ir/verifier.py` with `verify(plan: LogicalPlan) -> list[CompilerError]`. Tree-walks the operator DAG and checks five structural invariants: op_id uniqueness, dangling child references, predicate scope (non-empty expression and alias-visibility against `output_schema` on PatternMatch/Filter/IndexScan), output schema validity (all columns are LogicalType instances), and optional-arm nullability (PatternMatch optional=True requires non-empty arm_id and nullable ScalarType outputs). Traversal is cycle-safe via DFS ancestor tracking. `BoundPredicate` gains a `references: FrozenSet[str]` field; when non-empty the verifier checks that all referenced aliases appear in the operator's `output_schema.columns`. `verify` is exported from `ir/__init__.py`. 67 focused positive/negative tests; one documented xfail for ListType element_type deep recursion (future PR) (#1127).
- **GFQL / compiler**: Added initial `LogicalPlanner` skeleton in `graphistry/compute/gfql/logical_planner.py` with monotonic per-plan `IdGen` `op_id` assignment, minimal covered clause planning (`MATCH`/`WHERE`/`WITH`/`RETURN`/`UNWIND` + deterministic fallback), fail-fast guards for unsupported skeleton shapes (multi-alias/non-node `MATCH`, `DISTINCT` projections, `OPTIONAL MATCH`, unknown clauses), and contract tests for importability, deterministic purity, typed schema propagation, and unique op IDs (#1126).
- **GFQL / Cypher compiler**: CALL/GRAPH compatibility routing now maps supported Cypher CALL shapes to verified logical `ProcedureCall` operators instead of hard-defer, and graph-constructor artifacts (`CompiledCypherGraphQuery` + `CompiledGraphBinding`) now carry explicit logical route metadata (`logical_plan`, defer reason, route kind) while preserving existing runtime execution semantics (#1128).
- **GFQL / Cypher compiler**: `CompiledCypherQuery` escape-hatch surface was consolidated into grouped `post_processing` and `execution_extras` containers with backward-compatible properties, compile output now includes explicit logical route metadata (`logical_plan`, defer reason, and route kind), and connected OPTIONAL lowering now emits `QueryGraph` + logical-operator metadata while preserving existing execution parity checks (#1136).
- **GFQL / Cypher**: Extracted `ASTNormalizer` into `graphistry/compute/gfql/cypher/ast_normalizer.py` and moved shortestPath + WHERE-pattern-predicate rewrite ownership out of `lowering.py`, with parity-preserving wiring in compile/lowering flows and focused regression coverage for rewrite behavior and invocation order (#1117).
- **GFQL / Cypher compiler**: Lowering now functionally consumes `BoundIR` metadata for the M1 integration slice: binder-provided params are merged into effective lowering params (runtime overrides preserved) with binder metadata keys filtered out of runtime-param resolution, scope membership narrowing uses the active scope frame for WITH-boundary correctness, semantic-table entity kinds inform alias table routing, and nullable alias metadata is wired into optional-only alias detection. `_StageScope` duplicated table bookkeeping was reduced, binder now runs pre- and post-normalization in compile flow, and binder-path regression tests were added for these code paths (#1116).

### Changed
- **Collections**: Autofix validation now drops invalid collections (e.g., invalid GFQL ops) and non-string collection color fields instead of string-coercing them; warnings still emit when `warn=True`.
- **Collections**: `collections(...)` now always canonicalizes to URL-encoded JSON (string inputs are parsed + re-encoded); the `encode` parameter was removed to avoid ambiguous behavior.
- **Collections**: Set collections require an `id` field for server-side subgraph storage. In `strict` validation mode, missing IDs fail validation; in `autofix`, missing IDs are warned and deterministically auto-generated (`set-<index>` / `intersection-<index>`).
- **Collections**: Intersection collections now cross-validate that referenced set IDs exist; dangling references are warned and dropped in autofix mode.
- **Collections**: GFQL parsing consolidated to use `_wrap_gfql_expr` from `collections.py` as the canonical implementation with precise exception handling.

### Tests
- **Collections**: Added `test_collections.py` covering encoding, GFQL Chain/AST normalization, wire-protocol acceptance, validation modes, and helper constructors.
- **GFQL / Cypher binder**: Added PR-4 white-box binder semantic conformance coverage for name resolution success/failure (including unresolved alias errors), WITH scope-reset visibility, OPTIONAL MATCH `null_extended_from` lineage as `frozenset` clause ids, label narrowing from MATCH labels + conjunctive `WHERE alias:Label` checks, and SchemaConfidence rules (min-rule propagation, operand inheritance, and strong literal/`COUNT` behavior). Parser/lowering regression lanes remain green (#1114).
- **Plugins / cuDF**: 14 GPU tests in `TestCpuOnlyPluginsCudfRoundTrip` (`test_call_operations_gpu.py`) verifying real cuDF→pandas→cuDF round-trip for `compute_igraph` (pagerank, spanning_tree Graph-returning path, articulation_points list-return path, edge-attribute merge path), `layout_igraph`, `layout_graphviz`, `render_graphviz`, `execute_call`, `ensure_pandas` nullable dtype preservation, and `restore_engine` conversion. Requires `TEST_CUDF=1` and RAPIDS.
- **GFQL / Cypher**: Added bound-alias `WHERE NOT (pattern)` regression coverage for issue `#1237`, including an IC10-shaped direct-Cypher case (`MATCH (root {id: 'a'})-[:R]->(mid)-[:R]->(cand) WHERE NOT (root)-[:R]->(cand)`) plus compile-shape and mixed row+NOT predicate tests in `graphistry/tests/compute/gfql/cypher/test_lowering.py`.

## [0.54.1 - 2026-04-08]

### Infrastructure
- **CI / speedup**: All CI install steps migrated from floating `pip` to `uv` with per-Python-version hashed lockfiles, cutting the gating `test-minimal-python` job by **~47%** (e.g. 3.12: 733s → 406s, 3.8: 673s → 332s) and `python-lint-types` by **~47%** (49s → 26s avg). A single `generate-lockfiles` job (≤30s) runs in parallel with lint, so the new lockfile overhead does not extend the critical path (#1050).
- **CI / supply-chain security**: Introduced `UV_EXCLUDE_NEWER: "6 days"` globally across all CI jobs (belt-and-suspenders on top of hashed lockfiles) to reject packages published within the last 6 days, preventing 0-day supply chain attacks from reaching CI. All install steps use `--require-hashes` where feasible; umap/AI jobs exempt due to torch local-version-tag incompatibility with `--require-hashes` (#1050).
- **CI / GFQL split**: 944 GFQL-heavy tests (`test_lowering.py` + `test_row_pipeline_ops.py` + `test_parser.py`) moved from the serial `test-minimal-python` gate to a new parallel `test-gfql-core` job, further reducing the gating critical path (#1050).
- **Dockerfiles / supply-chain**: Added `ARG PIP_EXCLUDE_NEWER=6d` with matching `PIP_EXCLUDE_NEWER` env to all test Dockerfiles (`test-cpu`, `test-gpu`, `test-rapids-official`), applying the same 6-day cooldown to Docker-based installs (#1050).

### Added
- **GFQL / compiler**: Added initial IR type layer under `graphistry/compute/gfql/ir/` with `types.py` (`CypherAST`, `NodeRef`, `EdgeRef`, `RelSpec`, `LogicalType`, `QueryGraph`) and `bound_ir.py` (`BoundVariable`, `SemanticTable`, `ScopeFrame`, `BoundIR`) as frozen dataclasses for M0 binder scaffolding. No runtime behavior changes (#1091).

### Fixed
- **GFQL / Cypher**: `OPTIONAL MATCH` execution no longer materializes the full opt-arm result before joining. A semi-join filter now restricts the optional arm to join-key values present in the base MATCH result before the left-outer-join. On real benchmark graphs (LDBC SNB sf1 IS7) this eliminates the intermediate cross-product that previously caused ~120 GB RSS and a SIGKILL (#1052).
- **GFQL / Cypher**: `MATCH ... WITH scalar1, scalar2 ... MATCH ...` re-entry now supports multi-row WITH prefixes (N rows, not just 1). Each prefix row's scalars are broadcast independently to the base graph and the suffix MATCH runs once per row; results are unioned. This unblocks IC6 (`tag-cooccurrence`): the UNWIND fanout before the tag-cooccurrence MATCH produces 4+ prefix rows (#1047).
- **GFQL / cudf**: `graph.gfql(query, engine="cudf")` no longer crashes with SIGSEGV (returncode 139) on RAPIDS 25.02. Root cause: `Series.map(dict)` and `Series.to_pandas()` on cudf Series trigger numba JIT (via `numba_cuda.as_cuda_array`) which SIGSEGVs in RAPIDS 25.02. Fixed by introducing `safe_map_series()` in `Engine.py` that uses a GPU-native merge-based lookup with `to_arrow()` for cudf (stays on GPU, avoids numba path entirely), and replacing all eight affected call sites in `hop.py` (4 sites), `chain.py`, `df_executor.py` (2 sites), and `pipeline.py`. Validated on both RAPIDS 25.02 and 26.02. GPU performance lane restored (#977).
- **GFQL / Cypher**: Bounded zero-min variable-length relationship patterns (`*0..N`, e.g. `[:HAS_TYPE|IS_SUBCLASS_OF*0..3]`) are now supported. Previously `*0..N` raised an "unsupported" error at parse time while the open form `*0..` already worked. Fixes LDBC SNB IC12 and any query using bounded zero-min ranges (#983).
- **GFQL / Cypher**: Multi-row scalar prefix combined with `OPTIONAL MATCH` re-entry now raises a clear error instead of silently dropping null-fill rows. The multi-row fan-out path returns early before the optional null-fill branch; until null-fill is implemented for N>1 prefix rows, the engine raises `GFQLValidationError` with an actionable message (#1047).
- **GFQL / Cypher**: `CASE x WHEN null THEN ... ELSE ... END` now correctly matches when `x` is null. Previously `__cypher_case_eq__` used `pd.Series == None` which always evaluates to `False` in pandas; the fix returns the null-mask of the non-null operand when the other is a scalar null literal. This was the root blocker for IS7-style `CASE r WHEN null THEN false ELSE true END` over edge aliases from an OPTIONAL MATCH arm (#996).

### Tests
- **GFQL / comparison conformance**: Added exhaustive cross-backend comparison matrix coverage in `test_comparison_conformance.py` for pandas+cuDF over mixed scalar domains (`int`, `float`, `bool`, `string`) and null sentinels (`None`, `pd.NA`, `NaN`, `NaT`), with parity checks plus focused Cypher lowering regressions validated on RAPIDS 25.02 and 26.02 (#1219, #1223).
- **GFQL / oracle parity**: Extended reference enumerator (`graphistry/gfql/ref/enumerator.py`) to support multi-hop edge aliases — previously raised `ValueError`; oracle validates node/edge membership only so the engine's bare presence-marker column requires no special handling. Cleared `TestOracleLimitations` xfail; added `TestMultiHopEdgeAlias` with three parity tests. Added `_MIN_PARITY_CASES` count guard in `test_enumerator_parity.py` so future chain-kernel features must extend the parity suite or CI fails at import time. Added ref suite (`tests/gfql/ref/`) to `test-gfql-core` and `test-pandas-compat-gfql` CI jobs so oracle drift is caught automatically (#1086).
- **GFQL / Cypher**: Multi-row scalar prefix reentry — 13 new/updated tests: two-tag fanout, carried scalar visible in RETURN, empty prefix, single-row regression, bag-semantics duplicate-seed (explicit 4-row assertion), partial-hit (some prefix rows empty), partial-hit zero-contribution has no null rows, DISTINCT collapses duplicate prefix rows to single-row path, single-row + OPTIONAL MATCH still works (guard only fires for N>1), optional_reentry + multi-row guard raises `GFQLValidationError` specifically, empty base graph with multi-row prefix, cuDF path (#1047).
- **GFQL / Cypher**: Parser and execution tests for bounded zero-min variable-length relationships: `*0..N`, typed, multi-type, `*0..1` (seed + direct neighbors), `*0..100` (large bound stops at graph depth), `*0..0` (documents current empty-return behavior), `*0` exact still rejected; negative hop bounds `*-1`, `*-5..10`, `*1..-3` raise `GFQLSyntaxError` specifically (#983).
- **Tests / DRY**: Extracted `_mk_prefix_scalar_reentry_data()` and `_mk_multi_row_scalar_prefix_data()` helpers; pandas and cuDF factory twins now delegate to the shared data function eliminating duplicated DataFrame literals.
- **GFQL / Cypher**: Added 19 connected `MATCH + OPTIONAL MATCH` regression tests covering expression breadth (`type()`, `coalesce()`, arithmetic, `CASE WHEN ... IS NULL`), join edge cases (no matches, all match, multi-row, empty base, two shared aliases, integer IDs, custom node column, longer optional chains), and post-projection ops (ORDER BY, SKIP, LIMIT, DISTINCT) (#996).
- **GFQL / Cypher**: Added IS7-shape regression test for `CASE r WHEN null THEN false ELSE true END` over a connected MATCH + OPTIONAL MATCH with edge alias, covering both matching and non-matching OPTIONAL arms (#996).
- **GFQL / cudf**: 23 new tests for the `safe_map_series` cudf SIGSEGV fix (#977): unit tests for all mapping types (dict, `pd.Series`, `cudf.Series`, cudf+`pd.Series` path), edge cases (missing keys, empty mapping, NaN values, non-default index, mixed value types); hop integration tests (reverse, undirected, node-only, edge-only, empty result with edge hops, multi-hop ordering, missing-mask guard); chain integration test; Cypher label-filter, single-hop, and ORDER BY regression tests; DGX smoke script validating both pandas and cudf paths on RAPIDS 25.02 and 26.02 hardware (#977).
- **GFQL / chain**: Regression test — `safe_merge()` does not mutate the caller's right DataFrame (#892).
- **GFQL / chain**: Regression test — chain hop tag columns emit no `FutureWarning` on `fillna` (#881).
- **GFQL / schema**: Regression test — filtering on `bool` `label__*` columns does not raise `GFQLSchemaError` (#876).

## [0.54.0 - 2026-04-04]

### Added
- **Layout / treemap**: Built-in squarified treemap layout implementation (`graphistry/layout/gib/_squarify.py`); removes external layout dependency from `install_requires`. Coordinate transforms in `partitioned_layout.py` are now vectorized via DataFrame merge instead of per-row dict lookups (~3× faster CPU baseline). GPU path uses engine-native DataFrame throughout. 353 unit tests added (#1059).
- **GFQL / Cypher**: Support multi-alias `WITH DISTINCT` projections from connected MATCH patterns, including aggregation (e.g., `WITH DISTINCT tag, post RETURN tag.name, count(post)`). Bare alias references in row expressions now resolve to the alias-prefixed identity column on bindings-row tables (#880).
- **GFQL / Cypher**: Support multi-stage WITH chains on bindings-row tables (e.g., `WITH DISTINCT tag, post WITH tag, post.creationDate AS cd RETURN tag.name, sum(cd)`). Mixed whole-row + scalar WITH projections now use `extend` mode to add scalar columns without dropping alias-prefixed bindings columns (#880).
- **GFQL / Cypher**: Support connected `MATCH ... OPTIONAL MATCH ... RETURN` queries where the non-optional MATCH is a connected path (not just a single node). The compiler lowers each clause independently, left-outer-joins the binding-row tables on shared node aliases, and delegates RETURN / ORDER BY / SKIP / LIMIT to the standard row pipeline. This unblocks LDBC SNB Interactive `interactive-short-7` (IS7) and any query shape that combines a connected traversal with an optional extension and mixed-alias or `CASE`-expression projections (#996).
- **GFQL / Cypher**: Support `WHERE` clauses scoped to `OPTIONAL MATCH`, including label predicates and property comparisons. Each `WHERE` is associated with its preceding `MATCH` clause at parse time via `MatchClause.where` (#1024).
- **GFQL / Cypher**: Support multiple `OPTIONAL MATCH` clauses (1 non-optional + N optional) via chained left-outer-joins. Queries like `MATCH (a)-->(b) OPTIONAL MATCH (b)-->(c) OPTIONAL MATCH (c)-->(d) RETURN ...` now execute correctly (#1025).
- **GFQL / Cypher**: Support direct local Cypher `shortestPath(...)` scalar execution for the benchmark-facing subset, including `length(path)`, `path IS NULL`, `CASE path IS NULL WHEN true THEN -1 ELSE length(path) END`, and the official comma-seeded `interactive-complex-13` shape. Generic path-carrier projections and `allShortestPaths(...)` remain explicit fail-fast boundaries (#1010).

### Fixed
- **GFQL / Cypher (#1038, #1261 slices S1-S5)**: Hardened RETURN-side `CASE` handling in the local Cypher path for IC4-shaped query forms. Added a lowering fail-fast guard so aggregates nested inside row CASE expressions are rejected at compile-time with `E108` (instead of leaking to runtime row-expression failures), and stabilized regression-lock expectations across pandas compatibility lanes by asserting null-like semantics (`None`/`NaN`) where appropriate.
- **GFQL / Cypher**: Non-final `WITH alias, agg()` aggregate stages on bindings-row tables (e.g., `WITH tag, sum(cd) AS total`) now correctly group per alias and preserve `alias.*` property columns for subsequent stages. Previously the group key used the wrong column name (`id` instead of `tag.id`), the entity-blob serializer made every row unique, and property columns like `tag.name` were dropped before the next `RETURN` stage could access them (#1054).
- **GFQL / Cypher**: Extended scalar columns from a bindings-row `WITH` stage (e.g., `WITH tag, post.creationDate AS cd`) are now visible in subsequent non-aggregate stages (`WITH cd ... RETURN cd`). Previously, the next stage resolved the projected column name as an alias-qualified property path and prepended the active alias a second time, producing `None` instead of the actual scalar value (#1045).

- **GFQL / Cypher**: `MATCH (x) WITH x OPTIONAL MATCH (x)-->(y) RETURN ...` now null-fills unmatched rows (left-outer-join semantics) instead of silently dropping them. Previously produced wrong results with inner-join semantics (#1026).

- **GFQL / bindings rows**: `rows(binding_ops=...)` on `engine="cudf"` now preserves active-engine empty/intermediate tables instead of leaking pandas objects into the GPU path, and undirected multihop no-backtrack masking no longer uses a mixed-type sentinel that cuDF rejects. This fixes the `#1040` IC6 cudf/GPU failure class and the adjacent direct multihop/cartesian bindings-row replay cases.
- **GFQL / Cypher**: Support `WITH scalar, collect(alias) AS list UNWIND list AS alias MATCH ... RETURN` queries where carried scalars accompany the `collect()`. Previously only the single-item `WITH collect(alias) AS list` shape was supported (#1000).
- **GFQL / Cypher**: Pattern property maps in direct local Cypher now accept expression-valued entries such as `MATCH (a)-[:R]->(b {num: a.num})` and reentry-carried identifiers such as `MATCH (b {id: bid})`. Literal and parameter-valued property maps keep their existing lowering path.
- **GFQL / Cypher**: Alternating reentry shapes like `MATCH ... WITH ... MATCH ... WHERE ... WITH ... MATCH ... RETURN` are now supported in the local Cypher compiler. This moves the exact `#1000` IC6 query past the earlier post-reentry-`WHERE` barrier and up to the next `UNWIND after post-WITH MATCH` blocker.
- **GFQL / Cypher**: Alternating reentry shapes now preserve one `WHERE` per post-`WITH MATCH` stage, so queries like `MATCH ... WITH ... MATCH ... WHERE ... WITH ... MATCH ... WHERE ... WITH ... RETURN` no longer fail on the earlier single-global-`WHERE` compiler limit (#1000).
- **GFQL / Cypher**: Bounded reentry now supports scalar-only prefix `WITH ... MATCH ...` continuation for single-row prefix shapes, so queries like `MATCH (t) WITH t.id AS tid MATCH (p)-[:R]->(u {id: tid}) RETURN ...` no longer require carrying a whole-row alias through the prefix `WITH` stage (#1000).
- **GFQL / Cypher**: Connected comma-pattern row execution now lowers through joined bindings rows in the local compiler, so non-linear fanout shapes and grouped overlap cases like `MATCH (p)<-[:R]-(x), (x)-[:S]->(a), (x)-[:S]->(b)` and `RETURN a.id, count(b)` can execute through downstream row stages instead of failing on the older single-linear-path / single-source-row limitations (#1000).
- **GFQL / Cypher**: Exact official `interactive-complex-6` / `tag-cooccurrence` Cypher now runs in the local compiler. The remaining `#1000` gaps were cleared by attaching nested scalar-prefix reentry to the terminal consumer stage and by allowing joined-row multihop bindings to carry undirected, branching endpoint rows without dropping bare alias ids needed by row filters (#1000).
- **GFQL / bindings rows**: Native `rows()` and direct `rows(binding_ops=...)` now preserve open-range / fixed-point edge semantics during bindings serialization instead of collapsing those segments back to a single hop. This restores IS6-style multihop continuation row shaping from native GFQL chains under `#880`.
- **GFQL / Cypher**: Removed guard that rejected multi-hop connected patterns with edge alias projections (e.g., `MATCH (a)-[r:R]->(b)-[s:S]->(c) RETURN r.since, c.id`). The runtime bindings table already handles this correctly (#880).
- **GFQL / Cypher**: Direct local Cypher now supports comma-separated node-only `MATCH` cartesian products such as `MATCH (n), (m) RETURN n.num, m.num`, including cross-alias row filters, grouped/global aggregates, and `WITH`-staged row execution. This unlocks the `#990` prerequisite lane for `#1010`.
- **Tests / RAPIDS compat**: Replace `cudf.DataFrame.from_pandas()` with `cudf.from_pandas()` across GFQL Cypher test suite for RAPIDS 25.02 + 26.02 compatibility.

### Tests
- **GFQL / Cypher (#1038, #1261 slices S1-S5)**: Added/expanded regression coverage in `test_lowering.py`, `test_parser.py`, and `test_binder.py` for IC4-style RETURN-side CASE, parser rejection of malformed CASE, binder scope/name-resolution behavior for CASE after `WITH DISTINCT`, and explicit rejection of aggregate calls nested in row CASE expressions.
- **GFQL / bindings rows**: Added benchmark-shaped regressions for native IS6-style multihop continuation plus direct Cypher IS1 / IS3 / IS6 projection shapes.
- **GFQL / Cypher**: Added cartesian `MATCH` regressions covering scalar projection, non-simple row expressions, grouped/global aggregates, staged `WITH` filters, and direct `rows(binding_ops=[Node, Node])` cartesian row materialization.
- **GFQL / Cypher**: Added exact IC6 runtime regression coverage plus multihop joined-row regressions for undirected no-backtracking, branching reentry fanout, and direct `rows(binding_ops=...)` bare-alias row expressions (#1000).

## [0.53.16 - 2026-04-01]

### Added
- **GFQL / native chain**: Support connected multi-alias bindings-table materialization from native GFQL chains ending in bare `rows()`. Named traversal chains like `g.gfql([n(name="a"), e_forward(), n(name="b"), rows()])` now reuse connected `binding_ops` row materialization and expose alias-prefixed columns such as `a.id`, `b.id`, and edge alias properties (#880).

### Fixed
- **GFQL / bindings rows**: Reject duplicate aliases both in legacy named-chain execution and in direct `rows(binding_ops=...)` execution instead of silently overwriting labels or producing corrupted duplicate-prefixed row output.
- **GFQL / bindings rows**: Missing alias-prefixed properties on bindings tables now project as null in row expressions instead of failing with the generic `unsupported token in row expression` error. This aligns native-chain `rows() ... select()` behavior with the existing direct Cypher multi-alias projection path.

### Tests
- **GFQL / bindings rows**: Added regression coverage for direct `binding_ops` duplicate aliases and for native-chain injection guards that must preserve explicit `rows(source=...)`, `rows(alias_endpoints=...)`, `rows(binding_ops=...)`, and non-traversal middle segments.

## [0.53.15 - 2026-04-01]

### Added
- **GFQL / Cypher**: `shortestPath()` and `allShortestPaths()` syntax now parses and raises a clear "not yet supported" validation error instead of a generic syntax error (#997).
- **GFQL / Cypher**: Pattern existence expressions (`not((a)-[:R]-(b))`, `exists { ... }`) now detected and raise a clear validation error instead of a generic syntax error (#998).

### Fixed
- **GFQL / Cypher**: Fixed undirected MATCH returning the seed node instead of the peer when the stored edge is incoming (#994). The connected bindings row materializer now correctly orients undirected edges in both directions.

## [0.53.14 - 2026-04-01]

### Added
- **GFQL / Cypher**: Support bounded alternating multi-stage reentry read shapes in direct local Cypher, including `MATCH ... WITH ... MATCH ... WITH ... MATCH ... RETURN` for connected vectorized pipelines, plus nested reentry runtime composition and bindings-backed whole-row/scalar projection handling needed by this shape (#999).

## [0.53.13 - 2026-04-01]

### Added
- **GFQL / Cypher**: Support connected multi-alias row bindings for direct Cypher scalar projections and bounded `WITH ... MATCH` reentry. Connected single-path and connected multi-pattern shapes like `MATCH (a)-[:R]->(b), (b)-[:S]->(c) RETURN a.id, c.id` and the benchmark-facing `WITH ... MATCH` continuation behind `interactive-short-2` now materialize row bindings correctly while branching multihop and connected relationship-alias projection shapes remain explicit fail-fast boundaries.

## [0.53.12 - 2026-04-01]

### Added
- **GFQL / Cypher**: Support variable-length relationships in connected multi-relationship patterns. `MATCH (a)-[:R*2]->()-[:R]->(c) RETURN c` now works — variable-length hops at any position (start, middle, end) are supported with exact, range, and open-range forms (#973).

### Infra
- **CI**: Bump `test-minimal-python` timeout from 8 to 12 minutes (Python 3.14 suite exceeds 8 minutes on hosted runners).

## [0.53.11 - 2026-03-31]

### Fixed
- **GFQL / Cypher**: Added direct local Cypher support for the narrow graph-backed `MATCH ... With collect([DISTINCT] alias) AS list UNWIND list AS alias MATCH ... RETURN` reentry shape, moving those queries past the earlier parser rejection while preserving explicit fail-fast behavior for older unsupported multi-alias row-scope cases.

## [0.53.10 - 2026-03-31]

### Added
- **GFQL / Cypher**: Added bounded direct local Cypher reentry support for the vectorized same-alias `MATCH ... WITH ... MATCH ...` subset, including carried scalar projections and trailing `RETURN` / `ORDER BY` use on the carried alias.

### Fixed
- **GFQL / Cypher**: Unsupported cross-alias, fresh row-seeded, and prefix-order-dependent bounded reentry shapes now fail fast instead of silently miscompiling.

### Tests
- **GFQL / Cypher / cuDF**: Added pandas and cuDF regression coverage for bounded reentry at both the helper and end-to-end lowering layers, including targeted DGX validation on official RAPIDS `26.02` `cuda13`.

## [0.53.9 - 2026-03-31]

### Fixed
- **GFQL / GPU traversal**: Added a narrow one-hop undirected `hop()` fast path that avoids doubled edge-pair materialization for the common no-predicate traversal shape. On DGX-backed RAPIDS validation, warm `gplus` pipeline time improved `-39.67%` on `25.02` and `-39.27%` on `26.02`.

### Added
- **GFQL / Cypher**: Support multi-alias scalar `RETURN` projections in direct Cypher queries. `MATCH (a)-[:R]->(b) RETURN a.id AS a_id, b.id AS b_id` now works by building a bindings table from edges joined with node properties (#981).
- **GFQL / Cypher**: Edge alias property access in multi-alias `RETURN`. `MATCH (a)-[r:KNOWS]->(b) RETURN a.id, r.creationDate, b.firstName` now works — edge properties are accessible alongside node properties (#982).

## [0.53.8 - 2026-03-31]

### Added
- **GFQL / Cypher**: Support `*0..`, `*1..`, and other open-max variable-length relationship ranges in direct Cypher queries. For example, `MATCH (m)-[:REPLY_OF*0..]->(p) RETURN p` now parses and executes correctly, matching zero or more hops (#983).

### Docs
- **GFQL**: Show Cypher string syntax above the fold in "10 Minutes to GFQL" and "Overview" pages so the first code a reader sees is familiar Cypher, not the native chain API.

### Infra
- **CI**: Make GFQL doc example test CI-blocking instead of warning-only. Broken doc examples now fail the `test-docs` job.

## [0.53.7 - 2026-03-29]

### Fixed
- **GFQL docs**: Fixed 29 broken code examples across 12 doc files — wrong API params (`n(edge_match=...)`→`e_forward(edge_match=...)`), wrong imports (`remote`, `Chain`, `eq`), nonexistent classes (`PlottableValidator`, `is_not_in`), and wrong param names (`col_in`/`col_out`, `entity_cols`).
- **GFQL / Let**: Fixed nested `let()` inside `let()` failing at execution time despite passing `validate()`. The dependency resolver now treats nested `ASTLet` bindings as opaque execution units instead of walking into their internal bindings, which caused false "references undefined nodes" errors (#968).

### Tests
- **GFQL doc examples**: Added automated test harness (`docs/test_doc_examples.py`) that extracts and runs code examples from all GFQL doc files. Runs in CI via the docs docker build. Uses `.. doc-test: skip` / `xfail` markers for blocks that need special handling. 33 skip + 23 xfail markers on genuinely untestable blocks.
- **GFQL / Let**: Fixed nested let runtime scope isolation. Inner let bindings no longer leak into the outer scope's `ExecutionContext`. Added `child_context()` with read-through (lexical closure) and write-local semantics. Fixes name collisions between sibling inner lets and shadowing corruption (#968).
- **GFQL / Let**: Fixed nested let receiving the accumulated result instead of the original graph. Inner lets now filter from the outer scope's original graph independently, matching the behavior of Chain/Node bindings (#968).

### Docs
- **GFQL**: Documented nested let lexical scope rules (read-through, write-local, shadowing, sibling isolation) in wire protocol spec and LLM guide.

### Added
- **Docker / RAPIDS**: Added `docker/test-rapids-official-matrix.sh` to run the official RAPIDS compatibility matrix sequentially across `25.02` and `26.02` for `basic`, `gfql`, and `ai`.

### Changed
- **Docker / RAPIDS AI**: Official RAPIDS `ai` validation now preinstalls `torch==2.11.0+cpu` from the PyTorch CPU index before `-e .[test,testai,ai]`, preventing pip from layering CUDA 13 Torch wheels onto CUDA 12 RAPIDS images.

### Tests
- **Docker / RAPIDS matrix**: Validated no-GPU official-image cells for `25.02/basic`, `25.02/gfql`, `25.02/ai`, `26.02/basic`, `26.02/gfql`, and `26.02/ai`, using smoke imports of `cudf`, `cugraph`, `cuml`, `dask_cudf`, and `graphistry` plus focused `basic`, `gfql`, and `ai` test slices.

## [0.53.6 - 2026-03-28]

### Fixed
- **RAPIDS 26.02 compatibility**: Added backward-compatible shim in `lazy_cudf_import()` restoring `cudf.DataFrame.from_pandas()` and `cudf.Series.from_pandas()` class methods removed in RAPIDS 26.02, so all existing code paths and downstream users keep working across RAPIDS 24.12–26.02+.
- **RAPIDS 26.02 compatibility**: Fixed pre-existing bug where `calc_core_sample_indices=True` was passed to `cuml.DBSCAN.fit()` (never a valid `fit()` parameter on any cuml version — it is an `__init__()` parameter only).
- **RAPIDS 26.02 compatibility**: Added fallback for `cugraph.jaccard_w`, `cugraph.overlap_w`, and `cugraph.sorensen_w` weighted similarity functions removed in cugraph 24.04 — automatically reroutes to base algorithm with `use_weight=True` when unavailable, native call on older versions.
- **GFQL**: Fixed timezone-aware temporal comparisons (GT, LT, GE, LE, EQ, NE) crashing on cudf with `NotImplementedError: Binary operations with timezone aware operands is not supported`. Values are now compared as naive timestamps after tz conversion. Also fixed `s.dt.tz` attribute access that was missing on cudf < 26.02.
- **TigerGraph**: Replaced removed `pandas.Series.append()` with `pd.concat()` for pandas 2.0+ compatibility.

### Tests
- **Hypergraph**: Fixed contradictory dtype assertion in `honeypot_pdf()` test helper where datetime-parsed columns were asserted as both `float64` and `datetime64`.

## [0.53.5 - 2026-03-17]

### Fixed
- **GFQL**: Fixed `g.gfql()` rejecting pre-serialized Let dict envelopes (`{"type": "Let", "bindings": {...}}`). The dict dispatch now recognizes typed Let envelopes and deserializes via `ASTLet.from_json()` instead of misinterpreting them as bare binding dicts. This unblocks the ETL server from passing `gfql_query` Let payloads to `g.gfql()`.
- **GFQL / Remote**: Fixed `_step_to_json` emitting `"ChainRef"` instead of `"Ref"` for wire-protocol Ref entries. The wire type has always been `"Ref"` — removed the spurious `"ChainRef"` alias from `from_json()` and all documentation.

### Added
- **GFQL / Remote**: `gfql_remote()` now accepts `output` parameter for selecting which Let/DAG binding to return (sent as `gfql_output` in the request body).

## [0.53.4 - 2026-03-17]

### Fixed
- **GFQL / Remote**: Fixed `gfql_remote()` silently dropping WHERE clauses — queries with same-path constraints (e.g., `WHERE a.x = b.y`) now send the full Chain envelope via a new `gfql_query` request field. The existing `gfql_operations` flat array is still sent for backward compatibility with older servers.

### Added
- **GFQL / Remote**: `gfql_remote()` now accepts ASTLet/Let dict input for DAG queries, serialized as `{"type": "Let", ...}` in the `gfql_query` field.
- **GFQL / Remote**: `gfql_remote()` now accepts Cypher strings (compiled locally, sent as Chain or Let wire format). Supports `MATCH ... RETURN`, `GRAPH { ... }`, and `GRAPH g = ... USE g ...` forms.
- **GFQL / Remote**: `gfql_remote()` now accepts `params` for parameterized Cypher queries (e.g., `params={"cutoff": 10}` for `$cutoff` references).

### Docs
- **GFQL / Remote docs**: Updated remote mode guide with Cypher string, GRAPH constructor, multi-stage pipeline, and params examples.

## [0.53.3 - 2026-03-16]

### Docs
- **GFQL / Cypher Benchmark**: Fixed double-brace rendering in docs code example by replacing f-string with plain string.

## [0.53.2 - 2026-03-16]

### Docs
- **GFQL**: Added GFQL mascot to key docs pages (10 Minutes to GFQL, Overview, Cypher Benchmark).
- **GFQL / Cypher Benchmark**: Added GPU DataFrame ecosystem mention (Apache Arrow, NVIDIA RAPIDS / cuDF, cugraph) to benchmark notebook.

## [0.53.1 - 2026-03-16]

### Changed
- **GFQL / Cypher Benchmark**: Rewrote benchmark to use compound ``GRAPH { }`` pipeline (single ``g.gfql(...)`` call), retitled to "GFQL Cypher Benchmark: CPU/GPU DataFrames vs Neo4j", added Neo4j ETL timing, tightened to 2 lifecycle charts (ETL/Search/Analytics stacked bars, Neo4j → CPU → GPU ordering), and reran all benchmarks with warmup=2 runs=5. GPlus: 22.7x GPU vs CPU; Twitter 3-way: GPU 46x faster than Neo4j.

### Docs
- **GFQL / translate guide**: Added Cypher-string and ``GRAPH { }`` examples to the SQL/Pandas/Cypher/GFQL translation guide.
- **GFQL / Cypher Benchmark docs**: Added "Why the GFQL pipeline is shorter" design callout covering first-class graph values, multi-language single engine, and modern columnar execution.

## [0.53.0 - 2026-03-16]

### Added
- **GFQL / Cypher**: Added `GRAPH { MATCH ... }` graph constructors, `GRAPH g = GRAPH { ... }` named bindings, and `USE g` scoped graph switching as GFQL extensions to Cypher. These replace the earlier `RETURN GRAPH` syntax with a design aligned with GQL's deferred graph constructor direction and G-CORE's composable graph-in/graph-out vision. Graph constructors keep query results in graph state instead of flattening to rows, enabling multi-stage graph pipelines.
- **GFQL / Cypher**: Added `CALL graphistry.*.write()` support inside `GRAPH { }` constructors, enabling single-expression graph pipelines that chain search, enrichment, and analytics.
- **Benchmarks**: Added end-to-end GFQL CPU/GPU/Neo4j benchmark suite under `benchmarks/gfql/filter_pagerank/` with a filter → PageRank → filter pipeline on SNAP social graphs. Includes benchmark scripts, RTD docs page, presentation notebook, and rendered SVG charts. GPlus warm speedup: 25x GPU vs CPU; Twitter 3-way: GPU 54x faster than Neo4j.

### Changed
- **GFQL / Cypher**: Replaced `RETURN GRAPH` with `GRAPH { }` constructors to avoid overloading `RETURN` (row projection) with graph semantics.

### Docs
- **GFQL / Cypher docs**: Documented `GRAPH { }` constructors, `GRAPH g = ...` bindings, and `USE g` graph switching as GFQL extensions aligned with the GQL/G-CORE direction.
- **GFQL / translate guide**: Added Cypher-string and `GRAPH { }` examples to the SQL/Pandas/Cypher/GFQL translation guide, showing `g.gfql("MATCH ...")` alongside native chain syntax for each translation pattern.
- **GFQL / Benchmark docs**: Added story-first RTD benchmark page at `docs/source/gfql/benchmark_filter_pagerank.rst` with comparison charts for Graphistry GPU, Graphistry CPU, and Neo4j across Twitter and GPlus datasets.

## [0.52.0 - 2026-03-15]

### Added
- **GFQL / Cypher**: Added direct local Cypher multihop support for single variable-length relationship endpoint traversals through `g.gfql("MATCH ...")`, including `[*n]`, `[*m..n]`, and `[*]` in forward, reverse, undirected, and typed forms.

### Fixed
- **GFQL / Cypher WHERE lowering**: Supported direct local Cypher queries that combine one positive `WHERE` pattern predicate with ordinary row filters through top-level `AND`, including cases where the pattern predicate appears at either end or in the middle of the conjunction chain.
- **GFQL / Cypher validation**: Tightened direct local Cypher fail-fast boundaries for unsupported variable-length subfamilies, including path/list-carrier uses of relationship aliases, named path alias projections like `RETURN p` / `length(p)` / `relationships(p)`, exact/bounded `WHERE` pattern predicates, connected patterns that mix variable-length and standard relationships, and unsupported multi-alias `RETURN *` projections.
- **GFQL / Cypher pandas parity**: Normalized direct local Cypher row semantics across pandas 2/3 for stringified-list subscripts and string `min` / `max` aggregations with nulls, removing the last sibling TCK contract split on the current supported surface.
- **GFQL / multihop semantics**: Tightened undirected fixed-point wave-front output semantics so local Cypher `-[*]-` endpoint queries and the underlying GFQL/hop runtime exclude trivial seed backtracking while still keeping seeds that are rediscovered through a real cycle or another seed.

### Docs
- **GFQL / Cypher docs**: Clarified the current direct `g.gfql("MATCH ...")` multihop support boundary: endpoint-only `[*n]`, `[*m..n]`, and `[*]` relationship patterns are supported, while path-carrier and mixed-pattern residuals remain explicit validation failures.

## [0.52.1 - 2026-03-15]

### Added
- **GFQL / Cypher**: Added shared-registry-backed local Cypher `CALL graphistry.degree`, `CALL graphistry.igraph.<alg>`, and `CALL graphistry.cugraph.<alg>` procedure families. Bare calls stay row-returning for supported node/edge algorithms, while `.write()` preserves graph state and also covers topology-returning algorithms. The branch also keeps a smaller `graphistry.nx.*` compatibility subset on the same row/write contract: `pagerank`, `betweenness_centrality`, `edge_betweenness_centrality`, and `k_core.write()`.

### Fixed
- **GFQL / Cypher cugraph CALL parity**: Aligned local `CALL graphistry.cugraph.*` row/write output naming with real `compute_cugraph()` behavior for edge algorithms and multi-column node algorithms, and added backend-backed regression coverage for representative cuGraph and igraph CALL paths.

### Docs
- **GFQL docs**: Documented the local graph-preserving `CALL graphistry.*.write()` subset, clarified that omitting `.write()` keeps row-returning behavior, and updated enrich-then-match examples to show when local Cypher stays in graph state versus when `CALL ... YIELD ... RETURN ...` moves into row state.

## [0.51.3 - 2026-03-14]

### Docs
- **GFQL docs**: Clarified that `g.gfql_remote([...])` is the remote GFQL path for larger datasets and remote GPU execution, while `graphistry.cypher(...)` / `g.cypher(...)` is a separate remote database Cypher integration rather than the GFQL execution surface.
- **GFQL / Cypher docs**: Added a dedicated guide and helper reference for Cypher syntax through `g.gfql("MATCH ...")`, plus updated spec/reference docs to clarify the bound-graph execution path, helper APIs, and the current translation boundaries.

## [0.51.2 - 2026-03-13]

### Fixed
- **GFQL / Cypher validation**: Hardened local Cypher fail-fast handling for additional unsupported read-only query shapes so valid-but-unsupported queries raise `GFQLValidationError` instead of surfacing syntax errors, runtime errors, or wrong rows.

## [0.51.1 - 2026-03-12]

### Added
- **GFQL / Cypher**: Added local Cypher-string execution through `g.gfql(query, params=...)`. String queries that look like Cypher now execute locally through GFQL; remote `g.cypher(...)` semantics stay unchanged.
- **GFQL / Cypher compiler**: Added the `graphistry.compute.gfql.cypher` parser/compiler surface, including `parse_cypher`, `compile_cypher`, `cypher_to_gfql`, and `gfql_from_cypher`.
- **GFQL / Cypher runtime**: Added local lowering/execution support for the current Cypher fragment, including `MATCH` / `OPTIONAL MATCH`, `WHERE`, `WITH`, `RETURN`, `ORDER BY`, `SKIP`, `LIMIT`, `UNWIND`, `UNION`, and `CALL graphistry.*`.
- **GFQL / Row pipeline**: Added and expanded the row-pipeline operators used by Cypher lowering: `rows()`, `where_rows()`, `return_()`, `with_()`, `order_by()`, `skip()`, `limit()`, `distinct()`, `unwind()`, and `group_by()`.
- **GFQL / Row results**: Added whole-row entity projection helpers for Cypher node/edge/map outputs plus temporal/entity text normalization.

### Fixed
- **GFQL / Cypher validation/runtime parity**: Hardened lowering and row-pipeline boundaries so unsupported Cypher/query shapes fail fast with validation errors instead of falling through to unsupported execution paths.
- **GFQL / Cypher compatibility**: Fixed local Cypher execution across pandas 2/3 and Python 3.8-3.14 for temporal constructor rendering/parsing, whole-row projection text normalization, and Arrow-string null RHS handling in string predicates.
- **GFQL / Cypher aggregates**: Reject unsound aggregate multiplicity query shapes during local Cypher lowering instead of returning incorrect results.
- **GFQL / cuDF row projections**: Preserve list/map/entity row projections through pandas<->cuDF fallback paths, harden schema validation and struct/map access, and keep RAPIDS CUDA 13 execution aligned with pandas for supported local string-Cypher queries.

### Tests
- **GFQL / Cypher**: Added parser, lowering, execution, procedure-call, UNION, temporal, whole-row projection, and compatibility coverage for local Cypher execution.
- **GFQL / Row pipeline**: Expanded unit coverage for projection aliasing, ordering/grouping semantics, DISTINCT/UNWIND flows, and parser/precedence regressions across pure-vector execution paths.
- **GFQL / cuDF**: Added RAPIDS-focused regression coverage for schema validation, row projection precedence/list cases, and row-table preservation, keeping the supported string-Cypher TCK slice green on the latest GPU validation flow.

### Infra
- **Docker / GPU tests**: Added an opt-in RAPIDS CUDA 13 GPU validation path for local/DGX cuDF testing while keeping the default Graphistry GPU image flow unchanged.
- **CI**: Scoped the `docs_only_latest` optimization to push workflows so shallow PR refs do not suppress required jobs.

## [0.51.0 - 2026-03-02]

### Added
- **GEXF**: Added GEXF import/export with viz attribute bindings (color/size/position/thickness/opacity), validation, tests, and demo notebook.
- **GEXF**: Map node viz shapes to FA4 point icons on import.
- **GFQL / WHERE** (experimental): Added `Chain.where` field for same-path WHERE clause constraints. New modules: `same_path_types.py`, `df_executor.py`, and `same_path/` submodules implementing Yannakakis-style semijoin reduction for efficient WHERE filtering. Supports equality, inequality, and comparison operators on named alias columns.
- **GFQL / WHERE**: `gfql([...], where=[...])` list form now supports same-path WHERE constraints (no need to wrap in `Chain(...)`).
- **GFQL / cuDF same-path**: Added execution-mode gate `GRAPHISTRY_CUDF_SAME_PATH_MODE` (auto/oracle/strict) for GFQL cuDF same-path executor. Auto falls back to oracle when GPU unavailable; strict requires cuDF or raises.
- **GFQL / WHERE**: Added opt-in `GRAPHISTRY_NON_ADJ_WHERE_MULTI_EQ_SEMIJOIN` for multi-equality semijoin pruning (2-hop, experimental).
- **GFQL / WHERE**: Added opt-in `GRAPHISTRY_NON_ADJ_WHERE_INEQ_AGG` for aggregated inequality pruning on 2-hop non-adj clauses (experimental).
- **Telemetry**: Added optional OpenTelemetry helper and propagated trace headers through plot/upload/remote GFQL paths.

### Fixed
- **GFQL / chain**: Fixed `from_json` to validate `where` field type before casting, preventing type errors on malformed input.
- **GFQL / WHERE**: Fixed undirected edge handling in WHERE clause filtering to check both src→dst and dst→src directions.
- **GFQL / WHERE**: Fixed multi-hop path edge retention to keep all edges in valid paths, not just terminal edges.
- **GFQL / WHERE**: Fixed unfiltered start node handling with multi-hop edges in native path executor.
- **GFQL / WHERE**: Fixed vector-strategy guard to initialize start/end domains before pair-est gating (prevents UnboundLocalError).

### Performance
- **Compute / hop**: Optimized hop traversal with simplified domain operations and reduced redundant checks (8-35% faster across scenarios).
- **GFQL / WHERE**: Use DF-native forward pruning for cuDF equality constraints to avoid host syncs (pandas path unchanged).
- **GFQL / WHERE**: Default non-adjacent WHERE mode now `auto`, enabling value-mode + domain semijoin auto, with edge semijoin auto for edge clauses (opt-out via env).
- **GFQL / WHERE**: Auto mode skips value-mode on multi-clause non-adjacent WHERE when pair estimates exceed the semijoin threshold (guardrail against blowups).
- **GFQL / WHERE**: Avoid building semijoin pair tables when AUTO semijoin stays inactive; uses cheap pair estimates to gate work.
- **GFQL / WHERE**: Reduce semijoin dedup overhead and reuse cached edge pairs per edge when `allowed_edges` is unset.

### Tests
- **GFQL / df_executor**: Added comprehensive test suite (core, amplify, patterns, dimension) with 200+ tests covering Yannakakis semijoin, WHERE clause filtering, multi-hop paths, and pandas/cuDF parity.
- **GFQL / cuDF same-path**: Added strict/auto mode coverage for cuDF executor fallback behavior.

### Infra
- **Benchmarks**: Added GFQL benchmark scripts and a CI job that runs a small subset on GFQL/benchmark changes and uploads `gfql-bench.md`.
- **GFQL / same_path**: Modular architecture for WHERE execution: `same_path_types.py` (types), `df_executor.py` (execution), plus `same_path/` submodules for BFS, edge semantics, multihop, and WHERE filtering.
- **GFQL / WHERE**: Added OTel detail counters for semijoin pair sizes and mid-intersection sizes to help diagnose dense multi-clause blowups.
- **Linting**: Replace flake8 with ruff for linting (closes #466). Config in `pyproject.toml`, scripts in `bin/ruff.sh` / `bin/lint.sh`. Cleaned stale `# noqa` comments for W503/W504/E126 (codes not applicable in ruff).

## [0.50.6 - 2026-01-27]

### Fixed
- **GFQL / hypergraph**: Avoid `DataFrame.style` access when `return_as` yields a DataFrame, preventing Jinja2 import errors in minimal environments without Jinja2 (PR #909).

### Tests
- **Temporal**: Added datetime unit parity coverage (ms/us/ns) for ring layouts, GFQL time ring layouts, and temporal comparison predicates; relaxed honeypot hypergraph datetime unit expectations.

## [0.50.5 - 2026-01-25]

### Fixed
- **Predicates / str**: Centralized pandas/cuDF string predicate evaluation and NA handling; case-insensitive match/fullmatch now uses flags to align pandas 3 semantics.
- **Temporal**: Use pandas timezone localization/conversion for GFQL temporal parsing; drop `pytz` runtime dependency.
- **NodeXL**: Require `openpyxl>=3.1.5` in the `nodexl` extra for pandas 3 compatibility.

### Tests
- **Predicates / str**: Added NA=True and empty-tuple NA coverage for pandas and cuDF startswith/endswith/match/fullmatch.
- **Temporal**: Added timezone comparison/parity tests (pandas and cuDF) and ring layout microsecond vs nanosecond equivalence coverage.
- **Bolt util**: Relaxed dtype assertions to accept pandas 2/3 string and datetime unit differences.

### Docs
- **Docs / GFQL**: Updated datetime filtering and wire protocol examples to use pandas tz strings (no `pytz`).
- **Demos**: Removed `pytz` from demo notebooks/HTML.

### Infra
- **CI**: Added pandas 2.2.3/3.0.0 compatibility jobs and minimal suite coverage.

## [0.50.4 - 2026-01-15]

### Fixed
- **Compute / hop (cuDF)**: Hop label tracking could error or force host sync because it used pandas Index conversions; now stays engine-native. Example: `g.chain([n(), e(hops=2, label_node_hops='nh', label_edge_hops='eh', label_seeds=True)])`.

## [0.50.3 - 2026-01-14]

### Fixed
- **GFQL / let-ref**: Documented and tested existing letrec semantics for ref; list bindings now accept implicit Chains.
- **GFQL / schema**: Apply call() schema effects during validation so enrichments like `get_degrees` are recognized by downstream filters, and prioritize boundary-call validation before schema errors.
- **GFQL / filters**: Treat boolean literal filters on object-typed columns as booleans instead of numeric mismatches.

### Tests
- **Tests / GFQL**: Fixed cuDF compatibility in test files by using `to_set()` helper instead of `.tolist()` (cuDF doesn't support `tolist()`)

### Docs
- **Docs / GFQL**: Updated the GFQL remote notebook examples to use valid ref/edge traversals.

## [0.50.2 - 2026-01-11]

### Performance
- **Compute / hop**: Use scalar broadcast `Series(True, index=...)` instead of Python list splatting `Series([True] * len(...), ...)` for efficient GPU-friendly constant initialization
- **Predicates / str**: Use scalar broadcast for constant Series in `startswith`/`endswith` empty tuple edge cases
- **DGL**: Use `np.ones()` instead of `np.array([1] * len(...))` for efficient array initialization

### Fixed
- **Hypergraph**: Fixed engine auto-detection to use input DataFrame type instead of defaulting to cuDF when available
- **GFQL / chain**: Fixed cuDF compatibility in backward pass by removing `set()` wrappers around Series passed to `.isin()` (cuDF `.isin()` works directly with Series)
- **GFQL / chain**: Fixed cuDF compatibility by replacing `.combine_first()` with `.where()` pattern (cuDF lacks `combine_first`)
- **Compute / hop**: Fixed cuDF compatibility by making all operations engine-agnostic: vectorized `.isin()` instead of Python `set()`, engine-aware Series/concat construction, and `s_na(engine)` instead of `pd.NA` (fully GPU-accelerated)

### Infra
- **Engine.py**: Added `s_to_numeric(engine)` and `s_na(engine)` polymorphic utilities for engine-agnostic numeric conversion and null assignment

### Tests
- **GFQL / chain**: Added `engine_mode` parametrized fixture for automatic pandas/cuDF parity testing (enabled via `TEST_CUDF=1`). Chain optimization tests now run 156 tests (78 pandas + 78 cuDF) when GPU is available.

## [0.50.1 - 2026-01-09]

### Added
- **Compute / hop**: `hop()` supports `min_hops`/`max_hops` traversal bounds plus optional hop labels for nodes, edges, and seeds, and post-traversal slicing via `output_min_hops`/`output_max_hops` to keep outputs compact while traversing wider ranges.
- **Docs / hop**: Added bounded-hop walkthrough notebook (`docs/source/gfql/hop_bounds.ipynb`), cheatsheet and GFQL spec updates, and examples showing how to combine hop ranges, labels, and output slicing.
- **GFQL / reference**: Extended the pandas reference enumerator and parity tests to cover hop ranges, labeling, and slicing so GFQL correctness checks include the new traversal shapes.
- **Docs / GFQL**: Documented the external `tck-gfql` conformance harness and local run instructions in GFQL docs.

### Performance
- **GFQL / chain**: Optimized backward pass for simple single-hop edges by skipping full `hop()` call and using vectorized merge filtering instead (~50% faster on small graphs). Added `is_simple_single_hop()` method on `ASTEdge` for optimization eligibility checks.

### Fixed
- **Compute / hop**: Exact-hop traversals now prune branches that do not reach `min_hops`, avoid reapplying min-hop pruning in reverse passes, keep seeds in wavefront outputs, and reuse forward wavefronts when recomputing labels so edge/node hop labels stay aligned (fixes 3-hop branch inclusion issues and mislabeled slices).
- **GFQL / chain**: Fixed `output_min_hops`/`output_max_hops` semantics to correctly slice output nodes/edges matching oracle behavior.
- **GFQL / chain**: Fixed multi-hop detection in `_is_simple_single_hop` to check `to_fixed_point` flag and correctly identify optimization-eligible edges.
- **GFQL / enumerator**: Fixed hop labeling for paths outside `min_hops` range to use shortest path distance instead of enumeration order.
- **Compute / hop**: Fixed `min_hops` goal node calculation to use edge endpoints instead of lossy node merge, ensuring correct branch pruning.

### Tests
- **GFQL / hop**: Expanded `test_compute_hops.py` and GFQL parity suites to assert branch pruning, bounded outputs, label collision handling, and forward/reverse slice behavior.
- **Reference enumerator**: Added oracle parity tests for hop ranges and output slices to guard GFQL integrations.
- **GFQL / chain**: Added 78 tests for backward pass and combine_steps optimizations covering edge cases, direction semantics, hop labels, and multi-step chains.

### Infra
- **Tooling**: `bin/flake8.sh` / `bin/mypy.sh` now require installed tools (no auto-install), honor `FLAKE8_CMD` / `MYPY_CMD` and optional `MYPY_EXTRA_ARGS`; `bin/lint.sh` / `bin/typecheck.sh` resolve via uvx → python -m → bare.
- **CI / typecheck**: Stop forcing `PYTHON_VERSION` for mypy; rely on the job interpreter and `mypy.ini` defaults.
- **CI / GFQL**: Run the external `tck-gfql` conformance harness only when GFQL-related paths change (or on manual/scheduled runs).

## [0.50.0 - 2025-12-24]

### Added
- **Graphviz / plot_static**: Added text outputs (`graphviz-dot`, `mermaid-code`) and image outputs (`graphviz-svg`, `graphviz-png`), plus `graphviz` for any Graphviz format. Honors `reuse_layout` for bound positions, supports optional file outputs, and returns display-ready SVG/Image objects for notebooks; Graphviz render accepts passthrough args/positions for consistent layouts.

### Docs
- **Graphviz guides (RST)**: Updated [10min](https://pygraphistry.readthedocs.io/en/latest/10min.html), [Visualization 10min](https://pygraphistry.readthedocs.io/en/latest/visualization/10min.html), [GFQL overview](https://pygraphistry.readthedocs.io/en/latest/gfql/overview.html), [GFQL about](https://pygraphistry.readthedocs.io/en/latest/gfql/about.html), [Ecosystem](https://pygraphistry.readthedocs.io/en/latest/ecosystem.html), and [Layout catalog](https://pygraphistry.readthedocs.io/en/latest/visualization/layout/catalog.html) with static Graphviz examples, DOT/Mermaid text output snippets where relevant, and improved diagrams.
- **Graphviz reference (RST)**: Expanded [Graphviz plugin docs](https://pygraphistry.readthedocs.io/en/latest/api/plugins/compute/graphviz.html) with static rendering guidance.
- **Graphviz notebooks (ipynb)**: Refreshed outputs in [Graphviz demo](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/graphviz/graphviz.html), [Static rendering](https://pygraphistry.readthedocs.io/en/latest/demos/demos_databases_apis/graphviz/static_rendering.html), and [Hop/Chain graph pattern mining](https://pygraphistry.readthedocs.io/en/latest/demos/more_examples/graphistry_features/hop_and_chain_graph_pattern_mining.html).
- **Docs tooling**: Enabled `sphinx.ext.graphviz`, allowed rstcheck `graphviz` directive, and installed graphviz/pygraphviz/gcc in the docs Docker image; `docs/ci.sh` now builds only the sphinx service for CI.
- **Related MyST/Markdown**: GFQL spec docs remain at [GFQL spec index](https://pygraphistry.readthedocs.io/en/latest/gfql/spec/index.html).
- **RTD**: Fixed graphviz on Read the Docs by switching from `build.commands` to `build.jobs` (RTD's `apt_packages` is incompatible with `commands`). Docs now fail fast with clear error if `dot` is missing.

### Tests
- **Graphviz**: Added coverage for `plot_static` DOT/Mermaid engines and position reuse.

## [0.48.0 - 2025-12-20]

### Added
- **Arrow:** New `to_arrow()` public method for debugging DataFrame->Arrow conversion issues (#867).
- **Validation:** New `validate` parameter for `plot()` and `upload()` with modes:
  - `'autofix'` (default): Auto-coerce mixed-type columns to string + emit warning
  - `'strict'`: Raise `ArrowConversionError` on mixed types; may do data-level inspection
  - `'strict-fast'`: Same as strict but stays at metadata/schema level only (for high-throughput pipelines)

- **Validation:** New `validate` + `warn` parameters for `plot()`, `upload()`, and `to_arrow()` to control conversion behavior. `warn` controls warning emission during autofix conversions (default `True`; `validate=False` forces `warn=False`). Backward compatible: `validate=True` maps to `'strict'`, `validate=False` maps to `'autofix'` with `warn=False`. Encoding validation errors are also controlled by these modes. Works with pandas, cuDF, dask, dask_cudf, and Spark/Databricks DataFrames (#867).

### Changed
- **Arrow:** Auto-coerce mixed-type columns to string during Arrow conversion, preventing `ArrowTypeError` and `ArrowInvalid` exceptions when DataFrames contain columns with mixed types (e.g., bytes/float/string or list/scalar). Emits `RuntimeWarning` listing coerced columns. Applies to pandas, cuDF, and distributed DataFrames (dask, Spark) after collection (#867).

## [0.47.0 - 2025-12-15]

### Breaking 🔥
- **API v1 Removal**: Removed legacy VGraph/protobuf API v1 support in favor of API v3.
  * Removed `_etl1()`, `_etl_url()`, `_check_url()` methods from `pygraphistry.py`
  * Removed API v1 dispatch path from `PlotterBase.py`
  * Changed `register(api=...)` parameter type from `Literal[1, 3]` to `Literal[3]`
  * Updated `client_session.py` type from `Literal["arrow", "vgraph"]` to `Literal["arrow"]`
  * **Server Compatibility**: Graphistry server v2.45.7+ no longer supports api=1/2 uploads
  * **Migration**: Users calling `graphistry.register(api=1)` must switch to `graphistry.register(api=3)` or omit the parameter (defaults to v3)
  * **Auth Migration**: Users previously using `register(api=1, key="...")` must switch to JWT-based auth:
    - Username/password: `graphistry.register(api=3, username="...", password="...")`
    - Personal keys (recommended for scripts): `graphistry.register(api=3, personal_key_id="...", personal_key_secret="...")`
    - SSO: `graphistry.register(api=3, org_name="...", idp_name="...")`
    - See [authentication docs](https://pygraphistry.readthedocs.io/en/latest/server/register.html) for full options

## [0.46.1 - 2025-12-10]

### Added
- **GFQL**: Added auto_graph_renderer_switching config option for Kepler maps.

### Fixed
- **GFQL:** `Chain` now validates on construction (matching docs) and rejects invalid hops immediately; pass `validate=False` to defer validation when assembling advanced flows (fixes #860).
- **GFQL / eq:** `eq()` now accepts strings in addition to numeric/temporal values (use `isna()`/`notna()` for nulls); added coverage across validator, schema validation, JSON, and GFQL runtime (fixes #862).

### Docs
- **GFQL validation:** Clarified `Chain` constructor validation defaults, `validate=False` defer option, validation phases, and guidance for large/nested ASTs to reduce redundant validation (issue #860).

## [0.46.0 - 2025-12-01]

### Added
- **Plot: Geographic visualization support with Kepler.gl integration** (#799)
  * New bindings: `point_longitude` and `point_latitude` for specifying geographic coordinates
  * Kepler.gl encoding methods: `encode_kepler()`, `encode_kepler_dataset()`, `encode_kepler_layer()`, `encode_kepler_options()`, `encode_kepler_config()`
  * Configuration classes: `KeplerEncoding`, `KeplerDataset`, `KeplerLayer`, `KeplerOptions`, `KeplerConfig`
  * `mercator_layout()` - Convert lat/lon to Mercator projection with GPU/CPU support
  * Comprehensive user guide and API documentation
  * Added 55 tests in `test_kepler.py` and 6 tests in `test_layout.py`

## [0.45.10 - 2025-11-19]

### Added
- **GFQL / Oracle**: Introduced `graphistry.gfql.ref.enumerator`, a pandas-only reference implementation that enumerates fixed-length chains, enforces local + same-path predicates, applies strict null semantics, enforces safety caps, and emits alias tags/optional path bindings for use as a correctness oracle.

### Changed
- **GFQL / Chain**: AST graph operators now expose `execute(...)` instead of `__call__(...)`, and `chain.py` now invokes operators via explicit `op.execute(...)` / `op.reverse().execute(...)` to make the execution boundary explicit for binder extraction follow-up work.

### Fixed
- **Auth:** Work around server issue [graphistry/graphistry#2933](https://github.com/graphistry/graphistry/issues/2933) by automatically calling `/api/v2/o/<slug>/switch/` in ArrowUploader logins, SSO, and token refresh paths so `org_name` is honored for entitlements. Track the last `(org, token)` we switched to in the session to avoid redundant calls. (PR [#832](https://github.com/graphistry/pygraphistry/pull/832))
- **GFQL / Hypergraph:** Prevent `return_as=*` calls from importing pandas Styler/Jinja by skipping Protocol `isinstance()` checks whenever hypergraph legitimately returns DataFrames, fixing CI/test failures on setups without Jinja2.
- **UMAP / CuPy 13**: `umap_graph_to_weighted_edges()` now converts cuML COO matrices via `cupy.asnumpy()` / legacy `.get()` fallbacks instead of the removed `cupyx.scipy.sparse.coo_matrix.get()` so GFQL UMAP operations work again on RAPIDS 25.10 / CUDA 13 while staying compatible with older stacks (#844).
- **Init / Exports**: Re-export `graphistry.compute` from the top-level package so GFQL/remote tests importing `graphistry.compute` via `import graphistry` keep working across Python versions (#844).

### Tests
- **CI / Python**: Expand GitHub Actions coverage to Python 3.13 + 3.13/3.14 for CPU lint/type/test jobs, while pinning RAPIDS-dependent CPU/GPU suites to <=3.13 until NVIDIA publishes 3.14 wheels (ensures lint/mypy/pytest signal on the latest interpreter without breaking RAPIDS installs).
- **GFQL**: Added deterministic + property-based oracle tests (triangles, alias reuse, cuDF conversions, Hypothesis) plus parity checks ensuring pandas GFQL chains match the oracle outputs.
- **GFQL / Chain**: Added regression coverage that forces runtime `__call__` on AST operators to raise, confirming chain execution uses explicit `execute(...)` paths only.
- **Layouts**: Added comprehensive test coverage for `circle_layout()` and `group_in_a_box_layout()` with partition support (CPU/GPU)

### Infra
- **CI / HuggingFace cache**: Prewarm sentence-transformers models into a shared `HF_HOME` cache via Actions cache (per-OS key) using anonymous downloads, then run `test-core-umap` and `test-full-ai` with `HF_HUB_OFFLINE=1` to avoid HF 429s in `feature_utils` tests; no `HF_TOKEN` required (#853).
- **CI / DGL isolation**: Moved DGL/embed tests into a legacy CPU job pinned to torch 2.0.1/torchdata 0.6.1/dgl 2.1 (CPU wheels only) and removed DGL from the main matrix. Extras split into `dgl-cpu` (pinned) and `dgl-gpu` (torch 2.4.1 + dgl-cu12 2.4.0, opt-in); main jobs stay on newer torch without DGL install to avoid wheel/ABI churn (#856).

### Breaking 🔥
* **DGL extras**: `graphistry[ai]` no longer installs DGL. Use `graphistry[dgl-cpu]` (torch 2.0.1/torchdata 0.6.1/dgl 2.1, CPU-only) or `graphistry[dgl-gpu]` (torch 2.4.1 + dgl-cu12 2.4.0). `pip install graphistry[ai]` users must add one of the new DGL extras to keep DGL.

## [0.45.9 - 2025-11-10]

### Fixed
- **Layouts**: Fixed `group_in_a_box_layout` with GPU engine failing on older cuDF versions where `groupby.transform('size')` is not supported. Replaced with cuDF-compatible `groupby.size()` + `map()` pattern that works across all pandas and cuDF versions (#829).

## [0.45.8 - 2025-11-04]

### Changed
- **Docker**: Update `docker/test-cpu.Dockerfile` to default to Python 3.12 for parity with supported environments.

### Fixed

- **GFQL - let, ref**: `graphistry.compute.let` and `graphistry.compute.ref` are now accessible through the top-level `graphistry` module.

## [0.45.7 - 2025-10-21]

### Fixed
- **Auth**: `register(token=...)` now marks sessions authenticated, disables unused credential caching, and verifies tokens by default (opt-out) to unblock GFQL persistence workflows (fixes #824).

## [0.45.6 - 2025-10-21]

### Added
- **GFQL / Layouts:** Safelisted `ring_continuous_layout()`, `ring_categorical_layout()`, and `time_ring_layout()` for GFQL use, including stricter parameter validation, typing, and CPU/GPU coverage.

### Fixed
- **Layouts / FA2:** GFQL `fa2_layout()` now surfaces a clear GPU requirement instead of silently substituting a different CPU layout, while existing CPU workflows opt-in through an explicit fallback hook (PR #819).

## [0.45.5 - 2025-10-19]

### Added
- **GFQL: Relax chain homogeneity to allow call() at boundaries** (#792)
  - **Enhancement**: Relaxes v0.45.0 restriction to allow call() at chain start/end
  - Enables patterns like `[call('filter'), n(), e(), call('enrich')]` without requiring `let()`
  - Interior mixing still disallowed: `[n(), call(), e()]` raises GFQLValidationError
  - **Migration from v0.45.0**: Code that used `let()` for boundary patterns can now use simpler boundary syntax
  - Provides convenience for common filter/enrich patterns

### Docs
* **GFQL: Add LLM JSON generation guide to documentation site** (#807)
  * Comprehensive guide for LLMs (Claude, GPT, etc.) to generate valid GFQL JSON queries
  * Covers Core Types, Predicates, Common Patterns, Graph Algorithms, Visualization, Domain Examples
  * Includes generation rules, common mistakes, and call functions reference
  * Available at `docs/source/gfql/spec/llm_guide.md` with reference label `gfql-spec-llm-guide`
  * Includes maintenance process guide at `ai/prompts/GFQL_LLM_GUIDE_MAINTENANCE.md` for keeping guide updated

### Infra
- Refactored DataFrame type coercion into Engine module (#784)
  - Moved `safe_concat()` and `safe_merge()` from `compute/primitives.py` to `Engine.py`
  - Colocated with other polymorphic DataFrame methods (`df_concat`, `df_to_engine`, `resolve_engine`)
  - Centralized pandas/cuDF type handling prevents future type mismatch bugs
  - Fixed bug in `materialize_nodes()` using wrong concat method for Series
  - Comprehensive CPU and GPU Docker test coverage (80 tests passing with cuDF)

### Fixed
- Fixed potential None dereference crashes in DGL utilities and Sugiyama layout (#801)
  - Added None check for `_entity_to_index` in `dgl_utils.py:309` before calling `isin()` and `len()`
  - Added None checks for `layer` attributes in `sugiyamaLayout.py` before passing to `range()` (lines 474, 761)
  - Fixed `cast()` usage in `call_executor.py` to use proper `isinstance()` checks for polymorphic returns
  - Prevents TypeError crashes when optional attributes are None and when accessing DataFrame attributes
  - Added regression tests in `test_dgl_utils.py` and `test_sugiyama_none_dereference.py`

## [0.45.4 - 2025-10-18]

### Added
* **GFQL Remote: engine='auto' support for automatic DataFrame engine detection**
  * Remote GFQL operations now default to `engine='auto'` instead of manual engine specification
  * `chain_remote()`, `python_remote_g()`, `python_remote_table()`, `python_remote_json()` all support 'auto'
  * Client-side resolution inspects graph's DataFrame type (pandas/cudf) and passes resolved engine to server
  * **Example**: `g.gfql_remote(call('hypergraph', {'entity_types': ['user', 'product']}))` - automatically uses the right engine
  * **Migration**: Users who need explicit engine control can still pass `engine='pandas'` or `engine='cudf'`
  * **Benefits**: Seamless GPU/CPU usage without manual configuration, consistent with local GFQL behavior
  * Validates that dask engines aren't used remotely (server only supports pandas/cudf)
  * Updated type signatures across 5 files: `chain_remote.py`, `python_remote.py`, `chain_let.py`, `ComputeMixin.py`, `Plottable.py`

### Fixed
* **GFQL: Fix get_degrees parameter validation in remote operations**
  * Fixed `call('get_degrees', {'degree_in': '...', 'degree_out': '...'})` validation rejecting correct parameter names
  * **Problem**: Safelist had incorrect parameter names (`col_in`, `col_out`) instead of actual method signature (`degree_in`, `degree_out`)
  * **Solution**: Updated safelist to match `ComputeMixin.get_degrees()` signature (line 295-300)
  * Also fixed `GetDegreesParams` TypedDict in `graphistry/models/gfql/types/call.py` which had same incorrect names
  * Example that now works: `g.gfql_remote(call('get_degrees', {'degree_in': 'in_deg', 'degree_out': 'out_deg'}))`
  * Prevents confusing "Unknown parameters" errors for valid parameter names
* **Type Safety: Resolved all mypy type errors in graphistry codebase**
  * Fixed 6 mypy errors across 5 files (down from 10 total - remaining 4 are in external pyarrow-stubs package)
  * **arrow_uploader.py**: Added None checks before passing Optional[pa.Table] to methods expecting pa.Table
  * **PlotterBase.py**: Fixed `_table_to_arrow` return type to Optional[pa.Table] and updated `_make_arrow_dataset` signature
  * **Plottable.py & umap_utils.py**: Fixed overload signature overlaps for `umap()` method by removing default from `inplace: Literal[True]` parameter
  * **Impact**: Better IDE type checking and autocomplete, prevents type-related bugs at compile time
* **Hypergraph: Fix int32 NaN handling in Arrow-optimized columns**
  * Fixed `IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer` when hypergraph operations introduce NaN in int32 columns
  * **Problem**: Arrow serialization optimizes int64 → int32, then pandas merge/reindex introduces NaN. `coerce_col_safe()` handled int64 with NaN but not int32
  * **Solution**: Extended `coerce_col_safe()` in `hyper_dask.py` to handle both `int32` and `int64` dtypes with NaN values (fillna(0) before conversion)
  * **Root cause**: When client uploads clean int64 data, Arrow optimizes to int32, then hypergraph merge operations introduce NaN, causing conversion failures
  * **Impact**: Prevents cryptic server errors during GFQL remote hypergraph operations and local hypergraph with Arrow-optimized data

## [0.45.3 - 2025-10-17]

### Breaking 🔥
* **Hypergraph: Engine parameter now defaults to 'auto'**
  * Default changed from `engine='pandas'` to `engine='auto'`
  * Auto-detects DataFrame engine from graph's current nodes/edges (pandas/cudf/dask/dask_cudf)
  * **Migration**: Explicitly pass `engine='pandas'` to maintain old behavior
  * **Example**: `g.hypergraph(entity_types=['user', 'product'], engine='pandas')`
  * **Rationale**: Enables seamless GPU/CPU usage and proper engine propagation through GFQL operations
  * **Impact**: Users who relied on pandas-only behavior may see different DataFrame types returned
  * Most users benefit from automatic engine detection - only explicitly set if you need pandas-only

### Added
* **Type Safety: Introduced EngineAbstractType for engine parameters**
  * New type alias: `EngineAbstractType = Union[EngineAbstract, Literal['pandas', 'cudf', 'dask', 'dask_cudf', 'auto']]`
  * Provides compile-time type checking for engine string values (prevents typos)
  * IDE autocomplete now shows valid engine options
  * Supports both enum values (`EngineAbstract.AUTO`) and string literals (`'auto'`)
  * Updated 9 files with proper type annotations

### Changed
* **Hypergraph: Engine parameter propagation improvements**
  * Benefits outer engine parameter propagation: `g.gfql(call('hypergraph'), engine='cudf')` now works as expected
  * Updated all hypergraph signatures to use `EngineAbstractType` for type-safe API
  * Example that now works: `g.gfql(call('hypergraph', {'entity_types': ['a', 'b']}), engine='cudf')` - hypergraph respects outer engine
  * Files updated: `Engine.py`, `hyper_dask.py`, `hyper.py`, `Plottable.py`, `PlotterBase.py`, `pygraphistry.py`

### Fixed
* **Plot: Fix UnboundLocalError and improve API version error handling**
  * Fixed `UnboundLocalError` when calling `plot()` with `api_version=1` and certain render modes
  * Added proper initialization of `uploader` variable before conditional branches
  * Added clear error message for unsupported API versions (only 1 and 3 are supported)
  * Prevents confusing crashes and provides actionable error messages for configuration issues
* **GFQL Remote: Client-side validation now enforced before sending to server**
  * Fixed remote GFQL operations not validating `call()` parameters against safelist before execution
  * `ASTCall._validate_fields()` now calls `validate_call_params()` during deserialization
  * Catches invalid parameters during `Chain.from_json()` before sending requests to server
  * Prevents server-side errors from malformed client requests (e.g., invalid `engine`, `return_as` values, unknown parameters)
  * Added comprehensive test suite in `test_gfql_call_validation.py` with 28 validation tests covering:
    * Graph transformations: `hypergraph`, `umap`
    * Graph algorithms: `compute_igraph`, `get_degrees`
    * Graph traversals: `hop`
    * Visual encodings: `encode_point_color`
    * Metadata: `name`, `description`
  * Example that now fails fast client-side: `g.gfql_remote(call('hypergraph', {'engine': 'invalid'}))` → `GFQLTypeError` instead of server error

### Refactored
* **Import organization**: Hoisted dynamic imports to module level following PEP 8 conventions
  * Moved `validate_call_params` import to top-level in `ast.py`
  * Moved `resolve_engine` import to top-level in `hyper_dask.py`
* **Code cleanup**: Removed redundant comments per DECOMMENT protocol in test files

## [0.45.2 - 2025-10-17]

### Fixed
* **GFQL Remote: Metadata hydration after server-computed operations** (#798)
  * Server-computed metadata (bindings, encodings, styles, name/description) now returned to client after GFQL operations like `call('umap')`
  * Centralized metadata serialization/deserialization in `graphistry.io.metadata` with TypedDict structures
  * Properly typed `_complex_encodings` as `ComplexEncodingsDict` throughout codebase
* **Plot: Fix dataset_id invalidation in encoding and style methods** (#797)
  * Fixed 7 methods not invalidating `dataset_id` after modifying encodings/metadata/styles: `__encode()`, `encode_axis()`, `name()`, `description()`, `bind()` (conditional), `style()`, `addStyle()`
  * Added 25 tests in `test_dataset_id_invalidation.py` with parametric coverage of all encoding methods
  * Found via AST pattern matching + Pysa call graph analysis (Pysa found 2 additional bugs AST missed)

### Refactored
* **Import organization**: Hoisted dynamic imports to module level following PEP 8 conventions (#798)

## [0.45.1 - 2025-10-16]

### Fixed
* **UMAP: Fix mixed DataFrame types when engine='cudf' causing GFQL chain concatenation to fail** (#794)
  * **Problem**: UMAP with `engine='cuml'` was returning pandas nodes + cuDF edges, causing `TypeError: can only concatenate objects which are instances of...` in GFQL chain operations
  * **Example that now works**: `g.gfql([call('umap', {'X': ['col1', 'col2'], 'engine': 'cuml'}), call('name', {'name': 'result'})], engine='cudf')`
  * **Solution**: Now ensures both nodes and edges match the specified engine type (cuDF or pandas)
  * Added comprehensive test coverage in `TestCudfUmap` class

### Docs
* **GFQL: Fix critical documentation bugs where graph algorithms were called without edges** (#795)
  * **Critical fixes (3 instances):** PageRank and Louvain examples were calling algorithms on node-only patterns `n({...})` that lack the edges these algorithms require for meaningful computation
    * PageRank computes centrality from link structure - calling on nodes without edges produces meaningless scores
    * Louvain detects communities through connections - calling on nodes without edges cannot identify communities
  * Fixed 2 instances in quick.rst (lines ~377, ~407 PageRank Let binding examples)
  * Fixed 1 instance in about.rst (line ~305 Louvain multi-stage fraud analysis)
  * Changed patterns from `n({...})` to `[n({...}), e(), n()]` to include graph structure
  * **Style improvements:**
    * Removed unnecessary `Chain([...])` wrappers in Let binding examples (quick.rst lines ~335, ~353; about.rst ~287; overview.rst ~178)
    * Replaced low-level `ASTCall()` with user-friendly `call()` API in gfql_remote.ipynb (4 instances)
  * Examples now follow best practices and produce meaningful results when executed

## [0.45.0 - 2025-01-15]

### Breaking 🔥
* **GFQL: Chains must be homogeneous** (#786, #791)
  * Chains must be either all `call()` or all `n()`/`e()` operations, cannot mix
  * Previous behavior was likely buggy - mixed chains had unpredictable results
  * Mixed chains now raise `GFQLValidationError` with clear guidance
  * Migration: Use `let()` to compose sequences, e.g., `let({'filtered': [n(), e()], 'enriched': ref('filtered', [call(...)])})`
  * Affects both `.gfql()` and `.gfql_remote()`
  * **Note**: Boundary call patterns re-enabled in v0.45.5 (#792)

### Added
* **GFQL: Type-safe call() operations** - `from graphistry import call, CallMethodName` (#789)
  * TypedDict parameter classes with IDE autocomplete (e.g., `HopParams`, `UmapParams`)
  * Overloaded signatures for MyPy type checking
* **GFQL: Internal column validation** (#788)
  * Prevents filtering on or using `__gfql_*__` pattern in column names

### Fixed
* **Compute: Fix get_degrees() to respect degree_in/degree_out parameters** (#788)
* **GFQL: Fix chained ASTCall operations** - Pure `call()` chains now correctly apply sequentially (#786)

### Infra
* **Tests: Column name restriction coverage** (#788)

## [0.44.1 - 2025-10-13]

### Added
* **GFQL: Enhanced string predicates** (#697, #774)
  * **Case-insensitive matching**: `startswith()` and `endswith()` now support `case` parameter - `startswith('test', case=False)`
  * **Tuple pattern matching**: OR logic via tuples - `startswith(('test', 'prod'))` matches either pattern
  * **Full-string matching**: New `fullmatch(pat, case=True)` predicate for exact pattern validation
  * Examples:
    * `n({'filename': endswith(('.txt', '.csv'), case=False)})` - Case-insensitive file extensions
    * `n({'email': fullmatch(r'.*@company\.com')})` - Email validation
  * Works in both pandas and cuDF backends

### Fixed
* **Hypergraph: Restore backward compatibility with smart type detection** (#785)
  * **Breaking change introduced in version 0.43.2 has been eliminated** - all three API styles now work:
    * Old positional: `hypergraph(g, df, ['cols'])` - works again!
    * New convenience: `hypergraph(g, ['cols'])` - auto-selects dataframe from graph
    * Keyword: `hypergraph(g, entity_types=['cols'])` - explicit keyword style
  * **Root cause**: Version 0.43.2 added `*` marker forcing keyword-only arguments, breaking existing user code
  * **Solution**: Removed keyword-only argument restriction and added smart type detection
  * When second parameter is a list, treats it as `entity_types` and auto-selects dataframe from graph
  * When second parameter is DataFrame-like, treats it as `raw_events`
  * **User impact**: No migration needed - existing code continues to work as before version 0.43.2
  * Added comprehensive API compatibility test suites (`TestHypergraphAPICompatibility`, `TestHypergraphAPICompatibilityCudf`) with 9 tests total to prevent future regressions
* **GFQL: Extend engine coercion to let() and call() operations** (#783)
  * Fixed `gfql(ASTLet, engine='pandas'|'cudf')` and `call('umap', ...)` to honor engine parameter
  * Schema-changing operations (UMAP, hypergraph) in let/call context now correctly coerce DataFrames
  * Added `ensure_engine_match()` helper with graceful degradation for defensive type conversion
  * Added 40 comprehensive tests (8 helper + 16 let + 14 call + 2 get_indegrees integration) covering pandas↔cuDF coercion
  * **Root cause fixes**:
    * `is_legacy_cuml()` now catches both `ModuleNotFoundError` and `ImportError` for robust cuML handling
    * `CGFull` test fixture now includes `UMAPMixin` and `FeatureMixin` for proper UMAP support
  * Solves `get_indegrees()` merge error when UMAP creates cuDF edges but nodes remain pandas
  * Complements #777 (chain engine coercion) with consistent behavior across all GFQL operations
* **GFQL: Fix column name conflicts with internal tracking columns** (#776)
  * Fixed `collapse(column='index')` and similar operations failing when user columns conflicted with GFQL internal columns
  * Auto-generates unique internal column names to avoid all collisions
* **Compute: Fix engine-aware merge in get_degrees for cuDF/pandas compatibility** (#778)
  * Fixed `get_degrees()`, `get_indegrees()`, `get_topological_levels()` failing when merging mixed DataFrame types
  * Added engine detection and DataFrame conversion before merge operations
  * Pattern follows #777 fix - detect engine mismatch, convert to compatible types before merge
  * **UMAP lazy init behavior**: Enhanced UMAP engine initialization with three-level fallback (import detection, lazy import validation, runtime initialization) to gracefully handle broken GPU libraries (e.g., broken RMM). Note: `umap_lazy_init()` creates a new UMAP instance when parameters change - consider memoization for workflows with repeated calls using same parameters.
* **GFQL Chain: Fix engine parameter to correctly convert DataFrames after schema-changing operations** (#777)
  * Fixed `chain(engine='pandas'|'cudf')` returning wrong DataFrame type after UMAP/hypergraph operations
  * Added comprehensive test coverage (19 tests for pandas↔cuDF coercion with UMAP)
* **Search: Fix `search(..., fuzzy=True)` after `umap(y=['label'])` AssertionError** (#773, #629)

### Docs
* README: Added connector tutorials table with 16 categorized badges linking to demo notebooks (#771)
* ai: Compact CONVENTIONAL_COMMITS guide and enforce PLAN.md usage (3e537db)
* ai: Streamline PLAN.md template with phases, timestamps, and linear structure (1a85e6e)

## [0.44.0 - 2025-10-11]

### Added
* **GFQL Policy System enhancements** (#764)
  * **Policy shortcuts** - Reduce from 10 keys to 2: `'pre'`/`'post'` expand to all pre*/post* hooks, `'load'`/`'let'`/`'chain'`/`'binding'`/`'call'` for scope-specific hooks. Automatic composition with predictable order (general → scope → specific). Use `debug_policy()` for visibility.
  * **Complete execution hierarchy** - 10 hooks covering all levels: query (preload/postload), let/chain (prelet/postlet/prechain/postchain), binding (preletbinding/postletbinding), call (precall/postcall)
  * **OpenTelemetry fields** - `execution_depth`, `operation_path`, `parent_operation` enable proper span parent-child relationships
  * **Binding context** - Per-binding control with `binding_name`, `binding_index`, `total_bindings`, `binding_dependencies`, `binding_ast`
  * **Error context** - All post* hooks receive `success`, `error`, `error_type` fields
  * **Use cases**: OpenTelemetry tracing (2 keys instead of 10), server multi-policy composition, per-binding/DAG/chain-level control
  * 69 tests passing (48 core + 24 shortcuts + 14 let/chain + 5 binding hooks)

### Docs
* **Notebooks: Fix broken register link in memgraph demo** (#772, #627)
  * Follow up to #692 which fixed 47 notebooks but missed the memgraph tutorial
  * Changed broken link from `github.com/graphistry/pygraphistry#configure` to working docs: `pygraphistry.readthedocs.io/en/latest/server/register.html`

### Fixed
* **Hypergraph: Fix empty DataFrame structure when single entity + direct=True** (#766)
  * **Problem**: `hypergraph(entity_types=['single'], direct=True)` returned empty DataFrame with NO columns, causing `get_degrees()` to fail with `KeyError`
  * **Solution**: Empty edges DataFrame now has proper column structure (src, dst, edgeType, EventID, etc.)
  * **Engines**: All engines fixed (pandas, cudf, dask, dask_cudf)
  * **Additional fixes**:
    * `mt_series()` - Properly creates empty Series for dask/dask_cudf by wrapping pandas Series
    * `format_direct_edges()` - Uses `df_coercion()` for cross-engine empty DataFrame creation
    * `get_indegrees()` - Uses `.assign()` for cross-engine compatibility
    * `_safe_len()` - Fixed meta parameter for dask_cudf (was `meta=int`, now `meta=pd.Series([], dtype='int64')`)
    * Removed unnecessary `persist()` calls that broke lazy evaluation in dask_cudf
* **UMAP: Handle cuDF DataFrames with string node columns + improve error messages** (#765, #770)
  * **Fixed `TypeError: String arrays are not supported by cupy`** when using UMAP with cuDF DataFrames containing string-typed node ID columns (#765)
    * cuDF string columns are now converted to pandas before value extraction, avoiding cupy limitations
    * Simplified node ID extraction logic since values are always numpy arrays after conversion
  * **Improved error message when no numeric features available** (#770)
    * Validates after featurization to detect when all columns are dropped (all-strings edge case)
    * Provides helpful guidance: "No numeric features available for UMAP after featurization. Please provide at least one numeric column, or install 'skrub' for automatic string encoding: pip install skrub"
    * Applied validation to both nodes and edges paths
* **GFQL Policy: Ensure post* hooks always fire even on errors** (#764)
  * post* hooks now use consistent try/except/finally pattern
  * PolicyException takes precedence over operation errors
  * Error chaining preserved (`raise PolicyException from error`)
  * All post* hooks receive error context (success, error, error_type)

### Documentation
* **GFQL Policy**: Updated `docs/source/gfql/policy.rst` with new hooks and fields
* **OpenTelemetry Integration**: New guide `docs/source/gfql/policy_opentelemetry.rst`
  * Complete working examples for span tracing
  * Integration patterns for Jaeger, OTLP, and custom exporters
  * Best practices and performance considerations

## [0.43.2 - 2025-10-09]

### Breaking 🔥
* **Hypergraph: All parameters after `raw_events` now require keyword arguments** (#763) - **FIXED in version 0.44.1**
  * Added `*` marker in `hypergraph()` signature forcing keyword-only arguments
  * **Old code breaks**: `hypergraph(g, df, ['cols'])` → Must use `hypergraph(g, df, entity_types=['cols'])`
  * **Migration**: Upgrade to version 0.44.1+ for backward compatibility restoration - no code changes needed
  * **Note**: This breaking change only affected versions 0.43.2 - 0.44.0 (short-lived)

### Added
* **Hypergraph `from_edges` and `return_as` parameters now available in ALL contexts** (#763)
  * Works everywhere: GFQL, instance methods (`g.hypergraph()`), and module-level (`graphistry.hypergraph()`)
  * **`from_edges` parameter**: Use edges dataframe instead of nodes as input
    * `g.edges(df).hypergraph(from_edges=True)` - Direct method call
    * `g.gfql(call('hypergraph', {'from_edges': True}))` - GFQL
    * Default: `from_edges=False` (backward compatible)
  * **`return_as` parameter**: Control what hypergraph returns
    * `'graph'` - Returns Plottable (for method chaining)
    * `'all'` - Returns full dict with 5 keys: graph, entities, events, edges, nodes
    * `'entities'/'events'/'edges'/'nodes'` - Returns specific DataFrame
  * **Context-specific defaults** for optimal UX:
    * Module-level `graphistry.hypergraph(df)`: `return_as='all'` (backward compatible - returns dict)
    * Instance method `g.hypergraph()`: `return_as='graph'` (chainable - returns Plottable)
    * GFQL `g.gfql(call('hypergraph'))`: `return_as='graph'` (chainable - returns Plottable)
  * **Type safety**: Added `@overload` decorators for MyPy type inference based on `return_as` value
  * **Examples**:
    * Chainable: `g.hypergraph().plot()` - Returns Plottable
    * Full dict: `result = graphistry.hypergraph(df); g = result['graph']` - Backward compatible
    * Explicit all: `result = g.hypergraph(return_as='all')` - Get all 5 components
    * Extract DataFrame: `entities = g.hypergraph(return_as='entities')` - Just entities
    * From edges: `g.edges(df).hypergraph(from_edges=True, entity_types=['src', 'dst'])`

### Fixed
* **Hypergraph: Critical bug fix for return_as='graph' routing** (#763)
  * **Before**: `g.hypergraph()` incorrectly returned full dict (preventing method chaining)
  * **After**: `g.hypergraph()` correctly returns Plottable (enables `g.hypergraph().plot()`)
  * Impact: Instance methods and GFQL calls now chainable as designed
  * Added 'all' option for explicit full dict access when needed
* **Type safety: Resolved all mypy errors in PlotterBase and time ring layout**
  * Fixed Protocol signature mismatch in PlotterBase.hypergraph - added explicit `raw_events: Optional[Any]` type annotation
  * Fixed 10 pre-existing numpy type errors in layout/ring/time.py with proper `np.int64`, `np.datetime64`, `np.timedelta64` annotations
  * Added type: ignore comments for genuine numpy datetime arithmetic stub limitations
  * All 42 layout tests pass, full codebase now passes mypy type checks

## [0.43.1 - 2025-10-09]

### Added
* GFQL: Schema-changing operations (UMAP, hypergraph) now supported in chains (#761)
  * **UMAP in chains**: Can now use `call('umap', {...})` mixed with filters and other operations
  * **Hypergraph in chains**: Removed mixing restriction - now allows `[n(...), call('hypergraph', {...})]`
  * **Usage**: `g.gfql([n({'type': 'person'}), call('umap', {'n_neighbors': 15}), e()])`
  * Implemented via recursive dispatch that splits chains at schema-changer boundaries

### Fixed
* GFQL: Fix `"Column 'index' not found in edges"` error in schema-changing operations (#761)
  * Schema-changers now execute as: `before → schema_changer → rest` for proper isolation
  * Prevents tracking column conflicts when UMAP/hypergraph create new graph structures
* GFQL: Respect validate_schema flag for singleton schema-changers
  * Added validation check before execute_call() to honor user's validate_schema setting
* GFQL: Replace assertion with GFQLTypeError for proper error handling
  * Schema-changer type validation now uses structured exception instead of assert

## [0.43.0 - 2025-10-08]

### Added
* GFQL: Policy hook system for external query control and Hub integration
  * **Four-phase hooks**: `preload`, `postload`, `precall`, `postcall`
  * **Accept/Deny pattern**: Policies return `None` (accept) or raise `PolicyException` (deny)
  * **Context-rich**: Full access to query, graph stats, operation details, and timing
  * **Features**:
    * `preload` - Control before data loading (JWT validation, dataset access)
    * `postload` - Validate after data loading (size limits, content checks)
    * `precall` - Control before operation execution (feature gating, parameter validation)
    * `postcall` - Monitor after execution (performance tracking, result validation)
  * **Remote data support**: Special handling for `ASTRemoteGraph` with `is_remote` flag
  * **Thread-safe**: Uses thread-local storage with recursion prevention
  * **Usage**: `g.gfql(query, policy={'preload': check_auth, 'postcall': track_perf})`
* GFQL: Added UMAP call operation support
  * **UMAP embeddings**: UMAP dimensionality reduction available via `call('umap', {...})`
  * **Full parameter support**: All UMAP parameters validated through call safelist
  * **Policy integration**: UMAP operations controllable through precall/postcall policy hooks
  * **Usage**: `g.gfql(call('umap', {'X': ['x', 'y'], 'n_neighbors': 15}))`

### Fixed
* GFQL: Fixed remote operations incorrectly treating HTTP error responses as zip files
  * Added proper HTTP status code checking before attempting to parse server responses
  * Server validation errors now surface correctly instead of being masked by zip parsing failures

## [0.42.4 - 2025-10-05]

### Added
* GFQL: Remote GFQL persistence with persist=True parameter (#760)
  * Added `persist=False` parameter to `gfql_remote()` and `gfql_remote_shape()` methods
  * Added `url` property to access visualization URLs after server-side persistence
  * Enables server-side dataset persistence eliminating client round-trips

## [0.42.3 - 2025-10-04]

### Fixed
* GFQL: Fix dict-to-AST conversion in list contexts (#758)
  * Fixed critical regression where `g.gfql([{"type": "Node", ...}])` failed with `TypeError: 'dict' object is not callable`
  * Added dict-to-AST conversion loop in list processing branch using `from_json()`
  * Maintains backward compatibility while enabling dict convenience syntax within list queries
  * Added comprehensive `TestGFQLDictConversion` class with 6 regression tests for all dict conversion scenarios

## [0.42.2 - 2025-10-03]

### Fixed
* Tests: Fix NoAuthTestCase authentication bypass pattern that was broken due to incorrect operation order
  * `register(api=1)` was resetting `_is_authenticated` back to False after it was set to True
  * Fixed by calling `register()` first, then setting `_is_authenticated = True`
  * Added `setUp()` method to reset authentication state before each test for better isolation
  * Added comprehensive documentation explaining this is a temporary hack

## [0.42.1 - 2025-10-02]

### Fixed
* neo4j: Handle new spatial object format in driver 5.28+ for backwards compatibility

## [0.42.0 - 2025-10-02]

### Fixed
* GFQL: Fix hypergraph typing - add method to Plottable Protocol, resolve circular import

### Added
* GFQL: Renamed `chain()` methods to `gfql()` for clarity
  * New methods available: `g.gfql([...])`, `g.gfql_remote([...])`, `g.gfql_remote_shape([...])`
  * The old `chain*` methods are deprecated with no sunset date - continue working as before
  * All functionality remains the same, only method names have been added
* GFQL: Let bindings now accept ASTNode/ASTEdge matchers directly (#751)
  * Direct syntax: `let({'persons': n({'type': 'person'})})` without Chain wrapper
  * Auto-converts list syntax to Chain for backward compatibility
  * **Important**: Uses FILTER semantics - `n()` returns nodes only (no edges)
  * Independent bindings operate on root graph unless using `ref()`
* Development: Host-level convenience scripts for local testing
  * `./bin/pytest.sh` - Runs tests with highest available Python (3.8-3.14)
  * `./bin/mypy.sh` - Type checking without Docker overhead
  * `./bin/flake8.sh` - Linting with auto-detection of Python version
* GFQL: Add policy hook system for external policy injection with schema validation
  * Three-phase hooks: preload (before data), postload (after data), call (per operation)
  * Enable accept/deny/modify capabilities for GFQL queries
  * Schema validation for all policy modifications
  * Recursion prevention at depth 1 for safety
  * Enriched PolicyException with phase, reason, query_type, and data_size
  * Safe graph statistics extraction for pandas, cudf, dask, and dask-cudf
  * Closure-based state management pattern for Hub integration
  * Comprehensive test coverage with 48 unit tests
* GFQL: Add hypergraph transformation support for creating entity relationships from event data
  * Simple transformation: `g.gfql(hypergraph(entity_types=['user', 'product']))`
  * Typed builder with IDE support: `from graphistry.compute import hypergraph`
  * Full parameter support: entity_types, drop_na, direct, engine (pandas/cudf/dask), etc.
  * Remote execution: `g.gfql_remote(hypergraph(...))`
  * DAG composition: Use with `let()` for complex transformations
  * Safelist validation for all hypergraph parameters
  * Enhanced `opts` parameter validation for nested structures (CATEGORIES, EDGES, SKIP)
  * 19 unit tests including mocked remote execution
* GFQL: Add comprehensive validation framework with detailed error reporting
  * Built-in validation: `Chain()` constructor validates syntax automatically
  * Schema validation: `validate_chain_schema()` validates queries against DataFrame schemas
  * Pre-execution validation: `g.gfql(ops, validate_schema=True)` catches errors before execution
  * Structured error types: `GFQLValidationError`, `GFQLSyntaxError`, `GFQLTypeError`, `GFQLSchemaError`
  * Error codes (E1xx syntax, E2xx type, E3xx schema) for programmatic error handling
  * Collect-all mode: `validate(collect_all=True)` returns all errors instead of fail-fast
  * JSON validation: `Chain.from_json()` validates during parsing for safe LLM integration
  * Helpful error suggestions for common mistakes
  * Example notebook: `demos/gfql/gfql_validation_fundamentals.ipynb`

## [0.41.2 - 2025-08-28]

### Fixed
* Improve types by surfacing more on `Plottable`
  * `umap`, `search` and `embed`
  * shared types in `embed_types.py` and `umap_types.py`
* Add `mode_action` to `.privacy`
* Fixed `contains`, `startswith`, `endswith`, and `match` predicates to prevent error when run with cuDF

## [0.41.1 - 2025-08-15]

### Fixed
* Auth: Fix `chain_remote()` and `python_remote_g()` to use client session instead of global singleton (#733)
  * Fixes authentication errors when using `client.register()` with API keys
  * Ensures proper session isolation for multi-client scenarios
* Session: Fix missing certificate_validation support in remote operations (#734)
  * Fixed `chain_remote_generic()` to respect `self.session.certificate_validation`
  * Fixed `python_remote_generic()` to respect `self.session.certificate_validation`
  * Added comprehensive test coverage for certificate validation behavior
  * Ensures SSL verification can be properly disabled when needed for self-signed certificates
* Docs: Fix case sensitivity in server toctree to link concurrency.rst (#723)
* Docs: Correct hallucinated method names in GFQL documentation (#732)
  * Fixed `chain_remote_python` → `python_remote_g` (non-existent method)
  * Fixed `remote_python_table` → `python_remote_table`
  * Fixed `remote_python_json` → `python_remote_json`
  * Fixed code indentation issues in about.rst and combo.rst
* Logging: stop attaching a handler if one exists, let caller decide how to log

### Infra
* Docs: Add RST validation tooling to prevent documentation syntax errors
  * Added rstcheck configuration to validate RST files
  * Integrated validation into Docker build process
  * Added validation script for local development

## [0.41.0 - 2025-07-26]

### Added
* GFQL: Add comprehensive validation framework with detailed error reporting
  * Built-in validation: `Chain()` constructor validates syntax automatically
  * Schema validation: `validate_chain_schema()` validates queries against DataFrame schemas
  * Pre-execution validation: `g.chain(ops, validate_schema=True)` catches errors before execution
  * Structured error types: `GFQLValidationError`, `GFQLSyntaxError`, `GFQLTypeError`, `GFQLSchemaError`
  * Error codes (E1xx syntax, E2xx type, E3xx schema) for programmatic error handling
  * Collect-all mode: `validate(collect_all=True)` returns all errors instead of fail-fast
  * JSON validation: `Chain.from_json()` validates during parsing for safe LLM integration
  * Helpful error suggestions for common mistakes
  * Example notebook: `demos/gfql/gfql_validation_fundamentals.ipynb`

### Fixed
* Docs: Fix case sensitivity in server toctree to link concurrency.rst (#723)
* Docs: Fix notebook validation error in hop_and_chain_graph_pattern_mining.ipynb by adding missing 'outputs' field to code cell

### Infra
* CI: Add explicit timeout-minutes to all CI jobs to prevent stuck workflows (#710)
* CI: Add smart change detection to optimize CI runtime (#710)
  * Python changes trigger lint/type checks and Python tests
  * Documentation changes only trigger docs-related tests
  * Dedicated lint/type job runs once instead of per Python version
  * Reduces CI time for focused changes (e.g., docs-only PRs)

## [0.40.0 - 2025-07-23]

### Added
* GFQL: Add comprehensive validation framework with detailed error reporting
  * Built-in validation: `Chain()` constructor validates syntax automatically
  * Schema validation: `validate_chain_schema()` validates queries against DataFrame schemas
  * Pre-execution validation: `g.chain(ops, validate_schema=True)` catches errors before execution
  * Structured error types: `GFQLValidationError`, `GFQLSyntaxError`, `GFQLTypeError`, `GFQLSchemaError`
  * Error codes (E1xx syntax, E2xx type, E3xx schema) for programmatic error handling
  * Collect-all mode: `validate(collect_all=True)` returns all errors instead of fail-fast
  * JSON validation: `Chain.from_json()` validates during parsing for safe LLM integration
  * Helpful error suggestions for common mistakes
  * Example notebook: `demos/gfql/gfql_validation_fundamentals.ipynb`

### Fixed
* Engine: Fix resolve_engine() to use dynamic import for Plottable isinstance check to avoid Jinja dependency from pandas df.style getter (#701)
* Engine: Fix resolve_engine() to check both Plotter and Plottable classes for proper type detection

### Infra
* CI: Enable parallel test execution in GPU CI with pytest-xdist (#703)

### Docs
* Update copyright year from 2024 to 2025 in documentation and LICENSE.txt
* GFQL: Add comprehensive specification documentation (#698)
  * Core language specification with formal grammar, operations, predicates, and type system
  * Cypher to GFQL translation guide with Python and wire protocol examples
  * Python embedding guide with pandas/cuDF integration details
  * Wire protocol JSON format for client-server communication
  * Fix terminology: clarify g._node (node ID column) vs g._nodes (DataFrame)
  * Emphasize GFQL's declarative nature for graph-to-graph transformations
  * Add validation framework documentation with error code reference

## [0.39.1 - 2025-07-07]

### Fixed
* KQL: Fix Kusto syntax error: missing closing parenthesis in graph_query method (#689)

### Docs
* KQL: Add Microsoft Azure Data Explorer (Kusto) demo notebook to documentation TOC
* KQL: Update tutorial text

## [0.39.0 - 2025-06-30]

### Added
* Multi-tenant support: `graphistry.client()` and `graphistry.set_client_for()`
  * Global interface `PyGraphistry` class => Global `GraphistryClient` instance
  * `graphistry.client()` creates an independent `GraphistryClient` instance
* Type annotations added, especially in PlotterBase.py, arrow_uploader.py, and pygraphistry.py

### Changed
* Refactored Kusto and Spanner plugins:
  * Renamed `kustograph.py` to `kusto.py` for consistency
  * Renamed `spannergraph.py` to `spanner.py` for consistency
  * Improved configuration handling and error messages
* Enhanced test coverage with new tests for client_session, kusto, and spanner modules

### Breaking 🔥
* Plugin module renames: `graphistry.plugins.kustograph` → `graphistry.plugins.kusto` and `graphistry.plugins.spannergraph` → `graphistry.plugins.spanner`
* Configuration for `Spanner` now uses `g.configure_spanner(...)` instead of `g.register(spanner_config={...})`
* Configuration for `Kusto` now uses `g.configure_kusto(...)` instead of `g.register(kusto_config={...})`

### Fixed
* Fixed overzelous cache in `ArrowFileUploader`, now uses hashes to memoize uploads

## [0.38.3 - 2025-06-24]

### Fixed
* Fix relative imports in GFQL modules that broke pip install (#681)
  * Replace all `..` relative imports with absolute `graphistry.` imports
  * Add missing `__init__.py` files in `graphistry.models.gfql` subdirectories
  * Add lint check in `bin/lint.sh` to prevent future relative imports
  * Add Docker-based pip install test to CI pipeline

## [0.38.2 - 2025-06-24]

### Added
* GFQL temporal predicates and type system for date/time comparisons
  * Support for datetime, date, and time comparisons with operators: `gt`, `lt`, `ge`, `le`, `eq`, `ne`, `between`, `is_in`
  * Proper timezone handling for datetime comparisons
  * Type-safe temporal value handling with TypeGuard annotations
  * Temporal value classes: `DateTimeValue`, `DateValue`, `TimeValue` for explicit temporal types
  * Wire protocol support for JSON serialization of temporal predicates
  * Comprehensive documentation: datetime filtering guide, wire protocol reference, and examples notebook

## [0.38.1 - 2025-06-24]

### Breaking
* Plottable is now a Protocol
* py.typed added, type checking active on PyGraphistry!
* transform() and transform_umap() now require some parameters to be keyword-only

### Added
* CI: Enable notebook validation by default in docs builds (set VALIDATE_NOTEBOOK_EXECUTION=0 to disable)
* CI: Run notebook validation after doc generation for faster error detection

### Fixed
* Fix Sphinx documentation build errors in docstrings for kusto and spanner methods
* Fix toctree references to use correct file names without extensions
* Remove inherited members from PyTorch nn.Module in RGCN documentation to avoid formatting conflicts
* Fix Unicode characters in datetime_filtering.md for LaTeX compatibility

## [0.38.0 - 2025-06-17]

### Changed
* PyPI publish workflow now uses Trusted Publishing (OIDC) instead of password authentication

## [0.38.0 - 2025-06-17]

### Feat
* Kusto/Azure Data Explorer integration. `PyGraphistry.kusto()`, `kusto_query()`, `kusto_query_graph()`
* Extra kusto install target `pip install graphistry[kusto]` installs azure-kusto-data, azure-identity

### Fixed
* Fix sentence transformer model name handling to support both legacy format and new organization-prefixed formats (e.g., `mixedbread-ai/mxbai-embed-large-v1`)

### Changed
* Legacy `Plottable.spanner_init()` & `PyGraphistry.spanner_init()` helpers no longer shipped. Use `spanner()`

### Breaking
* Kusto device authentication doesn't persist.

### Test
* Add comprehensive tests for sentence transformer model name formats including legacy, organization-prefixed, and local path formats

## [0.37.0 - 2025-06-05]

### Fixed

* Fix embed_utils.py modifying global logging.StreamHandler.terminator ([#660](https://github.com/graphistry/pygraphistry/issues/660)) ([8480cd06](https://github.com/graphistry/pygraphistry/commit/8480cd06))

### Breaking 🔥

* `FeatureMixin.transform()` now raises `ValueError` for invalid `kind` parameter instead of silently continuing ([25e4bf51](https://github.com/graphistry/pygraphistry/commit/25e4bf51))
* `FeatureMixin._transform()` now raises `ValueError` when encoder is not initialized instead of returning `None` ([25e4bf51](https://github.com/graphistry/pygraphistry/commit/25e4bf51))
* `UMAPMixin.transform_umap()` now always returns `pd.DataFrame` (possibly empty) instead of `None` for `y_` in tuple return ([d2941ec4](https://github.com/graphistry/pygraphistry/commit/d2941ec4))

### Chore

* Switch to setup_logger utility in multiple modules ([842fb904](https://github.com/graphistry/pygraphistry/commit/842fb904))
* Add AI_PROGRESS/ and PLAN.md to .gitignore ([f0c18b3b](https://github.com/graphistry/pygraphistry/commit/f0c18b3b), [ac25a356](https://github.com/graphistry/pygraphistry/commit/ac25a356))

### Docs

* Add AI assistant prompt templates and conventional commits guidance ([a52048a7](https://github.com/graphistry/pygraphistry/commit/a52048a7))
* Simplify CLAUDE.md to point to ai_code_notes README ([e5393381](https://github.com/graphistry/pygraphistry/commit/e5393381))
* Update AI assistant documentation with Docker-first testing ([db5496eb](https://github.com/graphistry/pygraphistry/commit/db5496eb))

## [0.36.2 - 2025-05-16]

### Feat

* GFQL: Hop pattern matching now supports node ID column having same name as edge source or destination column

### Perf

* GFQL: Optimize hop operations with improved memory usage and reduced redundancy 

### Test

* GFQL: Comprehensive tests for column name conflicts in chain pattern matching

### Infra

* Add CLAUDE.md with performance guidelines

## [0.36.1 - 2025-04-17]

### Feat

* Add "erase_files_on_fail" option to plot and upload functions

## [0.36.0 - 2025-02-05]

### Breaking

* `from_cugraph` returns using the src/dst bindings of `cugraph.Graph` object instead of base `Plottable`
* `pip install graphistry[umap-learn]` and `pip install graphistry[ai]` are now Python 3.9+ (was 3.8+)
* `Plottable`'s fields `_node_dbscan` / `_edge_dbscan` are now `_dbscan_nodes` / `_dbscan_edges`

### Feat

* Switch to `skrub` for feature engineering
* More AI methods support GPU path
* Support cugraph 26.10+, numpy 2.0+
* Add more umap, dbscan fields to `Plottable`

### Infra

* `[umap-learn]` + `[ai]` unpin deps - scikit, scipy, torch (now 2), etc

### Refactor

* Move more type models to models/compute/{feature,umap,cluster}
* Turn more print => logger

### Fixes

* Remove lint/type ignores and fix root causes

### Tests

* Stop ignoring warnings in featurize and umap
* python version tests use corresponding python version for mypy
* ci umap tests: py 3.8, 3.9 => 3.9..3.12
* ci ai tests: py 3.8, 3.9 => 3.9..3.12
* ci tests dgl
* plugin tests check for module imports

## [0.35.10 - 2025-01-24]

### Fixes: 

* Spanner: better handling of spanner_config issues: #634, #644

## [0.35.9 - 2025-01-22]

### Docs 

* Spanner: minor changes to html and markdown in notebook for proper rendering in readthedocs 

## [0.35.8 - 2025-01-22]

### Docs

* Spanner: fix for plots rendering in readthedocs demo notebooks


## [0.35.7 - 2025-01-22]

### Feat 

* added support for Google Spanner Graph and Google Spanner `spanner_gql_to_g` and `spanner_query_to_df` 
* added new Google Spanner Graph demo notebook 

## [0.35.6 - 2025-01-11]

### Docs

* Fix typo in new shaping tutorial
* Privacy-preserving analytics

## [0.35.5 - 2025-01-10]

### Docs

* New tutorial on graph shaping

## [0.35.4 - 2024-12-28]

### Fixes

* `Plottable._render` now typed as `RenderModesConcrete`
* Remote GFQL - Handle `output_type is None` 

## [0.35.3 - 2024-12-24]

### Docs

* Rename CONTRIBUTE.md to CONTRIBUTING.md to match OSS standards (snyk)
* setup.py: add project_urls
* Add FUNDING.md
* Add CODE_OF_CONDUCT.md

### Feat

* GFQL serialization: Edge AST node deserializes as more precise `ASTEdge` subclasses

### Fixes

* GFQL Hop: Detect #614 of node id column name colliding with edge src/dst id column name and raise `NotImplementedError`
* MyPy: Remove explicit type annotation from Engine

## [0.35.2 - 2024-12-13]

### Docs

* Python remote mode notebook: Fixed engine results
* Python remote mode: Add JSON example

## [0.35.1 - 2024-12-11]

### Fixes

* Fix broken imports in new GFQL remote mode

## [0.35.0 - 2024-12-10]

### Docs

* New tutorial on GPU memory management and capacity planning in the GPU section
* New tutorial on remote-mode GFQL
* New tutorial on remote-mode Python

### Feat

* `plot(render=)` supports literal-typed mode values: `"auto"`, `"g"`, `"url"`, `"ipython"`, `"databricks"`, where `"g"` is a new Plottable
* Remote metadata: Expose and track uploaded `._dataset_id`, `._url`, `._nodes_file_id`, and `._edges_file_id`
* Remote upload: Factor out explicit upload method `g2 = g1.upload(); assert all([g2._dataset_id, g2._nodes_file_id, g2._edges_file_id])` from plot interface
* Remote dataset: Remote dataset binding via `g1 = graphistry.bind(dataset_id='abc123')`
* Remote GFQL: Remote GFQL calls via `g2 = g1.chain_remote([...])` and `meta_df = g1.chain_remote_shape([...])`
* Remote GPU Python: Remote Python calls via `g2 = g1.python_remote_g(...)`, `python_remote_table()`, and `python_remote_json()` for different return types

### Changed

* `plot(render=)` now `Union[bool, RenderMode]`, not just `bool` 

## [0.34.17 - 2024-10-20]

### Added

* Layout: `circle_layout()` that moves points to one or more rings based on an ordinal field such as degree
* Layout: `fa2_layout()` that uses the ForceAtlas2 algorithm to lay out the graph, with some extensions.
* Layout: `group_in_a_box_layout()` is significantly faster in CPU and GPU mode
* Layout: `group_in_a_box_layout()` exposes attribute `_partition_offsets`
* Compute methods `g.to_pandas()` and `g.to_cudf()`

### Fix

* cugraph methods now handle numerically-typed node/edge index columns
* repeat calls to `fa2_layout` and `group_in_a_box_layout` now work as expected

### Docs

* Add group-in-a-box layout tutorial

### Infra

* Expose `scene_settings` in `Plottable`

## [0.34.16 - 2024-10-13]

### Docs

* Update and streamline readme.md
* Add quicksheet for overall
* More crosslinking

### Infra

* Add markdown support to docsite
* ReadTheDocs homepage reuses github README.md
* Docs pip install caches
* Drop SVGs and external images during latexpdf generation

### Changed

* Treemap import `squarify` deferred to use to allow core import without squarify installed, such as in `--no-deps`

## [0.34.15 - 2024-10-11]

### Docs

* Improve GFQL translation doc
* Add examples and API links: Shaping, Hypergraphs, AI & ML
* Add performance docs
* Add AI examples

## [0.34.14 - 2024-10-09]

### Added

* HTTP responses with error status codes log an `logging.ERROR`-level message of the status code and response body

## [0.34.13 - 2024-10-07]

### Docs

* Add more GFQL cross-references

## [0.34.12 - 2024-10-07]

### Docs

* Fix ipynb examples in ReadTheDocs distribution

## [0.34.11 - 2024-10-07]

### Fix

* Types

### Infra

* Enable more Python version checks

## [0.34.10 - 2024-10-07]

### Fix

* Docs: Notebook builds

### Docs

* More links, especially around plugins
* Update color theme to match Graphistry branding

## [0.34.9 - 2024-10-07]

### Fix

* Docs: 10 Minutes to PyGraphistry links

## [0.34.8 - 2024-10-06]

### Fix

* Docs: PDF support
* Docs: Links

### Docs

* More accessible theme

## [0.34.7 - 2024-10-06]

### Docs

* RTD: Added notebook tutorials 
* RTD: Added various guides
* RTD: Added cross-references
* RTD: Cleaner navigation

### Infra

* Python: Add Python 12 to CI and document support
* Docs: Udated dependencies - Sphinx 8, Python 12, and various related
* Docs: Added nbsphinx - hub url grounding, ...
* Docs: Redo as a docker compose flow with incremental builds (docker, sphinx)
* Docs: Updated instructions for new flow

### Fix

* Docs: 2024
* Notebooks: Compatibility with nbsphinx - exactly one title heading, no uncommented `!`, correct references, ...

## [0.34.6 - 2024-10-04]

### Added

* Plugins: graphviz bindings, such as `g.layout_graphviz("dot")`

### Docs

* Reorganized readthedocs
* Added intro tutorials: `10 Minutes to PyGraphistry`, `10 Minutes to GFQL`, `Login and Sharing`

## [0.34.5 - 2024-09-23]

### Fixed

* GFQL: Fix `chain()` regression around an incorrectly disabled check manifesting as https://github.com/graphistry/pygraphistry/issues/583
* GFQL: Fix `chain()`, `hop()` traverse filtering logic for a multi-hop edge scenarios
* GFQL: Fix `hop()` predicate handling in multihop scenarios

### Infra

* GFQL: Expand test suite around multihop edge predicates in `hop()` and `chain()`

## [0.34.4 - 2024-09-20]

### Added

* UMAP: Optional kwargs passthrough to umap library constructor, fit, and transform methods: `g.umap(..., umap_kwargs={...}, umap_fit_kwargs={...}, umap_transform_kwargs={...})`
* Additional GPU support in featurize paths

### Changed

* Replace `verbose` with `logging`

### Refactor

* Narrow `use_scaler` and `use_scaler_target` typing to `ScalerType` (`Literal[...]`) vs `str`
* Rename `featurize_or_get_nodes_dataframe_if_X_is_None` (and edges variant) as non-private due to being shared

### Fixed

* get_indegrees: Fix warning https://github.com/graphistry/pygraphistry/issues/587

## [0.34.3 - 2024-08-03]

### Added

* Layout `modularity_weighted_layout` that uses edge weights to more strongly emphasize community structure

### Docs

* Tutorial for `modularity_weighted_layout`

### Infra

* Upgrade tests to`docker compose` from `docker-compose` 
* Remove deprecated `version` to address warnings

## [0.34.2 - 2024-07-22]

### Fixed

* Graceful CPU fallbacks: When lazy GPU dependency imports throw `ImportError`, commonly seen due to broken CUDA environments or having CUDA libraries but no GPU, warn and fall back to CPU.

* Ring layouts now support filtered inputs, giving expected positions

* `encode_axis()` updates are now functional, not inplace

### Changed

* Centralize lazy imports into `graphistry.utils.lazy_import`
* Lazy imports distinguish `ModuleNotFound` (=> `False`) from `ImportError` (warn + `False`)

## [0.34.1 - 2024-07-17]

### Infra

* Upgrade pypi automation to py3.8

## [0.34.0 - 2024-07-17]

### Added

* Ring layouts: `ring_categorical_layout()`, `ring_continuous_layout()`, `time_ring_layout()`
* Plottable interface includes `encode_axis()`, `settings()`
* Minimal global config manager

### Infra

* Test GPU infra updated to Graphistry 2.41 (RAPIDS 23.10, CUDA 11.8)
* Faster test preamble
* More aggressive low-memory support in GPU UMAP unit tests 

### Fixed

* cudf materialize nodes auto inference
* workaround feature_utils typecheck fail 

### Docs

* Ring layouts

### Breaking 🔥

* Dropping support for Python 3.7 (EOL)

## [0.33.9 - 2024-07-04]

### Added

* Added `personalized_pagerank` to the list of supported `compute_igraph` algorithms.

## [0.33.8 - 2024-04-30]

### Fixed

* Fix from_json when json object contains predicates.

## [0.33.7 - 2024-04-06]

### Fixed

* Fix refresh() for SSO

## [0.33.6 - 2024-04-05]

### Added

* `featurize()`, on error, coerces `object` dtype cols to `.astype(str)` and retries

## [0.33.5 - 2024-03-11]

### Fixed

* Fix upload-time validation rejecting graphs without a nodes table

## [0.33.4 - 2024-02-29]

### Added

* Fix validations import.

## [0.33.3 - 2024-02-28]

### Added

* Validations for dataset encodings.

## [0.33.2 - 2024-02-24]

### Added

* GFQL: Export shorter alias `e` for `e_undirected`
* Featurize: More auto-dropping of non-numerics when no `dirty_cat`

### Fixed

* GFQL: `hop()` defaults to `debugging_hop=False`
* GFQL: Edge cases around shortest-path multi-hop queries failing to enrich against target nodes during backwards pass

### Infra

* Pin test env to work around test fails: `'test': ['flake8>=5.0', 'mock', 'mypy', 'pytest'] + stubs + test_workarounds,` +  `test_workarounds = ['scikit-learn<=1.3.2']`
* Skip dbscan tests that require umap when it is not available

## [0.33.0 - 2023-12-26]

### Added

* GFQL: GPU acceleration of `chain`, `hop`, `filter_by_dict`
* `AbstractEngine`  to `engine.py::Engine` enum
* `compute.typing.DataFrameT` to centralize df-lib-agnostic type checking

### Refactor

* GFQL and more of compute uses generic dataframe methods and threads through engine

### Infra

* GPU tester threads through LOG_LEVEL

## [0.32.0 - 2023-12-22]

### Added

* GFQL `Chain` AST object
* GFQL query serialization - `Chain`, `ASTObject`, and `ASTPredict` implement `ASTSerializable`
  - Ex:`Chain.from_json(Chain([n(), e(), n()]).to_json())`
* GFQL predicate `is_year_end`

### Docs

* GFQL in readme.md

### Changes

* Refactor `ASTEdge`, `ASTNode` field naming convention to match other `ASTSerializable`s

### Breaking 🔥

* GFQL `e()` now aliases `e_undirected` instead of the base class `ASTEdge`

## [0.31.1 - 2023-12-05]

### Docs

* Update readthedocs yml to work around ReadTheDocs v2 yml interpretation regressions
* Make README.md pass markdownlint
* Switch markdownlint docker channel to official and pin

## [0.30.0 - 2023-12-04]

### Added

* Neptune: Can now use PyGraphistry OpenCypher/BOLT bindings with Neptune, in addition to existing Gremlin bindings
* chain/hop: `is_in()` membership predicate, `.chain([ n({'type': is_in(['a', 'b'])}) ])`
* hop: optional df queries - `hop(..., source_node_query='...', edge_query='...', destination_node_query='...')`
* chain: optional df queries:
  - `chain([n(query='...')])`
  - `chain([e_forward(..., source_node_query='...', edge_query='...', destination_node_query='...')])`
* `ASTPredicate` base class for filter matching
* Additional predicates for hop and chain match expressions:
  - categorical: is_in (example above), duplicated
  - temporal: is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end, is_leap_year
  - numeric: gt, lt, ge, le, eq, ne, between, isna, notna
  - str: contains, startswith, endswith, match, isnumeric, isalpha, isdigit, islower, isupper, isspace, isalnum, isdecimal, istitle, isnull, notnull

### Fixed

* chain/hop: source_node_match was being mishandled when multiple node attributes exist
* chain: backwards validation pass was too permissive; add `target_wave_front` check`
* hop: multi-hops with `source_node_match` specified was not checking intermediate hops
* hop: multi-hops reverse validation was mishandling intermediate nodes
* compute logging no longer default-overrides level to DEBUG

### Infra

* Docker tests support LOG_LEVEL

### Changed

* refactor: move `is_in`, `IsIn` implementations to `graphistry.ast.predicates`; old imports preserved
* `IsIn` now implements `ASTPredicate`
* Refactor: use `setup_logger(__name__)` more consistently instead of `logging.getLogger(__name__)`
* Refactor: drop unused imports
* Redo `setup_logger()` to activate formatted stream handler iff verbose / LOG_LEVEL

### Docs

* hop/chain: new query and predicate forms
* hop/chain graph pattern mining tutorial: [ipynb demo](demos/more_examples/graphistry_features/hop_and_chain_graph_pattern_mining.ipynb)
* Neptune: Initial tutorial for using PyGraphistry with Amazon Neptune's OpenCypher/BOLT bindings

## [0.29.7 - 2023-11-02]

### Added

* igraph: support `compute_igraph('community_optimal_modularity')`
* igraph: `compute_igraph('articulation_points')` labels nodes that are articulation points

### Fixed

* Type error in arrow uploader exception handler
* igraph: default coerce Graph-type node labels to strings, enabling plotting of g.compute_igraph('k_core')
* igraph: fix coercions when using numeric IDs that were confused by igraph swizzling

### Infra

* dask: Fixed parsing error in hypergraph dask tests
* igraph: Ensure in compute_igraph tests that default mode results coerce to arrow tables
* igraph: Test chaining
* tests: mount source folders to enable dev iterations without rebuilding

## [0.29.6 - 2023-10-23]

### Docs

* Memgraph: Add tutorial (https://github.com/graphistry/pygraphistry/pull/507 by https://github.com/karmenrabar)

### Fixed

* Guard against potential `requests`` null dereference in uploader error handling

### Security

* Add control `register(..., sso_opt_into_type='browser' | 'display' | None)`
* Fix display of SSO URL

## [0.29.5 - 2023-08-23]

### Fixed

* Lint: Update flake8 in test
* AI: UMAP ignores reserved columns and fewer exceptions on low dimensionaltiy

## [0.29.4 - 2023-08-22]

### Fixed

* Lint: Dynamic type checks

### Infra

* Adding Python 3.10, 3.11 to more of test matrix
* Unpin `setuptools` and `pandas`
* Fix tests that were breaking on pandas 2+

## [0.29.3 - 2023-07-19]

### Changed

* igraph: Change dependency to new package name before old deprecates (https://github.com/graphistry/pygraphistry/pull/494)

## [0.29.2 - 2023-07-10]

### Fixed

* Security: Allow token register without org
* Security: Refresh logic

## [0.29.1 - 2023-05-03]

### Fixed 

* AI: cuml OOM fix [#482](https://github.com/graphistry/pygraphistry/pull/482)

## [0.29.0 - 2023-05-01]

### Breaking 🔥
* AI: moves public `g.g_dgl` from KG `embed` method to private method `g._kg_dgl`
* AI: moves public `g.DGL_graph` to private attribute `g._dgl_graph`
* AI: To return matrices during transform, set the flag:  `X, y = g.transform(df, return_graph=False)` default behavior is ~ `g2 = g.transform(df)` returning a `Plottable` instance. 

## [0.28.7 - 2022-12-22]

### Added
* AI: all `transform_*` methods return graphistry Plottable instances, using an infer_graph method. To return matrices, set the `return_graph=False` flag. 
* AI: adds `g.get_matrix(**kwargs)` general method to retrieve (sub)-feature/target matrices
* AI: DBSCAN -- `g.featurize().dbscan()` and `g.umap().dbscan()` with options to use UMAP embedding, feature matrix, or subset of feature matrix via `g.dbscan(cols=[...])`
* AI: Demo cleanup using ModelDict & new features, refactoring demos using `dbscan` and `transform` methods.
* Tests: dbscan tests
* AI: Easy import of featurization kwargs for `g.umap(**kwargs)` and `g.featurize(**kwargs)`
* AI: `g.get_features_by_cols` returns featurized submatrix with `col_part` in their columns
* AI: `g.conditional_graph` and `g.conditional_probs` assessing conditional probs and graph
* AI Demos folder: OSINT, CYBER demos
* AI: Full text & semantic search (`g.search(..)` and `g.search_graph(..).plot()`)
* AI: Featurization: support for dataframe columns that are list of lists -> multilabel targets
                  set using `g.featurize(y=['list_of_lists_column'], multilabel=True,...)`
* AI: `g.embed(..)` code for fast knowledge graph embedding (2-layer RGCN) and its usage for link scoring and prediction
* AI: Exposes public methods `g.predict_links(..)` and `g.predict_links_all()`
* AI: automatic naming of graphistry objects during `g.search_graph(query)` -> `g._name = query`
* AI: RGCN demos - Infosec Jupyterthon 2022, SSH anomaly detection

### Fixed

* GIB: Add missing import during group-in-a-box cudf layout of 0-degree nodes
* Tests: SSO login tests catch more unexpected exns

## [0.28.6 - 2022-11-29]


### Added

* Personal keys: `register(personal_key_id=..., personal_key_secret=...)`
* SSO: `register()` (no user/pass), `register(idp_name=...)` (org-specific IDP)

### Fixed

* Type errors

## [0.28.4 - 2022-10-22]

### Added

* AI: `umap(engine='cuml')` now supports older RAPIDS versions via knn fallback for edge creation. Also: `"umap_learn"`, defaults to `"auto"`
* `prune_self_edges()` to drop any edges where the source and destination are the same

### Fixed

* Infra: Updated github actions versions and Ubuntu environment for publishing 

## [0.28.3 - 2022-10-12]

### Added

* AI: full text & semantic search (`g.search(..)` and `g.search_graph(..).plot()`)
* Featurization: support for dataframe columns that are list of lists -> multilabel targets
                  set using `g.featurize(y=['list_of_lists_column'], multilabel=True,...)`
                  Only supports single-column data targets


## [0.28.2 - 2022-10-11]

### Changed

* Infra: Updated github actions

### Fixed

* `encode_axis()` now correctly sets axis
* work around mypy mistyping operator & on pandas series 

## [0.28.1 - 2022-10-06]

### Changed

* Speed up `g.umap()` >100x by using cuML UMAP engine
* Drop official support for Python 3.6 - its LTS security support stopped 9mo ago
* neo4j: v5 support - backwards-compatible changing derefs from id to element_id

### Added

* umap: Optional `engine` parameter (default `cuml`) for `UMAP()`
* ipynb: UMAP purpose, functionality and parameter details, with general UMAP notebook planned in future (features folder)

### Fixed

* has_umap: removed as no longer necessary

## [0.28.0 - 2022-09-23]

### Added

* neo4j: v5 support (experimental)

### Changed

* Infra: suppress igraph pandas FutureWarnings

## [0.27.3 - 2022-09-07]

### Changed

* Infra: Remove heavy AI dependencies from `pip install graphistry[dev]`

### Added

* igraph: Optional `use_vids` parameter (default `False`) for `to_igraph()` and its callers (`layout_igraph`, `compute_graph`)
* igraph: add `coreness` and `harmonic_centrality` to `compute_igraph`

### Fixed

* igraph: CI errors around igraph
* igraph: Tolerate deprecation warning of `clustering`
* Docs: Typos and updates - thanks @gadde5300 + @szhorvat !

## [0.27.2 - 2022-09-02]

### Changed

* Speed up `import graphistry` 10X+ by lazily importing AI dependencies. Use of `pygraphistry[ai]` features will still trigger slow upstream dependency initialization times upon first use.

### Fixed

* Docs: Update Labs references to Hub

## [0.27.1 - 2022-07-25]

### Fixed

* `group_in_a_box_layout()`: Remove verbose output
* `group_in_a_box_layout()`: Remove synthesized edge weight

## [0.27.0 - 2022-07-25]

### Breaking 🔥

* Types: Switch `materialize_nodes` engine param to explicitly using`Engine` typing (no change to untyped user code)

### Added

* `g.keep_nodes(List or Series)`
* `g.group_in_a_box_layout(...)`: Both CPU (pandas/igraph) and (cudf/cugraph) versions, and various partitioning/layout/styling settings
* Internal clientside Brewer palettes helper for categorical point coloring

### Changed

* Infra: CI early fail on deeper lint
* Infra: Move Python 3.6 from core to minimal tests due to sklearn 1.0 incompatibility

### Fixed

* lint
* suppress known dgl bindings test type bug

## [0.26.1 - 2022-07-01]

### Breaking 🔥

* `_table_to_arrow()` for `cudf`: Updated for RAPIDS 2022.02+ to handle deprecation of `cudf.DataFrame.hash_columns()` in favor of new `cudf.DataFrame.hash_values()`

### Added

* `materialize_nodes()`: Supports `cudf`, materializing a `cudf.DataFrame` nodes table when `._edges` is an instance of `cudf.DataFrame`
* `to_cugraph()`, `from_cugraph()`, `compute_cugraph()`, `layout_cugraph()`
* docs: [cugraph demo notebook](demos/demos_databases_apis/gpu_rapids/cugraph.ipynb)

### Changed

* Infra: Update GPU test env settings
* `materialize_nodes`: Return regular index

### Fixed

* `hypergraph()` in dask handles failing metadata type inference
* tests: gpu env tweaks
* tests: umap logging was throwing warnings

## [0.26.0 - 2022-06-03]

### Added
* `g.transform()`
* `g.transform_umap()`
* `g.scale()`
* Memoization on UMAP and Featurize calls
* Adds **kwargs and propagates them through to different function calls (featurize, umap, scale, etc)

### Breaking 🔥

* Final deprecation of `register(api=2)` protobuf/vgraph mode - also works around need for protobuf test upgrades

## [0.25.3 - 2022-06-22]

### Added

* `register(..., org_name='my_org')`: Optionally upload into an organization
* `g.privacy(mode='organization')`: Optionally limit sharing to within your organization

### Changed

* docs: `org_name` in `README.md` and sharing tutorial

## [0.25.2 - 2022-05-11]

### Added

* `compute_igraph()`
* `layout_igraph()`
* `scene_settings()`

### Fixed

* `from_igraph` uses `g._node` instead of `'name'` in more cases 

## [0.25.1 - 2022-05-08]

### Fixed

* `g.from_igraph(ig)` will use IDs (ex: strings) for src/dst values instead of igraph indexes

## [0.25.0 - 2022-05-01]

Major version bump due to breaking igraph change

### Added

* igraph handlers: `graphistry.from_igraph`, `g.from_igraph`, `g.to_igraph`
* docs: README.md examples of using new igraph methods

### Changed

* Deprecation warnings in old igraph methods: `g.graph(ig)`, `igraph2pandas`, `pandas2igraph`
* Internal igraph handlers upgraded to use new igraph methods 

### Breaking 🔥

* `network2igraph` and `igraph2pandas` renamed output node ID column to `_n_implicit` (`constants.NODE`)

## [0.24.1 - 2022-04-29]

### Fixed

* Expose symbols for `.chain()` predicates as top-level: previous `ast` export was incorrect

## [0.24.0 - 2022-04-29]

Major version bump due to large dependency increases for kitchen-sink installs and overall sizeable new feature

### Added

* Use buildkit with pip install caching for test dockerfiles
* Graph AI branch: Autoencoding via dirty_cat and sentence_transformers (`g.featurize()`)
* Graph AI branch: UMAP via umap_learn (`g.umap()`)
* Graph AI branch: GNNs via DGL (`g.build_dgl_graph()`)
* `g.reset_caches()` to clear upload and compute caches (last 100)
* Central `setup_logger()`
* Official Python 3.10 support

### Changed

* Logging: Refactor to `setup_logger(__name__)`

### Fixed

* hypergraph: use default logger instead of DEBUG

## [0.23.3 - 2022-04-23]

### Added

* `g.collapse(node='root_id', column='some_col', attribute='some_val')

## [0.23.2 - 2022-04-11]

### Fixed

* Avoid runtime import exn when on GPU-less systems with cudf/dask_cudf installed

## [0.23.1 - 2022-04-08]

### Added

* Docs: `readme.md` digest of compute methods

### Fixed

* Docs: `get_degree()` -> `get_degrees()` (https://github.com/graphistry/pygraphistry/issues/330)
* Upload memoization handles column renames (https://github.com/graphistry/pygraphistry/issues/326)

## [0.23.0 - 2022-04-08]

### Breaking 🔥

* `g.edges()` now takes an optional 4th named parameter `edge` ID

Code that looks like `g.edges(some_fn, None, None, some_arg)` should now be like `g.edges(some_fn, None, None, None, some_arg)`

* Similar new optional `edge` ID parameter in `g.bind()`

### Changed

* `g.hop()` now takes optional `return_as_wave_front=False`, primarily for internal use by `chain()`

### Added

* `g.chain([...])` with `graphistry.ast.{n, e_forward, e_reverse, e_undirected}`

## [0.22.0 - 2022-04-06]

### Added

* Node dictionary-based filtering: `g.filter_nodes_by_dict({"some": "value", "another": 2})`
* Edge dictionary-based filtering: `g.filter_edges_by_dict({"some": "value", "another": 2})`
* Hops support edge filtering: `g.hop(hops=2, edge_match={"type": "transaction"})`
* Hops support pre-node filtering: `g.hop(hops=2, source_node_match={"type": "account"})`
* Hops support post-node filtering: `g.hop(hops=2, destination_node_match={"type": "wallet"})`
* Hops defaults to full graph if no initial nodes specified: `g.hop(hops=2, edge_match={"type": "transaction"})`

## [0.21.4 - 2022-03-30]

### Added

* Horizontal and radial axis using `.encode_axis(rows=[...])`

### Fixed

* Docs: Work around https://github.com/sphinx-doc/sphinx/issues/10291

## [0.21.0 - 2022-03-13]

### Added

* Better implementation of `.tree_layout(...)` using Sugiyama; good for small/medium DAGs
* Layout rotation method `.rotate(degree)`
* Compute method `.hops(nodes, hops, to_fixed_point, direction)`

### Changed

* Infra: `test-cpu-local-minimum.sh` accepts params

## [0.20.6 - 2022-03-12]

### Fixed

* Docs: Point color encodings

## [0.20.5 - 2021-12-06]

### Changed

* Unpin Networkx

### Fixed

* Docs: Removed deprecated `api=1`, `api=2` registration calls (#280 by @pradkrish)
* Docs: Fixed bug in honeypot nb (#279 by @pradkrish)
* Tests: Networkx test version sniffing

## [0.20.3 - 2021-11-21]

### Added

* Databricks notebook connector ([PR](https://github.com/graphistry/pygraphistry/pull/277))
* Databricks notebook + dashboard example ([PR](https://github.com/graphistry/pygraphistry/pull/277), [ipynb](https://github.com/graphistry/pygraphistry/blob/ad31a227136430bcd578feac1c18e90920ab4f00/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb))

## [0.20.2 - 2021-10-18]

### Added

* Docs: [umap_learn tutorial notebook](https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/umap_learn/umap_learn.ipynb)

## [0.20.1 - 2021-08-24]

### Added

* Docs: Sharing control [demos/more_examples/graphistry_features/sharing_tutorial.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb)

## [0.20.0 - 2021-08-24]

### Added
* Feature: global `graphistry.privacy()` and compositional `Plotter.privacy()`
* Docs: How to use `privacy()`

### Changed

* Docs: Start removing deprecated 1.0 API docs

## [0.19.6 - 2021-08-15]

### Fixed

* Fix: NetworkX 2.5+ support - accept minor version tags

## [0.19.5 - 2021-08-15]

### Fixed

* Fix: igraph `.plot()` arrow coercion syntax error (https://github.com/graphistry/pygraphistry/issues/257)
* Fix: Lint duplicate import warning

### Changed

* CI: Treat lint warnings as CI failures


## [0.19.4 - 2021-07-22]

### Added
* Infra: Add CI stage that installs and tests with minimal core deps (https://github.com/graphistry/pygraphistry/issues/254)

### Fixed
* Fix: Core tests pass with minimal install dependencies (https://github.com/graphistry/pygraphistry/issues/253, https://github.com/graphistry/pygraphistry/issues/254)

## [0.19.3 - 2021-07-16]

### Added
* Feature: Compute methods `materialize_nodes`, `get_degrees`, `drop_nodes`, `get_topological_levels`
* Feature: Layout methods `tree_layout`, `layout_settings`
* Docs: New compute and layout methods

## [0.19.2 - 2021-07-14]

### Added
* Feature: `g.fetch_edges()` for neptune/gremlin edge attributes

### Fixed
* Fix: `g.fetch_nodes()` for neptune/gremlin node attrbutes

## [0.19.1 - 2021-07-09]

### Added
* Docs: [demos/demos_databases_apis/neptune/neptune_tutorial.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/neptune/neptune_tutorial.ipynb)

### Changed
* Docs: Updated [demos/for_analysis.ipynb](https://github.com/graphistry/pygraphistry/blob/master/demos/for_analysis.ipynb) to `api=3`

### Fixed
* Fix: Gremlin (Neptune) connector deduplicates nodes/edges

## [0.19.0 - 2021-07-09]

### Added
* Feature: Gremlin connector (GraphSONSerializersV2d0)
* Feature: Cosmos connector
* Feature: Neptune connector
* Feature: Chained composition operators:
  * `g.pipe((lambda g, a1, ...: g2), a1, ...)`
  * `g.edges((lambda g, a1, ...: df), None, None, a1, ...)`
  * `g.nodes((lambda g, a1, ...: df), None, a1, ...)`
* Feature: plotter::infer_labels: Guess node label names when not set, instead of defaulting to node_id. Runs during plots.
* Infra: Jupyter notebook: `cd docker && docker-compose build jupyter && docker-compose up jupyter`
* Docs: Neptune, Cosmos, chained composition

### Changed
* Refactor: Split out PlotterBase, interface Plottable

### Fixed
* Fix: Plotter has `hypergraph()`

## [0.18.2 - 2021-04-29]

### Added
* Docs: security.md

### Fixed

* Hypergraphs - detect and handle mismatching types across partitions

### Changed

* Infra: Speedup testing containers via incrementalization and docker settings
* Infra: Update testing container base builds

## [0.18.1 - 2021-03-26]

### Added

* Feature: Hypergraphs in dask, dask_cudf modes. Mixed nan support. (https://github.com/graphistry/pygraphistry/pull/225)
* Feature: Dask/dask_cuda frames can be passed in, which will be .computed(), memoized, and converted to arrow (https://github.com/graphistry/pygraphistry/pull/225)
* Infra: Test env var controls - WITH_LINT=1, WITH_TYPECHECK=1, WITH_BUILD=1 (https://github.com/graphistry/pygraphistry/pull/225)
* Docs: Inline hypergraph examples  (https://github.com/graphistry/pygraphistry/pull/225)

### Changed
* CI: Disable seccomp during test (docker perf) (https://github.com/graphistry/pygraphistry/pull/225)

## [0.18.0 - 2021-03-21]

### Added

* Feature: cudf mode for hypergraph (https://github.com/graphistry/pygraphistry/pull/224)
* Feature: pandas mode for hypergraph uses all-vectorized operations (https://github.com/graphistry/pygraphistry/pull/224)
* Infra: Engine class for picking dataframe engine - pandas/cudf/dask/dask_cudf (https://github.com/graphistry/pygraphistry/pull/224)
* CI: mypy type checking (https://github.com/graphistry/pygraphistry/pull/222)
* CI: GPU test harness (https://github.com/graphistry/pygraphistry/pull/223)

### Changed

* Hypergraph: Uses new pandas/cudf implementations (https://github.com/graphistry/pygraphistry/pull/224)

### Added

* Infra: Issue templates for bugs and feature requests

## [0.17.0 - 2021-02-08]

### Added

* Docs: Overhaul Sphinx docs - Update, clean all warnings, add to CI, reject commits that fail
* Docs: Setup.py (pypi) now includes full README.md
* Docs: Added ARCHITECTURE, CONTRIBUTE, and expanded DEVELOP
* Garden: DRY for CI + local dev via shared bin/ scripts
* Docker: Downgrade local dev 3.7 -> 3.6 to more quickly catch minimum version errors
* CI: Now tests building docs (fail on warnings), pypi wheels distro, and neo4j connector

### Breaking 🔥

* Changes in setup.py extras_require: 'all' installs more

## [0.16.3 - 2021-02-08]

### Added

* Docs: ARCHITECTURE.md and CONTRIBUTE.md

### Changed
* Quieted memoization fail warning
* CI: Removed TravisCI in favor of GHA
* CD: GHA now handles PyPI publish on tag push
* Docs: Readme install clarifies Python 3.6+
* Docs: Update DEVELOP.md dev flow

## [0.16.2 - 2021-02-08]

### Added

* Friendlier error message for calling .cypher(...) without setting BOLT auth/driver (https://github.com/graphistry/pygraphistry/issues/204) 
* CI: Run containerized neo4j connector tests
* Infrastructure: Set Python 3.9 support metadata

### Fixed

* Memoization: When memoize hashes throw exceptions, emit warning and fallback to unmemoized (b7a25c74e)

## [0.16.1] - 2021-02-07

### Added

* Friendlier error message for api=1, 2 server non-json responses (https://github.com/graphistry/pygraphistry/pull/187)
* CI: Moved to GitHub Actions for CI + optional manual publish
* CI: Added Python 3.9 to test matrix
* Infrastructure: Upgraded Versioneer to 0.19
* Infrastructure: Fewer warnings and enforce flake8 CI checks

### Breaking 🔥

* None known; many small changes to fix warnings so version bump out of caution

## [0.15.0] - 2021-01-11

### Added
* File API: Enable via `.plot(as_files=True)`. By default, auto-skips file re-uploads (disable via `.plot(memoize=False)`) for tables with same hash as those already uploaded in the same session. Use with `.register(api=3)` clients on Graphistry `2.34`+ servers. More details at  (https://github.com/graphistry/pygraphistry/pull/195) .
* Dev: More docs and logging as part of https://github.com/graphistry/pygraphistry/pull/195
* Auth service account docs in README.md (12.2.2020)

## [0.14.1] - 2020-11-16

### Added
* Examples for icons, badges, and new node/edge bindings
* graph-app-kit links
* Slack link

### Fixed
* Python test matrix: Removed 3.9
* Propagate misformatted etl1/2 server errors 

## [0.14.0] - 2020-10-12

### Breaking 🔥
* Warnings: Standardizing on Python's warnings.warn

### Fixed
* Neo4j: Improve handling of empty query results (https://github.com/graphistry/pygraphistry/issues/178)

### Added
* Icons: Add new as_text, blend_mode, border, and style options (Graphistry 2.32+)
* Badges: Add new badge encodings (Graphistry 2.32.+)
* Python 3.8, 3.9 in test matrix
* New binding shortcuts `g.nodes(df, col)` and `g.nodes(df, src_col, dst_col)`

### Changed
* Python 2.7: Removed __future__ (Python 2.7 has already been EOL so not breaking)
* Redid ipython detection
* Imports: Refactoring for more expected style
* Testing: Fixed most warnings in preperation for treating them as errors
* Testing: Integration tests against self-contained neo4j instance

## [0.13.0] - 2020-09-17

### Added

- Chainable methods `.addStyle()` and `.style()` in `api=3` for controlling foreground, background, logo, and page metadata. Requires Graphistry 2.31.10+ [08eddb8](https://github.com/graphistry/pygraphistry/pull/175/commits/08eddb8fe3ce8ebe66ad2773fc8ee57dfad2dc58)
- Chainable methods `.encode_[point|edge]_[color|icon|size]()` for more powerful *complex* encodings, and underlying generic handler `__encode()`. Requires Graphistry 2.31.10+ [f370ca8](https://github.com/graphistry/pygraphistry/pull/175/commits/f370ca82931e8fb61e40d62ba397b95e0d474f7f)
- More usage examples in README.md

### Changed

- Split `ArrowLoader::*encoding*`methods to `*binding*` vs. `*encoding*` ones to more precisely reflect the protocol. Not considered breaking as an internal method.

## [0.12.0] - 2020-09-08

### Added

- Neo4j 4 temporal and spatial type support - [#172](https://github.com/graphistry/pygraphistry/pull/172)
- CHANGELOG.md

### Changed
- Removed deprecated docker test harness in favor of `docker/` - [#172](https://github.com/graphistry/pygraphistry/pull/172)
