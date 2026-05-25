# GFQL IR Seam Inventory

Last updated: 2026-05-24 PT / 2026-05-25 UTC, from `origin/master` at
`79a61f9ab7d3efa4984007c0ae30660adbd6df4e`, after PR #1457 and subsequent
follow-ons through PR #1630.

This document is the durable inventory for pygraphistry#1580. It marks the
`graphistry.compute.gfql.ir` surfaces that future pygraphistry#1567
IR/verifier shrink work must preserve unless a coordinator-approved migration
plan and replacement tests land first.

## Status Vocabulary

| Status | Meaning for #1567 work |
|---|---|
| `typed-schema-seam` | Protected by #1457/#1337 public schema or typed-schema follow-ons. Do not delete, narrow, or move without a public-schema migration. |
| `IR-metadata-seam` | Protected by the #1568/#1572 metadata restoration audit. Do not inline/delete as "helper residue". |
| `planner/verifier-seam` | Protected compiler-plan, route, pass, query-graph, or verifier contract. Shrink only with explicit replacement proof. |
| `shrink-eligible` | Private implementation detail that may be simplified with caller proof and focused tests. |
| `unknown-investigate` | Treat as protected until a fresh audit proves ownership and replacement coverage. |

## Audit References

- #1568 (`553bc3210ec5ec214f42e671b772939db7d05019`) deleted the
  `graphistry.compute.gfql.ir.metadata` helper layer and re-exports while
  simplifying remaining nullable reads.
- #1572 (`79aec983be8da83a6323acf8d91fb8e0dc6bfcda`) restored
  `metadata.py`, IR package re-exports, and metadata contract tests. The
  #1058 collateral audit found no additional protected-surface deletion beyond
  the intentional direct `.nullable` simplifications retained by #1572.
- #1457 (`3d85344d083295a05c7619baa1a5e415a8d69b8f`) landed the experimental
  public declarative schema model for #1337. That made `GraphSchemaCatalog`,
  `RowSchema`, logical type dataclasses, and the Arrow bridge public-schema
  dependencies, not shrink-only internals.
- #1058 comment
  https://github.com/graphistry/pygraphistry/issues/1058#issuecomment-4501642729
  recorded the #1568/#1572 collateral audit.
- #1058 comment
  https://github.com/graphistry/pygraphistry/issues/1058#issuecomment-4502272980
  parked #1567 until #1457 landed and this inventory was updated against the
  final typed-schema seam surface.

## Package Exports

`graphistry/compute/gfql/ir/__init__.py` is a compatibility surface. The
symbols in its `__all__` are importable package-level contracts.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `__all__` | public package export list | `typed-schema-seam` / `IR-metadata-seam` / `planner/verifier-seam` | Preserve package-level exports unless a migration keeps compatibility imports working. |
| `BoundIR`, `BoundQueryPart`, `BoundVariable`, `ScopeFrame`, `SemanticTable` | public re-export | `IR-metadata-seam` / `planner/verifier-seam` | Preserve binder-to-planner contracts and nullable/scope fields. |
| `CoercionMode`, `SchemaConfidence`, `from_arrow`, `to_arrow` | public re-export | `typed-schema-seam` | Preserve for public schema Arrow export/import. |
| `Decomposable`, `Monotonicity`, `OpCapability` | public re-export | `planner/verifier-seam` | Preserve pass/planner scheduling vocabulary. |
| `BackendCapabilities`, `CompilationState`, `CompilerConfig`, `CompilerError`, `GFQLSchema`, `GraphSchemaCatalog`, `IndexDescriptor`, `PhysicalPlan`, `PlanContext`, `QueryLanguage`, `StatsQuery` | public re-export | `typed-schema-seam` / `planner/verifier-seam` | Preserve schema catalog, planning context, route, diagnostics, and state contracts. |
| Logical-plan operators and helpers (`Aggregate`, `AntiSemiApply`, `Apply`, `Distinct`, `EdgeScan`, `Filter`, `GraphToRows`, `IndexScan`, `Join`, `Limit`, `LogicalPlan`, `NodeScan`, `OrderBy`, `PathProjection`, `PatternMatch`, `ProcedureCall`, `ProcedureOutputColumn`, `Project`, `RowSchema`, `RowsToGraph`, `SemiApply`, `Skip`, `Union`, `Unwind`) | public re-export | `typed-schema-seam` / `planner/verifier-seam` | Preserve operator dataclasses and `RowSchema`; these are verifier, lowering, physical planner, and public schema dependencies. |
| Metadata helpers (`bound_variable_is_nullable`, `bound_variable_type`, `column_is_nullable`, `column_logical_type`, `is_nullable`, `merge_types`, `widen_to_nullable`, `with_nullable`) | public re-export | `IR-metadata-seam` | Do not remove/reinline; #1572 restored this exact seam. |
| Pushdown safety helpers (`is_null_rejecting`, `is_null_safe`, `with_barrier_blocks_pushdown`) | public re-export | `planner/verifier-seam` | Preserve optional-arm predicate-pushdown safety contracts. |
| Query graph helpers (`ConnectedComponent`, `OptionalArm`, `QueryGraph`, `extract_query_graph`) | public re-export | `planner/verifier-seam` | Preserve join-ordering and optional-arm scaffold semantics. |
| Logical types (`BoundPredicate`, `EdgeRef`, `ListType`, `LogicalType`, `NodeRef`, `NodeSpec`, `PathType`, `PatternGraph`, `RelSpec`, `ScalarType`) | public re-export | `typed-schema-seam` / `planner/verifier-seam` | Preserve logical type and pattern contracts; public schema imports several of these directly. |
| `verify` | public re-export | `planner/verifier-seam` | Preserve verifier extension/safety gate. |

## Module Inventory

### `arrow_bridge.py`

Arrow/schema bridge for `RowSchema` and `LogicalType`, now consumed by the
public schema model in `graphistry/schema.py`.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `SchemaConfidence` | public type alias | `typed-schema-seam` | Preserve confidence vocabulary: `declared`, `propagated`, `inferred`. |
| `CoercionMode` | public type alias | `typed-schema-seam` | Preserve `strict`/`widen` behavior for schema Arrow round-trips. |
| `_METADATA_VERSION_KEY`, `_METADATA_VERSION_VALUE`, `_METADATA_LOGICAL_TYPE_KEY`, `_METADATA_CONFIDENCE_KEY` | private constants | `typed-schema-seam` | Private names, but wire metadata keys are a round-trip contract. Do not rename/drop without migration. |
| `_CONFIDENCE_VALUES` | private constant | `typed-schema-seam` | Private support for public `SchemaConfidence`; shrink only if validator behavior is identical. |
| `_SCALAR_KIND_TO_ARROW_FACTORY` | private constant | `typed-schema-seam` | Private mapping, but its scalar coverage is part of Arrow compatibility. |
| `_require_pyarrow` | private helper | `typed-schema-seam` | Keep import-time optional dependency behavior and error text unless tests change. |
| `_ensure_confidence` | private helper | `typed-schema-seam` | Preserve strict/widen fallback diagnostics. |
| `_logical_type_to_payload` | private helper | `typed-schema-seam` | Preserve exact logical-type JSON metadata shape. |
| `_payload_to_logical_type` | private helper | `typed-schema-seam` | Preserve metadata import and tolerant/widen behavior. |
| `_logical_type_to_arrow_type` | private helper | `typed-schema-seam` | Preserve strict rejection and widen string-bridge behavior for structural types. |
| `_arrow_type_to_logical_type` | private helper | `typed-schema-seam` | Preserve Arrow dtype to `ScalarType`/`ListType` mapping. |
| `to_arrow` | public function | `typed-schema-seam` | Preserve public schema export entry point. |
| `from_arrow` | public function | `typed-schema-seam` | Preserve public schema import entry point. |
| `__all__` | public export list | `typed-schema-seam` | Keep export list aligned with public imports. |

### `bound_ir.py`

Frontend binder output consumed by logical planning, query-graph extraction,
pushdown safety, and metadata helpers.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `BoundVariable` | public dataclass | `IR-metadata-seam` / `planner/verifier-seam` | Preserve `logical_type`, `nullable`, `null_extended_from`, `entity_kind`, and `scope_id`. |
| `SemanticTable` | public dataclass | `planner/verifier-seam` | Preserve variable table shape used by binder/lowering/tests. |
| `ScopeFrame` | public dataclass | `IR-metadata-seam` / `planner/verifier-seam` | Preserve `visible_vars`, `schema`, and `origin_clause`; pushdown and typed schema depend on them. |
| `BoundQueryPart` | public dataclass | `planner/verifier-seam` | Preserve clause/input/output/predicate/metadata payloads; query graph and lowering consume them. |
| `BoundIR` | public dataclass | `planner/verifier-seam` | Preserve binder-to-planner package shape. |

### `capabilities.py`

Pass/planner capability vocabulary.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `Decomposable` | public enum | `planner/verifier-seam` | Preserve enum values unless pass scheduling migrates. |
| `Monotonicity` | public enum | `planner/verifier-seam` | Preserve enum values unless pass scheduling migrates. |
| `OpCapability` | public dataclass | `planner/verifier-seam` | Preserve field names and defaults for future pass/planner scheduling. |

### `compilation.py`

Compilation state, planning context, schema catalog, diagnostics, and physical
route contract.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `NodeId` | module alias | `planner/verifier-seam` | Treat as low-risk but keep while private state dictionaries use it. |
| `QueryLanguage` | public enum | `planner/verifier-seam` | Preserve frontend dialect names. |
| `GraphSchemaCatalog` | public dataclass | `typed-schema-seam` | Preserve field names/accessors and `from_schema_parts`; public `GraphSchema` adapts into it. |
| `GraphSchemaCatalog.from_schema_parts` | public classmethod | `typed-schema-seam` | Preserve iterable normalization and metadata copy behavior. |
| `GraphSchemaCatalog.node_id`, `edge_source`, `edge_destination` | public properties | `typed-schema-seam` | Preserve canonical accessors. |
| `GraphSchemaCatalog.has_node_column`, `has_edge_column` | public methods | `typed-schema-seam` | Preserve schema validation helpers. |
| `GFQLSchema` | public alias | `typed-schema-seam` | Compatibility alias; do not remove without migration. |
| `StatsQuery` | public dataclass | `planner/verifier-seam` | Planner hook placeholder; keep unless CBO hooks migrate. |
| `IndexDescriptor` | public dataclass | `planner/verifier-seam` | Preserve index rewrite contract. |
| `BackendCapabilities` | public dataclass | `planner/verifier-seam` | Preserve backend selection hook. |
| `CompilerConfig` | public dataclass | `planner/verifier-seam` | Preserve compiler flags. |
| `PlanContext` | public dataclass | `typed-schema-seam` / `planner/verifier-seam` | Preserve catalog/stats/index/backend/config/scope-stack context. |
| `CompilerError` | public dataclass | `planner/verifier-seam` | Preserve diagnostic payload type until structured diagnostics replace it. |
| `PhysicalPlan` | public dataclass | `planner/verifier-seam` | Preserve `route`, `operators`, `logical_op_ids`, and `metadata`; route names drive execution dispatch. |
| `CompilationState` | public dataclass | `planner/verifier-seam` | Preserve phase state and private accumulator fields unless a pass-manager migration replaces them. |

### `logical_plan.py`

Logical operator dataclasses shared by lowering, rewrites, verifier, and
physical planning.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `CHILD_SLOTS` | public constant | `planner/verifier-seam` | Preserve centralized traversal slot contract. |
| `RowSchema` | public dataclass | `typed-schema-seam` / `IR-metadata-seam` | Preserve row type map; public schema and metadata helpers depend on it. |
| `LogicalPlan` | public dataclass | `planner/verifier-seam` | Preserve `op_id` and `output_schema` base fields. |
| `iter_children` | public function | `planner/verifier-seam` | Preserve traversal semantics for verifier/physical planner/rewrite passes. |
| `NodeScan`, `EdgeScan`, `IndexScan`, `PatternMatch`, `PathProjection`, `Filter`, `Project`, `Aggregate`, `Distinct`, `OrderBy`, `Limit`, `Skip`, `Unwind`, `Union`, `Join`, `GraphToRows`, `RowsToGraph`, `Apply`, `SemiApply`, `AntiSemiApply` | public dataclasses | `planner/verifier-seam` | Preserve operator field names and child slots unless all producers/consumers migrate together. |
| `ProcedureOutputColumn` | public dataclass | `planner/verifier-seam` | Preserve CALL output mapping contract. |
| `ProcedureCall` | public dataclass | `planner/verifier-seam` | Preserve `procedure`, `backend`, `algorithm`, `call_function`, `result_kind`, `row_kind`, `output_columns`, and `call_params`; physical dispatch depends on them. |

### `metadata.py`

Stable type/nullability compatibility seam restored by #1572 after #1568.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `__all__` | public export list | `IR-metadata-seam` | Preserve helper exports. |
| `is_nullable` | public function | `IR-metadata-seam` | Preserve ScalarType-only nullable convention. |
| `with_nullable` | public function | `IR-metadata-seam` | Preserve dataclass-copy/nullability update behavior. |
| `widen_to_nullable` | public function | `IR-metadata-seam` | Preserve optional-arm/outer-join widening helper. |
| `column_logical_type` | public function | `IR-metadata-seam` | Preserve RowSchema lookup helper. |
| `column_is_nullable` | public function | `IR-metadata-seam` | Preserve tri-state scalar/nullability semantics. |
| `merge_types` | public function | `IR-metadata-seam` | Preserve union/outer-join logical type least-upper-bound behavior. |
| `bound_variable_type` | public function | `IR-metadata-seam` | Preserve BoundVariable nullable reconciliation behavior. |
| `bound_variable_is_nullable` | public function | `IR-metadata-seam` | Preserve variable-level nullable source of truth. |

### `pushdown_safety.py`

Predicate pushdown safety primitives for optional-arm null semantics and WITH
barriers.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `_NULL_SAFE_FORMS` | private constant | `planner/verifier-seam` | Private support for null-safety classification; keep unless exact classifier tests migrate. |
| `_mask_quoted_and_backticked` | private helper | `shrink-eligible` | May simplify only with quoted/backticked predicate tests. |
| `_normalize_unquoted_whitespace` | private helper | `shrink-eligible` | May simplify only with whitespace/token tests. |
| `is_null_rejecting` | public function | `planner/verifier-seam` | Preserve conservative optional-arm safety semantics. |
| `is_null_safe` | public function | `planner/verifier-seam` | Preserve inverse helper behavior. |
| `with_barrier_blocks_pushdown` | public function | `planner/verifier-seam` | Preserve WITH-scope barrier check against `ScopeFrame.visible_vars`. |

### `query_graph.py`

Query graph and optional-arm scaffold extracted from `BoundIR`.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `OptionalArm` | public dataclass | `planner/verifier-seam` | Preserve arm id, join aliases, and nullable aliases. |
| `ConnectedComponent` | public dataclass | `planner/verifier-seam` | Preserve component fields for future join ordering. |
| `QueryGraph` | public dataclass | `planner/verifier-seam` | Preserve components, boundary aliases, and optional arms. |
| `_uf_find` | private helper | `shrink-eligible` | May simplify if extraction tests prove identical connected components. |
| `_uf_union` | private helper | `shrink-eligible` | Same as `_uf_find`. |
| `_SCOPE_SPLIT_CLAUSES` | private constant | `planner/verifier-seam` | Preserve WITH/RETURN scope split behavior. |
| `_normalize_clause` | private helper | `shrink-eligible` | May simplify only with scope split tests. |
| `extract_query_graph` | public function | `planner/verifier-seam` | Preserve boundary alias, component, and optional-arm semantics. |

### `types.py`

Logical type and pattern dataclasses used by binder, planner, public schema,
Arrow bridge, metadata helpers, and verifier.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `BoundPredicate` | public dataclass | `planner/verifier-seam` | Preserve expression/reference payload shape. |
| `NodeRef` | public dataclass | `typed-schema-seam` | Preserve label-set semantics. |
| `EdgeRef` | public dataclass | `typed-schema-seam` | Preserve type/source-label/destination-label semantics. |
| `ScalarType` | public dataclass | `typed-schema-seam` / `IR-metadata-seam` | Preserve `kind` and `nullable`; metadata and public schema depend on it. |
| `PathType` | public dataclass | `typed-schema-seam` / `planner/verifier-seam` | Preserve hop bounds. |
| `ListType` | public dataclass | `typed-schema-seam` / `IR-metadata-seam` | Preserve recursive element type. |
| `LogicalType` | public alias | `typed-schema-seam` / `IR-metadata-seam` | Preserve union membership unless all type consumers migrate together. |
| `NodeSpec` | public dataclass | `planner/verifier-seam` | Preserve pattern-graph node shape. |
| `RelSpec` | public dataclass | `planner/verifier-seam` | Preserve relationship pattern fields, direction, hop bounds, fixed-point flag, and predicates. |
| `PatternGraph` | public dataclass | `planner/verifier-seam` | Preserve flat pattern representation. |

### `verifier.py`

Structural verifier safety gate after rewrites and before routed execution.

| Symbol | Public/private | Status | #1567 guidance |
|---|---|---|---|
| `_LOGICAL_TYPES` | private constant | `planner/verifier-seam` | Preserve accepted logical type families. |
| `_MISSING` | private sentinel | `shrink-eligible` | Private implementation detail; change only if dangling-child checks remain equivalent. |
| `_walk` | private helper | `planner/verifier-seam` | Preserve DAG traversal and ancestor-cycle detection. |
| `_check_predicate` | private helper | `planner/verifier-seam` | Preserve predicate expression/reference diagnostics. |
| `verify` | public function | `planner/verifier-seam` | Preserve verifier entry point and invariants. |
| `_check_logical_type` | private helper | `planner/verifier-seam` | Preserve recursive `ListType.element_type` validation. |
| `_check_schema` | private helper | `planner/verifier-seam` | Preserve output schema validation. |
| `_NULLABILITY_NARROWING_OPS` | private constant | `IR-metadata-seam` / `planner/verifier-seam` | Preserve row-dropping operator allowlist for nullability narrowing. |
| `_check_propagation_continuity` | private helper | `IR-metadata-seam` / `planner/verifier-seam` | Preserve shared-column type/nullability continuity checks. |

## Recommended #1567 Guardrails

#1567 can resume only as a guarded audit/shrink lane after coordinator accepts
this inventory. It should not resume as a broad "IR metadata/verifier helper
deletion" lane.

Required guardrails:

1. Classify every touched symbol against this document before editing.
2. Avoid all `typed-schema-seam`, `IR-metadata-seam`, and
   `planner/verifier-seam` symbols unless the PR includes an explicit migration,
   compatibility re-export where applicable, and focused tests proving old and
   new contracts.
3. Treat `metadata.py`, `__init__.py` re-exports, `RowSchema`,
   `GraphSchemaCatalog`, `LogicalType` families, `PhysicalPlan.route`,
   `PhysicalPlan.metadata`, `CHILD_SLOTS`, `iter_children`, `verify`, and
   Arrow metadata keys as explicit do-not-touch seams.
4. Restrict shrink to private helpers marked `shrink-eligible`, and only when
   net production LOC drops without weakening diagnostic/source-span,
   typed-schema, route-selection, optional-arm, or nullability behavior.
5. If a candidate is not listed as `shrink-eligible`, start with
   `unknown-investigate` and file a coordinator note instead of cutting.

Current recommendation: #1567 may resume with these guardrails for narrow
private-helper cleanup, but should remain paused for any deletion that touches
metadata, typed-schema, package exports, logical/physical planning contracts,
or verifier invariants.
