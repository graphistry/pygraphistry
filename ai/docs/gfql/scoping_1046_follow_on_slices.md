# #1046 Follow-on Scoping: Next Executable Child Slices

## Why This Exists

`#1046` is still open, but the staged decomposition program (`#1262`) completed T1-T5 and preflight lane `#1320`. The remaining work is now about productizing those internals into stable user-facing schema APIs and workflows.

## Current Delivered Foundation

Completed lanes already provide:

- Internal schema catalog contracts for binder/planner (`GraphSchemaCatalog`)
- Compile-time and preflight schema validation paths (`gfql_validate`, binder hooks)
- Internal Arrow bridge for logical types (`graphistry.compute.gfql.ir.arrow_bridge`)

These are sufficient as internal building blocks, but they do not yet satisfy the original user-facing shape from `#1046` (declarative schema objects, inference API, integrated Arrow boundary APIs).

## Gap Matrix vs #1046

1. Schema specification: Partially complete (internal catalog exists; public declarative API absent)
2. Schema inference: Not complete (no first-class public `infer_schema()` API)
3. Query validation against schema: Mostly complete for current binder/preflight model; needs integration with declarative schema object model
4. Arrow representation: Internal bridge complete; public Plottable-boundary APIs not complete

## Proposed Child Slices (Execution Order)

### Slice A: Public Declarative Schema Model + Stable Exports (`#1337`)

Goal:
- Add stable, public schema model types for declaration and introspection.

In scope:
- Public module(s) for schema contracts (declarative node/edge/schema types)
- Canonical stable import path(s) with explicit exports
- Adapter between public model and internal `GraphSchemaCatalog`
- Strict/permissive validation policy knobs at schema-binding seam

Out of scope:
- Full inference implementation
- Arrow/plottable I/O integration

Acceptance:
- Public API allows declaring typed node/edge schemas with topology constraints
- Binder/preflight can consume declared schema through adapter path
- Tests cover declaration, adapter, and compile-time validation integration

### Slice B: Schema Inference API + Typed Topology Extraction (`#1338`)

Goal:
- Provide public schema inference from bound node/edge dataframes.

In scope:
- `infer_schema()` entrypoint (or equivalent stable API) returning public schema model
- Property type inference + nullability inference
- Typed topology inference (align with existing typed-topology issue #483)
- Override/refinement flow for user adjustments after inference

Out of scope:
- End-to-end Arrow export enforcement

Acceptance:
- Inference works on representative pandas (and cuDF where available) graphs
- Typed topology extraction is deterministic and test-covered
- Inference output feeds Slice A declaration/binding path without lossy translation

### Slice C: Public Schema/Arrow Boundary APIs for Plottable Workflows (`#1339`)

Goal:
- Expose schema-aware Arrow boundary operations at user-facing APIs.

In scope:
- Public API(s) to convert schema model to/from Arrow schema
- Public Plottable-boundary helpers to apply schema enforcement/coercion on Arrow load/export
- Deterministic coercion/confidence semantics surfaced in public docs/tests

Out of scope:
- Non-Arrow storage backends

Acceptance:
- Round-trip schema conversion is stable and tested
- Schema enforcement/coercion behavior is explicit and regression-tested
- API docs link to internal bridge semantics and clarify strict vs widen modes

## Suggested Issue Wiring

- Keep `#1046` as umbrella
- Track A/B/C as direct child issues: `#1337`, `#1338`, `#1339`
- Cross-link typed topology issue `#483` under Slice B
- Keep `#1259` as execution DAG reference for scheduling

## Exit Criteria for Closing #1046

- Slice A/B/C merged with tests and docs
- Public API and stable import paths documented
- Explicit compatibility notes for existing users (if any behavior changes)
