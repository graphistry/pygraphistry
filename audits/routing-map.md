# Routing Map Audit: `compile_cypher_query`

Date: 2026-04-07 PDT
Scope: `graphistry/compute/gfql/cypher/api.py`, `graphistry/compute/gfql/cypher/lowering.py`

## Entry Points

- `cypher_to_gfql(query, params)` (`api.py:12`) parses via `parse_cypher()`, compiles via `compile_cypher_query()`, asserts `CompiledCypherQuery`, returns `.chain`.
- `compile_cypher(query, params)` (`api.py:79`) parses via `parse_cypher()`, compiles via `compile_cypher_query()`, returns compiled object.

## Top-Level Dispatch Call Graph

```text
parse_cypher()
  -> compile_cypher_query()
     -> (CypherGraphQuery) _compile_graph_bindings + _compile_graph_constructor
     -> (CypherUnionQuery) recursive compile_cypher_query per branch
     -> (CypherQuery)
        -> pre-routing rewrite/reject pipeline
        -> branch ladder (first-match wins)
```

## Pre-Routing Rewrite/Reject Pipeline

Executed in this order inside `compile_cypher_query()`:

1. `_rewrite_shortest_path_query` (`lowering.py:8057`)
2. `_reject_unsupported_variable_length_where_pattern_predicates` (`lowering.py:8058`)
3. `_reject_variable_length_path_alias_references` (`lowering.py:8059`)
4. `_rewrite_where_pattern_predicates_to_matches` (`lowering.py:8060`)
5. `_reject_unsupported_where_expr_forms` (`lowering.py:8061`)
6. `_reject_nonterminal_variable_length_relationship_patterns` (`lowering.py:8062`)

## Branch Ladder (First Match Wins)

| # | Condition | Lowering Path | Output Type | Cypher Construct |
|---|---|---|---|---|
| 1 | `isinstance(query, CypherGraphQuery)` | `_compile_graph_bindings` + `_compile_graph_constructor` | `CompiledCypherGraphQuery` | `USE GRAPH` / multi-graph |
| 2 | `isinstance(query, CypherUnionQuery)` | recursive `compile_cypher_query` per branch | `CompiledCypherUnionQuery` | `UNION`, `UNION ALL` |
| 3 | `query.reentry_matches` | `_compile_bounded_reentry_query` (`7196`) | `CompiledCypherQuery` | `MATCH ... WITH ... MATCH` re-entry |
| 4 | `query.call is not None` | `_compile_call_query` (`7457`) | `CompiledCypherQuery` | `CALL ...` |
| 5 | `query.row_sequence` | `_lower_row_only_sequence` | `CompiledCypherQuery` | row-only / `UNWIND`-only |
| 6 | single MATCH + connected multi-pattern + guard set | `_compile_connected_match_join` (`7669`) | `CompiledCypherQuery` | connected multi-pattern join |
| 7 | `_is_connected_optional_match_query` | `_compile_connected_optional_match` (`7895`) | `CompiledCypherQuery` | connected `MATCH ... OPTIONAL MATCH` |
| 8 | `query.with_stages` | `_build_initial_row_scope` + stage loop + final stage lowering | `CompiledCypherQuery` | `WITH` pipeline |
| 9 | merged MATCH + no UNWIND + shortest path pattern | `_lower_general_row_projection` (`6412`) | `CompiledCypherQuery` | shortest path projections |
| 10 | merged MATCH + no UNWIND + aggregate RETURN | `_lower_general_row_projection` | `CompiledCypherQuery` | aggregate projections |
| 11 | merged MATCH + no UNWIND + simple projection plan success | `_build_projection_plan` + `_lower_projection_chain` | `CompiledCypherQuery` | simple alias projection |
| 12 | merged MATCH + no UNWIND + fallback | `_lower_general_row_projection` | `CompiledCypherQuery` | multi-alias / fallback |
| 13 | terminal fallback | `_lower_general_row_projection` | `CompiledCypherQuery` | UNWIND + RETURN or bare RETURN |

Notes:
- Branches 9-12 are internal to the merged-MATCH fast path block.
- Branch 13 is the global fallback return at end of function.

## WITH-Stage Internal Dispatch

Inside Branch 8 (`query.with_stages`):

- If `scope.mode == "match_alias"`:
  - `_lower_match_alias_stage` (`4522`)
  - aggregate stage path: `_lower_match_alias_aggregate_stage` (`4681`)
- If `scope.mode == "row_columns"`:
  - `_lower_row_column_stage` (`5049`)
  - aggregate stage path: `_lower_row_column_aggregate_stage` (`5342`)

Final RETURN stage dispatches by the same `scope.mode` split.

## Explicit Unsupported Routing-Level Constructs

`compile_cypher_query()` can raise `_unsupported()` for these routing-level cases:

1. Bounded-hop WHERE pattern predicates (only fixed-point `*` allowed).
2. Bare grouped-alias WHERE forms (`WHERE (n)`).
3. Nested UNION branches that compile to non-`CompiledCypherQuery`.
4. `MATCH ... OPTIONAL MATCH` projections returning seed-only alias.
5. OPTIONAL-only projection requiring null-extension without valid seed/guard.

## M1 Gate Relevance

This audit captures the full routing surface area that M1 binder extraction must preserve:

- pre-routing rewrite/reject order,
- branch ordering and first-match semantics,
- mode-based stage dispatch,
- routing-level unsupported gates.
