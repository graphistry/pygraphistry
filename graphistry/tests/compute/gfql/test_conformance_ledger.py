"""Conformance COVERAGE LEDGER for the GFQL polars engine (Phase-0 prong #4).

PURE INTROSPECTION — NO engine execution, NO GPU, runs on any box (CPU-only).

This ledger keeps the differential-conformance matrix HONEST about its own coverage. It:
  * DERIVES the predicate universe from the live code registry
    `type_to_predicate` (graphistry/compute/predicates/from_json.py) — so a newly
    registered predicate automatically enters the ledger.
  * DERIVES the "exercised" set by parsing the labels of `_predicate_queries()` in
    test_engine_polars_conformance_matrix.py (labels shaped `pred:<ClassName>(...)`).
  * Keeps a hand-maintained KNOWN_UNCOVERED waiver of predicates deliberately not (yet)
    asserted through `_predicate_queries()` — each with a one-line honest reason
    (native/NIE rationale, "covered by dedicated test X", or an explicit TODO).

The tests then FAIL CI when those drift apart, naming the exact gap:
  (a) a registry predicate that is neither exercised nor waived  -> coverage hole
  (b) an exercised label that is not a real registry predicate   -> stale/typo label
  (c) a KNOWN_UNCOVERED name that is not in the registry          -> stale waiver
  (d) a KNOWN_UNCOVERED name that IS exercised                    -> redundant waiver

NB: `_predicate_queries()` is the ONLY exercised-set source the parser reads. Predicates
asserted via OTHER dedicated tests in the matrix file (e.g. the temporal IsLeapYear /
boundary-predicate tests, the EQ DateValue temporal test) are invisible to this parser and
therefore MUST carry a KNOWN_UNCOVERED waiver pointing at their dedicated test.
"""
from __future__ import annotations

import re

from graphistry.compute.predicates.from_json import type_to_predicate
from graphistry.compute.gfql.language_defs import GFQL_SCALAR_FUNCTIONS
from graphistry.compute.gfql.call.validation import SAFELIST_V1
from graphistry.compute.gfql.row.pipeline import ROW_PIPELINE_CALLS
from graphistry.tests.compute.gfql.test_engine_polars_conformance_matrix import (
    _predicate_queries,
    _cypher_expression_queries,
    _call_exercised_functions,
    _rowop_exercised,
)


# --------------------------------------------------------------------------------------
# Hand-maintained waiver: registry predicates NOT exercised by `_predicate_queries()`.
# Derived once by set-differencing the registry against the parsed exercised set; keep it
# in sync (the tests below fail loudly if it drifts). Each value is an honest one-liner.
# --------------------------------------------------------------------------------------
KNOWN_UNCOVERED: dict[str, str] = {
    # --- logical / categorical combinators (compositional; matrix asserts leaves) ---
    "AllOf": "logical AND-combinator over sub-predicates; compositional, not yet asserted in the matrix. TODO: add a nested-predicate conformance case.",
    "Duplicated": "categorical duplicate-row predicate; not yet asserted for parity/NIE. TODO: add a chain+dag conformance case.",

    # --- equality (EQ/NE) ---
    "EQ": "equality predicate; exercised only via the temporal DateValue path (label 'eq-date' in test_conformance_temporal_datevalue_chain), not the scalar `_predicate_queries()` path. TODO: add scalar EQ.",
    "NE": "not-equal predicate; not yet asserted for parity/NIE. TODO: add a scalar NE conformance case.",

    # --- string char-class predicates (.str.is*) ---
    "IsNumeric": "str .is_numeric char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsAlpha": "str .is_alpha char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsDecimal": "str .is_decimal char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsDigit": "str .is_digit char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsLower": "str .is_lower char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsUpper": "str .is_upper char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsSpace": "str .is_space char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsAlnum": "str .is_alnum char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",
    "IsTitle": "str .is_title char-class predicate; not yet asserted (native-or-NIE undecided). TODO.",

    # --- string null predicates (distinct from numeric IsNA/NotNA which ARE exercised) ---
    "IsNull": "str-module null predicate (distinct from numeric IsNA, which is exercised); not yet asserted. TODO.",
    "NotNull": "str-module not-null predicate (distinct from numeric NotNA, which is exercised); not yet asserted. TODO.",

    # --- temporal date-part predicates: asserted by DEDICATED tests, not _predicate_queries ---
    "IsLeapYear": "native polars lowering; covered by dedicated tests test_conformance_temporal_is_leap_year_parity / test_temporal_is_leap_year_runs_natively_polars (not via _predicate_queries).",
    "IsMonthStart": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
    "IsMonthEnd": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
    "IsQuarterStart": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
    "IsQuarterEnd": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
    "IsYearStart": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
    "IsYearEnd": "no faithful polars boolean accessor -> honest NIE; covered by dedicated test test_temporal_boundary_predicates_honest_nie_polars (not via _predicate_queries).",
}


# `_predicate_queries()` labels are built as f"pred:{P.__name__}({col},{kw})"; the class
# name is everything between the "pred:" prefix and the first "(" (kw repr may itself
# contain parens, e.g. tuple patterns, so anchor on the FIRST paren only).
_PRED_LABEL_RE = re.compile(r"^pred:(?P<name>\w+)\(")


def _exercised_predicate_names() -> set:
    """Parse the predicate class names actually exercised by `_predicate_queries()`."""
    names = set()
    for label, _query in _predicate_queries():
        m = _PRED_LABEL_RE.match(label)
        assert m is not None, f"unparseable predicate label (format drift?): {label!r}"
        names.add(m.group("name"))
    return names


def _registry_names() -> set:
    return set(type_to_predicate)


# --------------------------------------------------------------------------------------
# Sanity: the introspection sources are actually wired up.
# --------------------------------------------------------------------------------------
def test_registry_is_nonempty():
    assert _registry_names(), "type_to_predicate registry is empty — import wiring broken"


def test_exercised_set_is_nonempty():
    assert _exercised_predicate_names(), (
        "_predicate_queries() yielded no parseable `pred:<Class>(...)` labels — "
        "label format drift would silently zero out coverage tracking"
    )


# --------------------------------------------------------------------------------------
# (a) Every registry predicate is EITHER exercised by _predicate_queries() OR waived.
# --------------------------------------------------------------------------------------
def test_every_registry_predicate_is_exercised_or_known_uncovered():
    registry = _registry_names()
    accounted = _exercised_predicate_names() | set(KNOWN_UNCOVERED)
    gap = registry - accounted
    assert not gap, (
        "registry predicates are neither exercised by _predicate_queries() nor listed in "
        "KNOWN_UNCOVERED — add a conformance case OR a KNOWN_UNCOVERED reason for: "
        f"{sorted(gap)}"
    )


# --------------------------------------------------------------------------------------
# (b) Every exercised label maps to a REAL registry predicate (no typo / dead label).
# --------------------------------------------------------------------------------------
def test_exercised_predicates_are_all_in_registry():
    bogus = _exercised_predicate_names() - _registry_names()
    assert not bogus, (
        "_predicate_queries() exercises predicate names absent from the registry "
        f"(typo or removed predicate?): {sorted(bogus)}"
    )


# --------------------------------------------------------------------------------------
# (c) No STALE KNOWN_UNCOVERED entries: every waived name is a real registry predicate.
# --------------------------------------------------------------------------------------
def test_known_uncovered_entries_are_real_registry_predicates():
    stale = set(KNOWN_UNCOVERED) - _registry_names()
    assert not stale, (
        "KNOWN_UNCOVERED waives names not in the registry (stale waiver — remove them): "
        f"{sorted(stale)}"
    )


# --------------------------------------------------------------------------------------
# (d) Nothing is BOTH exercised and waived (redundant waiver hiding real coverage).
# --------------------------------------------------------------------------------------
def test_known_uncovered_entries_are_not_already_exercised():
    overlap = set(KNOWN_UNCOVERED) & _exercised_predicate_names()
    assert not overlap, (
        "KNOWN_UNCOVERED waives predicates that _predicate_queries() already exercises "
        f"(remove the redundant waiver): {sorted(overlap)}"
    )


# --------------------------------------------------------------------------------------
# Every waiver carries an honest, non-empty reason.
# --------------------------------------------------------------------------------------
def test_known_uncovered_reasons_are_nonempty():
    blank = sorted(n for n, r in KNOWN_UNCOVERED.items() if not (r and r.strip()))
    assert not blank, f"KNOWN_UNCOVERED entries with empty reasons: {blank}"


# ======================================================================================
# AXIS 2: CYPHER SCALAR FUNCTIONS
# Universe = GFQL_SCALAR_FUNCTIONS (language_defs.py). Exercised = scalar-function names
# parsed out of the cypher strings of `_cypher_expression_queries()`. Same four drift
# tests (a)-(d). Aggregations (count/count_distinct) live in a DIFFERENT registry and are
# intentionally not in this universe.
# ======================================================================================

# A cypher call is `name(...)`; lowercase and intersect with the registry (cypher uses
# camelCase `toInteger`, registry keys are lowercase `tointeger` — both engines `.lower()`).
_FN_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Registry scalar functions NOT exercised by a parseable cypher call in the matrix. Each an
# honest one-liner (all currently honest-NIE-or-unasserted; none has a dedicated test that the
# parser misses, unlike the predicate temporal entries).
KNOWN_UNCOVERED_FUNCTIONS: dict[str, str] = {
    "tofloat": "pandas-native (astype float); polars _lower_function has NO branch -> honest NIE, not yet asserted. TODO: add a tofloat native-or-NIE case.",
    "keys": "map/entity key-extraction; polars declines (no _lower_function branch) -> NIE; not yet asserted. TODO.",
    "labels": "node-label text function; polars declines -> NIE; not yet asserted. TODO.",
    "type": "edge-type function; polars declines -> NIE; not yet asserted. TODO.",
    "properties": "entity-properties map function; polars declines -> NIE; not yet asserted. TODO.",
    "range": "list-generating range(a,b[,step]); polars declines -> NIE; not yet asserted. TODO.",
    "__node_keys__": "internal cypher-lowering helper (node entity keys); not a standalone user function. internal.",
    "__edge_keys__": "internal cypher-lowering helper (edge entity keys); not a standalone user function. internal.",
    "__node_entity__": "internal cypher-lowering helper (node entity text); not a standalone user function. internal.",
    "__edge_entity__": "internal cypher-lowering helper (edge entity text); not a standalone user function. internal.",
    "__cypher_case_eq__": "internal simple-CASE equality helper; matrix exercises the SEARCHED CASE form (CaseWhen AST), not this function. internal.",
}


def _exercised_function_names() -> set:
    """Scalar-function names actually exercised by `_cypher_expression_queries()` cypher
    strings (lowercased, intersected with the registry so non-function tokens like
    aggregations / `count` / bare identifiers drop out)."""
    names = set()
    for _label, cypher in _cypher_expression_queries():
        for tok in _FN_CALL_RE.findall(cypher):
            names.add(tok.lower())
    return names & set(GFQL_SCALAR_FUNCTIONS)


def test_function_registry_is_nonempty():
    assert set(GFQL_SCALAR_FUNCTIONS), "GFQL_SCALAR_FUNCTIONS registry is empty — import wiring broken"


def test_exercised_function_set_is_nonempty():
    assert _exercised_function_names(), (
        "_cypher_expression_queries() yielded no parseable scalar-function calls — "
        "cypher format drift would silently zero out function coverage tracking"
    )


def test_every_registry_function_is_exercised_or_known_uncovered():
    registry = set(GFQL_SCALAR_FUNCTIONS)
    accounted = _exercised_function_names() | set(KNOWN_UNCOVERED_FUNCTIONS)
    gap = registry - accounted
    assert not gap, (
        "GFQL_SCALAR_FUNCTIONS entries are neither exercised by _cypher_expression_queries() "
        f"nor listed in KNOWN_UNCOVERED_FUNCTIONS — add a conformance case OR a waiver for: {sorted(gap)}"
    )


def test_exercised_functions_are_all_in_registry():
    bogus = _exercised_function_names() - set(GFQL_SCALAR_FUNCTIONS)
    assert not bogus, f"exercised function names absent from GFQL_SCALAR_FUNCTIONS: {sorted(bogus)}"


def test_known_uncovered_functions_are_real_registry_functions():
    stale = set(KNOWN_UNCOVERED_FUNCTIONS) - set(GFQL_SCALAR_FUNCTIONS)
    assert not stale, (
        "KNOWN_UNCOVERED_FUNCTIONS waives names not in GFQL_SCALAR_FUNCTIONS (stale — remove): "
        f"{sorted(stale)}"
    )


def test_known_uncovered_functions_are_not_already_exercised():
    overlap = set(KNOWN_UNCOVERED_FUNCTIONS) & _exercised_function_names()
    assert not overlap, (
        "KNOWN_UNCOVERED_FUNCTIONS waives functions already exercised (remove redundant waiver): "
        f"{sorted(overlap)}"
    )


def test_known_uncovered_function_reasons_are_nonempty():
    blank = sorted(n for n, r in KNOWN_UNCOVERED_FUNCTIONS.items() if not (r and r.strip()))
    assert not blank, f"KNOWN_UNCOVERED_FUNCTIONS entries with empty reasons: {blank}"


# ======================================================================================
# AXIS 3: call() SAFELIST
# Universe = SAFELIST_V1 (call/validation.py) — every function invocable via call() / a
# let() DAG binding. Exercised = `_call_exercised_functions()` (matrix). The bulk of the
# safelist is architecturally pandas/cuDF-only (layouts / encoders / igraph / cugraph /
# umap / hypergraph) and honest-NIEs under polars via the no-silent-bridge guard — those
# are waived with that reason. The ledger fails CI if a NEW safelist entry lands with
# neither a conformance assertion nor a waiver (e.g. a new unsafe call slips in untracked).
# ======================================================================================

CALL_KNOWN_UNCOVERED: dict[str, str] = {
    # row-pipeline ops native under polars via the call()/DAG executor; not yet asserted via call()
    "rows": "native polars row-pipeline op; exercised via call() only in test_engine_polars_row_pipeline.py, not this matrix. TODO.",
    "skip": "native polars row-pipeline op; not yet asserted via a matrix call() case. TODO.",
    "distinct": "native polars row-pipeline op; not yet asserted via a matrix call() case. TODO.",
    "drop_cols": "native polars row-pipeline op; not yet asserted via a matrix call() case. TODO.",
    # row-pipeline ops native only on the polars CHAIN surface; honest-NIE via the call()/DAG executor
    "select": "row projection; native on polars chain, NIE via call()/DAG executor; not asserted via call(). TODO.",
    "return_": "RETURN projection (alias of select); chain-native / call()-NIE; not asserted via call(). TODO.",
    "with_": "WITH projection; chain-native / call()-NIE; not asserted via call(). TODO.",
    "where_rows": "row filter; native on polars chain, NIE via call()/DAG executor; not asserted via call(). TODO.",
    "order_by": "row sort; native on polars chain, NIE via call()/DAG executor; not asserted via call(). TODO.",
    "unwind": "list explode; native on polars chain, NIE via call()/DAG executor; not asserted via call(). TODO.",
    "group_by": "grouped aggregation; native on polars chain, NIE via call()/DAG executor; not asserted via call(). TODO.",
    "semi_apply_mark": "correlated EXISTS-mark; row-pipeline op honest-NIE under polars; not asserted. TODO.",
    "anti_semi_apply": "anti-semi correlated filter; row-pipeline op honest-NIE under polars; not asserted. TODO.",
    "join_apply": "correlated row join; row-pipeline op honest-NIE under polars; not asserted. TODO.",
    # Plottable-method calls: no native polars impl; pandas/cuDF only -> no-silent-bridge NIE under polars.
    # Class covered by test_engine_polars_no_silent_call_bridge (hypergraph representative); each TODO individually.
    "filter_nodes_by_dict": "Plottable-method; pandas/cuDF only, polars no-bridge NIE; class covered by test_engine_polars_no_silent_call_bridge. TODO.",
    "filter_edges_by_dict": "Plottable-method; pandas/cuDF only, polars no-bridge NIE; class covered by test_engine_polars_no_silent_call_bridge. TODO.",
    "materialize_nodes": "Plottable-method; native node-materialize is the chain surface, call() is pandas/cuDF-only -> polars no-bridge NIE. TODO.",
    "hop": "Plottable-method; native hop is the polars chain surface, call()/DAG hop is pandas-only -> polars no-bridge NIE. TODO.",
    "compute_cugraph": "GPU cuGraph algo; cuDF-only, polars no-bridge NIE (honest). TODO.",
    "compute_igraph": "igraph algo; pandas-only, polars no-bridge NIE (honest). TODO.",
    "layout_cugraph": "GPU layout; cuDF-only, polars no-bridge NIE (honest). TODO.",
    "layout_igraph": "igraph layout; pandas-only, polars no-bridge NIE (honest). TODO.",
    "layout_graphviz": "graphviz layout; pandas-only, polars no-bridge NIE (honest). TODO.",
    "ring_continuous_layout": "radial layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "ring_categorical_layout": "radial layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "time_ring_layout": "time-ring layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "circle_layout": "circular layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "tree_layout": "tree layout; pandas-only, polars no-bridge NIE (honest). TODO.",
    "mercator_layout": "mercator layout; pandas-only, polars no-bridge NIE (honest). TODO.",
    "modularity_weighted_layout": "community-weighted layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "fa2_layout": "ForceAtlas2 layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "prune_self_edges": "Plottable-method; pandas/cuDF only, polars no-bridge NIE (honest). TODO.",
    "collapse": "node collapse; pandas-only, polars no-bridge NIE (honest). TODO.",
    "drop_nodes": "Plottable-method; pandas/cuDF only, polars no-bridge NIE (honest). TODO.",
    "keep_nodes": "Plottable-method; pandas/cuDF only, polars no-bridge NIE (honest). TODO.",
    "get_topological_levels": "DAG-level compute; pandas-only, polars no-bridge NIE (honest). TODO.",
    "encode_point_color": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_color": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_point_size": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_size": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_weight": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_point_opacity": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_opacity": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_point_label": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_label": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_point_title": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_title": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_point_icon": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_edge_icon": "visual encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "encode_axis": "axis encoding; pandas/cuDF-only Plottable method, polars no-bridge NIE (honest). TODO.",
    "name": "metadata setter; engine-agnostic but not asserted via a parseable call() label. TODO.",
    "description": "metadata setter; engine-agnostic but not asserted via a parseable call() label. TODO.",
    "group_in_a_box_layout": "GIB layout; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
    "umap": "UMAP embedding; pandas/cuDF-only, polars no-bridge NIE (honest). TODO.",
}


def test_call_safelist_is_nonempty():
    assert set(SAFELIST_V1), "SAFELIST_V1 is empty — import wiring broken"


def test_exercised_call_set_is_nonempty():
    assert _call_exercised_functions(), "_call_exercised_functions() is empty — coverage tracking would zero out"


def test_every_safelist_entry_is_exercised_or_known_uncovered():
    registry = set(SAFELIST_V1)
    accounted = _call_exercised_functions() | set(CALL_KNOWN_UNCOVERED)
    gap = registry - accounted
    assert not gap, (
        "SAFELIST_V1 entries are neither exercised by the matrix nor listed in CALL_KNOWN_UNCOVERED — "
        f"add a call() conformance case OR a waiver for: {sorted(gap)}"
    )


def test_exercised_calls_are_all_in_safelist():
    bogus = _call_exercised_functions() - set(SAFELIST_V1)
    assert not bogus, f"exercised call() names absent from SAFELIST_V1 (typo/removed?): {sorted(bogus)}"


def test_call_known_uncovered_entries_are_real_safelist_entries():
    stale = set(CALL_KNOWN_UNCOVERED) - set(SAFELIST_V1)
    assert not stale, f"CALL_KNOWN_UNCOVERED waives names not in SAFELIST_V1 (stale — remove): {sorted(stale)}"


def test_call_known_uncovered_entries_are_not_already_exercised():
    overlap = set(CALL_KNOWN_UNCOVERED) & _call_exercised_functions()
    assert not overlap, f"CALL_KNOWN_UNCOVERED waives already-exercised calls (remove redundant waiver): {sorted(overlap)}"


def test_call_known_uncovered_reasons_are_nonempty():
    blank = sorted(n for n, r in CALL_KNOWN_UNCOVERED.items() if not (r and r.strip()))
    assert not blank, f"CALL_KNOWN_UNCOVERED entries with empty reasons: {blank}"


# ======================================================================================
# AXIS 4: ROW-PIPELINE OPS
# Universe = ROW_PIPELINE_CALLS (row/pipeline.py) — the cypher row-pipeline ops. Exercised =
# `_rowop_exercised()` (ops asserted as a labeled subject; today just `with_`). The rest are
# either native-but-only-implicitly-exercised (via cypher RETURN/WHERE/grouped-count) or
# honest-NIE correlated-subquery ops; all waived with that reason. (The degree calls
# get_degrees/in/out are NOT in ROW_PIPELINE_CALLS — they are tracked on the call() axis.)
# The ledger fails CI if a NEW row-pipeline op lands without an assertion or a waiver.
# ======================================================================================

ROW_OP_KNOWN_UNCOVERED: dict[str, str] = {
    # native polars frame ops, simply not exercised as a labeled subject yet
    "skip": "native polars frame op; no conformance case yet. TODO: add skip(N) chain+dag parity.",
    "drop_cols": "native polars frame op; no conformance case yet. TODO: add drop_cols parity.",
    "distinct": "native polars frame op; only count(DISTINCT) cypher AGGREGATION is tested, not the distinct ROW op. TODO.",
    "limit": "native polars frame op; exercised only for chain-vs-dag consistency (call axis fn='limit'), no labeled row-op parity. TODO.",
    # native lowerings exercised only IMPLICITLY via cypher text (no labeled op subject)
    "rows": "native frame op; exercised only as a [n(), rows(), ...] query component, never a labeled subject. TODO.",
    "select": "native via select_polars; exercised only implicitly via cypher RETURN. TODO: add a labeled select subject.",
    "return_": "native via select_polars (alias of select); exercised only via cypher RETURN / a return_() AST. TODO.",
    "where_rows": "native via where_rows_polars; exercised only implicitly via cypher WHERE; the where_rows AST op is never directly asserted. TODO.",
    "order_by": "native via order_by_polars; NOT exercised anywhere in the matrix. TODO: add ORDER BY parity+NIE.",
    "group_by": "native via group_by_polars; exercised only implicitly via cypher grouped count(); group_by AST never directly asserted. TODO.",
    "unwind": "native via unwind_polars; NOT exercised anywhere in the matrix. TODO: add UNWIND parity+NIE.",
    # honest NIE — correlated-subquery ops with no native polars lowering (_try_native_row_op returns None)
    "semi_apply_mark": "honest NIE — correlated EXISTS-mark op has no native polars lowering. TODO: add an explicit NIE-assertion case.",
    "anti_semi_apply": "honest NIE — correlated anti-semi op has no native polars lowering. TODO: add an explicit NIE-assertion case.",
    "join_apply": "honest NIE — correlated join op has no native polars lowering. TODO: add an explicit NIE-assertion case.",
}


def test_row_pipeline_registry_is_nonempty():
    assert set(ROW_PIPELINE_CALLS), "ROW_PIPELINE_CALLS registry is empty — import wiring broken"


def test_exercised_rowop_set_is_nonempty():
    assert _rowop_exercised(), "_rowop_exercised() is empty — coverage tracking would zero out"


def test_every_rowop_is_exercised_or_known_uncovered():
    registry = set(ROW_PIPELINE_CALLS)
    accounted = _rowop_exercised() | set(ROW_OP_KNOWN_UNCOVERED)
    gap = registry - accounted
    assert not gap, (
        "ROW_PIPELINE_CALLS ops are neither exercised by the matrix nor listed in ROW_OP_KNOWN_UNCOVERED — "
        f"add a labeled conformance case OR a waiver for: {sorted(gap)}"
    )


def test_exercised_rowops_are_all_in_registry():
    bogus = _rowop_exercised() - set(ROW_PIPELINE_CALLS)
    assert not bogus, f"exercised row-op names absent from ROW_PIPELINE_CALLS (typo/removed?): {sorted(bogus)}"


def test_rowop_known_uncovered_entries_are_real():
    stale = set(ROW_OP_KNOWN_UNCOVERED) - set(ROW_PIPELINE_CALLS)
    assert not stale, f"ROW_OP_KNOWN_UNCOVERED waives names not in ROW_PIPELINE_CALLS (stale — remove): {sorted(stale)}"


def test_rowop_known_uncovered_entries_are_not_already_exercised():
    overlap = set(ROW_OP_KNOWN_UNCOVERED) & _rowop_exercised()
    assert not overlap, f"ROW_OP_KNOWN_UNCOVERED waives already-exercised ops (remove redundant waiver): {sorted(overlap)}"


def test_rowop_known_uncovered_reasons_are_nonempty():
    blank = sorted(n for n, r in ROW_OP_KNOWN_UNCOVERED.items() if not (r and r.strip()))
    assert not blank, f"ROW_OP_KNOWN_UNCOVERED entries with empty reasons: {blank}"
