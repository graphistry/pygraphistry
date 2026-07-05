"""Conformance COVERAGE LEDGER for the GFQL polars engine (Phase-0 prong #4). PURE
INTROSPECTION — no engine execution, no GPU; runs on any CPU-only box. Keeps the conformance
matrix honest about its own coverage across FOUR axes (see ``AXES``). Each axis DERIVES its
universe from the live code registry (type_to_predicate / GFQL_SCALAR_FUNCTIONS / SAFELIST_V1 /
ROW_PIPELINE_CALLS — new registrations auto-enter the ledger), DERIVES the "exercised" set from
test_engine_polars_conformance_matrix.py (parsed labels / cypher strings / exported case lists),
and keeps a hand-maintained waiver dict, each entry with an honest one-line reason (native/NIE
rationale, "covered by dedicated test X", or an explicit TODO).

The tests FAIL CI naming the exact gap: (a) registry entry neither exercised nor waived =
coverage hole; (b) exercised name not in registry = stale/typo label; (c) waived name not in
registry = stale waiver; (d) waived name that IS exercised = redundant waiver. Nonempty canaries
per axis: empty registry = import wiring broke; empty exercised set = label/cypher FORMAT drift
silently zeroing coverage tracking.

NB: each axis's parser reads ONE source (see AXES) — entries asserted only by other dedicated
tests (e.g. temporal IsLeapYear / boundary predicates) are invisible to it and MUST carry a
waiver pointing at their dedicated test.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, NamedTuple

import pytest

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


# ---- AXIS 1: PREDICATES — universe `type_to_predicate` (predicates/from_json.py); exercised =
# class names parsed from `_predicate_queries()` labels (shaped `pred:<ClassName>(...)`) ----

# Hand-maintained waiver: registry predicates NOT exercised by _predicate_queries(). Derived
# once by set-differencing; the tests below fail loudly if it drifts. Honest one-liner each.
KNOWN_UNCOVERED: dict[str, str] = {
    # --- logical / categorical combinators (compositional; matrix asserts leaves) ---
    "AllOf": "logical AND-combinator over sub-predicates; compositional, not yet asserted in the matrix. TODO: add a nested-predicate conformance case.",
    "Duplicated": "categorical duplicate-row predicate; not yet asserted for parity/NIE. TODO: add a chain+dag conformance case.",

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

    # --- temporal date-part predicates: asserted by DEDICATED tests, not _predicate_queries ---
    "IsLeapYear": "native polars lowering; covered by dedicated tests test_conformance_temporal_is_leap_year_parity / test_temporal_is_leap_year_runs_natively_polars (not via _predicate_queries).",
    "IsMonthStart": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
    "IsMonthEnd": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
    "IsQuarterStart": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
    "IsQuarterEnd": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
    "IsYearStart": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
    "IsYearEnd": "native polars (provable calendar-field derivation); covered by dedicated test test_temporal_boundary_predicates_native_parity (not via _predicate_queries).",
}


# Labels are f"pred:{P.__name__}({col},{kw})"; the class name ends at the FIRST "(" — kw repr
# may itself contain parens (tuple patterns), so anchor on the first paren only.
_PRED_LABEL_RE = re.compile(r"^pred:(?P<name>\w+)\(")


def _exercised_predicate_names() -> set:
    """Parse the predicate class names actually exercised by `_predicate_queries()`."""
    names = set()
    for label, _query in _predicate_queries():
        m = _PRED_LABEL_RE.match(label)
        assert m is not None, f"unparseable predicate label (format drift?): {label!r}"
        names.add(m.group("name"))
    return names


# ---- AXIS 2: CYPHER SCALAR FUNCTIONS — universe GFQL_SCALAR_FUNCTIONS (language_defs.py);
# exercised = function names parsed from `_cypher_expression_queries()` cypher strings.
# Aggregations (count/count_distinct) live in a DIFFERENT registry: intentionally excluded ----

# A cypher call is `name(...)`; lowercase and intersect with the registry (cypher camelCase
# `toInteger` vs lowercase registry key `tointeger` — both engines `.lower()`).
_FN_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Registry scalar functions NOT exercised by a parseable cypher call in the matrix — all
# honest-NIE-or-unasserted; none has a parser-invisible dedicated test (unlike the temporals).
KNOWN_UNCOVERED_FUNCTIONS: dict[str, str] = {
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
    """Scalar-function names in _cypher_expression_queries() cypher strings (lowercased;
    registry-intersected so non-function tokens like aggregations/`count` drop out)."""
    names = set()
    for _label, cypher in _cypher_expression_queries():
        for tok in _FN_CALL_RE.findall(cypher):
            names.add(tok.lower())
    return names & set(GFQL_SCALAR_FUNCTIONS)


# ---- AXIS 3: call() SAFELIST — universe SAFELIST_V1 (call/validation.py): everything invocable
# via call() / a let() DAG binding; exercised = `_call_exercised_functions()` (matrix). The
# bulk is architecturally pandas/cuDF-only (layouts/encoders/igraph/cugraph/umap/hypergraph)
# and honest-NIEs under polars via the no-silent-bridge guard — waived with that reason.
# Fails CI if a NEW safelist entry lands with neither an assertion nor a waiver (untracked) ----

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
    "count_table": "count(*) short-circuit fast path (table height / source-mask sum); native frame op emitted by the cypher lowering, exercised as a labeled subject via _ROW_OP_CASES + the count_all_nodes/edges cypher cases, not via a direct call() consistency label. TODO.",
    "semi_apply_mark": "correlated EXISTS-mark; NATIVE on polars (viz-filter L1), exercised implicitly by the matrix EXISTS cypher cases; no direct labeled subject.",
    "anti_semi_apply": "anti-semi correlated filter; NATIVE on polars (viz-filter L1), exercised implicitly by the matrix NOT-EXISTS cypher case; no direct labeled subject.",
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


# ---- AXIS 4: ROW-PIPELINE OPS — universe ROW_PIPELINE_CALLS (row/pipeline.py); exercised =
# `_rowop_exercised()` (ops asserted as a labeled subject). The rest are either native-but-
# only-implicitly-exercised (via cypher RETURN/WHERE/grouped-count) or honest-NIE correlated-
# subquery ops — waived with that reason. (get_degrees/in/out are NOT in ROW_PIPELINE_CALLS;
# they're tracked on the call() axis.) Fails CI if a NEW op lands unasserted and unwaived ----

ROW_OP_KNOWN_UNCOVERED: dict[str, str] = {
    # honest NIE — correlated-subquery ops with no native polars lowering (_try_native_row_op returns None)
    "semi_apply_mark": "native single-join-alias polars lowering (viz-filter L1); multi-alias correlation still declines NIE; exercised via cypher EXISTS cases, no labeled rowop subject.",
    "anti_semi_apply": "native single-join-alias polars lowering (viz-filter L1); multi-alias correlation still declines NIE; exercised via cypher NOT-EXISTS case, no labeled rowop subject.",
    "join_apply": "honest NIE — correlated join op has no native polars lowering. TODO: add an explicit NIE-assertion case.",
}


# ---- LEDGER MACHINERY: one axis table, seven parametrized drift checks; a new coverage
# axis = one AXES entry (registry + exercised + waivers + names for messages) ----

class _Axis(NamedTuple):
    registry: Callable[[], set]      # live-code universe
    exercised: Callable[[], set]     # parsed from the conformance matrix module
    waivers: Dict[str, str]          # hand-maintained; the ledger content itself
    registry_name: str               # for failure messages
    exercised_name: str
    waiver_name: str


AXES: Dict[str, _Axis] = {
    "predicates": _Axis(
        lambda: set(type_to_predicate), _exercised_predicate_names, KNOWN_UNCOVERED,
        "type_to_predicate", "_predicate_queries()", "KNOWN_UNCOVERED"),
    "functions": _Axis(
        lambda: set(GFQL_SCALAR_FUNCTIONS), _exercised_function_names, KNOWN_UNCOVERED_FUNCTIONS,
        "GFQL_SCALAR_FUNCTIONS", "_cypher_expression_queries()", "KNOWN_UNCOVERED_FUNCTIONS"),
    "calls": _Axis(
        lambda: set(SAFELIST_V1), lambda: set(_call_exercised_functions()), CALL_KNOWN_UNCOVERED,
        "SAFELIST_V1", "_call_exercised_functions()", "CALL_KNOWN_UNCOVERED"),
    "rowops": _Axis(
        lambda: set(ROW_PIPELINE_CALLS), lambda: set(_rowop_exercised()), ROW_OP_KNOWN_UNCOVERED,
        "ROW_PIPELINE_CALLS", "_rowop_exercised()", "ROW_OP_KNOWN_UNCOVERED"),
}

_axis = pytest.mark.parametrize("axis", list(AXES), ids=list(AXES))


@_axis
def test_registry_is_nonempty(axis):
    ax = AXES[axis]
    assert ax.registry(), f"{ax.registry_name} registry is empty — import wiring broken"


@_axis
def test_exercised_set_is_nonempty(axis):
    # Anti-format-drift canary: a parser/label/cypher format change must not silently
    # zero out coverage tracking (every check below would then vacuously pass).
    ax = AXES[axis]
    assert ax.exercised(), (
        f"{ax.exercised_name} yielded no parseable exercised entries — format drift "
        "would silently zero out coverage tracking"
    )


@_axis
def test_every_registry_entry_is_exercised_or_waived(axis):
    ax = AXES[axis]
    gap = ax.registry() - (ax.exercised() | set(ax.waivers))
    assert not gap, (
        f"{ax.registry_name} entries are neither exercised by {ax.exercised_name} nor "
        f"listed in {ax.waiver_name} — add a conformance case OR a waiver reason for: "
        f"{sorted(gap)}"
    )


@_axis
def test_exercised_entries_are_all_in_registry(axis):
    ax = AXES[axis]
    bogus = ax.exercised() - ax.registry()
    assert not bogus, (
        f"{ax.exercised_name} exercises names absent from {ax.registry_name} "
        f"(typo or removed entry?): {sorted(bogus)}"
    )


@_axis
def test_waived_entries_are_real_registry_entries(axis):
    ax = AXES[axis]
    stale = set(ax.waivers) - ax.registry()
    assert not stale, (
        f"{ax.waiver_name} waives names not in {ax.registry_name} (stale waiver — "
        f"remove them): {sorted(stale)}"
    )


@_axis
def test_waived_entries_are_not_already_exercised(axis):
    ax = AXES[axis]
    overlap = set(ax.waivers) & ax.exercised()
    assert not overlap, (
        f"{ax.waiver_name} waives entries {ax.exercised_name} already exercises "
        f"(remove the redundant waiver): {sorted(overlap)}"
    )


@_axis
def test_waiver_reasons_are_nonempty(axis):
    ax = AXES[axis]
    blank = sorted(n for n, r in ax.waivers.items() if not (r and r.strip()))
    assert not blank, f"{ax.waiver_name} entries with empty reasons: {blank}"
