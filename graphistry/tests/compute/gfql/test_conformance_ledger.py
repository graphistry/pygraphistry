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
from graphistry.tests.compute.gfql.test_engine_polars_conformance_matrix import (
    _predicate_queries,
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
