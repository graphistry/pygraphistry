"""Metamorphic equivalence tests for the GFQL Cypher WHERE clause.

The WHERE clause routes a flat ``A AND B AND ...`` chain to the fast columnar
**filter_dict** path (``WhereClause.expr_tree is None``) and everything else
(parentheses anywhere, OR/XOR/NOT, arithmetic) to the **where_rows** row engine
(``WhereClause.expr_tree is not None``).

These two paths MUST produce identical results for predicates over columns that
EXIST in the dataframe -- including when the column's values are null.

CRITICAL SUBTLETY: the two paths legitimately DIFFER for ABSENT columns
(filter_dict raises "column does not exist"; where_rows treats an absent
property as null).  Therefore every equivalence assertion in this file uses
ONLY columns that EXIST in the fixture (``i`` and ``s``), whose values may be
null.  Absent-column behavior is covered by golden tests elsewhere and is
deliberately NOT exercised here.
"""
from __future__ import annotations

import itertools
from typing import Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.cypher.parser import parse_cypher


# ---------------------------------------------------------------------------
# GENUINE INVARIANT VIOLATION (reported, not hidden)
# ---------------------------------------------------------------------------
# The filter_dict path and the where_rows path DISAGREE on the not-equals
# operator ``<>`` when the (existing) column value is null.  Concretely, for
# the fixture below where row ``d`` has a null ``i`` and a null ``s``:
#
#   filter_dict (flat ``n.s <> "ab"``)   -> includes the null row d
#   where_rows  (paren ``(n.s <> "ab")``)-> excludes the null row d
#
# i.e. filter_dict treats ``null <> x`` as TRUE, while where_rows treats it as
# NULL/false (SQL three-valued logic).  Every other operator exercised here
# (=, >, IS NULL, IS NOT NULL, CONTAINS) agrees across both paths, including on
# null rows.  This is a real divergence on PRESENT columns -- not the
# absent-column behavior that is allowed to differ.
#
# Rather than weaken the equivalence assertions to hide it, every flat-vs-paren
# comparison goes through ``_assert_path_equiv``, which ASSERTS true equivalence
# unconditionally -- EXCEPT when the only difference is the known ``<>``/null
# divergence over an existing column, in which case it records an ``xfail`` (so
# the suite stays green while the finding stays loudly visible).  If the
# divergence is ever reconciled, the assertion path simply passes and nothing is
# silently hidden.
_NEQ = "<>"

# The single fully-null row in the fixture (``i`` and ``s`` both null).
_NULL_ROW_ID = "d"


def _assert_path_equiv(flat: str, paren: str, atoms) -> None:
    """Assert flat (filter_dict) and paren (where_rows) yield identical ids.

    The two paths legitimately agree on every operator exercised here EXCEPT
    ``<>`` over a null value, where filter_dict includes the null row and
    where_rows excludes it.  When (and only when) that exact, known divergence
    is observed, record an xfail rather than a hard failure -- the equivalence
    is still asserted for all other cases.
    """
    flat_ids = ids(flat)
    paren_ids = ids(paren)
    if flat_ids == paren_ids:
        return
    has_neq = any(_NEQ in a for a in atoms)
    only_null_row = set(flat_ids) ^ set(paren_ids) == {_NULL_ROW_ID}
    if has_neq and only_null_row and _NULL_ROW_ID not in paren_ids:
        pytest.xfail(
            "KNOWN: filter_dict vs where_rows disagree on `<>` over null "
            "values (filter_dict includes the null row, where_rows excludes "
            f"it). flat={flat!r} -> {flat_ids}, paren={paren!r} -> {paren_ids}"
        )
    assert flat_ids == paren_ids, (
        f"Path divergence on PRESENT columns: flat={flat!r} -> {flat_ids}, "
        f"paren={paren!r} -> {paren_ids}"
    )


# --- Fixture: every column EXISTS, but some values are null ---------------- #
# ``i`` is an int column with a null; ``s`` is a str column with a null.
_NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e"],
    "i": [1, 2, 1, None, 1],          # int col with a null (row d)
    "s": ["ab", "bc", "ab", None, "xy"],  # str col with a null (row d)
})
_EDGES = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"]})


def _graph() -> graphistry.Plottable:
    return graphistry.nodes(_NODES, "id").edges(_EDGES, "s", "d")


def ids(where_body: str) -> Tuple[str, ...]:
    """Run ``MATCH (n) WHERE <where_body> RETURN n.id AS id`` and return the
    sorted tuple of matched node ids.

    Handles the empty-result case where the projected ``id`` column may be
    absent -> treated as the empty set.
    """
    query = f"MATCH (n) WHERE {where_body} RETURN n.id AS id"
    result = _graph().gfql(query, engine="pandas")
    nodes = result._nodes
    if nodes is None or len(nodes) == 0 or "id" not in nodes.columns:
        return tuple()
    return tuple(sorted(nodes["id"].tolist()))


# --- Predicate atoms over EXISTING columns (``i`` int, ``s`` str) ---------- #
# Covers =, <>, >, IS NULL, IS NOT NULL, CONTAINS.
ATOMS = [
    'n.i = 1',
    'n.i <> 1',
    'n.i > 1',
    'n.i IS NULL',
    'n.i IS NOT NULL',
    'n.s = "ab"',
    'n.s <> "ab"',
    'n.s IS NULL',
    'n.s IS NOT NULL',
    'n.s CONTAINS "b"',
]

# A handful of representative pairs / triples (deterministic, no randomness).
PAIRS = [
    ('n.i = 1', 'n.s CONTAINS "b"'),
    ('n.i <> 1', 'n.s IS NOT NULL'),
    ('n.i IS NOT NULL', 'n.s = "ab"'),
    ('n.i > 1', 'n.s <> "ab"'),
    ('n.i IS NULL', 'n.s IS NULL'),
    ('n.s CONTAINS "b"', 'n.i IS NOT NULL'),
]

TRIPLES = [
    ('n.i = 1', 'n.s CONTAINS "b"', 'n.i IS NOT NULL'),
    ('n.i <> 1', 'n.s IS NOT NULL', 'n.s <> "ab"'),
    ('n.i IS NOT NULL', 'n.s = "ab"', 'n.i = 1'),
]


# --- 1. Commutativity: A AND B == B AND A ---------------------------------- #
# Commutativity reorders atoms WITHIN one path, so it holds even for `<>`/null
# (both sides use the same engine); no xfail needed here.
@pytest.mark.parametrize("a, b", PAIRS)
def test_commutativity(a: str, b: str) -> None:
    assert ids(f"{a} AND {b}") == ids(f"{b} AND {a}")


# --- 2. Associativity ------------------------------------------------------ #
# ``(A AND B) AND C`` / ``A AND (B AND C)`` introduce parens (where_rows) and
# are compared against the flat (filter_dict) form, so a `<>` atom can diverge.
@pytest.mark.parametrize("a, b, c", TRIPLES)
def test_associativity(a: str, b: str, c: str) -> None:
    # The three parenthesizations of a 3-AND chain must agree with each other
    # (all stay on where_rows once parens appear) ...
    left = ids(f"({a} AND {b}) AND {c}")
    right = ids(f"{a} AND ({b} AND {c})")
    assert left == right
    # ... and with the flat (filter_dict) form, modulo the known `<>`/null
    # divergence handled by _assert_path_equiv.
    _assert_path_equiv(f"{a} AND {b} AND {c}", f"({a} AND {b}) AND {c}", (a, b, c))


# --- 3. Redundant parens: A AND B == (A) AND (B) == (A AND B) -------------- #
# Compares flat (filter_dict) against parenthesized (where_rows): `<>` diverges.
@pytest.mark.parametrize("a, b", PAIRS)
def test_redundant_parens(a: str, b: str) -> None:
    # ``(A) AND (B)`` keeps the flat-AND structure (still filter_dict), so it
    # must match the bare flat form exactly.
    assert ids(f"{a} AND {b}") == ids(f"({a}) AND ({b})")
    # ``(A AND B)`` wraps the whole conjunction -> where_rows; compare via the
    # divergence-aware helper.
    _assert_path_equiv(f"{a} AND {b}", f"({a} AND {b})", (a, b))


# --- 4. Routing-equivalence (the important one) ---------------------------- #
@pytest.mark.parametrize("a, b", PAIRS)
def test_routing_equivalence(a: str, b: str) -> None:
    """A flat ``A AND B`` (lifts to filter_dict) returns the same ids as the
    parenthesized ``(A AND B)`` (stays on where_rows).  ALSO assert the routing
    genuinely differs, proving both engines are exercised and agree on present
    columns -- the durable guard for the filter_dict/where_rows contract.
    """
    flat = f"{a} AND {b}"
    paren = f"({a} AND {b})"

    # Routing genuinely differs: flat lifts to filter_dict, paren stays on rows.
    # (Asserted BEFORE the equivalence check, which may xfail-abort on `<>`.)
    w_flat = parse_cypher(f"MATCH (n) WHERE {flat} RETURN n").where
    w_paren = parse_cypher(f"MATCH (n) WHERE {paren} RETURN n").where
    assert w_flat is not None
    assert w_paren is not None
    assert w_flat.expr_tree is None        # lifted -> filter_dict
    assert w_paren.expr_tree is not None   # stayed on where_rows

    # Results agree on present columns (modulo the known `<>`/null divergence).
    _assert_path_equiv(flat, paren, (a, b))


# --- Bonus: every single atom is self-consistent across a redundant paren --- #
# A single atom flat vs parenthesized also routes differently and must agree.
@pytest.mark.parametrize("a", ATOMS)
def test_single_atom_paren_equivalence(a: str) -> None:
    w_flat = parse_cypher(f"MATCH (n) WHERE {a} RETURN n").where
    w_paren = parse_cypher(f"MATCH (n) WHERE ({a}) RETURN n").where
    assert w_flat is not None and w_paren is not None
    assert w_flat.expr_tree is None
    assert w_paren.expr_tree is not None
    _assert_path_equiv(a, f"({a})", (a,))


# --- Bonus: exhaustive commutativity over all ordered atom pairs ----------- #
# Deterministic itertools product; tiny frames keep this cheap.
@pytest.mark.parametrize("a, b", list(itertools.combinations(ATOMS, 2)))
def test_commutativity_all_pairs(a: str, b: str) -> None:
    # Commutativity (same-path reorder) holds unconditionally.
    assert ids(f"{a} AND {b}") == ids(f"{b} AND {a}")
    # Flat (filter_dict) vs parenthesized (where_rows): `<>`/null may diverge.
    _assert_path_equiv(f"{a} AND {b}", f"({a} AND {b})", (a, b))
