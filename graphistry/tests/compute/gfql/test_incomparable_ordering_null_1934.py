"""#1934: incomparable ordering comparisons yield NULL on every engine, never False.

openCypher orders only within a type: STRING vs NUMBER ordering is incomparable ->
null, so BOTH ``WHERE pred`` and ``WHERE NOT pred`` drop the row. #1919 fixed the
pandas arm; the cuDF arm kept an early-return-False, so ``NOT (n.str_col < 1.0)``
KEPT every non-null row on cuDF while pandas dropped it. Pinned here per engine:
pandas serves null, cuDF serves null (post-fix), polars declines typed
(parity-or-error by design). Equality is deliberately NOT rerouted: mixed-type
``=`` stays the pre-existing typed GFQLSchemaError on both engines.
"""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry


NODES = pd.DataFrame({
    "id": ["a", "b", "c"],
    "str_col": ["x", "y", None],
})
EDGES = pd.DataFrame({"src": ["a"], "dst": ["b"]})


def _graph(engine: str):
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(NODES), "id").edges(
            cudf.from_pandas(EDGES), "src", "dst"
        )
    return graphistry.nodes(NODES, "id").edges(EDGES, "src", "dst")


def _ids(result):
    nodes = result._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return sorted(nodes["id"].tolist())


# Oracle (hand-computed): every ordering of str_col vs 1.0 is null for rows a/b
# (incomparable) and null for row c (null input), so the row set is [] for the
# predicate AND for its negation.
DIRECT_ORDERINGS = [
    ("lt", "MATCH (n) WHERE n.str_col < 1.0 RETURN n.id AS id"),
    ("le", "MATCH (n) WHERE n.str_col <= 1.0 RETURN n.id AS id"),
    ("gt", "MATCH (n) WHERE n.str_col > 1.0 RETURN n.id AS id"),
    ("ge", "MATCH (n) WHERE n.str_col >= 1.0 RETURN n.id AS id"),
    ("lt_rev", "MATCH (n) WHERE 1.0 < n.str_col RETURN n.id AS id"),
    ("gt_rev", "MATCH (n) WHERE 1.0 > n.str_col RETURN n.id AS id"),
]

# The user-visible tell: null and False agree under WHERE, but NOT null is null
# while NOT False is True -- the silent-False arm returned ['a', 'b'] here.
NOT_ORDERINGS = [
    ("not_lt", "MATCH (n) WHERE NOT (n.str_col < 1.0) RETURN n.id AS id"),
    ("not_le", "MATCH (n) WHERE NOT (n.str_col <= 1.0) RETURN n.id AS id"),
    ("not_gt", "MATCH (n) WHERE NOT (n.str_col > 1.0) RETURN n.id AS id"),
    ("not_ge", "MATCH (n) WHERE NOT (n.str_col >= 1.0) RETURN n.id AS id"),
    ("not_lt_rev", "MATCH (n) WHERE NOT (1.0 < n.str_col) RETURN n.id AS id"),
    ("not_gt_rev", "MATCH (n) WHERE NOT (1.0 > n.str_col) RETURN n.id AS id"),
]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("name,query", DIRECT_ORDERINGS, ids=[c[0] for c in DIRECT_ORDERINGS])
def test_incomparable_ordering_matches_nothing(engine, name, query):
    assert _ids(_graph(engine).gfql(query, engine=engine)) == []


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("name,query", NOT_ORDERINGS, ids=[c[0] for c in NOT_ORDERINGS])
def test_not_of_incomparable_ordering_also_matches_nothing(engine, name, query):
    """#1934: red at master on cuDF (returned ['a', 'b']); pandas is the fence."""
    assert _ids(_graph(engine).gfql(query, engine=engine)) == []


@pytest.mark.parametrize(
    "name,query",
    DIRECT_ORDERINGS + NOT_ORDERINGS,
    ids=[c[0] for c in DIRECT_ORDERINGS + NOT_ORDERINGS],
)
def test_polars_declines_typed_on_incomparable_ordering(name, query):
    """polars: NotImplementedError (parity-or-error), never a silent row set."""
    pytest.importorskip("polars")
    with pytest.raises(NotImplementedError):
        _graph("polars").gfql(query, engine="polars")


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_mixed_type_equality_stays_typed_not_null(engine):
    """Equality is NOT ordering: mixed-type ``=`` keeps its pre-existing typed
    decline on both engines -- the #1934 null reroute must not swallow it."""
    from graphistry.compute.exceptions import GFQLSchemaError

    with pytest.raises(GFQLSchemaError):
        _graph(engine).gfql(
            "MATCH (n) WHERE n.str_col = 1.0 RETURN n.id AS id", engine=engine
        )


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_mixed_type_inequality_serves_spec_answer(engine):
    """openCypher ``'x' <> 1.0`` is true (not null): non-null rows survive."""
    assert _ids(_graph(engine).gfql(
        "MATCH (n) WHERE n.str_col <> 1.0 RETURN n.id AS id", engine=engine
    )) == ["a", "b"]
