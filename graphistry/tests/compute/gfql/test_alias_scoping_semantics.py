"""Alias-scoping semantics pins for #1911 (round-008 adversarial scoping probe).

Four defects, all pinned on BOTH engines with hand-computed openCypher oracles:

1. ``WITH a AS b`` where ``b`` is another MATCH-bound alias — the rename took its ROWS
   from ``a`` but resolved ``.property`` against the SHADOWED ``b`` (all-NULL on pandas
   over disjoint node sets, correct on polars). Now a typed decline on both.
2. ``WITH a AS b, b AS a`` (and ``WITH a, b WITH a AS b``) — openCypher WITH projections
   are SIMULTANEOUS, so those are a swap / a rebind; both engines silently returned the
   UNSWAPPED / shadowed values. Now the same typed decline.
3. A user EDGE column literally named like the internal per-edge identity — pandas
   adopted it AS the identity (nulling the user's values in ``r.<col>`` and dropping it
   from whole-entity ``RETURN r``); polars died with ``duplicate column name`` on EVERY
   relationship query for that graph. Now the internal name is resolved with
   ``generate_safe_column_name`` against the user's columns.
4. An alias named identically to a column it projects — the alias marker
   (``<alias> = True``) overwrote the user column. Fixed for the connected-pattern
   binding path (node + single-hop relationship aliases) and typed for the node-ID
   collision; the remaining shapes are pinned below as explicit residuals.

Discriminating controls from the probe are pinned alongside each fix so a future
regression cannot pass by repairing only one side.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

pl = pytest.importorskip("polars")

ENGINES = ["pandas", "polars"]


# ---------------------------------------------------------------- fixtures

# a Alice 30, b Bob 25, c Carol 35, d Dave 40 (:Person); x Acme (:Company)
# a-[:KNOWS w1]->b, b-[:KNOWS w2]->c, a-[:KNOWS w3]->c, c-[:KNOWS w4]->d, a-[:WORKS w5]->x
PEOPLE_NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d", "x"],
    "name": ["Alice", "Bob", "Carol", "Dave", "Acme"],
    "age": [30, 25, 35, 40, 0],
    "label__Person": [True, True, True, True, False],
    "label__Company": [False, False, False, False, True],
})
PEOPLE_EDGES = pd.DataFrame({
    "s": ["a", "b", "a", "c", "a"],
    "d": ["b", "c", "c", "d", "x"],
    "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS", "WORKS"],
    "w": [1, 2, 3, 4, 5],
})

# user columns literally named like GFQL internals
RESERVED_NODES = pd.DataFrame({
    "id": ["a", "b", "c"],
    "name": ["Alice", "Bob", "Carol"],
    "label__Person": [True, True, True],
    "__gfql_edge_ident__": ["N1", "N2", "N3"],
    "__gfql_edge_index_0__": ["M1", "M2", "M3"],
    "__cypher_group__": ["G1", "G2", "G3"],
})
RESERVED_EDGES = pd.DataFrame({
    "s": ["a", "b"],
    "d": ["b", "c"],
    "type": ["KNOWS", "KNOWS"],
    "__gfql_edge_ident__": ["E1", "E2"],
    "__gfql_edge_index_0__": ["F1", "F2"],
})

# node column named like the aliases the tests bind
SHADOW_NODES = pd.DataFrame({
    "id": ["n1", "n2", "n3"],
    "name": ["One", "Two", "Three"],
    "kind": ["K1", "K2", "K3"],
    "label__P": [True, True, True],
})
SHADOW_EDGES = pd.DataFrame({
    "s": ["n1", "n2"],
    "d": ["n2", "n3"],
    "type": ["K", "K"],
    "w": [7, 8],
})


def _graph(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str):
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(g, query: str, engine: str):
    frame = g.gfql(query, engine=engine)._nodes
    if frame is None:
        return []
    if isinstance(frame, pl.DataFrame):
        return frame.to_dicts()
    return frame.to_dict("records")


def _run(nodes, edges, query: str, engine: str):
    return _rows(_graph(nodes, edges, engine), query, engine)


# ------------------------------------------------- fix 1 + 2: WITH rebind

REBIND_DECLINED = [
    # #1911 defect 1: rows from `a`, properties from the shadowed `b`.
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b RETURN b.name",
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH b AS a RETURN a.name",
    # whole-entity projection of the same rename (was CORRECT before, but the rename is
    # unsupported either way -- pin that it declines rather than silently splitting).
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b RETURN b",
    # #1911 defect 2: openCypher WITH is SIMULTANEOUS, so this is a swap; both engines
    # returned the UNSWAPPED rows.
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b, b AS a RETURN a.name, b.name",
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH b AS a, a AS b RETURN a.name, b.name",
    # #1911 defect 2, staged form: the second WITH returned the SHADOWED original.
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a, b WITH a AS b RETURN b.name",
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a WITH a AS b RETURN b.name",
    # ORDER BY / WHERE do not launder it either
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b ORDER BY b.name RETURN b.name",
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b WHERE b.age > 26 RETURN b.name",
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", REBIND_DECLINED)
def test_with_rebind_onto_pattern_alias_declines(query: str, engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert exc_info.value.code == ErrorCode.E108
    assert "rebind an entity alias" in str(exc_info.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_with_rebind_decline_is_engine_identical(engine: str) -> None:
    """Both engines must produce the SAME typed decline -- polars previously answered
    `WITH a AS b RETURN b.name` CORRECTLY while pandas nulled it, so a one-sided fix
    would leave the divergence in place."""
    query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS b RETURN b.name"
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert exc_info.value.context["value"] == "a AS b"


@pytest.mark.parametrize("engine", ENGINES)
def test_with_rename_to_fresh_name_still_declines_as_unknown_alias(engine: str) -> None:
    """CONTROL (probe discriminator): the NON-colliding rename `WITH a AS fresh` has
    always declined -- lowering has no whole-entity alias renaming at all. The #1911 bug
    was that the colliding rename slipped past that same wall. Both must decline."""
    query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS fresh RETURN fresh.name"
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert exc_info.value.code == ErrorCode.E108
    assert "Unknown Cypher alias 'fresh'" in str(exc_info.value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # self-rename is a no-op, not a rebind
        (
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS a RETURN a.name",
            [{"a.name": "Alice"}, {"a.name": "Bob"}, {"a.name": "Carol"}],
        ),
        # plain carry is untouched
        (
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a, b RETURN b.name",
            [{"b.name": "Bob"}, {"b.name": "Carol"}, {"b.name": "Carol"}, {"b.name": "Dave"}],
        ),
        # a PROPERTY projected onto a live alias name is a scalar; openCypher lets a new
        # scope shadow freely, and lowering handles it -- the guard must NOT fire here.
        (
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a.name AS b RETURN b",
            [{"b": "Alice"}, {"b": "Alice"}, {"b": "Bob"}, {"b": "Carol"}],
        ),
    ],
)
def test_with_non_rebind_shapes_are_unaffected(query: str, expected, engine: str) -> None:
    rows = _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    key = sorted(expected[0].keys())
    assert sorted([tuple(r[k] for k in key) for r in rows]) == sorted(
        [tuple(r[k] for k in key) for r in expected]
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_terminal_return_rename_onto_live_alias_still_works(engine: str) -> None:
    """CONTROL: a terminal `RETURN a AS b` only names an output column -- no later clause
    resolves against it -- so it must keep working on both engines."""
    rows = _run(PEOPLE_NODES, PEOPLE_EDGES,
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a AS b", engine)
    assert sorted(r["b.name"] for r in rows) == ["Alice", "Bob", "Carol"]


@pytest.mark.parametrize("engine", ENGINES)
def test_with_rebind_into_reentry_match_is_not_a_rebind(engine: str) -> None:
    """CONTROL: `WITH a AS x MATCH (x)-->(b)` introduces `x` and the trailing MATCH
    CONSUMES it -- the reentry pattern alias is not a competing binding, so the guard
    must not fire (this is the LDBC-style WITH->MATCH re-entry shape)."""
    rows = _run(PEOPLE_NODES, PEOPLE_EDGES,
                "MATCH (a:Person) WITH a AS x MATCH (x)-[:KNOWS]->(b:Person) RETURN b.name AS bn",
                engine)
    assert sorted(r["bn"] for r in rows) == ["Bob", "Carol", "Carol", "Dave"]


# --------------------------------- fix 3: user column named like edge identity

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("column", ["__gfql_edge_ident__", "__gfql_edge_index_0__"])
def test_user_edge_column_named_like_internal_identity_is_preserved(
    column: str, engine: str
) -> None:
    """#1911 defect 3: pandas read these back as NULL (its identity column had replaced
    them); polars raised `duplicate column name __gfql_edge_ident__` from the
    unconditional `with_row_index`, on EVERY relationship query for such a graph."""
    rows = _run(
        RESERVED_NODES, RESERVED_EDGES,
        f"MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name AS n, r.{column} AS e",
        engine,
    )
    expected = {"__gfql_edge_ident__": ["E1", "E2"], "__gfql_edge_index_0__": ["F1", "F2"]}[column]
    assert sorted((r["n"], r["e"]) for r in rows) == sorted(zip(["Alice", "Bob"], expected))


@pytest.mark.parametrize("engine", ENGINES)
def test_relationship_query_on_reserved_named_edge_graph_does_not_crash(engine: str) -> None:
    """The polars crash hit queries that never MENTION the colliding column."""
    rows = _run(RESERVED_NODES, RESERVED_EDGES,
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name AS n", engine)
    assert sorted(r["n"] for r in rows) == ["Alice", "Bob"]


def test_whole_entity_relationship_omits_reserved_named_user_columns() -> None:
    """Whole-entity `RETURN r` omits user columns shaped like GFQL internals
    (``__gfql_*__``) on BOTH sides of the fix -- the whole-entity renderer filters that
    name pattern by design. What #1911 changed is that the values are no longer
    DESTROYED: they read back correctly through explicit `r.<col>` access (pinned in
    `test_user_edge_column_named_like_internal_identity_is_preserved`). Polars declines
    whole-entity relationship rendering, so this pin is pandas-only."""
    rows = _run(RESERVED_NODES, RESERVED_EDGES,
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r", "pandas")
    assert rows == [{"r.type": "KNOWS"}, {"r.type": "KNOWS"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_node_side_reserved_names_stay_correct(engine: str) -> None:
    """CONTROL: node-side `__gfql_edge_ident__` / `__cypher_group__` were already handled
    correctly -- the defect was edge-identity-specific."""
    rows = _run(RESERVED_NODES, RESERVED_EDGES,
                "MATCH (a:Person) RETURN a.__gfql_edge_ident__ AS v, a.__cypher_group__ AS g",
                engine)
    assert sorted((r["v"], r["g"]) for r in rows) == [("N1", "G1"), ("N2", "G2"), ("N3", "G3")]


@pytest.mark.parametrize("engine", ENGINES)
def test_trail_semantics_still_hold_on_reserved_named_edge_graph(engine: str) -> None:
    """The identity column exists to enforce openCypher relationship uniqueness; renaming
    it must not disable that. n1->n2->n3 has exactly one 2-hop trail."""
    rows = _run(RESERVED_NODES, RESERVED_EDGES,
                "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
                "RETURN a.name AS an, c.name AS cn", engine)
    assert sorted((r["an"], r["cn"]) for r in rows) == [("Alice", "Carol")]


# ------------------------------- fix 4: alias named like the column it projects

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # node alias == node column, connected pattern: pandas returned the alias marker
        # `True` while polars returned the real value.
        ("MATCH (kind:P)-[:K]->(b:P) RETURN kind.kind AS k", ["K1", "K2"]),
        ("MATCH (a:P)-[:K]->(kind:P) RETURN kind.kind AS k", ["K2", "K3"]),
        # relationship alias == edge column, single hop with a node property attached
        # (the connected-bindings route): same divergence, now fixed.
        ("MATCH (a:P)-[w:K]->(b:P) RETURN w.w AS k, a.id AS ai", [7, 8]),
    ],
)
def test_connected_pattern_alias_named_like_its_column_reads_the_user_value(
    query: str, expected, engine: str
) -> None:
    """#1911 defect 4 (connected-binding path): `ASTNode`/`ASTEdge.execute` stamp the
    alias flag as `<alias> = True`, which overwrote the user column that the property
    lookup then read."""
    rows = _run(SHADOW_NODES, SHADOW_EDGES, query, engine)
    assert sorted(r["k"] for r in rows) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_alias_named_like_a_different_column_is_unaffected(engine: str) -> None:
    """CONTROL (probe discriminator): the corruption never spread to OTHER properties."""
    rows = _run(SHADOW_NODES, SHADOW_EDGES,
                "MATCH (kind:P)-[:K]->(b:P) RETURN kind.name AS n, b.kind AS bk", engine)
    assert sorted((r["n"], r["bk"]) for r in rows) == [("One", "K2"), ("Two", "K3")]


@pytest.mark.parametrize("engine", ENGINES)
def test_alias_named_after_node_id_binding_is_a_typed_decline(engine: str) -> None:
    """#1911 defect 4 variant: `MATCH (id:P) RETURN id.id` overwrote the node-ID column
    with the alias flag -- pandas died with a RAW `ValueError: The column label 'id' is
    not unique`, polars answered `True`. Both now decline with the same typed error."""
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(SHADOW_NODES, SHADOW_EDGES, "MATCH (id:P) RETURN id.id AS i", engine)
    assert exc_info.value.code == ErrorCode.E108
    assert "node-ID binding column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_entity_projection_of_shadowing_alias_omits_the_column(engine: str) -> None:
    """Whole-entity `RETURN kind` drops the shadowed column rather than emitting the
    marker -- identical on both engines, unchanged by #1911. Pinned so a later marker
    redesign has to decide this case deliberately."""
    rows = _run(SHADOW_NODES, SHADOW_EDGES, "MATCH (kind:P) RETURN kind", engine)
    assert "kind.kind" not in rows[0]
    assert sorted(r["kind.name"] for r in rows) == ["One", "Three", "Two"]


# ------------------------------------------------------------------ residuals

@pytest.mark.parametrize("engine", ENGINES)
def test_residual_single_alias_and_cartesian_alias_marker_still_leaks(engine: str) -> None:  # noqa: ARG001
    """RESIDUAL (#1911 defect 4, not fixed this cycle): the single-alias
    `rows(table='nodes', source=...)` and the cartesian binding paths read properties off
    the chain output frame, where the alias marker has ALREADY overwritten the user
    column -- the pre-marker values are gone by then, so the fix needs the marker itself
    to move to a safe internal name (which chain.py's own labeling machinery reads back).
    Pinned CONSISTENT across engines so the divergence cannot reappear silently."""
    for query in ["MATCH (kind:P) RETURN kind.kind AS k",
                  "MATCH (kind:P), (b:P) RETURN kind.kind AS k, b.id AS bi",
                  # a bare relationship-property projection also skips the connected-
                  # bindings property attach that the fix hooks.
                  "MATCH (a:P)-[w:K]->(b:P) RETURN w.w AS k"]:
        rows = _run(SHADOW_NODES, SHADOW_EDGES, query, "pandas")
        assert {r["k"] for r in rows} == {True}, f"{query} -> {rows}"


def test_residual_multihop_relationship_alias_marker_divergence() -> None:
    """RESIDUAL (#1911 defect 4): a relationship alias in a MULTI-hop pattern still takes
    a different path than the single-hop one fixed above -- pandas reads the marker,
    polars reads the value (and also over-multiplies the rows, a separate pre-existing
    polars defect independent of the alias name). Pinned so the state is explicit."""
    query = "MATCH (a:P)-[w:K]->(b:P)-[:K]->(c:P) RETURN w.w AS x"
    assert [r["x"] for r in _run(SHADOW_NODES, SHADOW_EDGES, query, "pandas")] == [True]
    assert sorted(r["x"] for r in _run(SHADOW_NODES, SHADOW_EDGES, query, "polars")) == [7, 8]


def test_residual_empty_result_drops_a_shadowed_alias_column_on_pandas() -> None:
    """RESIDUAL (#1911 defect 4): when an alias is named after ANY existing node column
    and the match is EMPTY, pandas' chain output loses that column outright, so the
    downstream `rows` op fails its schema check; polars returns the correct empty result.
    Reproduces with a non-colliding property too (`kind.name`), so it is a chain
    empty-frame column-preservation bug rather than a property-resolution one."""
    from graphistry.compute.exceptions import GFQLSchemaError

    query = "MATCH (kind:P) WHERE kind.name = 'ZZ' RETURN kind.name AS n"
    with pytest.raises(GFQLSchemaError):
        _run(SHADOW_NODES, SHADOW_EDGES, query, "pandas")
    assert _run(SHADOW_NODES, SHADOW_EDGES, query, "polars") == []


# ------------------------------------------------- error-message attribution

@pytest.mark.parametrize("engine", ENGINES)
def test_unresolved_identifier_error_names_the_visible_scope(engine: str) -> None:
    """#1911 (low severity): `MATCH (a) WITH a MATCH (q)-->(z) RETURN a.name` declines
    naming 'a' -- an alias the user DID bind -- because the trailing MATCH re-scoped.
    The decline itself is an NIE (openCypher makes this a cartesian), but the message
    must say what IS visible so the attribution is not read as 'you never declared a'."""
    query = "MATCH (a:Person) WITH a MATCH (q)-[:KNOWS]->(z) RETURN a.name AS an"
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert exc_info.value.code == ErrorCode.E204
    assert "Visible in this scope" in str(exc_info.value)
    assert exc_info.value.context["visible_scope"] == ["q", "z"]


# --------------------------------------------------- helper-level unit pins

def test_generate_safe_column_name_from_skips_taken_names() -> None:
    from graphistry.compute.util import generate_safe_column_name_from

    assert generate_safe_column_name_from("edge_ident", []) == "__gfql_edge_ident_0__"
    assert generate_safe_column_name_from(
        "edge_ident", ["__gfql_edge_ident_0__", "__gfql_edge_ident_1__"]
    ) == "__gfql_edge_ident_2__"


def test_unshadow_alias_marker_column_declines_when_there_is_nothing_to_restore() -> None:
    """The restore is a no-op unless the alias genuinely shadows a user column that the
    key column can be joined back on."""
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin as RPM

    frame = pd.DataFrame({"id": [1, 2], "kind": [True, True]})
    base = pd.DataFrame({"id": [1, 2], "kind": ["K1", "K2"]})
    assert RPM._gfql_unshadow_alias_marker_column(frame, "kind", None, "id") is frame
    assert RPM._gfql_unshadow_alias_marker_column(None, "kind", base, "id") is None
    # alias == key column: the marker already destroyed the join key, nothing to restore
    assert RPM._gfql_unshadow_alias_marker_column(frame, "id", base, "id") is frame
    # alias is not a user column
    assert RPM._gfql_unshadow_alias_marker_column(frame, "other", base, "id") is frame
    # key column missing from one side
    assert RPM._gfql_unshadow_alias_marker_column(
        frame, "kind", base.drop(columns=["id"]), "id"
    ) is frame
    restored = RPM._gfql_unshadow_alias_marker_column(frame, "kind", base, "id")
    assert list(restored["kind"]) == ["K1", "K2"]


def test_trail_edge_identity_col_is_resolved_against_user_edge_columns() -> None:
    from graphistry.compute.gfql_unified import _trail_edge_identity_col

    g = _graph(RESERVED_NODES, RESERVED_EDGES, "pandas")
    assert _trail_edge_identity_col(g) == "__gfql_edge_index_1__"  # _0__ is a user column
    assert _trail_edge_identity_col(g.edges(pd.DataFrame({"s": [], "d": []}), "s", "d")) == (
        "__gfql_edge_index_0__"
    )
    assert _trail_edge_identity_col(graphistry.bind()) == "__gfql_edge_index_0__"
