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
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher

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


def _null(v):
    return None if isinstance(v, float) and v != v else v


def _rows(g, query: str, engine: str):
    frame = g.gfql(query, engine=engine)._nodes
    if frame is None:
        return []
    records = frame.to_dicts() if isinstance(frame, pl.DataFrame) else frame.to_dict("records")
    return [{k: _null(v) for k, v in r.items()} for r in records]


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


# ---------------------------------------- defect-4 remaining shapes (fixed)

@pytest.mark.parametrize("engine", ENGINES)
def test_single_alias_and_cartesian_shadowed_alias_reads_the_user_value(engine: str) -> None:
    """#1911 defect 4 (was RESIDUAL): the single-alias ``rows(table=..., source=...)``
    route and the cartesian binding path read properties off the chain output frame,
    where the alias marker had overwritten the user column. The rows route now restores
    the user values from the base frame (dotted self-column, marker kept boolean); the
    cartesian paths unshadow like the connected one. Anti-vacuity: 3 node rows / 9
    cartesian rows / 2 relationship rows, distinct values -- an all-``True`` marker
    leak or an empty frame cannot pass."""
    rows = _run(SHADOW_NODES, SHADOW_EDGES, "MATCH (kind:P) RETURN kind.kind AS k", engine)
    assert sorted(r["k"] for r in rows) == ["K1", "K2", "K3"]
    rows = _run(SHADOW_NODES, SHADOW_EDGES,
                "MATCH (kind:P), (b:P) RETURN kind.kind AS k, b.id AS bi", engine)
    assert len(rows) == 9 and sorted({r["k"] for r in rows}) == ["K1", "K2", "K3"]
    # bare relationship-property projection (rows(table='edges', source=alias) route)
    rows = _run(SHADOW_NODES, SHADOW_EDGES, "MATCH (a:P)-[w:K]->(b:P) RETURN w.w AS k", engine)
    assert sorted(r["k"] for r in rows) == [7, 8]


def test_multihop_relationship_shadowed_alias_pandas_fixed_polars_multiplicity_residual() -> None:
    """#1911 defect 4: pandas now reads the user value in the MULTI-hop shape too
    (was the marker ``True``). polars still over-multiplies the rows -- a separate
    pre-existing multiplicity defect independent of the alias name -- pinned as-is."""
    query = "MATCH (a:P)-[w:K]->(b:P)-[:K]->(c:P) RETURN w.w AS x"
    assert [r["x"] for r in _run(SHADOW_NODES, SHADOW_EDGES, query, "pandas")] == [7]
    assert sorted(r["x"] for r in _run(SHADOW_NODES, SHADOW_EDGES, query, "polars")) == [7, 8]


def test_empty_result_keeps_a_shadowed_alias_column_on_both_engines() -> None:
    """#1911 defect 4 (was RESIDUAL): an EMPTY match on an alias named after an existing
    node column used to lose that column on pandas (chain empty-frame column drop), so the
    downstream ``rows`` op raised a schema error; the marker-aware coalesce now keeps the
    column even at zero rows and both engines return the correct empty result."""
    query = "MATCH (kind:P) WHERE kind.name = 'ZZ' RETURN kind.name AS n"
    assert _run(SHADOW_NODES, SHADOW_EDGES, query, "pandas") == []
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


# ------------------------------------------- round-005 amplification cells

def test_trail_edge_identity_col_resolves_against_arrow_edge_frames() -> None:
    """The bound frame may still be pyarrow when the identity name is resolved --
    arrow exposes names as `.column_names`, and `.columns` (the arrays) would
    make the collision scan crash or silently miss."""
    pa = pytest.importorskip("pyarrow")
    from graphistry.compute.gfql_unified import _trail_edge_identity_col

    plain = graphistry.edges(pa.table({"s": ["a"], "d": ["b"], "w": [1]}), "s", "d")
    assert _trail_edge_identity_col(plain) == "__gfql_edge_index_0__"
    colliding = graphistry.edges(
        pa.table({"s": ["a"], "d": ["b"], "__gfql_edge_index_0__": [9]}), "s", "d"
    )
    assert _trail_edge_identity_col(colliding) == "__gfql_edge_index_1__"


@pytest.mark.parametrize("engine", ENGINES)
def test_with_rebind_edge_alias_onto_edge_alias_declines(engine: str) -> None:
    """Edge-onto-edge is the same-kind rebind on the RELATIONSHIP side; before
    the guard it silently answered [None, 2, None] on pandas (rows from r,
    properties resolved against the shadowed q)."""
    query = ("MATCH (a:Person)-[r:KNOWS]->(b:Person)-[q:KNOWS]->(c:Person) "
             "WITH r AS q RETURN q.w")
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert exc_info.value.code == ErrorCode.E108
    assert "rebind an entity alias" in str(exc_info.value)
    assert exc_info.value.context["value"] == "r AS q"


@pytest.mark.parametrize(
    ("query", "value"),
    [
        (
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a AS r RETURN r.type AS t",
            "a AS r",
        ),
        (
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH r AS b RETURN b.w AS t",
            "r AS b",
        ),
    ],
    ids=["node_onto_edge", "edge_onto_node"],
)
def test_cross_kind_entity_rebinds_decline_at_compile_time(query: str, value: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    assert exc_info.value.code == ErrorCode.E108
    assert "rebind an entity alias" in str(exc_info.value)
    assert exc_info.value.context["value"] == value


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_rebind_stays_an_error(engine: str) -> None:
    query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a.name AS b RETURN b.name AS t"
    if engine == "pandas":
        with pytest.raises(GFQLTypeError) as exc_info:
            _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
        assert exc_info.value.code == ErrorCode.E303
        assert exc_info.value.context["field"] == "function"
        assert exc_info.value.context["value"] == "select"
    else:
        with pytest.raises(NotImplementedError):
            _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_shadow_in_where_keeps_the_bag(engine: str) -> None:
    """A scalar projected onto a live alias name flows through WHERE with bag
    multiplicity: a-side ages [30, 25, 30, 35], filter > 26 -> [30, 30, 35]."""
    rows = _run(PEOPLE_NODES, PEOPLE_EDGES,
                "MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH a.age AS b WHERE b > 26 "
                "RETURN b AS t", engine)
    assert sorted(r["t"] for r in rows) == [30, 30, 35]


@pytest.mark.parametrize("engine", ENGINES)
def test_unwind_alias_collision_still_declines_before_the_rebind_guard(engine: str) -> None:
    """The guard's UNWIND carve-out never has to defend a colliding UNWIND alias:
    the collision rejection fires first."""
    query = ("MATCH (a:Person)-[:KNOWS]->(b:Person) UNWIND [1, 2] AS a "
             "WITH b AS a RETURN a.name AS t")
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES, query, engine)
    assert "UNWIND alias collides" in str(exc_info.value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("alias", ["s", "d"])
def test_edge_alias_named_like_an_endpoint_binding_is_a_typed_decline(alias: str, engine: str) -> None:
    """#1911 defect-4 sibling (was RESIDUAL): an edge alias named after the edge
    SOURCE/DESTINATION binding column has its marker destroy the endpoints. It used to
    surface as an incidental GFQLSchemaError on pandas; the marker-aware coalesce would
    have turned it into a silent EMPTY result, so it is now the same typed decline as
    the node-ID collision, on both engines."""
    with pytest.raises(GFQLValidationError) as exc_info:
        _run(PEOPLE_NODES, PEOPLE_EDGES,
             f"MATCH (a:Person)-[{alias}:KNOWS]->(b:Person) RETURN {alias}.type AS t", engine)
    assert exc_info.value.code == ErrorCode.E108
    assert "edge endpoint binding column" in str(exc_info.value)


# -------------------- #1911 defect-4 round-2: single-alias rows-route restore

# NULL cell on purpose: the restore must carry the NULL through (the NULL-vs-membership
# class), never drop the row or backfill the marker.
SELF_NAMED_NODES = pd.DataFrame({
    "id": ["a1", "a2", "b1", "b2"],
    "name": ["Sa", "Sb", None, "Tb"],
    "label__P": [True, True, True, True],
})
SELF_NAMED_EDGES = pd.DataFrame({
    "s": ["a1", "a2"], "d": ["b1", "b2"], "type": ["K", "K"], "w": [7, 8],
})


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_route_alias_named_as_its_property_reads_user_values(engine: str) -> None:
    """#1911 defect-4 (rows route): ``MATCH (name:P) RETURN name.name`` answered the
    alias marker ``[True] x 4`` on both engines (cuDF crashed with a raw mixed-types
    TypeError). Anti-vacuity: 4 rows with 3 distinct values plus a preserved NULL."""
    rows = _run(SELF_NAMED_NODES, SELF_NAMED_EDGES, "MATCH (name:P) RETURN name.name", engine)
    assert [r["name.name"] for r in rows] == ["Sa", "Sb", None, "Tb"]


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_route_where_on_self_named_alias_filters_and_projects_user_values(engine: str) -> None:
    """WHERE already compared user values (1 row matched) while RETURN projected the
    marker ``True`` -- both sides must read the same user column."""
    rows = _run(SELF_NAMED_NODES, SELF_NAMED_EDGES,
                "MATCH (name:P) WHERE name.name = 'Sa' RETURN name.name AS n", engine)
    assert rows == [{"n": "Sa"}]


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_route_edge_alias_named_as_its_property_reads_user_values(engine: str) -> None:
    """Edge twin of the rows route (``rows(table='edges', source=alias)``): the marker
    also overwrote the same-named edge payload column."""
    rows = _run(SELF_NAMED_NODES, SELF_NAMED_EDGES,
                "MATCH (a:P)-[w:K]->(b:P) RETURN w.w AS x", engine)
    assert sorted(r["x"] for r in rows) == [7, 8]


def test_rows_route_edge_alias_colliding_with_its_own_type_filter() -> None:
    """``MATCH (a)-[type:K]->(b) RETURN type.type``: the alias shadows the very column
    its ``:K`` filter reads. pandas/cuDF now answer the user values; polars' chain
    machinery re-applies the type filter against the stamped marker and raises a typed
    GFQLSchemaError -- an honest decline, pinned so it cannot rot into a silent wrong
    answer (residual polish for #1911)."""
    from graphistry.compute.exceptions import GFQLSchemaError

    query = "MATCH (a:P)-[type:K]->(b:P) RETURN type.type AS t"
    assert _run(SELF_NAMED_NODES, SELF_NAMED_EDGES, query, "pandas") == [
        {"t": "K"}, {"t": "K"}]
    with pytest.raises(GFQLSchemaError):
        _run(SELF_NAMED_NODES, SELF_NAMED_EDGES, query, "polars")


@pytest.mark.parametrize("engine", ENGINES)
def test_rows_route_self_named_plus_other_property(engine: str) -> None:
    """The restore must not turn OTHER property reads of the same alias into NA: the
    row table is not a bindings table just because the shadowed value was re-keyed
    (an early restore design leaked exactly that, NA-ing ``name.id``)."""
    rows = _run(SELF_NAMED_NODES, SELF_NAMED_EDGES,
                "MATCH (name:P) RETURN name.name AS n, name.id AS i", engine)
    assert [(r["n"], r["i"]) for r in rows] == [
        ("Sa", "a1"), ("Sb", "a2"), (None, "b1"), ("Tb", "b2")]


def test_connected_join_carried_relationship_projection_unaffected() -> None:
    """CONTROL (regression found in-flight): a comma-pattern whose base graph is an
    intermediate dispatch frame must not have the restore misread the arm's marker as
    user data — ``r.weight`` stays 7, never NA."""
    nodes = pd.DataFrame({"id": ["a1", "b1"], "label__A": [True, False], "label__B": [False, True]})
    edges = pd.DataFrame({"s": ["a1", "b1"], "d": ["b1", "a1"], "type": ["R", "S"], "w": [7, 9]})
    rows = _run(nodes, edges,
                "MATCH (a:A {id: 'a1'})-[r:R]->(b:B), (b)-[:S]->(a) RETURN r.w AS w", "pandas")
    assert rows == [{"w": 7}]


@pytest.mark.parametrize("engine", ENGINES)
def test_mixed_whole_entity_and_self_named_property_projection(engine: str) -> None:
    """``RETURN name, name.name AS n``: the whole-entity flatten keeps omitting the
    shadowed column (pinned above) while the explicit property column must read the
    restored user values, NULL included."""
    rows = _run(SELF_NAMED_NODES, SELF_NAMED_EDGES,
                "MATCH (name:P) RETURN name, name.name AS n", engine)
    assert [r["n"] for r in rows] == ["Sa", "Sb", None, "Tb"]
    assert [r["name.id"] for r in rows] == ["a1", "a2", "b1", "b2"]


def test_restore_alias_shadowed_user_column_branches() -> None:
    """Helper-level pins for the rows-route restore: no-op without a base, a shadowed
    column, or when the base column is itself a boolean marker (intermediate dispatch
    graph); index-keyed restore; key-merge fallback when the index cannot re-key."""
    from types import SimpleNamespace

    from graphistry.compute.gfql.identifiers import shadow_restore_column
    from graphistry.compute.gfql.row.frame_ops import _restore_alias_shadowed_user_column

    restore_col = shadow_restore_column("kind")

    def ctx_for(base_graph):
        return SimpleNamespace(_gfql_rows_base_graph=base_graph, _g=None)

    marked = pd.DataFrame({"id": ["a", "b"], "kind": [True, True]})
    # no base graph / alias shadows nothing: unchanged
    assert _restore_alias_shadowed_user_column(ctx_for(None), marked, "nodes", "kind") is marked
    base_no_col = SimpleNamespace(_nodes=pd.DataFrame({"id": ["a", "b"]}), _edges=None, _node="id", _edge=None)
    assert _restore_alias_shadowed_user_column(ctx_for(base_no_col), marked, "nodes", "kind") is marked
    # base column is itself a boolean marker (an intermediate dispatch graph): unchanged
    base_marker = SimpleNamespace(_nodes=pd.DataFrame({"id": ["a", "b"], "kind": [True, False]}), _edges=None, _node="id", _edge=None)
    assert _restore_alias_shadowed_user_column(ctx_for(base_marker), marked, "nodes", "kind") is marked
    # index-keyed restore adds the internal restore column and keeps the marker boolean
    base = SimpleNamespace(_nodes=pd.DataFrame({"id": ["a", "b"], "kind": ["K1", "K2"]}), _edges=None, _node="id", _edge=None)
    out = _restore_alias_shadowed_user_column(ctx_for(base), marked, "nodes", "kind")
    assert list(out[restore_col]) == ["K1", "K2"] and list(out["kind"]) == [True, True]
    # base index cannot re-key (duplicate labels): fall back to the id-key merge
    dup_index_nodes = pd.DataFrame({"id": ["a", "b"], "kind": ["K1", "K2"]}, index=[0, 0])
    base_dup = SimpleNamespace(_nodes=dup_index_nodes, _edges=None, _node="id", _edge=None)
    out = _restore_alias_shadowed_user_column(ctx_for(base_dup), marked, "nodes", "kind")
    assert list(out[restore_col]) == ["K1", "K2"]
    # neither index nor key can re-key: unchanged (marker stays, as before)
    base_no_key = SimpleNamespace(_nodes=dup_index_nodes, _edges=None, _node=None, _edge=None)
    assert _restore_alias_shadowed_user_column(ctx_for(base_no_key), marked, "nodes", "kind") is marked
    # row-table labels absent from a unique base index: guarded .loc declines to the key merge
    shifted = pd.DataFrame({"id": ["a", "b"], "kind": [True, True]}, index=[10, 11])
    out = _restore_alias_shadowed_user_column(ctx_for(base), shifted, "nodes", "kind")
    assert list(out[restore_col]) == ["K1", "K2"]
    # polars: id-keyed join replaces the marker column in place; no key -> unchanged
    marked_pl = pl.DataFrame({"id": ["a", "b"], "kind": [True, True]})
    base_pl = SimpleNamespace(_nodes=pl.DataFrame({"id": ["a", "b"], "kind": ["K1", "K2"]}), _edges=None, _node="id", _edge=None)
    out_pl = _restore_alias_shadowed_user_column(ctx_for(base_pl), marked_pl, "nodes", "kind")
    assert out_pl["kind"].to_list() == ["K1", "K2"]
    base_pl_no_key = SimpleNamespace(_nodes=base_pl._nodes, _edges=None, _node=None, _edge=None)
    assert _restore_alias_shadowed_user_column(ctx_for(base_pl_no_key), marked_pl, "nodes", "kind") is marked_pl
    # polars marker-carrying base: unchanged
    base_pl_marker = SimpleNamespace(_nodes=pl.DataFrame({"id": ["a", "b"], "kind": [True, False]}), _edges=None, _node="id", _edge=None)
    assert _restore_alias_shadowed_user_column(ctx_for(base_pl_marker), marked_pl, "nodes", "kind") is marked_pl


def test_cudf_rows_route_self_named_alias_parity() -> None:
    """cuDF (dataframe ops only): the node shape crashed with a raw mixed-types
    TypeError from the marker/user-column coalesce; both shapes now match pandas."""
    cudf = pytest.importorskip("cudf")

    g = graphistry.nodes(cudf.from_pandas(SELF_NAMED_NODES), "id").edges(
        cudf.from_pandas(SELF_NAMED_EDGES), "s", "d"
    )
    out = g.gfql("MATCH (name:P) RETURN name.name", engine="cudf")._nodes.to_pandas()
    assert list(out["name.name"].fillna("<null>")) == ["Sa", "Sb", "<null>", "Tb"]
    out = g.gfql("MATCH (a:P)-[type:K]->(b:P) RETURN type.type AS t", engine="cudf")._nodes
    assert sorted(out.to_pandas()["t"]) == ["K", "K"]


def test_cudf_unshadow_and_rebind_guard_parity() -> None:
    """cuDF (dataframe ops only): the marker unshadow reads the user value and
    the rebind guard declines identically."""
    cudf = pytest.importorskip("cudf")

    def _cudf_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(edges), "s", "d"
        )

    g = _cudf_graph(SHADOW_NODES, SHADOW_EDGES)
    out = g.gfql("MATCH (kind:P)-[:K]->(b:P) RETURN kind.kind AS k", engine="cudf")._nodes
    assert sorted(out.to_pandas()["k"]) == ["K1", "K2"]
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql("MATCH (a:P)-[r:K]->(b:P) WITH a AS b RETURN b.name", engine="cudf")
    assert "rebind an entity alias" in str(exc_info.value)
