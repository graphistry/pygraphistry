"""Indexed-vs-scan agreement for the EXISTS / NOT EXISTS pattern-predicate family.

An index is an optimization: for every pattern shape it must produce the SAME rows as
the un-indexed graph, or decline identically. Never a third answer.

The matrix is generated (graph shape x pattern x polarity) rather than hand-listed, so
new adjacency shortcuts are held to the whole family and not just the reported input.
"""
import pytest

import graphistry

pl = pytest.importorskip("polars")


def _graphs():
    """(name, nodes, edges) shapes that stress every way edge-table keys can diverge
    from a node-table-intersected answer: self-loops, edge endpoints absent from the
    node table, isolated nodes, string ids, duplicate node rows, no edges at all."""
    return [
        ("self_loop_and_isolated",
         pl.DataFrame({"id": [0, 1, 2, 3]}), pl.DataFrame({"s": [0, 3], "d": [1, 3]})),
        ("endpoint_absent_from_nodes",
         pl.DataFrame({"id": [0, 1, 2]}), pl.DataFrame({"s": [0, 9], "d": [9, 9]})),
        ("chain_plus_self_loop",
         pl.DataFrame({"id": [0, 1, 2, 3, 4, 5]}), pl.DataFrame({"s": [0, 1, 4, 3], "d": [1, 2, 4, 0]})),
        ("string_ids",
         pl.DataFrame({"id": ["a", "b", "c", "d"]}), pl.DataFrame({"s": ["a", "d"], "d": ["b", "d"]})),
        ("duplicate_node_rows",
         pl.DataFrame({"id": [0, 0, 1, 2, 3]}), pl.DataFrame({"s": [0, 3], "d": [1, 3]})),
        ("every_edge_is_a_self_loop",
         pl.DataFrame({"id": [0, 1, 2]}), pl.DataFrame({"s": [0, 1], "d": [0, 1]})),
        ("no_edges",
         pl.DataFrame({"id": [0, 1, 2]}),
         pl.DataFrame({"s": [], "d": []}, schema={"s": pl.Int64, "d": pl.Int64})),
    ]


PATTERNS = [
    "(n)-->(n)", "(n)<--(n)", "(n)--(n)",
    "(n)-->(m)", "(n)<--(m)", "(n)--(m)",
    "(n)-->()", "(n)<--()", "(n)--()",
    "(n)-->()-->()", "(n)-->(m)-->(n)",
    "(n)-->(m) WHERE m <> n",
]


def _outcome(g, query):
    """('rows', sorted-ids) or ('declined', None) — the two outcomes an index may pick
    between; anything else is a third answer and fails the comparison outright."""
    try:
        out = g.gfql(query, engine="polars")
    except NotImplementedError:
        return ("declined", None)
    col = out._nodes.columns[0]
    return ("rows", tuple(sorted(out._nodes[col].to_list())))


@pytest.mark.parametrize("shape", [s[0] for s in _graphs()])
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("polarity", ["EXISTS", "NOT EXISTS"])
def test_indexed_pattern_predicate_matches_the_scan(shape, pattern, polarity):
    nodes, edges = next((n, e) for name, n, e in _graphs() if name == shape)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = "MATCH (n) WHERE %s { %s } RETURN n.id" % (polarity, pattern)
    scan = _outcome(g, query)
    indexed = _outcome(g.gfql_index_all(), query)
    assert indexed == scan, "indexed %r != scan %r for %s" % (indexed, scan, query)


def test_indexed_self_loop_exists_does_not_answer_has_any_out_edge():
    """The reported input: only node 3 self-loops, so 'has an out-edge' (nodes 0 and 3)
    is the wrong answer for ``(n)-->(n)``."""
    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2, 3]}), "id").edges(
        pl.DataFrame({"s": [0, 3], "d": [1, 3]}), "s", "d")
    query = "MATCH (n) WHERE EXISTS { (n)-->(n) } RETURN n.id"
    with pytest.raises(NotImplementedError):
        g.gfql(query, engine="polars")
    with pytest.raises(NotImplementedError):
        g.gfql_index_all().gfql(query, engine="polars")


def test_indexed_self_loop_not_exists_does_not_drop_satisfying_rows():
    """The mirror: ``NOT EXISTS { (n)-->(n) }`` is true of 0, 1 and 2 here, so the
    'has an out-edge' complement (1 and 2) drops a row that satisfies the predicate."""
    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2, 3]}), "id").edges(
        pl.DataFrame({"s": [0, 3], "d": [1, 3]}), "s", "d")
    query = "MATCH (n) WHERE NOT EXISTS { (n)-->(n) } RETURN n.id"
    with pytest.raises(NotImplementedError):
        g.gfql(query, engine="polars")
    with pytest.raises(NotImplementedError):
        g.gfql_index_all().gfql(query, engine="polars")


def _keys(g, ops, alias="n"):
    from graphistry.compute.ast import serialize_binding_ops
    from graphistry.compute.gfql.lazy.engine.polars.pattern_apply import _pattern_alias_keys_polars
    return _pattern_alias_keys_polars(g, serialize_binding_ops(ops), alias)


def test_adjacency_membership_declines_a_repeated_endpoint_alias():
    from graphistry.compute.ast import e_forward, n
    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2, 3]}), "id").edges(
        pl.DataFrame({"s": [0, 3], "d": [1, 3]}), "s", "d").gfql_index_all()
    assert _keys(g, [n(name="n"), e_forward(), n(name="n")]) is None
    assert _keys(g, [n(name="n"), e_forward(), n(name="m")]) is not None


def test_adjacency_membership_excludes_edges_whose_other_end_is_not_a_node():
    """Adjacency keys are edge-derived; the scan additionally requires BOTH endpoints in
    the node table. Node 0's only out-edge points at 9, which is not a node, so 0 does
    not participate — with or without an index."""
    from graphistry.compute.ast import e_forward, n
    ops = [n(name="n"), e_forward(), n(name="m")]
    dangling = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2]}), "id").edges(
        pl.DataFrame({"s": [0, 9], "d": [9, 9]}), "s", "d")
    covered = graphistry.nodes(pl.DataFrame({"id": [0, 1, 9]}), "id").edges(
        pl.DataFrame({"s": [0, 9], "d": [9, 9]}), "s", "d")

    def ids(g):
        keys = _keys(g, ops)
        assert keys is not None
        return sorted(keys.get_column("id").to_list())

    assert ids(dangling.gfql_index_all()) == [] == ids(dangling)
    assert ids(covered.gfql_index_all()) == [0, 9] == ids(covered)


@pytest.mark.parametrize("nodes,keys,expected", [
    (pl.DataFrame({"id": [0, 1, 2]}), [0, 2], True),
    (pl.DataFrame({"id": [0, 1, 2]}), [0, 9], False),
    (pl.DataFrame({"id": [0, None, 2]}), [0, 2], False),
    (pl.DataFrame({"id": ["a", "b"]}), [0, 1], False),
    (pl.LazyFrame({"id": [0, 1, 2]}), [0, 2], False),
    (None, [0], False),
    (pl.DataFrame({"other": [0, 1]}), [0], False),
])
def test_node_coverage_check_declines_anything_it_cannot_compare(nodes, keys, expected):
    """Every way the node table can fail to answer 'are these ids all nodes?' — absent,
    lazy, wrong column, null id, incomparable dtype — must read as NOT covered."""
    import numpy as np
    from graphistry.compute.gfql.lazy.engine.polars.pattern_apply import _nodes_cover_keys

    g = graphistry.edges(pl.DataFrame({"s": [0], "d": [1]}), "s", "d")
    if nodes is not None:
        g = g.nodes(nodes, "id" if "id" in nodes.collect_schema().names() else "other")
    assert _nodes_cover_keys(g, "id", np.asarray(keys)) is expected


@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_adjacency_membership_still_answers_the_distinct_alias_shapes(direction, monkeypatch):
    """The guards must not silently retire the adjacency route for the shapes it does
    answer — otherwise the agreement matrix above proves nothing."""
    from graphistry.compute.ast import e_forward, e_reverse, e_undirected, n
    import graphistry.compute.gfql.index.degrees as index_degrees

    edge_op = {"forward": e_forward, "reverse": e_reverse, "undirected": e_undirected}[direction]()
    g = graphistry.nodes(pl.DataFrame({"id": [0, 1, 2]}), "id").edges(
        pl.DataFrame({"s": [0, 1], "d": [1, 2]}), "s", "d").gfql_index_all()

    seen = []
    orig = index_degrees.adjacency_membership_keys

    def wrapped(registry, direction, edges_df, cols, engine):
        seen.append(direction)
        return orig(registry, direction, edges_df, cols, engine)

    monkeypatch.setattr(index_degrees, "adjacency_membership_keys", wrapped)
    keys = _keys(g, [n(name="n"), edge_op, n(name="m")])
    assert keys is not None
    assert seen, "the adjacency route was not consulted at all"
    expected = {"forward": [0, 1], "reverse": [1, 2], "undirected": [0, 1, 2]}[direction]
    assert sorted(keys.get_column("id").to_list()) == expected
