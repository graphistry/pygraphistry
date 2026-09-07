"""Alias / column collision matrix for native chains, every engine against the pandas full path.

An alias marker that shares its name with a frame column must never be re-read as that column
by a later filter, and the documented contract is marker-authoritative output. Pins: every
single-hop collision shape (edge alias = filtered / unfiltered edge column, node alias =
filtered node column on the seed or the destination, all three colliding at once) yields the
same node keys and edge pairs on pandas, cuDF and polars with and without resident indexes;
the multi-hop forms (#2049) and the binding-column aliases (#2050) are strict expected
failures that flip when those land.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, e_undirected, n
from graphistry.tests.compute.gfql.routes.registry import Frames, register

NODES = pd.DataFrame({"key": [1, 2, 3, 4], "id": [10, 20, 30, 40], "type": ["p", "p", "m", "m"], "w": [1, 2, 3, 4]})
EDGES = pd.DataFrame({"s": [3, 3, 4, 1], "d": [1, 2, 1, 4], "type": ["HAS_CREATOR", "OTHER", "HAS_CREATOR", "OTHER"],
                      "eid": [100, 101, 102, 103], "w": [5, 6, 7, 8]})
ENGINES = ["pandas", "cudf", "polars"]
FRAMES = Frames(NODES, EDGES, "key", "s", "d", "eid")


def _graph(engine, indexed):
    nodes, edges = NODES, EDGES
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine) if indexed else g


def _alias_marker(res, alias):
    """The op-list contract on every engine: the alias is a boolean marker column, shadowing any user column of that name."""
    ee = res._edges.to_pandas() if hasattr(res._edges, "to_pandas") else res._edges
    assert str(ee[alias].dtype) == "bool", ee.dtypes.to_dict()
    assert not any(str(c).endswith("_right") for c in ee.columns), list(ee.columns)
    return sorted(zip(ee["eid"].tolist(), ee[alias].tolist()))


def _sig(res):
    def topd(x):
        return x.to_pandas() if hasattr(x, "to_pandas") else x
    nn, ee = topd(res._nodes), topd(res._edges)
    return sorted(nn["key"].tolist()), sorted(map(tuple, ee[["s", "d"]].values.tolist()))


SERVED = {
    "edge alias = filtered edge column, forward": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "edge alias = filtered edge column, reverse": [n({"id": 10}, name="m"), e_reverse({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "edge alias = filtered edge column, undirected": [n({"id": 30}, name="m"), e_undirected({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "edge alias = unfiltered edge column": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="w"), n(name="p")],
    "edge alias = filtered column on the second of two steps": [n({"id": 30}, name="m"), e_forward(name="e1"), n(name="mid"), e_forward({"type": "OTHER"}, name="type"), n(name="p")],
    "edge alias = filtered column, unseeded": [n(name="m"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="p")],
    "seed alias = its filtered node column": [n({"id": 30}, name="id"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")],
    "destination alias = its filtered node column": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n({"type": "p"}, name="type")],
    "destination alias = an edge column name": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="type")],
    "seed alias = a source column name": [n({"id": 30}, name="s"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")],
    "single node alias = its filtered column": [n({"id": 30}, name="id")],
    "seed, edge and destination aliases all collide": [n({"id": 30}, name="id"), e_forward({"type": "HAS_CREATOR"}, name="type"), n({"type": "p"}, name="type")],
}

register("collision.served", [(k, (lambda v=v: list(v))) for k, v in SERVED.items()], FRAMES, tags=("alias-collision", "#2039"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("shape", list(SERVED))
def test_single_hop_collisions_match_the_pandas_full_path(engine, indexed, shape):
    ops = SERVED[shape]
    oracle = _sig(_graph("pandas", False).gfql(ops, engine="pandas", index_policy="off"))
    res = _graph(engine, indexed).gfql(ops, engine=engine, index_policy="use" if indexed else "off")
    assert _sig(res) == oracle
    edge_alias = ops[1]._name if len(ops) == 3 else None
    if edge_alias is not None:
        assert _alias_marker(res, edge_alias) == _alias_marker(_graph("pandas", False).gfql(ops, engine="pandas", index_policy="off"), edge_alias)


MULTI_HOP = {
    "edge alias = filtered column, hops=2": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, hops=2, name="type"), n(name="p")],
    "edge alias = filtered column, to_fixed_point": [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, to_fixed_point=True, name="type"), n(name="p")],
}

register("collision.multi_hop", [(k, (lambda v=v: list(v))) for k, v in MULTI_HOP.items()], FRAMES, tags=("alias-collision", "#2049"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(MULTI_HOP))
def test_multi_hop_collisions_match_the_control_alias(engine, shape):
    ops = MULTI_HOP[shape]
    g = _graph(engine, False)
    expect = _sig(g.gfql([n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, hops=ops[1].hops, to_fixed_point=ops[1].to_fixed_point, name="e"), n(name="p")], engine=engine))
    res = g.gfql(ops, engine=engine)
    assert _sig(res) == expect
    assert _alias_marker(res, ops[1]._name) == _alias_marker(g.gfql([n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, hops=ops[1].hops, to_fixed_point=ops[1].to_fixed_point, name="e"), n(name="p")], engine=engine), "e")


SURFACES = ["gfql", "chain"]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("alias", ["s", "d", "eid"])
def test_alias_named_like_a_binding_column_is_rejected_before_execution(surface, engine, alias):
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
    run = getattr(_graph(engine, False), surface)
    with pytest.raises(GFQLValidationError) as exc_info:
        run([n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name=alias), n(name="p")], engine=engine)
    assert exc_info.value.code == ErrorCode.E108


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("engine", ENGINES)
def test_alias_named_like_the_node_binding_is_rejected_before_execution(surface, engine):
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
    run = getattr(_graph(engine, False), surface)
    with pytest.raises(GFQLValidationError) as exc_info:
        run([n({"id": 30}, name="key"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")], engine=engine)
    assert exc_info.value.code == ErrorCode.E108


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("engine", ENGINES)
def test_cross_frame_binding_name_is_a_plain_alias_on_both_surfaces(surface, engine):
    """A node alias named like an EDGE binding column collides with nothing on the node frame: served, marker present."""
    run = getattr(_graph(engine, False), surface)
    res = run([n({"id": 30}, name="s"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")], engine=engine)
    nn = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    assert str(nn["s"].dtype) == "bool" and sorted(nn["key"].tolist()) == [1, 3]
