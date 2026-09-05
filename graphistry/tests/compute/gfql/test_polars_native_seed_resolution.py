"""Native Polars scalar chains preserve full-path tables and use resident seed indexes."""
import importlib

import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, n

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal

chain_polars = importlib.import_module("graphistry.compute.gfql.lazy.engine.polars.chain")


def _graph(reverse=False, indexed=True, padding=0):
    nodes = pl.DataFrame({
        "key": [4, 1, 6, 2, 5, 3],
        "id": [104, 101, 106, 102, 105, 103],
        "kind": ["Message", "Person", "Message", "Person", "Message", "Person"],
    })
    if padding:
        nodes = pl.concat([nodes, pl.DataFrame({
            "key": range(1000, 1000 + padding), "id": range(2000, 2000 + padding),
            "kind": ["Unrelated"] * padding,
        })])
    edges = pl.DataFrame({
        "s": [6, 4, 4, 5, 6, 4], "d": [1, 2, 1, 3, 99, None],
        "type": ["T", "T", "T", "OTHER", "T", "T"], "value": [0, 1, 2, 3, 4, 5],
    })
    if reverse:
        edges = edges.rename({"s": "d", "d": "s"}).select("s", "d", "type", "value")
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d")
    return g.gfql_index_all(engine="polars").gfql_index_node_props(["id"], engine="polars") if indexed else g


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("seed", [{"id": 104}, {"kind": "Message"}, {"id": 105}, {"id": 999}])
def test_native_named_typed_hop_preserves_full_path_tables(reverse, indexed, seed, monkeypatch):
    g = _graph(reverse, indexed)
    edge = e_reverse if reverse else e_forward
    ops = [n(seed, name="m"), edge({"type": "T"}, name="e"), n({"kind": "Person"}, name="p")]
    real = chain_polars._try_seeded_chain_polars
    served = []

    def spy(*args):
        out = real(*args)
        served.append(out is not None)
        return out

    monkeypatch.setattr(chain_polars, "_try_seeded_chain_polars", spy)
    fast = g.gfql(ops, engine="polars", index_policy="use")
    assert served == [True]
    monkeypatch.setattr(chain_polars, "_try_seeded_chain_polars", lambda *args: None)
    full = g.gfql(ops, engine="polars", index_policy="use")
    assert_frame_equal(fast._nodes, full._nodes)
    assert_frame_equal(fast._edges, full._edges)


@pytest.mark.parametrize("single_node", [False, True])
def test_native_property_seed_uses_resident_index(single_node, monkeypatch):
    import graphistry.compute.gfql.index.bindings as bindings
    g = _graph(padding=100)
    real = bindings._seed_rows_via_property_index
    hits = []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        hits.append(out is not None)
        return out

    monkeypatch.setattr(bindings, "_seed_rows_via_property_index", spy)
    ops = [n({"id": 104}, name="m")]
    if not single_node:
        ops += [e_forward({"type": "T"}), n({"kind": "Person"}, name="p")]
    out = g.gfql(ops, engine="polars", index_policy="use")
    assert any(hits)
    assert out._nodes.height == (1 if single_node else 3)


def test_stale_property_index_does_not_select_old_seed(monkeypatch):
    g = _graph()
    g = g.nodes(g._nodes.with_columns((pl.col("id") + 1000).alias("id")))
    ops = [n({"id": 104}, name="m"), e_forward({"type": "T"}), n(name="p")]
    fast = g.gfql(ops, engine="polars", index_policy="use")
    monkeypatch.setattr(chain_polars, "_try_seeded_chain_polars", lambda *args: None)
    full = g.gfql(ops, engine="polars", index_policy="use")
    assert_frame_equal(fast._nodes, full._nodes)
    assert_frame_equal(fast._edges, full._edges)
    assert fast._nodes.height == 0


@pytest.mark.parametrize("case", ["lazy", "mixed", "varlen", "collision", "duplicate", "undirected"])
def test_native_seeded_hop_declines_unsupported_shapes(case):
    from graphistry.compute.ast import e_undirected
    from graphistry.compute.chain_fast_paths import _try_seeded_chain_polars
    g = _graph()
    ops = [n({"id": 104}, name="m"), e_forward({"type": "T"}), n(name="p")]
    if case == "lazy":
        g = g.nodes(g._nodes.lazy())
    elif case == "mixed":
        g = g.edges(g._edges.to_pandas())
    elif case == "varlen":
        ops[1] = e_forward({"type": "T"}, hops=2)
    elif case == "collision":
        ops[0] = n({"id": 104}, name="kind")
    elif case == "duplicate":
        ops[2] = n(name="m")
    elif case == "undirected":
        ops[1] = e_undirected({"type": "T"})
    assert _try_seeded_chain_polars(g, ops) is None
