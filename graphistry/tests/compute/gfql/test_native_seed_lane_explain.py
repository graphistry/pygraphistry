"""Native op-list lanes that a resident index serves are visible in gfql_explain on every engine."""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.gfql.index import index_trace


def _frames():
    nodes = pd.DataFrame({
        "key": [1, 2, 3, 4] + list(range(100, 300)),
        "id": [10, 20, 30, 40] + list(range(1000, 1200)),
        "kind": ["Person", "Person", "Message", "Message"] + ["Other"] * 200,
    })
    edges = pd.DataFrame({"s": [3, 4, 3], "d": [1, 2, 2], "type": ["HAS_CREATOR", "HAS_CREATOR", "OTHER"]})
    return nodes, edges


def _graph(engine, indexed=True):
    nodes, edges = _frames()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine) if indexed else g


NODE_ONLY = [n({"id": 10, "kind": "Person"}, name="p")]
NAMED_HOP = [n({"id": 30, "kind": "Message"}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n({"kind": "Person"}, name="p")]
ENGINES = ["pandas", "polars", "cudf"]


@pytest.mark.route_engaged("native-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_node_only_lookup_served_by_the_property_index_is_explained(engine):
    g = _graph(engine)
    report = g.gfql_explain(NODE_ONLY, index_policy="use", engine=engine)
    assert report["used_index"] is True, report
    assert [(s["seam"], s["reason"], s["hops"]) for s in report["steps"]] == [("native_seed_lookup", "property_index", 0)]
    assert len(g.gfql(NODE_ONLY, engine=engine, index_policy="use")._nodes) == 1


@pytest.mark.route_engaged("native-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_typed_hop_served_by_the_resident_indexes_is_explained(engine):
    g = _graph(engine)
    report = g.gfql_explain(NAMED_HOP, index_policy="use", engine=engine)
    assert report["used_index"] is True, report
    assert [s["seam"] for s in report["steps"] if s["path"] == "index"] == ["native_seeded_hop"]
    with index_trace() as steps:
        out = g.gfql(NAMED_HOP, engine=engine, index_policy="use")
    assert [s["path"] for s in steps] == ["index"]
    assert len(out._edges) == 1


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ops", [NODE_ONLY, NAMED_HOP], ids=["node-only", "named-hop"])
def test_policy_off_records_no_engagement_and_keeps_the_answer(engine, ops):
    g = _graph(engine)
    report = g.gfql_explain(ops, index_policy="off", engine=engine)
    assert report["used_index"] is False
    assert report["decision_code"] == "policy_off"
    assert [s for s in report["steps"] if s["path"] == "index"] == []
    off = g.gfql(ops, engine=engine, index_policy="off")
    on = g.gfql(ops, engine=engine, index_policy="use")
    assert len(off._nodes) == len(on._nodes) and len(off._edges) == len(on._edges)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ops", [NODE_ONLY, NAMED_HOP], ids=["node-only", "named-hop"])
def test_no_resident_index_records_no_engagement(engine, ops):
    g = _graph(engine, indexed=False)
    report = g.gfql_explain(ops, index_policy="use", engine=engine)
    assert report["used_index"] is False, report
    assert [s for s in report["steps"] if s["path"] == "index"] == []
