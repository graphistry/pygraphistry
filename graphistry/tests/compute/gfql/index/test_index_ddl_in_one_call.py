"""Index DDL fused with the query that uses it, in ONE ``gfql()`` call (#2119).

Every admitted form must do two things: build the index, and have the traversal that follows be
served by it — the same seams the two-step ``g.gfql("CREATE ...").gfql(query)`` form reports.
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd
import pytest

import graphistry
from graphistry import call, e_forward, is_in, let, n, ref
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.compute.gfql.index.wire import CreateIndex, DropIndex, ShowIndexes

ENGINES = ["pandas", "polars", "cudf"]


def _require(engine: str) -> None:
    if engine == "polars":
        pytest.importorskip("polars")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        cupy = pytest.importorskip("cupy")
        try:
            cudf.Series([1]).sum()
            cupy.arange(3).sum().item()  # the index kernels JIT through cupy
        except Exception:
            pytest.skip("cudf not runnable here")


def _native(frame: pd.DataFrame, engine: str) -> Any:
    if engine == "cudf":
        import cudf
        return cudf.from_pandas(frame)
    if engine == "polars":
        import polars as pl
        return pl.from_pandas(frame)
    return frame


def _graph(engine: str = "pandas", with_edge_id: bool = False) -> Any:
    edges = pd.DataFrame({"src": [0, 0, 1, 1, 2, 3, 4, 5], "dst": [1, 2, 2, 3, 4, 4, 5, 0]})
    if with_edge_id:
        edges = edges.assign(eid=range(len(edges)))
    nodes = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5]})
    g = graphistry.edges(_native(edges, engine), "src", "dst", "eid" if with_edge_id else None)
    return g.nodes(_native(nodes, engine), "id")


def _dst(result: Any) -> List[int]:
    frame = result._edges
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    return sorted(int(v) for v in frame["dst"].tolist())


def _served(report: Any) -> List[str]:
    return sorted({step["seam"] for step in report["steps"] if step.get("served")})


TWO_INDEXES = [CreateIndex("edge_out_adj"), CreateIndex("node_id")]
SEEDED_HOP = [n({"id": 0}), e_forward(), n()]


@pytest.mark.route_engaged("native-fast", "index-hop", "indexed-kernel", "cypher-fast")
@pytest.mark.parametrize("engine", ["pandas", "cudf"])  # the native seeded lane is pandas/cuDF; polars has its own
@pytest.mark.parametrize("with_edge_id", [False, True], ids=["chain adds its edge index", "edge id bound"])
def test_wire_ops_at_the_front_of_a_chain_build_and_serve(engine: str, with_edge_id: bool) -> None:
    _require(engine)
    g = _graph(engine, with_edge_id)
    two_step = g.gfql(TWO_INDEXES[0], engine=engine).gfql(TWO_INDEXES[1], engine=engine)
    expected = _served(two_step.gfql_explain(SEEDED_HOP, engine=engine))
    assert expected and two_step.gfql_explain(SEEDED_HOP, engine=engine)["used_index"] is True
    fused = TWO_INDEXES + SEEDED_HOP
    report = g.gfql_explain(fused, engine=engine)
    assert report["used_index"] is True and _served(report) == expected
    assert _dst(g.gfql(fused, engine=engine)) == _dst(two_step.gfql(SEEDED_HOP, engine=engine)) == [1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_call_form_is_the_same_op(engine: str) -> None:
    _require(engine)
    g = _graph(engine)
    ddl = [call("create_index", {"kind": "edge_out_adj"}), call("create_index", {"kind": "node_id"})]
    resident = g.gfql(ddl, engine=engine).show_indexes(engine=engine)
    assert sorted(resident["name"].tolist()) == ["edge_out_adj:src", "node_id:id"]
    if engine == "polars":
        # the polars chain engine declines ANY call() ahead of a traversal (parity-or-error); the
        # string form below has no call in the chain and is the fused form that works there
        with pytest.raises(NotImplementedError, match=r"call\(\) before a traversal"):
            g.gfql(ddl + SEEDED_HOP, engine=engine)
    else:
        assert _dst(g.gfql(ddl + SEEDED_HOP, engine=engine)) == [1, 2]
    fused = "CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; MATCH (m {id: 0})-[e]->(p) RETURN p"
    frame = g.gfql(fused, engine=engine)._nodes
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    assert sorted(int(v) for v in frame["p.id"].tolist()) == [1, 2]


def test_wire_ops_normalize_to_calls_and_round_trip_through_json() -> None:
    g = _graph()
    chain = Chain(TWO_INDEXES + SEEDED_HOP)
    assert [op.function for op in chain.chain[:2]] == ["create_index", "create_index"]  # type: ignore[attr-defined]
    payload = chain.to_json()
    assert [op["type"] for op in payload["chain"][:2]] == ["Call", "Call"]
    assert _dst(g.gfql(Chain.from_json(payload))) == [1, 2]
    wire = {"type": "Chain", "chain": [CreateIndex("edge_out_adj").to_json(), *[op.to_json() for op in SEEDED_HOP]]}
    assert _dst(g.gfql(Chain.from_json(wire))) == [1, 2]  # a raw CreateIndex document is accepted on the wire too


@pytest.mark.route_engaged("native-fast", "index-hop", "indexed-kernel")
def test_a_let_binding_may_be_the_indexed_graph() -> None:
    g = _graph()
    dag = let({"b": [CreateIndex("edge_out_adj"), CreateIndex("node_id")], "q": ref("b", [n({"id": is_in([0])}), e_forward(), n()])})
    assert _dst(g.gfql(dag)) == [1, 2]
    single = let({"b": CreateIndex("edge_out_adj"), "q": ref("b", SEEDED_HOP)})
    assert _dst(g.gfql(single)) == [1, 2]
    assert g.gfql_explain(let({"b": TWO_INDEXES, "q": ref("b", SEEDED_HOP)}))["used_index"] is True


@pytest.mark.route_engaged("native-fast", "index-hop", "indexed-kernel", "cypher-fast")
def test_leading_ddl_statements_in_one_string_build_then_run_the_rest() -> None:
    g = _graph()
    query = "CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; MATCH (m {id: 0})-[e]->(p) RETURN p"
    two_step = g.gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id")
    assert _served(g.gfql_explain(query)) == _served(two_step.gfql_explain("MATCH (m {id: 0})-[e]->(p) RETURN p"))
    assert g.gfql_explain(query)["used_index"] is True
    assert sorted(g.gfql(query)._nodes["p.id"].tolist()) == [1, 2]
    multi = "CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; MATCH (m)-[e]->(p) WHERE m.id IN [0, 1] RETURN p"
    assert g.gfql_explain(multi)["used_index"] is True
    assert sorted(g.gfql(multi)._nodes["p.id"].tolist()) == [1, 2, 2, 3]


def test_ddl_only_strings_keep_working_with_and_without_semicolons() -> None:
    g = _graph()
    assert sorted(g.gfql("CREATE GFQL INDEX FOR edge_out_adj;").show_indexes()["name"]) == ["edge_out_adj:src"]
    both = g.gfql("CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id")
    assert sorted(both.show_indexes()["name"]) == ["edge_out_adj:src", "node_id:id"]
    assert len(g.gfql("CREATE GFQL INDEX FOR edge_out_adj; DROP GFQL INDEX FOR edge_out_adj").show_indexes()) == 0


def test_create_then_drop_inside_one_chain_leaves_nothing_resident() -> None:
    g = _graph()
    assert _dst(g.gfql([CreateIndex("edge_out_adj"), DropIndex(kind="edge_out_adj")] + SEEDED_HOP)) == [1, 2]
    assert len(g.gfql([CreateIndex("edge_out_adj"), DropIndex(kind="edge_out_adj")]).show_indexes()) == 0
    assert len(g.gfql([call("create_index", {"kind": "edge_out_adj"}), call("drop_index")]).show_indexes()) == 0


@pytest.mark.parametrize(
    "bad,error,needle",
    [
        ([ShowIndexes(), n()], GFQLTypeError, "show_indexes"),
        ([DropIndex(name="edge_out_adj:src"), n()], GFQLTypeError, "by name"),
        ([call("create_index", {"kind": "nope"}), n()], GFQLTypeError, "kind"),
        ([call("create_index"), n()], GFQLTypeError, "kind"),
        ("SHOW GFQL INDEXES; MATCH (m) RETURN m", ValueError, "SHOW GFQL INDEXES"),
        ("CREATE GFQL INDEX FOR bogus; MATCH (m) RETURN m", ValueError, "Malformed GFQL INDEX DDL"),
    ],
    ids=["ShowIndexes in a chain", "DropIndex by name in a chain", "unknown kind", "missing kind", "SHOW in a multi-statement", "malformed leading DDL"],
)
def test_what_cannot_be_fused_says_so(bad: Any, error: type, needle: str) -> None:
    g = _graph()
    with pytest.raises(error, match=needle):
        g.gfql(bad)


def test_a_semicolon_inside_the_query_is_not_a_statement_boundary() -> None:
    g = graphistry.edges(pd.DataFrame({"src": ["a;b", "c"], "dst": ["c", "a;b"]}), "src", "dst")
    g = g.nodes(pd.DataFrame({"id": ["a;b", "c"]}), "id")
    out = g.gfql("CREATE GFQL INDEX FOR edge_out_adj; MATCH (m {id: 'a;b'})-[e]->(p) RETURN p")
    assert out._nodes["p.id"].tolist() == ["c"]
