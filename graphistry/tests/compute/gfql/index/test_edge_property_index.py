"""Public edge-property lookups preserve scan answers, row order, and trail identity."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ast import e_forward, is_in, n
from graphistry.compute.gfql.index import CreateIndex, DropIndex, get_registry, index_trace


@pytest.fixture(params=["pandas", "polars", "cudf"])
def engine(request):
    if request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, txn=None):
    nodes = pd.DataFrame({"id": np.arange(500)})
    edges = pd.DataFrame({
        "s": np.arange(400), "d": np.arange(400) + 1,
        "txn": np.arange(400) if txn is None else txn,
        "keep": np.arange(400) % 2,
    }, index=np.arange(400)[::-1])
    concrete = Engine(engine)
    return graphistry.nodes(df_to_engine(nodes, concrete), "id").edges(
        df_to_engine(edges, concrete), "s", "d",
    )


def frame_records(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.reset_index(drop=True).to_dict("records")
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas().reset_index(drop=True).to_dict("records")
    return frame.to_dicts()


@pytest.mark.parametrize("pattern", [
    "MATCH (a)-[e {txn: 7}]->(b)",
    "MATCH (a)<-[e {txn: 7}]-(b)",
    "MATCH (a)-[e {txn: 7}]-(b)",
    "MATCH (a)-[e]->(b) WHERE e.txn IN [7, 7, 9]",
    "MATCH (a)-[e {txn: 7, keep: 1}]->(b)",
    "MATCH (a {id: 7})-[e {txn: 7}]->(b)",
])
def test_indexed_edge_seed_matches_scan_and_engages(engine, pattern):
    txn = np.arange(400)
    txn[9] = txn[11] = 7
    base = graph(engine, txn)
    indexed = base.gfql("CREATE GFQL INDEX FOR edge_prop ON (txn)", engine=engine)
    query = pattern + " RETURN a.id AS a, b.id AS b, e.txn AS txn"
    indexed_rows = indexed.gfql(query, engine=engine)._nodes
    scan_rows = indexed.gfql(query, engine=engine, index_policy="off")._nodes
    assert frame_records(indexed_rows) == frame_records(scan_rows)
    report = indexed.gfql_explain(query, engine=engine)
    assert report["error"] is None
    assert report["used_index"]
    assert report["decision_code"] == "index_selected"
    assert any(s["op"] == "property_lookup" and s["path"] == "index" for s in report["steps"])
    assert get_registry(base).is_empty()  # functional index build


def test_known_rows_and_input_order_with_duplicate_members(engine):
    txn = np.arange(400)
    txn[9] = txn[11] = 7
    indexed = graph(engine, txn).create_index("edge_prop", column="txn", engine=engine)
    with index_trace() as steps:
        out = indexed.filter_edges_by_dict({"txn": is_in([7, 7, 9]), "keep": 1}, engine=engine)
    assert [r["s"] for r in frame_records(out._edges)] == [7, 9, 11]
    assert any(s.get("decision_code") == "index_selected" for s in steps)


@pytest.mark.parametrize("value", [-1, 999, 2**64, 2**53 + 1])
def test_absent_and_out_of_domain_integer_queries(engine, value):
    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    assert len(indexed.filter_edges_by_dict({"txn": value}, engine=engine)._edges) == 0


def test_lifecycle_wire_reuse_multiple_columns_and_named_drop(engine):
    base = graph(engine)
    op = CreateIndex("edge_prop", "txn", name="transactions")
    assert CreateIndex.from_json(op.to_json()) == op
    indexed = base.gfql(op, engine=engine)
    original = get_registry(indexed).edge_props["txn"]
    same = indexed.gfql(op, engine=engine)
    assert get_registry(same).edge_props["txn"] is original
    indexed = indexed.create_index("edge_prop", column="s", engine=engine)
    shown = indexed.show_indexes(engine=engine)
    assert set(shown["kind"]) == {"edge_prop"}
    assert set(shown["key_col"]) == {"txn", "s"}
    assert shown["valid"].all() and shown["usable"].all()
    dropped = indexed.gfql("DROP GFQL INDEX transactions", engine=engine)
    assert set(get_registry(dropped).edge_props) == {"s"}
    assert dropped.gfql(DropIndex(kind="edge_prop"), engine=engine).show_indexes(engine=engine).empty
    assert indexed.drop_index().show_indexes(engine=engine).empty


def test_stale_frame_is_never_laundered(engine):
    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    changed = graph(engine, np.arange(400)[::-1])._edges
    stale = indexed.edges(changed)
    with index_trace() as steps:
        result = stale.filter_edges_by_dict({"txn": 7}, engine=engine)
    assert [r["s"] for r in frame_records(result._edges)] == [392]
    assert not any(s.get("path") == "index" for s in steps)
    assert not stale.show_indexes(engine=engine)["valid"].any()


def test_force_bypasses_cost_and_off_bypasses_lookup(engine):
    indexed = graph(engine, np.zeros(400, dtype=np.int64)).create_index("edge_prop", column="txn", engine=engine)
    ops = [n(), e_forward({"txn": 0}), n()]
    use = indexed.gfql_explain(ops, engine=engine)
    force = indexed.gfql_explain(ops, engine=engine, index_policy="force")
    off = indexed.gfql_explain(ops, engine=engine, index_policy="off")
    assert not use["used_index"]
    assert force["used_index"]
    assert not off["used_index"]
    assert any(s.get("op") == "property_lookup" and s.get("decision_code") == "scan_cost" for s in use["steps"])


def test_multihop_edge_identity_does_not_reuse_one_relationship(engine):
    # The property index gathers edge 0 in one hop and edge 1 in the next. Both
    # would get local row position 0 if identities were assigned AFTER gathering.
    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    query = "MATCH (a)-[x {txn: 7}]->(b)-[y {txn: 8}]->(c) RETURN a.id AS a, c.id AS c"
    assert frame_records(indexed.gfql(query, engine=engine)._nodes) == [{"a": 7, "c": 9}]
    assert frame_records(indexed.gfql(query, engine=engine)._nodes) == frame_records(
        indexed.gfql(query, engine=engine, index_policy="off")._nodes,
    )


def test_integer_signed_unsigned_boundary_has_no_lossy_match(engine):
    txn = np.arange(400, dtype=np.uint64) + 2**63
    indexed = graph(engine, txn).create_index("edge_prop", column="txn", engine=engine)
    assert [r["s"] for r in frame_records(indexed.filter_edges_by_dict({"txn": 2**63 + 7}, engine=engine)._edges)] == [7]
    assert len(indexed.filter_edges_by_dict({"txn": -1}, engine=engine)._edges) == 0


def test_missing_column_and_missing_drop_are_caller_errors(engine):
    base = graph(engine)
    with pytest.raises(ValueError):
        base.create_index("edge_prop", engine=engine)
    with pytest.raises(ValueError):
        base.create_index("edge_prop", column="absent", engine=engine)
    with pytest.raises(ValueError):
        base.gfql(DropIndex(kind="edge_prop", column="txn"), engine=engine)


def test_most_selective_column_is_used_and_residual_reapplied(engine):
    indexed = graph(engine).create_index("edge_prop", column="keep", engine=engine).create_index(
        "edge_prop", column="txn", engine=engine,
    )
    with index_trace() as steps:
        result = indexed.filter_edges_by_dict({"keep": 0, "txn": 7}, engine=engine)
    assert len(result._edges) == 0
    assert [s["column"] for s in steps if s.get("op") == "property_lookup"] == ["txn"]


def test_synthetic_column_rebind_preserves_only_valid_lineage(engine):
    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    original = indexed._edges
    if engine == "polars":
        augmented = original.with_columns(synthetic=np.arange(400))
    else:
        augmented = original.assign(synthetic=np.arange(400))
    registry = get_registry(indexed).rebind_edges(augmented, original)
    assert registry.get_property_valid("edges", "txn", augmented, Engine(engine)) is not None
    rebound = graph(engine, np.arange(400)[::-1])._edges
    stale = get_registry(indexed).rebind_edges(augmented, rebound)
    assert stale.get_property_valid("edges", "txn", augmented, Engine(engine)) is None


def test_endpoint_casts_survive_polars_property_gather():
    pl = pytest.importorskip("polars")
    base = graph("polars")
    edges = base._edges.with_columns(pl.col("s").cast(pl.Float64), pl.col("d").cast(pl.Float64))
    indexed = base.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    query = "MATCH (a)-[e {txn: 7}]->(b) RETURN a.id AS a, b.id AS b"
    assert frame_records(indexed.gfql(query, engine="polars")._nodes) == [{"a": 7, "b": 8}]
    assert frame_records(indexed.gfql(query, engine="polars")._nodes) == frame_records(
        indexed.gfql(query, engine="polars", index_policy="off")._nodes,
    )


def test_multihop_cannot_repeat_a_self_loop_relationship(engine):
    base = graph(engine)
    edges = pd.DataFrame({"s": [7], "d": [7], "txn": [7], "keep": [1]})
    indexed = base.edges(df_to_engine(edges, Engine(engine))).create_index("edge_prop", column="txn", engine=engine)
    query = "MATCH (a)-[x {txn: 7}]->(b)-[y {txn: 7}]->(c) RETURN a.id AS a"
    assert len(indexed.gfql(query, engine=engine, index_policy="force")._nodes) == 0
    assert len(indexed.gfql(query, engine=engine, index_policy="off")._nodes) == 0
