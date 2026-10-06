"""Public edge-property lookups preserve scan answers, row order, and trail identity."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ast import e_forward, is_in, n
from graphistry.compute.gfql.index import CreateIndex, DropIndex, get_registry, index_trace


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
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


def test_dense_numeric_property_declines_and_respects_explicit_cost_override(engine):
    from graphistry.compute.gfql.index import reset_cost_gate_frac, set_cost_gate_frac
    from graphistry.compute.gfql.index.api import with_index_policy

    indexed = graph(engine, np.arange(400) % 4).create_index("edge_prop", column="txn", engine=engine)
    original = frame_records(indexed._edges)
    expected = [row for row in original if row["txn"] == 1]
    with index_trace() as steps:
        result = indexed.filter_edges_by_dict({"txn": 1}, engine=engine)
    assert frame_records(result._edges) == expected
    assert any(step.get("decision_code") == "scan_cost" for step in steps)
    assert not any(step.get("path") == "index" for step in steps)
    for policy in ("off", "force"):
        actual = with_index_policy(indexed, policy).filter_edges_by_dict({"txn": 1}, engine=engine)
        assert frame_records(actual._edges) == expected
    try:
        set_cost_gate_frac(Engine(engine), 0.5)
        with index_trace() as tuned:
            actual = indexed.filter_edges_by_dict({"txn": 1}, engine=engine)
        assert frame_records(actual._edges) == expected
        assert any(step.get("path") == "index" for step in tuned)
    finally:
        reset_cost_gate_frac(Engine(engine))
    assert frame_records(indexed._edges) == original


@pytest.mark.parametrize("engine", ["pandas", "cudf"], indirect=True)
def test_nullable_residual_membership_preserves_duplicate_named_row_index(engine):
    from graphistry.compute.gfql.index.api import with_index_policy

    frame = pd.DataFrame({
        "id": range(6), "v": pd.array([1, None, 2, 1, 1, 2], dtype="Int64"),
        "keep": pd.array([False, True, True, None, True, True], dtype="boolean"),
    }, index=pd.Index([4, 4, 9, 1, 1, 2], name="row_key"))
    original = graphistry.edges(df_to_engine(frame, Engine(engine)), "id", "id")
    indexed = original.create_index("edge_prop", column="id", engine=engine)
    # Concrete membership filters promise null exclusion. Native IsIn delegates
    # to each engine's Series.isin(), whose legacy null behavior is separate.
    filters = {"id": [0, 1, 3, 4], "v": [1, None], "keep": True}
    for policy in ("off", "use", "force"):
        with index_trace() as steps:
            result = with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine=engine)
        assert [row["id"] for row in frame_records(result._edges)] == [4]
        index = result._edges.index if engine == "pandas" else result._edges.index.to_pandas()
        assert index.name == "row_key"
        assert index.tolist() == [1]
        if policy == "force":
            assert any(step.get("op") == "property_lookup" and step.get("path") == "index" for step in steps)
    assert [row["id"] for row in frame_records(original._edges)] == list(range(6))


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
    if engine in ("polars", "polars-gpu"):
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


def test_empty_edge_candidates_preserve_residual_type_errors(engine):
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index.api import with_index_policy

    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    filters = {"txn": 999, "keep": "bad"}
    outcomes = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine=engine)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("residual_first", [False, True])
def test_positive_edge_candidates_preserve_residual_type_errors(engine, residual_first):
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index.api import with_index_policy

    indexed = graph(engine).create_index("edge_prop", column="txn", engine=engine)
    filters = {"keep": "bad", "txn": 7} if residual_first else {"txn": 7, "keep": "bad"}
    outcomes = []
    for policy in ("off", "use", "force"):
        with pytest.raises(GFQLSchemaError) as caught:
            with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine=engine)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("policy", ["use", "force"])
def test_empty_property_candidates_preserve_temporal_residual(engine, policy):
    from graphistry.compute.gfql.index.api import with_index_policy

    base = graph(engine)
    nodes = pd.DataFrame({"id": np.arange(500), "time": pd.date_range("2026-01-01", periods=500)})
    edges = pd.DataFrame({"s": np.arange(400), "d": np.arange(400) + 1,
                          "txn": np.arange(400), "time": pd.date_range("2026-01-01", periods=400)})
    indexed = base.nodes(df_to_engine(nodes, Engine(engine))).edges(df_to_engine(edges, Engine(engine)))
    indexed = indexed.create_index("edge_prop", column="txn", engine=engine)
    filters = {"txn": 999, "time": "2026-01-01T00:00:00"}

    reference = with_index_policy(indexed, "off").filter_edges_by_dict(filters, engine=engine)
    with index_trace() as steps:
        actual = with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine=engine)
    for name in ("_nodes", "_edges"):
        expected_frame, actual_frame = getattr(reference, name), getattr(actual, name)
        assert frame_records(actual_frame) == frame_records(expected_frame)
        assert list(actual_frame.columns) == list(expected_frame.columns)
        assert list(actual_frame.dtypes) == list(expected_frame.dtypes)
    assert len(actual._edges) == 0
    assert frame_records(indexed._nodes) == frame_records(df_to_engine(nodes, Engine(engine)))
    assert frame_records(indexed._edges) == frame_records(df_to_engine(edges, Engine(engine)))
    if engine in ("polars", "polars-gpu"):
        assert any(s.get("decision_code") == "index_path_unavailable" for s in steps)
        assert not any(s.get("op") == "property_lookup" and s.get("decision_code") == "scan_cost" for s in steps)


@pytest.mark.parametrize("filters", [{"txn": 999}, {"txn": 7, "time": "2026-01-08T00:00:00"}])
def test_polars_temporal_decline_does_not_disable_safe_property_gathers(filters):
    pl = pytest.importorskip("polars")
    base = graph("polars")
    edges = base._edges.with_columns((
        pl.datetime(2026, 1, 1) + pl.duration(days=pl.int_range(0, pl.len()))
    ).alias("time"))
    indexed = base.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    with index_trace() as steps:
        actual = indexed.filter_edges_by_dict(filters, engine="polars")
    from graphistry.compute.gfql.index.api import with_index_policy
    reference = with_index_policy(indexed, "off").filter_edges_by_dict(filters, engine="polars")
    assert actual._edges.equals(reference._edges)
    assert any(s.get("decision_code") == "index_selected" for s in steps)


def test_polars_original_empty_temporal_filter_keeps_canonical_error():
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.index.api import with_index_policy

    base = graph("polars")
    edges = base._edges.clear().with_columns(pl.lit(None).cast(pl.Datetime).alias("time"))
    indexed = base.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    for policy in ("off", "use", "force"):
        with pytest.raises(pl.exceptions.InvalidOperationError):
            with_index_policy(indexed, policy).filter_edges_by_dict(
                {"txn": 999, "time": "2026-01-01T00:00:00"}, engine="polars",
            )


@pytest.mark.parametrize("kind,text", [
    ("Datetime", "2026-01-01T00:00:00"), ("Date", "2026-01-01"),
    ("Time", "00:00:00"), ("Duration", "1 day"),
])
def test_polars_empty_candidates_preserve_all_temporal_scalar_types(kind, text):
    pl = pytest.importorskip("polars")
    from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
    from graphistry.compute.gfql.index.api import with_index_policy

    indexed = graph("polars")
    edges = indexed._edges.with_columns(pl.lit(0).cast(getattr(pl, kind)).alias("time"))
    indexed = indexed.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    for policy in ("off", "use", "force"):
        result = with_index_policy(indexed, policy).filter_edges_by_dict({"txn": 999, "time": text}, engine="polars")
        assert result._edges.height == 0
        assert result._edges.schema == edges.schema
        with pytest.raises(GFQLSchemaError) as caught:
            with_index_policy(indexed, policy).filter_edges_by_dict(
                {"txn": 999, "time": "invalid-temporal"}, engine="polars",
            )
        assert caught.value.code == ErrorCode.E302
        assert caught.value.context["field"] == "time"


def test_polars_empty_temporal_guard_preserves_absent_filter_short_circuit():
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.index.api import with_index_policy
    from graphistry.compute.gfql.strictness import strictness_scope

    base = graph("polars")
    edges = base._edges.with_columns(pl.lit(0).cast(pl.Datetime).alias("time"))
    indexed = base.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    filters = {"txn": 999, "absent": 1, "time": "invalid-temporal"}
    with strictness_scope(level="quiet"):
        for policy in ("off", "use", "force"):
            with index_trace() as steps:
                result = with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine="polars")
            assert result._edges.height == 0
            if policy != "off":
                assert any(s.get("decision_code") == "index_selected" for s in steps)


def test_polars_empty_temporal_guard_resolves_label_alias_and_warns_once():
    pl = pytest.importorskip("polars")
    from graphistry.compute.ast import isna
    from graphistry.compute.gfql.index.api import with_index_policy
    from graphistry.compute.gfql.strictness import strictness_scope

    base = graph("polars")
    edges = base._edges.with_columns(pl.lit(0).cast(pl.Date).alias("labels"))
    indexed = base.edges(edges).create_index("edge_prop", column="txn", engine="polars")
    for policy in ("off", "use", "force"):
        filters = {"txn": 999, "absent": isna(), "label__2026-01-01": True}
        with strictness_scope(level="warn"), pytest.warns(UserWarning) as warnings:
            result = with_index_policy(indexed, policy).filter_edges_by_dict(filters, engine="polars")
        assert len(warnings) == 1
        assert result._edges.height == 0
        assert result._edges.schema == edges.schema


@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
@pytest.mark.parametrize("value", [1, 999, 2**65, True, 1.0, "bad", is_in([1, 2])])
def test_dense_polars_property_decline_keeps_canonical_values_errors_without_probe(kind, role, value, monkeypatch):
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index import property_lookup, with_index_policy

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": range(400), "s": range(400), "d": range(400), "v": np.arange(400) % 4})
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="v", engine="polars")
    index = get_registry(indexed).property_indexes(role)["v"]
    assert index.min_group_count == 100
    method = "filter_" + role + "_by_dict"
    filters = {"v": value}
    try:
        expected = getattr(getattr(base, method)(filters, engine="polars"), "_" + role)
    except (GFQLSchemaError, pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as error:
        expected = error

    def unnecessary_probe(*args, **kwargs):
        pytest.fail("Every stored bucket exceeds the crossover; an untraced scan needs no probe")

    with monkeypatch.context() as patch:
        patch.setattr(property_lookup, "prop_match_count", unnecessary_probe)
        if isinstance(expected, Exception):
            with pytest.raises(type(expected)) as actual:
                getattr(with_index_policy(indexed, "use"), method)(filters, engine="polars")
            assert getattr(actual.value, "code", None) == getattr(expected, "code", None)
            assert getattr(actual.value, "context", None) == getattr(expected, "context", None)
        else:
            actual = getattr(getattr(indexed, method)(filters, engine="polars"), "_" + role)
            from polars.testing import assert_frame_equal
            assert_frame_equal(actual, expected)
    assert frame.equals(base._nodes) and base._nodes is indexed._nodes


@pytest.mark.parametrize("value,count,path", [(1, 100, "scan"), (999, 0, "index")])
def test_dense_polars_trace_still_costs_requested_key(value, count, path):
    indexed = graph("polars", np.arange(400) % 4).create_index("edge_prop", column="txn", engine="polars")
    with index_trace() as steps:
        out = indexed.filter_edges_by_dict({"txn": value}, engine="polars")
    assert len(out._edges) == count
    assert any(s.get("op") == "property_lookup" and s["path"] == path and s["est_result_rows"] == count for s in steps)


def test_dense_polars_minimum_bucket_respects_skew_force_and_explicit_override(monkeypatch):
    from graphistry.compute.gfql.index import set_cost_gate_frac, reset_cost_gate_frac, with_index_policy

    values = np.zeros(400, dtype=np.int64)
    values[7] = 99
    rare = graph("polars", values).create_index("edge_prop", column="txn", engine="polars")
    assert get_registry(rare).edge_props["txn"].min_group_count == 1
    with index_trace() as steps:
        assert len(rare.filter_edges_by_dict({"txn": 99}, engine="polars")._edges) == 1
    assert any(s.get("path") == "index" for s in steps)
    dense = graph("polars", np.arange(400) % 4).create_index("edge_prop", column="txn", engine="polars")
    try:
        set_cost_gate_frac(Engine.POLARS, 0.5)
        with index_trace() as steps:
            assert len(dense.filter_edges_by_dict({"txn": 1}, engine="polars")._edges) == 100
        assert any(s.get("path") == "index" for s in steps)
    finally:
        reset_cost_gate_frac(Engine.POLARS)
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_POLARS", "invalid")
    assert len(with_index_policy(dense, "force").filter_edges_by_dict({"txn": 1}, engine="polars")._edges) == 100
    # Unencodable predicates keep canonical filtering before cost configuration.
    assert len(dense.filter_edges_by_dict({"txn": True}, engine="polars")._edges) == 100
