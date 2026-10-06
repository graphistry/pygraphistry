"""Dense indexed CPU scans keep canonical bags and native frame ownership."""
import pytest
import graphistry
from graphistry.compute.gfql.index import index_trace, with_index_policy


@pytest.mark.parametrize("role,kind", [("edges", "edge_prop")])
@pytest.mark.parametrize("nullable", [False])
@pytest.mark.parametrize("value", [1, 9])
def test_dense_integer_scan_preserves_schema_order_nulls_and_native_ownership(role, kind, nullable, value, monkeypatch):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    frame = pl.DataFrame({"id": range(400), "s": range(400), "d": range(400),
                          "v": [None if nullable and i % 11 == 0 else i % 4 for i in range(400)],
                          "payload": [[i, None] for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="v", engine="polars")
    method = "filter_" + role + "_by_dict"
    expected = getattr(getattr(base, method)({"v": value}, engine="polars"), "_" + role)
    original = frame.clone()
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("Dense native scan must not export columns or run a frame expression plan")
        patch.setattr(pl.Series, "to_numpy", forbidden)
        patch.setattr(pl.DataFrame, "filter", forbidden)
        actual = getattr(getattr(indexed, method)({"v": value}, engine="polars"), "_" + role)
    assert_frame_equal(actual, expected)
    if actual.height:
        actual.replace_column(0, pl.Series("id", [999] * actual.height))
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("policy", ["off", "use", "force"])
def test_dense_scan_preserves_explain_and_force_contract(policy):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    frame = pl.DataFrame({"id": range(400), "v": [i % 4 for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index("edge_prop", column="v", engine="polars"), policy)
    with index_trace() as steps:
        actual = indexed.filter_edges_by_dict({"v": 1}, engine="polars")._edges
    assert_frame_equal(actual, base.filter_edges_by_dict({"v": 1}, engine="polars")._edges)
    decisions = [step for step in steps if step.get("op") == "property_lookup"]
    assert not decisions if policy == "off" else decisions[-1]["decision_code"] == ("index_selected" if policy == "force" else "scan_cost")


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_dense_scan_preserves_unsupported_literal_error_priority(value, monkeypatch):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    frame = pl.DataFrame({"id": range(400), "v": [i % 4 for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = base.create_index("edge_prop", column="v", engine="polars")
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_POLARS", "invalid")
    # Compare both outputs and typed error contracts; unsupported predicates
    # must not newly validate the cost configuration.
    from graphistry.compute.exceptions import GFQLSchemaError
    def outcome(g):
        try:
            return g.filter_edges_by_dict({"v": value}, engine="polars")._edges
        except (GFQLSchemaError, ValueError, TypeError, OverflowError, pl.exceptions.PolarsError) as error:
            return error
    expected, actual = outcome(base), outcome(indexed)
    if isinstance(expected, Exception):
        assert type(actual) is type(expected)
        assert getattr(actual, "code", None) == getattr(expected, "code", None)
        assert getattr(actual, "context", None) == getattr(expected, "context", None)
    else:
        assert_frame_equal(actual, expected)


@pytest.mark.parametrize("value", [[1], 2**80, -2**80])
def test_dense_admitted_literal_retains_invalid_cost_configuration_error(value, monkeypatch):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": range(400), "v": [i % 4 for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = base.create_index("edge_prop", column="v", engine="polars")
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_POLARS", "invalid")
    with pytest.raises(ValueError):
        indexed.filter_edges_by_dict({"v": value}, engine="polars")


def test_dense_rebound_index_and_gpu_target_decline_native_scan(monkeypatch):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    from graphistry.compute import filter_by_dict as filters
    from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode
    frame = pl.DataFrame({"id": range(400), "v": [i % 4 for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = base.create_index("edge_prop", column="v", engine="polars")
    changed = base._edges.with_columns(pl.lit(1).alias("v"))
    rebound = indexed.edges(changed)
    def forbidden(*args, **kwargs):
        pytest.fail("Stale indexes and active GPU target must retain canonical filtering")
    monkeypatch.setattr(filters, "_filter_native_property_scalar", forbidden)
    assert_frame_equal(rebound.filter_edges_by_dict({"v": 1}, engine="polars")._edges, changed)
    with target_mode(ExecutionTarget.GPU):
        assert not filters._supports_native_property_scalar(base._edges, "v", 1)
