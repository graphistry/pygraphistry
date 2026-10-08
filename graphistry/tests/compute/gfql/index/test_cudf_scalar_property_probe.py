"""GPU scalar probes preserve canonical scans, traces, and owned native gathers."""
import numpy as np
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.gfql.index import index_trace, with_index_policy

cudf = pytest.importorskip("cudf")
cp = pytest.importorskip("cupy")
from cudf.testing import assert_frame_equal


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("value", [1, 999, -1, 2**65, True, 1.0])
@pytest.mark.parametrize("policy", ["off", "use", "force"])
def test_scalar_candidates_preserve_scan_schema_order_nulls_errors_and_ownership(role, kind, dense, value, policy):
    from graphistry.compute.filter_by_dict import _filter_property_candidates, filter_by_dict
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index.property_lookup import property_candidate_frame

    frame = cudf.DataFrame({"id": cp.arange(400), "v": cp.arange(400) % (4 if dense else 400),
                           "text": [None if i % 7 == 0 else str(i) for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), policy)
    filters = {"v": value}
    def execute():
        candidates = property_candidate_frame(indexed, role, frame, filters, Engine.CUDF)
        return _filter_property_candidates(frame, candidates, filters, Engine.CUDF,
                                           filter_validated=candidates is not frame)

    try:
        expected = filter_by_dict(frame, filters, "cudf")
    except (TypeError, ValueError, OverflowError, GFQLSchemaError) as error:
        with pytest.raises(type(error)) as observed:
            execute()
        assert str(observed.value) == str(error)
        return
    actual = execute()
    assert_frame_equal(actual, expected)
    original = frame.to_pandas()
    if len(actual):
        actual["id"].values[0] = 999
    np.testing.assert_array_equal(frame["id"].values.get(), original["id"].to_numpy())


@pytest.mark.parametrize("value,count,path", [(1, 100, "scan"), (999, 0, "index")])
@pytest.mark.parametrize("policy", ["use", "force"])
def test_dense_trace_costs_requested_key_and_force_still_gathers(value, count, path, policy):
    from graphistry.compute.gfql.index.property_lookup import property_candidate_positions

    frame = cudf.DataFrame({"id": cp.arange(400), "v": cp.arange(400) % 4})
    indexed = with_index_policy(graphistry.edges(frame, "id", "id").create_index(
        "edge_prop", column="v", engine="cudf"), policy)
    with index_trace() as steps:
        positions = property_candidate_positions(indexed, "edges", frame, {"v": value}, Engine.CUDF)
    expected_path = "index" if policy == "force" else path
    assert steps[-1]["path"] == expected_path
    assert steps[-1]["est_result_rows"] == count
    assert (positions is None) == (expected_path == "scan")
    if positions is not None:
        assert len(positions) == count


def test_scalar_cost_and_gather_share_one_probe_with_only_bounded_metadata_transfer(monkeypatch):
    from graphistry.compute.gfql.index import property_lookup

    frame = cudf.DataFrame({"id": cp.arange(400), "v": cp.arange(400)})
    indexed = graphistry.edges(frame, "id", "id").create_index("edge_prop", column="v", engine="cudf")
    hit, transfer = property_lookup._csr_hit_positions, cp.asnumpy
    calls = []

    def observe_hit(*args):
        calls.append(True)
        return hit(*args)

    def bounded_transfer(values, *args, **kwargs):
        assert values.size <= 2
        return transfer(values, *args, **kwargs)

    monkeypatch.setattr(property_lookup, "_csr_hit_positions", observe_hit)
    monkeypatch.setattr(cp, "asnumpy", bounded_transfer)
    positions = property_lookup.property_candidate_positions(indexed, "edges", frame, {"v": 1}, Engine.CUDF)
    assert len(calls) == 1
    assert int(positions[0]) == 1


@pytest.mark.parametrize("positions", [[0], [2], [-1], [3], [], [2, 0, 2], [1.0], [[1]], [True]])
def test_gpu_take_rows_matches_native_values_errors_and_owned_buffers(positions):
    from graphistry.compute.gfql.index.engine_arrays import take_rows

    frame = cudf.DataFrame({"id": [1, 2, 3], "text": ["雪", None, ""], "v": [1, None, 3]})
    indexer = cp.asarray(positions, dtype="int64" if not positions else None)
    try:
        expected = frame.iloc[indexer]
    except Exception as error:
        with pytest.raises(type(error)):
            take_rows(frame, indexer, Engine.CUDF)
    else:
        actual = take_rows(frame, indexer, Engine.CUDF)
        assert_frame_equal(actual, expected)
        if len(actual):
            actual["id"].values[0] = 99
            assert int(frame["id"].values[0]) == 1


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("value", ["key1", "absent", "雪", 1, True, "\ud800"])
@pytest.mark.parametrize("policy", ["off", "use", "force"])
def test_string_scalar_candidates_match_canonical_values_errors_and_owned_buffers(role, kind, dense, value, policy):
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.filter_by_dict import _filter_property_candidates, filter_by_dict
    from graphistry.compute.gfql.index.property_lookup import property_candidate_frame

    frame = cudf.DataFrame({"id": cp.arange(400), "v": [f"key{i % (4 if dense else 400)}" for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), policy)
    filters = {"v": value}

    def execute():
        candidates = property_candidate_frame(indexed, role, frame, filters, Engine.CUDF)
        return _filter_property_candidates(frame, candidates, filters, Engine.CUDF,
                                           filter_validated=candidates is not frame)

    try:
        expected = filter_by_dict(frame, filters, "cudf")
    except (TypeError, ValueError, OverflowError, GFQLSchemaError) as error:
        with pytest.raises(type(error)) as observed:
            execute()
        assert str(observed.value) == str(error)
        return
    actual = execute()
    assert_frame_equal(actual, expected)
    if len(actual):
        actual["id"].values[0] = 999
    np.testing.assert_array_equal(frame["id"].values.get(), np.arange(400))


def test_gpu_string_scalar_probe_never_exports_more_than_one_dictionary_key(monkeypatch):
    from graphistry.compute.gfql.index.property_lookup import property_candidate_positions

    frame = cudf.DataFrame({"id": cp.arange(400), "v": [f"key{i}" for i in range(400)]})
    indexed = graphistry.edges(frame, "id", "id").create_index("edge_prop", column="v", engine="cudf")
    original = cudf.Series.to_arrow
    exports = []

    def bounded_export(series, *args, **kwargs):
        assert len(series) <= 1
        exports.append(len(series))
        return original(series, *args, **kwargs)

    monkeypatch.setattr(cudf.Series, "to_arrow", bounded_export)
    positions = property_candidate_positions(indexed, "edges", frame, {"v": "key1"}, Engine.CUDF)
    assert int(positions[0]) == 1
    assert sum(exports) <= 1
