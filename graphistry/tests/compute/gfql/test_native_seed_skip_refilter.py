"""An exact index hit answers a one-predicate seed without re-filtering the candidates.

The node-id and node-property indexes hold integer keys, verify membership after the
searchsorted probe (a float id promotes exactly as the scan's ``==`` does), and decline
mismatched value families back to the scan, so for a
filter that is exactly one scalar equality on the served column the gathered rows are
already the canonical filter's rows. Pins: that shape skips the re-filter with parity to
the full path on every engine; every other shape (two predicates, ``label__`` rewrite,
bool/str values, float on a property, no index) still runs the canonical filter, keeps its typed error,
and stays value-identical.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql.index.bindings as bindings_mod
from graphistry.compute.ast import e_forward, n
from graphistry.compute.exceptions import GFQLSchemaError

ENGINES = ["pandas", "cudf"]


def _graph(engine, n_persons=500, n_messages=1500):
    rng = np.random.default_rng(3)
    persons = pd.DataFrame({"key": np.arange(n_persons), "id": np.arange(n_persons) + 10_000,
                            "label__Person": True, "label__Message": False,
                            "name": [f"p{i}" for i in range(n_persons)]})
    messages = pd.DataFrame({"key": np.arange(n_persons, n_persons + n_messages),
                             "id": np.arange(n_messages) + 50_000,
                             "label__Person": False, "label__Message": True, "name": None})
    nodes = pd.concat([persons, messages], ignore_index=True)
    edges = pd.DataFrame({"s": np.arange(n_persons, n_persons + n_messages),
                          "d": rng.integers(0, n_persons, n_messages), "type": "HAS_CREATOR"})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)


def _canon(frame):
    df = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float64")
    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)
    return df.sort_values(cols).reset_index(drop=True) if len(df) else df


def _run(g, ops, engine, policy):
    real = bindings_mod._filter_frame
    calls = {"n": 0}

    def spy(frame, filter_dict, eng):
        calls["n"] += 1
        return real(frame, filter_dict, eng)
    bindings_mod._filter_frame = spy
    try:
        return g.gfql(ops, engine=engine, index_policy=policy), calls["n"]
    finally:
        bindings_mod._filter_frame = real


def _assert_parity(g, ops, engine):
    served, _ = _run(g, ops, engine, "use")
    full, _ = _run(g, ops, engine, "off")
    pd.testing.assert_frame_equal(_canon(served._nodes), _canon(full._nodes))
    pd.testing.assert_frame_equal(_canon(served._edges), _canon(full._edges))
    return served


SKIP_SHAPES = {
    "node-id single": lambda: [n({"key": 7})],
    "node-id single named": lambda: [n({"key": 7}, name="p")],
    "property single": lambda: [n({"id": 10_007})],
    "property single named": lambda: [n({"id": 10_007}, name="p")],
    "node-id seeded hop": lambda: [n({"key": 700}), e_forward({"type": "HAS_CREATOR"}), n()],
    "property seeded hop named": lambda: [n({"id": 50_007}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="e"), n(name="p")],
    "node-id no match": lambda: [n({"key": 99_999})],
    "integral float on node id": lambda: [n({"key": 7.0})],
    "non-integral float on node id": lambda: [n({"key": 7.5})],
    "property no match": lambda: [n({"id": 1})],
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(SKIP_SHAPES))
def test_exact_index_hit_skips_the_refilter_with_parity(engine, shape):
    g = _graph(engine)
    ops = SKIP_SHAPES[shape]()
    _assert_parity(g, ops, engine)
    _, refilters = _run(g, ops, engine, "use")
    assert refilters == 0, "an exact single-predicate index hit must not re-filter its rows"


VERIFIED_SHAPES = {  # an index hit plus residual scalar equalities: verified on the hit rows, never re-filtered
    "two predicates": lambda: [n({"id": 10_007, "label__Person": True})],
    "two predicates hop": lambda: [n({"id": 50_007, "label__Message": True}), e_forward({"type": "HAS_CREATOR"}), n()],
}

REFILTER_SHAPES = {  # no index hit (the index declines a float on an integer property): the canonical filter runs
    "label only": lambda: [n({"label__Person": True})],
    "float on property": lambda: [n({"id": 10_007.0})],
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(VERIFIED_SHAPES))
def test_an_index_hit_with_residual_scalar_predicates_is_verified_not_refiltered(engine, shape):
    g = _graph(engine)
    ops = VERIFIED_SHAPES[shape]()
    _assert_parity(g, ops, engine)
    _, refilters = _run(g, ops, engine, "use")
    assert refilters == 0, "residual scalar equalities are verified on the hit rows, not re-filtered"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", list(REFILTER_SHAPES))
def test_every_other_shape_still_runs_the_canonical_filter(engine, shape):
    g = _graph(engine)
    ops = REFILTER_SHAPES[shape]()
    _assert_parity(g, ops, engine)
    _, refilters = _run(g, ops, engine, "use")
    assert refilters >= 1, "without an index hit the canonical filter runs"


def test_bool_on_integer_column_is_never_index_served():
    g = _graph("pandas")
    ops = [n({"id": True})]
    _assert_parity(g, ops, "pandas")
    _, refilters = _run(g, ops, "pandas", "use")
    assert refilters >= 1


def test_bool_on_integer_column_raises_the_same_way_on_cudf_either_policy():
    pytest.importorskip("cudf")
    import pyarrow
    g = _graph("cudf")
    for policy in ("use", "off"):
        with pytest.raises(pyarrow.lib.ArrowTypeError):
            g.gfql([n({"id": True})], engine="cudf", index_policy=policy)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ops", [[n({"key": "7"})], [n({"id": "10007"})]], ids=["node-id", "property"])
def test_string_on_integer_column_keeps_the_typed_error_on_both_policies(engine, ops):
    g = _graph(engine)
    for policy in ("use", "off"):
        with pytest.raises(GFQLSchemaError):
            g.gfql(ops, engine=engine, index_policy=policy)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_resident_index_still_filters(engine):
    g = _graph(engine)
    plain = graphistry.nodes(g._nodes, "key").edges(g._edges, "s", "d")
    ops = [n({"key": 7})]
    served, refilters = _run(plain, ops, engine, "use")
    full, _ = _run(plain, ops, engine, "off")
    pd.testing.assert_frame_equal(_canon(served._nodes), _canon(full._nodes))
    assert refilters >= 1


# ---- residual verify on an index hit (the multi-predicate seed) ----

def _full_path(g, ops, engine):
    from graphistry.tests.compute.gfql.routes.switch import routes_off, ROUTES
    with routes_off(ROUTES):
        return g.gfql(ops, engine=engine)


def _keys(res):
    nn = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    return sorted(nn["key"].tolist())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("ops", [
    [n({"id": 10_007, "label__Person": True})],
    [n({"id": 10_007, "label__Person": True}, name="a"), e_forward(), n(name="b")],
    [n({"id": 50_003, "label__Message": True}, name="m"), e_forward({"type": "HAS_CREATOR"}), n({"label__Person": True}, name="p")],
])
def test_two_predicate_seed_on_an_index_hit_matches_the_full_path(engine, ops):
    if engine == "cudf":
        pytest.importorskip("cudf")
    g = _graph(engine)
    assert _keys(g.gfql(ops, engine=engine)) == _keys(_full_path(g, ops, engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_index_hit_whose_residual_predicate_fails_answers_empty_like_the_full_path(engine):
    if engine == "cudf":
        pytest.importorskip("cudf")
    g = _graph(engine)
    ops = [n({"id": 10_007, "label__Message": True})]  # the id is a Person
    assert _keys(g.gfql(ops, engine=engine)) == [] == _keys(_full_path(g, ops, engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_index_hit_keeps_the_typed_error_of_the_canonical_filter(engine):
    if engine == "cudf":
        pytest.importorskip("cudf")
    g = _graph(engine)
    with pytest.raises(GFQLSchemaError):
        g.gfql([n({"id": 10_007, "name": 5})], engine=engine)  # string column, numeric value
    with pytest.raises(GFQLSchemaError):
        _full_path(g, [n({"id": 10_007, "name": 5})], engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_index_hit_with_a_null_residual_column_matches_the_full_path(engine):
    if engine == "cudf":
        pytest.importorskip("cudf")
    g = _graph(engine)
    ops = [n({"id": 50_003, "name": "p1"})]  # messages carry a null name
    assert _keys(g.gfql(ops, engine=engine)) == [] == _keys(_full_path(g, ops, engine))
    ops = [n({"id": 10_001, "name": "p1"})]
    assert _keys(g.gfql(ops, engine=engine)) == [1] == _keys(_full_path(g, ops, engine))


def test_polars_seed_path_is_unchanged_by_the_residual_verify():
    pl = pytest.importorskip("polars")
    g = _graph("pandas")
    g = graphistry.nodes(pl.from_pandas(g._nodes), "key").edges(pl.from_pandas(g._edges), "s", "d")
    g = g.gfql_index_all(engine="polars").gfql_index_node_props(["id"], engine="polars")
    ops = [n({"id": 10_007, "label__Person": True}, name="a"), e_forward(), n(name="b")]
    assert _keys(g.gfql(ops, engine="polars")) == _keys(_full_path(g, ops, "polars"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("values,expected", [([None], []), ([None, 4], [1]), ([3, 4], [1])])
def test_index_hit_residual_nulls_never_count_as_matches(engine, values, expected):
    from graphistry.Engine import Engine
    from graphistry.compute.chain_fast_paths import _verify_scalar_filters_on_hit
    frame = pd.DataFrame({"id": range(len(values)), "value": pd.Series(values, dtype="Int64")})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        frame = cudf.from_pandas(frame)
    result = _verify_scalar_filters_on_hit(frame, {"value": 4}, Engine(engine))
    actual = result.to_pandas() if engine == "cudf" else result
    assert actual["id"].tolist() == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("dtype,values,target", [("Int64", [0, None, 4], 4), ("string", ["x", None, "y"], "y")])
def test_numeric_tail_nullable_residual_rejects_null_without_error(engine, direction, dtype, values, target):
    from graphistry.compute.chain_specializations.hotpaths import _seeded_hop_tail_numeric
    nodes = pd.DataFrame({"id": [0, 1, 2], "value": pd.Series(values, dtype=dtype)})
    edges = pd.DataFrame({"s": [0, 0, 0], "d": [1, 2, 2]})
    if direction == "reverse":
        edges = edges.rename(columns={"s": "d", "d": "s"})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    result = _seeded_hop_tail_numeric(nodes, edges, {"value": target}, "s", "d",
                                      "d" if direction == "forward" else "s", "id")
    assert result is not None
    output = result[0].to_pandas() if engine == "cudf" else result[0]
    assert output["id"].tolist() == [0, 2]
    assert len(result[1]) == 2


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("count", [0, 1, 2, 129])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_numeric_tail_keeps_edge_bag_and_rejects_dangling_endpoints(engine, count, direction):
    from graphistry.compute.chain_specializations.hotpaths import _seeded_hop_tail_numeric
    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({"s": [0] * count + [0, 99], "d": [2] * count + [99, 2]})
    if direction == "reverse":
        edges = edges.rename(columns={"s": "d", "d": "s"})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    result = _seeded_hop_tail_numeric(nodes, edges, {}, "s", "d",
                                      "d" if direction == "forward" else "s", "id")
    assert result is not None
    assert type(result[0]) is type(nodes) and type(result[1]) is type(edges)
    actual = result[0].to_pandas() if engine == "cudf" else result[0]
    assert actual["id"].tolist() == ([0, 2] if count else [])
    assert len(result[1]) == count


@pytest.mark.parametrize("engine", ENGINES)
def test_numeric_tail_mixed_id_dtypes_decline(engine):
    from graphistry.compute.chain_specializations.hotpaths import _seeded_hop_tail_numeric
    nodes = pd.DataFrame({"id": pd.Series([2**63], dtype="uint64")})
    edges = pd.DataFrame({"s": pd.Series([2**63 - 1], dtype="int64"),
                          "d": pd.Series([2**63 - 1], dtype="int64")})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    assert _seeded_hop_tail_numeric(nodes, edges, {}, "s", "d", "d", "id") is None
