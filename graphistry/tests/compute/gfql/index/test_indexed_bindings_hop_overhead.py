"""Per-hop overhead cuts in the indexed bindings path keep exact results.

Three seams: the array-side join estimate, the predicate-first node gather, and the
identity semi-join skip. Each is pinned against the frame path it shortcuts.
"""
import os

import numpy as np
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.dataframe.join import (
    _estimate_inner_join_rows_arrays,
    estimate_inner_join_rows,
)
from graphistry.compute.gfql.index import bindings as B


def _engines():
    engines = [("pandas", Engine.PANDAS)]
    try:
        import polars  # noqa: F401
        engines.append(("polars", Engine.POLARS))
    except ImportError:
        pass
    if os.environ.get("TEST_CUDF"):
        engines.append(("cudf", Engine.CUDF))
    return engines


def _to_engine(df: pd.DataFrame, engine: Engine):
    if engine == Engine.POLARS:
        import polars as pl
        return pl.from_pandas(df)
    if engine == Engine.CUDF:
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(df)
    return df.copy()


def _to_pandas(frame, engine: Engine) -> pd.DataFrame:
    if engine == Engine.POLARS:
        return frame.to_pandas()
    if engine == Engine.CUDF:
        return frame.to_pandas()
    return frame


@pytest.mark.parametrize("name,engine", _engines())
@pytest.mark.parametrize("left_keys,right_keys", [
    ([2, 1, 2, 5], [2, 1, 2, 9]),
    ([7, 7, 7], [7]),
    ([1, 2, 3], [4, 5, 6]),
    ([3, 1, 2], [1, 1, 2, 2, 3, 3]),
    (list(range(50)) * 3, list(range(25, 75)) * 2),
])
def test_array_estimate_matches_frame_estimate(name, engine, left_keys, right_keys):
    if name == "cudf":
        pytest.importorskip("cudf")
    left = _to_engine(pd.DataFrame({"current": pd.Series(left_keys, dtype="int64")}), engine)
    right = _to_engine(pd.DataFrame({"from": pd.Series(right_keys, dtype="int64")}), engine)
    expected = pd.merge(
        pd.DataFrame({"k": left_keys}), pd.DataFrame({"k": right_keys}), on="k", how="inner",
    ).shape[0]
    via_arrays = _estimate_inner_join_rows_arrays(left, right, left_on="current", right_on="from", engine=engine)
    assert via_arrays == expected
    assert estimate_inner_join_rows(left, right, left_on="current", right_on="from", engine=engine) == expected


def test_array_estimate_defers_on_nulls_and_non_integer_keys():
    pl = pytest.importorskip("polars")
    nulls = pl.DataFrame({"k": [1, None, 2]})
    ints = pl.DataFrame({"k": [1, 2, 2]})
    assert _estimate_inner_join_rows_arrays(nulls, ints, left_on="k", right_on="k", engine=Engine.POLARS) is None
    assert _estimate_inner_join_rows_arrays(ints, nulls, left_on="k", right_on="k", engine=Engine.POLARS) is None
    strings = pl.DataFrame({"k": ["a", "b"]})
    assert _estimate_inner_join_rows_arrays(strings, strings, left_on="k", right_on="k", engine=Engine.POLARS) is None
    lazy = ints.lazy()
    assert _estimate_inner_join_rows_arrays(lazy, lazy, left_on="k", right_on="k", engine=Engine.POLARS) is None
    floats = pd.DataFrame({"k": [1.0, 2.0]})
    assert _estimate_inner_join_rows_arrays(floats, floats, left_on="k", right_on="k", engine=Engine.PANDAS) is None
    nullable = pd.DataFrame({"k": pd.array([1, None], dtype="Int64")})
    assert _estimate_inner_join_rows_arrays(nullable, nullable, left_on="k", right_on="k", engine=Engine.PANDAS) is None
    # frame path still resolves the deferred shapes
    assert estimate_inner_join_rows(nulls, ints, left_on="k", right_on="k", engine=Engine.POLARS) == 3
    assert estimate_inner_join_rows(nullable, nullable, left_on="k", right_on="k", engine=Engine.PANDAS) == 1  # groupby drops NA keys


def test_polars_array_estimate_skips_the_lazy_plan(monkeypatch):
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql import lazy
    calls = []
    original = lazy.collect
    monkeypatch.setattr(lazy, "collect", lambda frame: calls.append(frame) or original(frame))
    left = pl.DataFrame({"current": [2, 1, 2]})
    right = pl.DataFrame({"from": [2, 2, 1, 3]})
    assert estimate_inner_join_rows(left, right, left_on="current", right_on="from", engine=Engine.POLARS) == 5
    assert calls == []


NODES = pd.DataFrame({
    "id": [10, 11, 12, 13, 14, 15],
    "label__Person": [True, False, True, True, False, True],
    "kind": ["a", "b", "a", "c", "a", "a"],
    "payload": list("uvwxyz"),
})


@pytest.mark.parametrize("name,engine", _engines())
@pytest.mark.parametrize("positions,filter_dict", [
    ([0, 2, 3, 5], {"label__Person": True}),           # nothing dropped
    ([0, 1, 2, 4], {"label__Person": True}),           # drops rows 1 and 4
    ([5, 3, 1, 0], {"label__Person": True, "kind": "a"}),  # two predicates, unsorted positions
    ([0, 1, 2], {"missing": True}),                    # absent column -> wide path verdict
    ([0, 1, 2], {}),                                   # empty filter
    ([], {"label__Person": True}),                     # empty gather
])
def test_take_filtered_rows_matches_gather_then_filter(name, engine, positions, filter_dict):
    if name == "cudf":
        pytest.importorskip("cudf")
    frame = _to_engine(NODES, engine)
    pos = np.asarray(positions, dtype="int64")
    if engine == Engine.CUDF:
        import cupy
        pos = cupy.asarray(pos)
    if "missing" in filter_dict:
        from graphistry.compute.exceptions import GFQLSchemaError
        with pytest.raises(GFQLSchemaError):
            B._filter_frame(B.take_rows(frame, pos, engine), filter_dict, engine)
        with pytest.raises(GFQLSchemaError):
            B._take_filtered_rows(frame, pos, filter_dict, engine)
        return
    expected = B._filter_frame(B.take_rows(frame, pos, engine), filter_dict or None, engine)
    actual = B._take_filtered_rows(frame, pos, filter_dict or None, engine)
    exp_pd, act_pd = _to_pandas(expected, engine), _to_pandas(actual, engine)
    assert list(act_pd.columns) == list(exp_pd.columns)
    assert B._ROW_POS not in act_pd.columns
    pd.testing.assert_frame_equal(act_pd.reset_index(drop=True), exp_pd.reset_index(drop=True))
    if engine == Engine.PANDAS:
        assert list(act_pd.index) == list(exp_pd.index)


@pytest.mark.parametrize("name,engine", _engines())
def test_covers_ids_detects_dropped_endpoints(name, engine):
    if name == "cudf":
        pytest.importorskip("cudf")
    xp = np
    if engine == Engine.CUDF:
        import cupy
        xp = cupy
    ids = xp.asarray([10, 12, 13])
    full = _to_engine(NODES[NODES.id.isin([10, 12, 13])], engine)
    assert B._covers_ids(full, "id", ids, engine, xp)
    dropped = _to_engine(NODES[NODES.id.isin([10, 13])], engine)
    assert not B._covers_ids(dropped, "id", ids, engine, xp)
    empty = _to_engine(NODES[NODES.id.isin([])], engine)
    assert not B._covers_ids(empty, "id", ids, engine, xp)
    assert B._covers_ids(empty, "id", xp.asarray([], dtype="int64"), engine, xp)


def _chain_graph(engine: Engine):
    import graphistry
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "label__Person": [True, False, False, False, True, True],
        "label__Message": [False, True, True, True, False, False],
        "name": list("abcdef"),
    })
    edges = pd.DataFrame({
        "s": [2, 3, 4, 4, 2, 3],
        "d": [1, 1, 1, 5, 6, 6],
        "type": ["HAS_CREATOR"] * 4 + ["LIKES", "HAS_CREATOR"],
    })
    g = graphistry.nodes(_to_engine(nodes, engine), "id").edges(_to_engine(edges, engine), "s", "d")
    return g.gfql_index_all(engine=engine)


@pytest.mark.route_engaged("indexed-kernel")
@pytest.mark.parametrize("name,engine", _engines())
@pytest.mark.parametrize("seed_id,dst_filter,expect_pids,expect_semijoin", [
    (2, {"label__Person": True}, [1], False),        # every endpoint survives: skip is the identity
    (2, {"label__Person": True, "name": "a"}, [1], False),
    (2, {}, [1], False),
    (4, {}, [1, 5], False),                          # node 4 reaches two creators, both kept
    (4, {"id": 1}, [1], True),                       # ... and dropping one makes the semi-join RUN
])
def test_indexed_hop_rows_match_canonical_with_and_without_endpoint_drops(
    name, engine, seed_id, dst_filter, expect_pids, expect_semijoin, monkeypatch,
):
    """The kernel's answer equals a real oracle, on both sides of the endpoint-drop branch.

    Three things this has to do to mean anything, none of which it did before: run with the
    sibling routes off, since `polars-seeded` otherwise answers BOTH legs and the comparison
    says nothing about the kernel; assert the kernel actually served; and include a case where
    the endpoint filter really drops an id, so the semi-join skip is exercised on both sides
    rather than only where it happens to be the identity.
    """
    if name == "cudf":
        pytest.importorskip("cudf")
    from graphistry.compute.ast import n, e_forward, rows, select, order_by
    from graphistry.tests.compute.gfql.routes.switch import routes_off
    import graphistry.compute.gfql.index.bindings as bindings_module

    g = _chain_graph(engine)
    ops = [
        n({"id": seed_id, "label__Message": True}, name="m"),
        e_forward({"type": "HAS_CREATOR"}),
        n(dst_filter or None, name="p"),
        rows(),
        select([("pid", "p.id"), ("pname", "p.name"), ("mid", "m.id")]),
        order_by([("pid", "asc")]),
    ]
    calls = []
    served_by_kernel = {"n": 0}
    original_semijoin = B.semijoin_by_column
    original_kernel = bindings_module._try_indexed_connected_bindings_state

    def counting_kernel(*args, **kwargs):
        result = original_kernel(*args, **kwargs)
        served_by_kernel["n"] += result is not None
        return result

    monkeypatch.setattr(B, "semijoin_by_column", lambda *a, **k: calls.append(1) or original_semijoin(*a, **k))
    monkeypatch.setattr(bindings_module, "_try_indexed_connected_bindings_state", counting_kernel)
    with routes_off(["polars-point-rows", "point-rows", "polars-seeded", "native-fast", "cypher-fast"]):
        served = g.gfql(ops, engine=engine, index_policy="force")
    monkeypatch.setattr(B, "semijoin_by_column", original_semijoin)
    monkeypatch.setattr(bindings_module, "_try_indexed_connected_bindings_state", original_kernel)
    assert served_by_kernel["n"], "the indexed kernel never served; this case proves nothing"

    with routes_off(["polars-point-rows", "point-rows", "polars-seeded", "native-fast",
                     "cypher-fast", "indexed-kernel", "index-hop"]):
        canonical = g.gfql(ops, engine=engine, index_policy="off")
    served_pd = _to_pandas(served._nodes, engine).reset_index(drop=True)
    canonical_pd = _to_pandas(canonical._nodes, engine).reset_index(drop=True)
    pd.testing.assert_frame_equal(served_pd, canonical_pd, check_dtype=False)
    assert served_pd["pid"].tolist() == expect_pids, "the oracle moved; the case no longer means what it says"
    assert bool(calls) == expect_semijoin


def _frame_path_expand(state, step, **kwargs):
    """The single-collect polars plan the array path replaces (kept as the oracle)."""
    import polars as pl
    from graphistry.compute.gfql.lazy import collect
    po = kwargs["path_order_col"]
    joined = (
        state.lazy().with_row_index(po)
        .join(step.lazy(), left_on=kwargs["current_col"], right_on=kwargs["from_col"], how="inner")
        .sort([po, *kwargs["tiebreak_cols"]])
        .drop(kwargs["current_col"])
        .rename({kwargs["to_col"]: kwargs["current_col"]})
    )
    if isinstance(kwargs["alias"], str):
        joined = joined.with_columns(pl.col(kwargs["current_col"]).alias(kwargs["alias"]))
    drop_after = [kwargs["from_col"], po, *kwargs["tiebreak_cols"]]
    return collect(joined.drop([c for c in drop_after if c in joined.collect_schema().names()]))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("alias", ["tail", None])
def test_polars_array_expand_join_matches_frame_plan(seed, alias, monkeypatch):
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import path_ordered_expand_join, _path_ordered_expand_join_arrays
    from graphistry.compute.gfql import lazy
    rng = np.random.default_rng(seed)
    n_state, n_step = int(rng.integers(0, 40)), int(rng.integers(0, 60))
    key_space = int(rng.integers(1, 12))
    state = pl.DataFrame({
        "seed": rng.integers(0, 100, n_state),
        "cur": rng.integers(0, key_space, n_state),
        "hop1": rng.integers(0, 5, n_state),
    })
    step = pl.DataFrame({
        "from": rng.integers(0, key_space, n_step),
        "to": rng.integers(100, 200, n_step),
        "edge_ord": rng.permutation(n_step),
        "orient": rng.integers(0, 2, n_step),
        "payload": [f"p{i}" for i in range(n_step)],
    })
    kwargs = dict(current_col="cur", from_col="from", to_col="to", path_order_col="po",
                  tiebreak_cols=("orient", "edge_ord"), alias=alias, engine=Engine.POLARS)
    expected = _frame_path_expand(state, step, **kwargs)
    via_arrays = _path_ordered_expand_join_arrays(
        state, step, **{k: v for k, v in kwargs.items() if k != "path_order_col"},
    )
    assert via_arrays is not None
    assert via_arrays.columns == expected.columns
    assert via_arrays.schema == expected.schema
    assert via_arrays.rows() == expected.rows()
    calls = []
    original = lazy.collect
    monkeypatch.setattr(lazy, "collect", lambda frame: calls.append(frame) or original(frame))
    served = path_ordered_expand_join(state, step, **kwargs)
    assert served.rows() == expected.rows() and served.columns == expected.columns
    assert calls == []


def test_polars_array_expand_join_defers_on_nulls_floats_lazy_and_no_tiebreaks():
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import _path_ordered_expand_join_arrays
    kwargs = dict(current_col="cur", from_col="from", to_col="to", tiebreak_cols=("ord",), alias=None, engine=Engine.POLARS)
    state = pl.DataFrame({"cur": [1, 2]})
    step = pl.DataFrame({"from": [1, 2], "to": [3, 4], "ord": [0, 1]})
    assert _path_ordered_expand_join_arrays(state, step, **kwargs) is not None
    assert _path_ordered_expand_join_arrays(pl.DataFrame({"cur": [1, None]}), step, **kwargs) is None
    assert _path_ordered_expand_join_arrays(state, step.with_columns(pl.col("from").cast(pl.Float64)), **kwargs) is None
    assert _path_ordered_expand_join_arrays(state, step.with_columns(pl.col("ord").cast(pl.Float64)), **kwargs) is None
    assert _path_ordered_expand_join_arrays(state.lazy(), step, **kwargs) is None
    assert _path_ordered_expand_join_arrays(state, step, **{**kwargs, "tiebreak_cols": ()}) is None
    assert _path_ordered_expand_join_arrays(state, step, **{**kwargs, "engine": Engine.PANDAS}) is None


@pytest.mark.parametrize("seed", range(12))
def test_expand_plan_rows_is_the_estimate_without_a_second_pass(seed, monkeypatch):
    """``plan.rows`` equals ``estimate_inner_join_rows`` and, on the array path, costs nothing extra.

    The searchsorted range widths the expansion already computes sum to the join's row
    count, so a cost gate reads them instead of paying a group-by pass. Both halves are
    pinned: the number, and that the estimator is never called to produce it.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe import join as join_module

    rng = np.random.default_rng(seed)
    n_state, n_step = int(rng.integers(0, 40)), int(rng.integers(0, 60))
    key_space = int(rng.integers(1, 12))
    state = pl.DataFrame({"cur": rng.integers(0, key_space, n_state)})
    step = pl.DataFrame({
        "from": rng.integers(0, key_space, n_step),
        "to": rng.integers(100, 200, n_step),
        "edge_ord": rng.permutation(n_step),
        "orient": rng.integers(0, 2, n_step),
    })
    kwargs = dict(current_col="cur", from_col="from", to_col="to", path_order_col="po",
                  tiebreak_cols=("orient", "edge_ord"), alias=None, engine=Engine.POLARS)
    expected = join_module.estimate_inner_join_rows(
        state, step, left_on="cur", right_on="from", engine=Engine.POLARS,
    )
    estimator_calls = []
    monkeypatch.setattr(
        join_module, "estimate_inner_join_rows",
        lambda *a, **k: estimator_calls.append(a) or 0,
    )
    plan = join_module.plan_path_ordered_expand_join(state, step, **kwargs)
    assert plan.rows == expected
    assert estimator_calls == []
    assert int(plan.expand().shape[0]) == expected


def test_expand_plan_falls_back_to_the_estimator_when_the_array_path_declines():
    """A shape the array path declines still gets a costed plan, via the frame estimator."""
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import (
        estimate_inner_join_rows, plan_path_ordered_expand_join,
    )

    state = pl.DataFrame({"cur": [1, 2, 2]})
    step = pl.DataFrame({"from": [1, 2, 2], "to": [7, 8, 9], "ord": [0, 1, 2]})
    kwargs = dict(current_col="cur", from_col="from", to_col="to", path_order_col="po",
                  alias=None, engine=Engine.POLARS)
    # No tiebreak columns is exactly the case the array path declines.
    plan = plan_path_ordered_expand_join(state, step, tiebreak_cols=(), **kwargs)
    expected = estimate_inner_join_rows(state, step, left_on="cur", right_on="from", engine=Engine.POLARS)
    assert plan.rows == expected == 5
    assert int(plan.expand().shape[0]) == expected


@pytest.mark.parametrize("left_dtype,right_dtype", [
    ("Int64", "UInt64"), ("UInt64", "Int64"), ("Int32", "Int64"), ("Int64", "Int32"),
])
def test_mixed_key_dtypes_decline_rather_than_promoting(left_dtype, right_dtype):
    """Keys of different dtypes defer, because the compare that joins them promotes.

    These helpers are public, so the guard belongs here rather than relying on a
    caller's own schema check. The magnitude that makes the promotion lossy is the
    next test's job; this one pins that a mismatch alone is enough to decline.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import (
        _estimate_inner_join_rows_arrays, _path_ordered_expand_join_arrays,
        estimate_inner_join_rows, path_ordered_expand_join,
    )

    ids = [1, 2]
    left = pl.DataFrame({"k": pl.Series(ids, dtype=getattr(pl, left_dtype))})
    right = pl.DataFrame({"k": pl.Series(ids, dtype=getattr(pl, right_dtype))})
    assert _estimate_inner_join_rows_arrays(
        left, right, left_on="k", right_on="k", engine=Engine.POLARS,
    ) is None
    assert estimate_inner_join_rows(left, right, left_on="k", right_on="k", engine=Engine.POLARS) == 2

    state = left.rename({"k": "cur"})
    step = right.rename({"k": "from"}).with_columns(
        pl.Series("to", [7, 8], dtype=pl.Int64), pl.Series("ord", [0, 1], dtype=pl.Int64),
    )
    kwargs = dict(current_col="cur", from_col="from", to_col="to",
                  tiebreak_cols=("ord",), alias=None, engine=Engine.POLARS)
    assert _path_ordered_expand_join_arrays(state, step, **kwargs) is None
    served = path_ordered_expand_join(state, step, path_order_col="po", **kwargs)
    assert sorted(served.rows()) == [(7,), (8,)], "the frame path must still answer exactly"


def test_an_int64_uint64_pair_would_alias_ids_past_the_float_mantissa():
    """Why the mismatch guard exists: the promoted compare cannot tell these ids apart.

    Without the guard the array expansion emitted four rows where the frame path emits
    two, because float64 collapses 2**60+1 and 2**60+2 onto the same value.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import (
        _path_ordered_expand_join_arrays, path_ordered_expand_join,
    )

    base = 2 ** 60
    assert float(base + 1) == float(base + 2), "the premise: float64 aliases these ids"
    state = pl.DataFrame({"cur": pl.Series([base + 1, base + 2], dtype=pl.Int64)})
    step = pl.DataFrame({
        "from": pl.Series([base + 1, base + 2], dtype=pl.UInt64),
        "to": pl.Series([7, 8], dtype=pl.Int64),
        "ord": pl.Series([0, 1], dtype=pl.Int64),
    })
    kwargs = dict(current_col="cur", from_col="from", to_col="to",
                  tiebreak_cols=("ord",), alias=None, engine=Engine.POLARS)
    assert _path_ordered_expand_join_arrays(state, step, **kwargs) is None
    assert sorted(path_ordered_expand_join(state, step, path_order_col="po", **kwargs).rows()) == [(7,), (8,)]


@pytest.mark.parametrize("left_rows,right_rows", [(0, 3), (3, 0), (0, 0)])
def test_the_array_estimate_handles_empty_frames_without_raising(left_rows, right_rows):
    """The helper is public, so it cannot rely on its caller's ``len == 0`` guard.

    Without its own guard an empty left frame indexes ``left_keys[-1]`` of a zero-length
    array and raises IndexError instead of answering zero.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import (
        _estimate_inner_join_rows_arrays, estimate_inner_join_rows,
    )

    left = pl.DataFrame({"k": pl.Series(list(range(left_rows)), dtype=pl.Int64)})
    right = pl.DataFrame({"k": pl.Series(list(range(right_rows)), dtype=pl.Int64)})
    assert _estimate_inner_join_rows_arrays(
        left, right, left_on="k", right_on="k", engine=Engine.POLARS,
    ) == 0
    assert estimate_inner_join_rows(left, right, left_on="k", right_on="k", engine=Engine.POLARS) == 0


def test_a_rejected_hop_does_not_pay_for_an_ordering_it_discards():
    """Costing a hop must not sort the step rows; only materializing it may.

    The cost gate exists to reject a hop before paying for it, so building the plan
    must not do the (key, tiebreak...) sort. Pinned by counting the sort itself: a plan
    that is never expanded performs no lexsort, and expanding it performs exactly one.
    """
    pl = pytest.importorskip("polars")
    from graphistry.compute.dataframe.join import plan_path_ordered_expand_join

    rng = np.random.default_rng(7)
    state = pl.DataFrame({"cur": rng.integers(0, 50, 200)})
    step = pl.DataFrame({
        "from": rng.integers(0, 50, 400), "to": rng.integers(0, 1000, 400),
        "orient": rng.integers(0, 2, 400), "edge_ord": rng.permutation(400),
    })
    kwargs = dict(current_col="cur", from_col="from", to_col="to", path_order_col="po",
                  tiebreak_cols=("orient", "edge_ord"), alias=None, engine=Engine.POLARS)
    calls = {"n": 0}
    original = np.lexsort

    def counting(*args, **kwargs_):
        calls["n"] += 1
        return original(*args, **kwargs_)

    np.lexsort = counting
    try:
        plan = plan_path_ordered_expand_join(state, step, **kwargs)
        assert plan.rows > 0
        assert calls["n"] == 0, "costing the hop sorted rows the caller may never want"
        expanded = plan.expand()
        assert calls["n"] == 1
    finally:
        np.lexsort = original
    assert int(expanded.shape[0]) == plan.rows
