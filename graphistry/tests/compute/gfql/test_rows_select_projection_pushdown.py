"""``rows()`` followed by ``select`` attaches only the properties the select reads.

The pushdown is invisible in results: every case is checked against the same
query with the pushdown disabled, on every engine, including declines.
"""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTCall, e_forward, e_reverse, n, order_by, rows, select, where_rows
from graphistry.compute.chain import select_attach_prop_columns


def _engines():
    out = ["pandas"]
    try:
        import polars  # noqa: F401
        out.append("polars")
    except ImportError:
        pass
    if os.environ.get("TEST_CUDF"):
        out.append("cudf")
    return out


NODES = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "label__Person": [True, False, False, False, True, True],
    "label__Message": [False, True, True, True, False, False],
    "name": list("abcdef"),
    "age": [30, 0, 0, 0, 41, 52],
    "note": ["x", "y", "z", "w", "v", "u"],
})
EDGES = pd.DataFrame({
    "s": [2, 3, 4, 4, 2, 3],
    "d": [1, 1, 1, 5, 6, 6],
    "type": ["HAS_CREATOR"] * 4 + ["LIKES", "HAS_CREATOR"],
    "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
})


def _graph(engine: str):
    nodes, edges = NODES.copy(), EDGES.copy()
    if engine == "polars":
        import polars as pl
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _pandas(frame, engine: str) -> pd.DataFrame:
    if engine in ("polars", "cudf"):
        frame = frame.to_pandas()
    return frame.reset_index(drop=True)


MIDDLE = [
    n({"label__Person": True}, name="p"),
    e_reverse({"type": "HAS_CREATOR"}, name="r"),
    n({"label__Message": True}, name="m"),
]

CASES = {
    "two_aliases": ([select([("who", "p.name"), ("mid", "m.id"), ("age", "p.age")]), order_by([("who", "asc"), ("mid", "asc")])],
                    {"p": ["name", "age"], "m": ["id"]}),
    "one_alias_only": ([select([("who", "p.name")]), order_by([("who", "asc")])], {"p": ["name"], "m": []}),
    "bare_ids_and_edge_col": ([select([("pid", "p"), ("mid", "m"), ("w", "r.weight")]), order_by([("pid", "asc"), ("mid", "asc")])],
                              {"p": [], "m": []}),
    "literal_item": ([select([("who", "p.name"), ("k", 1)]), order_by([("who", "asc")])], {"p": ["name"], "m": []}),
    "plain_string_item": ([select(["p.name", "m.note"]), order_by([("p.name", "asc"), ("m.note", "asc")])], {"p": ["name"], "m": ["note"]}),
    "expression_declines": ([select([("who", "p.name"), ("older", "p.age + 1")]), order_by([("who", "asc")])], None),
    "absent_property_declines": ([select([("who", "p.name"), ("ghost", "m.missing")]), order_by([("who", "asc")])], None),
    "where_between_declines": ([where_rows({"p.name": "a"}), select([("who", "p.name")])], None),
    "no_select_after_rows": ([order_by([("p", "asc"), ("m", "asc")])], None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_pushdown_plan(case):
    suffix, expected = CASES[case]
    calls = [rows(), *suffix]
    plan = select_attach_prop_columns(MIDDLE, calls, list(NODES.columns), "id")
    assert plan == expected


@pytest.mark.parametrize("engine", _engines())
@pytest.mark.parametrize("case", sorted(CASES))
def test_pushdown_results_match_attach_all(engine, case):
    if engine == "polars":
        pytest.importorskip("polars")
    suffix, _ = CASES[case]
    g = _graph(engine)
    import graphistry.compute.chain as chain_mod

    def run(pushdown: bool):
        with pytest.MonkeyPatch.context() as mp:
            if not pushdown:
                mp.setattr(chain_mod, "select_attach_prop_columns", lambda *a, **k: None)
            try:
                return g.gfql([*MIDDLE, rows(), *suffix], engine=engine)
            except NotImplementedError as exc:  # polars declines absent properties either way
                return exc

    served, baseline = run(True), run(False)
    if isinstance(baseline, NotImplementedError):
        assert isinstance(served, NotImplementedError) and str(served) == str(baseline)
        return
    assert not isinstance(served, NotImplementedError)
    pd.testing.assert_frame_equal(_pandas(served._nodes, engine), _pandas(baseline._nodes, engine), check_dtype=False)
    assert len(served._nodes) > 0


@pytest.mark.parametrize("engine", _engines())
def test_pushdown_narrows_the_bindings_table(engine, monkeypatch):
    if engine == "polars":
        pytest.importorskip("polars")
    g = _graph(engine)
    seen = []
    if engine == "polars":
        import graphistry.compute.gfql.lazy.engine.polars.row_pipeline as rp
        original = rp._finish_binding_rows_polars

        def spy(*args, **kwargs):
            seen.append(kwargs.get("attach_prop_columns"))
            return original(*args, **kwargs)
        monkeypatch.setattr(rp, "_finish_binding_rows_polars", spy)
    else:
        from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
        original = RowPipelineMixin._gfql_connected_bindings_row_frame_from_state

        def spy(self, ops, state_df, alias_frames, attach_prop_aliases=None, attach_prop_columns=None):
            seen.append(attach_prop_columns)
            return original(self, ops, state_df, alias_frames, attach_prop_aliases, attach_prop_columns)
        monkeypatch.setattr(RowPipelineMixin, "_gfql_connected_bindings_row_frame_from_state", spy)
    out = g.gfql([*MIDDLE, rows(), select([("who", "p.name"), ("mid", "m.id")]), order_by([("who", "asc"), ("mid", "asc")])], engine=engine)
    assert seen == [{"p": ["name"], "m": ["id"]}]
    assert list(_pandas(out._nodes, engine).columns) == ["who", "mid"]
    assert _pandas(out._nodes, engine)["who"].tolist() == ["a", "a", "a", "e", "f"]
    # the bindings table itself (no select) still carries every property
    seen.clear()
    full = g.gfql([*MIDDLE, rows()], engine=engine)
    assert seen == [None]
    assert {"p.name", "p.age", "m.note", "m.id"} <= set(map(str, _pandas(full._nodes, engine).columns))


def test_rows_serializes_attach_prop_columns():
    op = rows(attach_prop_columns={"p": ("name", "age")})
    assert isinstance(op, ASTCall)
    assert op.params["attach_prop_columns"] == {"p": ["name", "age"]}
    op.validate()
    with pytest.raises(Exception):
        ASTCall("rows", {"attach_prop_columns": {"p": [1]}}).validate()
