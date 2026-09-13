"""Indexed join helpers retain path order and collect on the active target."""
import pytest

from graphistry.Engine import Engine
from graphistry.compute.dataframe.join import estimate_inner_join_rows, path_ordered_expand_join

pl = pytest.importorskip("polars")


@pytest.mark.parametrize("empty", [False, True])
def test_polars_join_helpers_collect_once_and_keep_path_bag(monkeypatch, empty):
    from graphistry.compute.gfql import lazy
    state = pl.DataFrame({"current": [2, 1, 2, None], "seed": [20, 10, 21, 30]})
    step = pl.DataFrame({"from": [2, 1, 2, None], "to": [8, 9, 7, 6], "edge_order": [2, 0, 1, 3]})
    if empty:
        step = step.clear()
    calls = []
    original = lazy.collect

    def collect(frame):
        calls.append(frame)
        return original(frame)

    monkeypatch.setattr(lazy, "collect", collect)
    estimate = estimate_inner_join_rows(state, step, left_on="current", right_on="from", engine=Engine.POLARS)
    assert estimate == (0 if empty else 5)
    assert len(calls) == (0 if empty else 1)
    calls.clear()
    result = path_ordered_expand_join(
        state, step, current_col="current", from_col="from", to_col="to",
        path_order_col="path_order", tiebreak_cols=["edge_order"], alias="tail", engine=Engine.POLARS,
    )
    assert len(calls) == 1
    assert result.columns == ["seed", "current", "tail"]
    assert result.schema == {"seed": pl.Int64, "current": pl.Int64, "tail": pl.Int64}
    assert result.rows() == ([] if empty else [(20, 7, 7), (20, 8, 8), (10, 9, 9), (21, 7, 7), (21, 8, 8)])
