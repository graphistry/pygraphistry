"""A WHERE that excludes every row returns zero rows instead of raising (#2116 item 3b).

The row evaluators type their masks from the values they see; a frame with no rows has
none, so ``NOT (x IN [...])`` that excluded every row came back as an empty float64 mask the
truth-mask gate declined ("AST evaluator unsupported"), cuDF rejected the empty object mask
("does not support mixed types"), and list compares broadcast 0 against 1. Tri-valued results
are now typed boolean even when empty and a comparison with a zero-row operand answers a
zero-row mask, so the predicate is evaluated on the empty frame like on any other: absent
properties are still reported under ``strict``.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLSchemaError, GFQLTypeError
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin, _gfql_expr_runtime_parser_bundle

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover - depends on test env
    _HAS_CUDF = False

_ENGINES = ["pandas"] + (["cudf"] if _HAS_CUDF else [])

_NODES = pd.DataFrame({
    "id": ["a", "b", "c"],
    "type": ["person", "company", "person"],
    "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
})
_EDGES = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})


def _ids(result):
    col = result._nodes["id"]
    col = col.to_pandas() if hasattr(col, "to_pandas") else col
    return sorted(map(str, col.tolist()))


def _graph():
    return graphistry.edges(_EDGES, "src", "dst").nodes(_NODES, "id")


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query", [
    # the colleague's 3b: NOT IN that excludes every destination
    "MATCH (a)-[e]->(t) WHERE NOT (t.type IN ['person','company']) RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE t.type IN ['robot'] RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE a.type IN ['robot'] RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE t.ts IN [datetime('2030-01-01T00:00:00')] RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE NOT (t.ts IN [datetime('2024-01-02T00:00:00'), datetime('2024-01-03T00:00:00')]) RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE [t.type] = ['robot'] RETURN t.id AS id",
    "MATCH (a)-[e]->(t) WHERE [t.id] < ['a'] RETURN t.id AS id",
])
def test_where_that_excludes_every_row_returns_no_rows(engine, query):
    assert _ids(_graph().gfql(query, engine=engine)) == []


@pytest.mark.parametrize("engine", _ENGINES)
def test_where_controls_still_select(engine):
    g = _graph()
    assert _ids(g.gfql("MATCH (a)-[e]->(t) WHERE t.type IN ['company'] RETURN t.id AS id", engine=engine)) == ["b"]
    assert _ids(g.gfql("MATCH (a)-[e]->(t) WHERE NOT (t.type IN ['company']) RETURN t.id AS id", engine=engine)) == ["c"]


def test_bad_predicate_is_still_rejected_when_no_row_survives():
    with pytest.raises(GFQLTypeError):
        _graph().gfql("MATCH (a)-[e]->(t) WHERE t.type IN ['robot'] AND nosuchfn(t.id) RETURN t.id AS id", engine="pandas")


def test_absent_property_on_an_emptied_frame_is_still_reported_under_strict():
    q = "MATCH (a)-[e]->(t) WHERE t.type IN ['robot'] AND t.nosuch = 1 RETURN t.id AS id"
    with pytest.raises(GFQLSchemaError):
        _graph().gfql(q, engine="pandas", strict="strict")
    assert _ids(_graph().gfql(q, engine="pandas")) == []


def test_empty_where_rows_validates_and_keeps_zero_rows():
    class _M(RowPipelineMixin):
        pass
    empty = pd.DataFrame({"t.type": pd.Series([], dtype=object)})
    with pytest.raises(ValueError, match="unsupported row expression"):
        _M()._gfql_parse_row_expr("t.type IN [")
    assert len(RowPipelineMixin._gfql_tri_valued_series(empty, [], "m")) == 0
    assert str(RowPipelineMixin._gfql_tri_valued_series(empty, [], "m").dtype) == "bool"
    assert list(RowPipelineMixin._gfql_tri_valued_series(pd.DataFrame({"x": [1, 2]}), [True, None], "m")) == [True, None]


def test_not_in_evaluates_on_an_empty_frame():
    # before: IN on an empty frame was float64 and NOT declined it (ast_ok False)
    parser, _checker, _mod = _gfql_expr_runtime_parser_bundle()
    class _M(RowPipelineMixin):
        pass
    empty = pd.DataFrame({"t.type": pd.Series([], dtype=object)})
    ok, value = _M()._gfql_eval_expr_ast(empty, parser("NOT (t.type IN ['person', 'company'])"))
    assert ok is True and len(value) == 0
