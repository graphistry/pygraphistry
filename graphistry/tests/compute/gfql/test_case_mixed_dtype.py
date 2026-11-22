"""Regression: CASE must never silently coerce incompatible Cypher value families.

pandas can preserve mixed branch values in an object column. Columnar engines with
homogeneous columns do not guess a Cypher type: they decline incompatible families.
"""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover
    _HAS_CUDF = False

try:
    import polars  # noqa: F401
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    _HAS_POLARS = False

_HAS_CUDF_POLARS = _HAS_CUDF
_ENGINES = ["pandas"] + (
    ["cudf"] if _HAS_CUDF and os.environ.get("TEST_CUDF") == "1" else []
)
FOUR_ENGINES = _ENGINES + [
    *(["polars"] if _HAS_POLARS else []),
    *(
        ["polars-gpu"]
        if _HAS_CUDF_POLARS and os.environ.get("TEST_POLARS_GPU") == "1"
        else []
    ),
]

_N = pd.DataFrame({"id": [0, 1, 2], "x": pd.Series([None, None, None], dtype="object")})
_E = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})


def _graph(engine):
    nodes = _N
    edges = _E
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d").bind(edge="eid")


def test_case_branch_family_classifier_pins_null_scalar_and_object_paths():
    classify = RowPipelineMixin._gfql_case_branch_value_kind

    assert classify(1) == "number"
    assert classify(True) == "bool"
    assert classify("x") == "str"
    assert classify(None) is None
    assert classify(pd.Series([None, None], dtype="object")) is None
    assert classify(pd.Series([True, False])) == "bool"
    assert classify(pd.Series([1, 2.5])) == "number"
    assert classify(pd.Series(["x", None], dtype="object")) == "str"
    assert classify(pd.Series([True, 1], dtype="object")) == "mixed"
    # temporal and categorical dtypes are not a Cypher family: no family, so two such branches never raise
    assert classify(pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))) is None
    assert classify(pd.Series(pd.Categorical(["a", "b"]))) is None


@pytest.mark.parametrize("engine", _ENGINES)
def test_case_mixed_dtype_branches_do_not_raise(engine):
    # int THEN branch vs all-null object ELSE branch: cuDF `.where` used to raise
    # "cudf does not support mixed types"; must now return -1 for every row like pandas.
    g = graphistry.nodes(_N, "id").edges(_E, "s", "d").bind(edge="eid")
    r = g.gfql(
        "MATCH (m) RETURN m.id AS id, CASE WHEN m.x IS NULL THEN -1 ELSE m.x END AS r ORDER BY id",
        engine=engine,
    )._nodes
    if hasattr(r, "to_pandas"):
        r = r.to_pandas()
    # value-equal across engines (cuDF coerces to float -1.0; pandas keeps int -1)
    assert [rec["id"] for rec in r.to_dict("records")] == [0, 1, 2]
    assert all(float(rec["r"]) == -1.0 for rec in r.to_dict("records"))


@pytest.mark.parametrize("engine", FOUR_ENGINES)
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("CASE WHEN m.id = 1 THEN true ELSE 5 END", [5, True, 5]),
        ("CASE WHEN m.id = 1 THEN 5 ELSE 'x' END", ["x", 5, "x"]),
    ],
)
def test_incompatible_case_value_families_never_coerce(engine, query, expected):
    if engine in {"cudf", "polars", "polars-gpu"}:
        with pytest.raises(GFQLUnsupportedError) as exc:
            _graph(engine).gfql(f"MATCH (m) RETURN {query} AS r", engine=engine)
        assert exc.value.code == ErrorCode.E110
        assert sorted(exc.value.context["value"]) in [["bool", "number"], ["number", "str"]]
        return

    r = _graph(engine).gfql(f"MATCH (m) RETURN {query} AS r", engine=engine)._nodes
    assert r["r"].tolist() == expected


@pytest.mark.parametrize("engine", FOUR_ENGINES)
def test_numeric_case_families_still_conform(engine):
    r = _graph(engine).gfql(
        "MATCH (m) RETURN CASE WHEN m.id = 1 THEN 1 ELSE 2.5 END AS r", engine=engine
    )._nodes
    if hasattr(r, "to_pandas"):
        r = r.to_pandas()
    assert r["r"].tolist() == [2.5, 1.0, 2.5]


_CASE_FRAME = pd.DataFrame({
    "id": [1, 2, 3],
    "ts_ms": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]).astype("datetime64[ms]"),
    "ts_ns": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]).astype("datetime64[ns]"),
    "cat": pd.Categorical(["a", "b", "c"]),
    "s": ["x", "y", "z"],
    "i": [1, 2, 3],
    "f": [1.5, 2.5, 3.5],
    "b": [True, False, True],
})


def _case_values(engine, expression):
    nodes = _CASE_FRAME
    edges = _E
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d").bind(edge="eid")
    out = graph.gfql(f"MATCH (n) RETURN n.id AS id, {expression} AS v ORDER BY id", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out["v"].tolist()


@pytest.mark.parametrize("engine", _ENGINES)
def test_case_over_same_family_branches_with_different_dtype_strings_answers(engine):
    """A dtype-string difference inside one Cypher family (timestamp units, categorical vs string) is not a family clash."""
    stamps = _case_values(engine, "CASE WHEN n.id = 1 THEN n.ts_ms ELSE n.ts_ns END")
    assert list(pd.to_datetime(stamps)) == list(pd.to_datetime(["2020-01-01", "2021-01-02", "2021-01-03"]))
    assert _case_values(engine, "CASE WHEN n.id = 1 THEN n.cat ELSE n.s END") == ["a", "y", "z"]


@pytest.mark.parametrize("engine", _ENGINES)
def test_case_int_then_float_branches_answer_in_either_order(engine):
    assert _case_values(engine, "CASE WHEN n.id = 1 THEN n.i ELSE n.f END") == [1.0, 2.5, 3.5]
    assert _case_values(engine, "CASE WHEN n.id = 1 THEN n.f ELSE n.i END") == [1.5, 2.0, 3.0]


@pytest.mark.parametrize("engine", FOUR_ENGINES)
@pytest.mark.parametrize(("expression", "pandas_values"), [
    ("CASE WHEN n.id = 1 THEN n.i ELSE n.b END", [1, False, True]),
    ("CASE WHEN n.id = 1 THEN n.b ELSE n.i END", [True, 2, 3]),
    ("CASE WHEN n.id = 1 THEN n.s ELSE n.i END", ["x", 2, 3]),
    ("CASE WHEN n.id = 1 THEN n.i WHEN n.id = 2 THEN n.f ELSE n.b END", [1, 2.5, True]),
], ids=["int-then-bool", "bool-then-int", "string-then-int", "three-branch-with-bool"])
def test_case_across_cypher_families_keeps_pandas_values_and_declines_elsewhere(engine, expression, pandas_values):
    """pandas keeps the mixed values in an object column; the homogeneous-column engines decline with E110."""
    if engine == "pandas":
        values = _case_values(engine, expression)
        assert values == pandas_values
        assert [type(v) is bool for v in values] == [type(v) is bool for v in pandas_values]
        return
    with pytest.raises(GFQLUnsupportedError) as exc:
        _case_values(engine, expression)
    assert exc.value.code == ErrorCode.E110
