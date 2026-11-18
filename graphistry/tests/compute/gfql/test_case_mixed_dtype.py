"""Regression: CASE must never silently coerce incompatible Cypher value families.

pandas can preserve mixed branch values in an object column. Columnar engines with
homogeneous columns do not guess a Cypher type: they decline incompatible families.
"""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError

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
