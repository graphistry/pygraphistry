import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError


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

_ENGINES = ["pandas"]
if _HAS_CUDF and os.environ.get("TEST_CUDF") == "1":
    _ENGINES.append("cudf")
if _HAS_POLARS:
    _ENGINES.append("polars")
if _HAS_CUDF and os.environ.get("TEST_POLARS_GPU") == "1":
    _ENGINES.append("polars-gpu")


@pytest.fixture(params=_ENGINES)
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    return request.param


def _graph(engine):
    nodes = pd.DataFrame({"id": [0], "base": [2.0], "exp": [3.0], "nullable": [None], "flag": [True]})
    edges = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})
    if engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d").bind(edge="eid")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RETURN 4 ^ 3 * 2 ^ 3 AS a, (4 ^ 3) * (2 ^ 3) AS b, 4 ^ (3 * 2) ^ 3 AS c", [512.0, 512.0, 68719476736.0]),
        ("RETURN 4 ^ 3 / 2 ^ 3 AS a, (4 ^ 3) / (2 ^ 3) AS b, 4 ^ (3 / 2) ^ 3 AS c", [8.0, 8.0, 64.0]),
        ("RETURN 4 ^ 3 % 2 ^ 3 AS a, (4 ^ 3) % (2 ^ 3) AS b, 4 ^ (3 % 2) ^ 3 AS c", [0.0, 0.0, 64.0]),
        ("RETURN 4 ^ 3 + 2 ^ 3 AS a, (4 ^ 3) + (2 ^ 3) AS b, 4 ^ (3 + 2) ^ 3 AS c", [72.0, 72.0, 1073741824.0]),
        ("RETURN 4 ^ 3 - 2 ^ 3 AS a, (4 ^ 3) - (2 ^ 3) AS b, 4 ^ (3 - 2) ^ 3 AS c", [56.0, 56.0, 64.0]),
        ("RETURN -3 ^ 2 AS a, (-3) ^ 2 AS b, -(3 ^ 2) AS c", [9.0, 9.0, -9.0]),
    ],
)
def test_power_precedence_matches_tck(engine, query, expected):
    row = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(row, "to_pandas"):
        row = row.to_pandas()
    values = row.iloc[0].tolist()
    assert values == expected


def test_power_series_scalar_null_and_bool_paths(engine):
    graph = _graph(engine)
    if engine == "polars":
        with pytest.raises((NotImplementedError, GFQLValidationError)):
            graph.gfql("MATCH (n) RETURN n.flag ^ n.exp AS v", engine=engine)
        return

    result = graph.gfql("MATCH (n) RETURN n.base ^ n.exp AS a, 2 ^ n.exp AS b, n.base ^ 2 AS c", engine=engine)._nodes
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result.iloc[0].tolist() == [8.0, 8.0, 4.0]

    with pytest.raises((GFQLValidationError, NotImplementedError)):
        graph.gfql("MATCH (n) RETURN n.flag ^ n.exp", engine=engine)


@pytest.mark.parametrize("engine", ["pandas", *(["polars"] if _HAS_POLARS else [])])
def test_power_scalar_float_edges_match_series(engine):
    result = _graph(engine).gfql(
        "RETURN 0.0 ^ -1.0 AS zero, (-2.0) ^ 0.5 AS negative",
        engine=engine,
    )._nodes
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result.iloc[0]["zero"] == float("inf")
    assert pd.isna(result.iloc[0]["negative"])


def _column_graph(engine):
    nodes = pd.DataFrame({
        "id": [1, 2, 3],
        "x": [2.0, None, 3.0],
        "name": ["a", "b", "c"],
        "s1": ["2", "3", "4"],
        "s2": ["2", "2", "2"],
    })
    edges = pd.DataFrame({"s": [1], "d": [2]})
    if engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _column_values(engine, expr):
    out = _column_graph(engine).gfql(f"MATCH (n) RETURN n.id AS id, {expr} AS v ORDER BY id", engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return [None if pd.isna(value) else value for value in out["v"].tolist()]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("n.x ^ 0", [1.0, None, 1.0]),
        ("1 ^ n.x", [1.0, None, 1.0]),
        ("n.x ^ 2", [4.0, None, 9.0]),
        ("2 ^ n.x", [4.0, None, 8.0]),
        ("null ^ 2", [None, None, None]),
        ("2 ^ null", [None, None, None]),
    ],
)
def test_power_of_a_null_row_is_null(engine, expr, expected):
    """IEEE power gives 1.0 for 1 ** NaN and NaN ** 0; Cypher gives null whenever either operand is null."""
    assert _column_values(engine, expr) == expected


@pytest.mark.parametrize("expr", ["n.name ^ 2", "2 ^ n.name", "n.name ^ n.name", "n.s1 ^ n.s2"])
def test_power_of_string_operands_raises_on_every_engine(engine, expr):
    """Strings (even numeric-looking ones) are not numbers: no engine may answer."""
    with pytest.raises((GFQLValidationError, NotImplementedError)):
        _column_values(engine, expr)
