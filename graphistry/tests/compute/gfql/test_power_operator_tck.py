import os

import pandas as pd
import pytest

import graphistry


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
    nodes = pd.DataFrame({"id": [0]})
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
