"""Grouped whole-entity keys preserve row identity instead of collapsing to null."""

from collections import Counter

import os

import pandas as pd
import pytest

import graphistry

try:
    import polars
except ImportError:
    polars = None

ENGINES = [
    "pandas",
    pytest.param("polars", marks=pytest.mark.skipif(polars is None, reason="polars not installed")),
    pytest.param(
        "polars-gpu",
        marks=[
            pytest.mark.skipif(polars is None, reason="polars not installed"),
            pytest.mark.skipif(
                os.environ.get("TEST_POLARS_GPU") != "1",
                reason="requires actual polars-gpu",
            ),
        ],
    ),
    pytest.param(
        "cudf",
        marks=pytest.mark.skipif(
            os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
        ),
    ),
]

NODES = pd.DataFrame({"id": ["a", "b1", "b2"], "label__L": [True, False, False]})
EDGES = pd.DataFrame({"s": ["a", "a"], "d": ["b1", "b2"], "type": ["A", "A"]})


def _graph(engine):
    if engine in {"polars", "polars-gpu"}:
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(NODES), pl.from_pandas(EDGES)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(NODES), cudf.from_pandas(EDGES)
    else:
        nodes, edges = NODES.copy(), EDGES.copy()
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _run(engine, query):
    frame = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    if isinstance(frame, pd.DataFrame):
        rows = frame.to_dict(orient="records")
        columns = frame.columns.tolist()
    else:
        rows = frame.to_dicts()
        columns = frame.columns
    return columns, rows


def _identity_counts(columns, rows):
    assert not any(str(column).startswith("__cypher_reentry_") for column in columns)
    return Counter(
        (row.get("a.id", row.get("rel")), int(row["n"])) for row in rows if isinstance(row, dict)
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_node_group_key_preserves_identity(engine):
    result = _run(engine, "MATCH (a:L)-[rel]->(b) RETURN a, count(*) AS n")
    assert _identity_counts(*result) == Counter({("a", 2): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_node_group_key_preserves_identity_after_with(engine):
    result = _run(
        engine,
        "MATCH (a:L)-[rel]->(b) WITH a MATCH (a)-[rel]->(b) RETURN a, count(*) AS n",
    )
    assert _identity_counts(*result) == Counter({("a", 4): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_distinct_whole_node_group_key_preserves_identity(engine):
    result = _run(engine, "MATCH (a:L)-[rel]->(b) RETURN DISTINCT a, count(*) AS n")
    assert _identity_counts(*result) == Counter({("a", 2): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_relationship_group_keys_stay_identity_distinct(engine):
    query = "MATCH (a:L)-[rel]->(b) RETURN rel, count(*) AS n"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises(NotImplementedError, match="row op 'with_'"):
            _run(engine, query)
        return
    result = _run(engine, query)
    assert _identity_counts(*result) == Counter({("[:A]", 1): 2})
