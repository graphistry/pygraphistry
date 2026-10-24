from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable


@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "polars",
        pytest.param(
            "cudf",
            marks=pytest.mark.skipif(
                os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
            ),
        ),
    ],
)
def test_scalar_reentry_defers_aggregation_until_after_union(engine: str) -> None:
    if engine != "pandas":
        pytest.importorskip(engine, reason=f"{engine} unavailable")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
    nodes = pd.DataFrame(
        {"nid": ["s", "a1", "a2", "a3"], "type": ["x", "u", "u", "u"]}
    )
    edges = pd.DataFrame(
        [("a1", "s"), ("a2", "s"), ("a3", "s")], columns=["s", "d"]
    )
    if engine == "cudf":
        graph = graphistry.nodes(cudf.from_pandas(nodes), "nid").edges(
            cudf.from_pandas(edges), "s", "d"
        )
    else:
        graph = graphistry.nodes(nodes, "nid").edges(edges, "s", "d")

    cases: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "MATCH (a)-[]->(s) WITH s MATCH (b)-[]->(s) RETURN s.type AS t, count(*) AS c",
            [{"t": "x", "c": 9}],
        ),
        (
            "MATCH (a)-[]->(s) WITH s MATCH (b)-[]->(s) RETURN count(*) AS c",
            [{"c": 9}],
        ),
        (
            "MATCH (a)-[]->(s) WITH DISTINCT s MATCH (b)-[]->(s) RETURN s.type AS t, count(*) AS c",
            [{"t": "x", "c": 3}],
        ),
    ]
    for query, expected in cases:
        result = graph.gfql(query, engine=engine)
        records = (
            result._nodes.to_dicts()
            if engine == "polars"
            else result._nodes.to_dict(orient="records")
        )
        assert sorted(records, key=lambda row: tuple(sorted(row.items()))) == expected


@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "polars",
        pytest.param(
            "cudf",
            marks=pytest.mark.skipif(
                os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
            ),
        ),
    ],
)
def test_scalar_reentry_preserves_unequal_seed_multiplicity(engine: str) -> None:
    if engine != "pandas":
        pytest.importorskip(engine, reason=f"{engine} unavailable")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
    nodes = pd.DataFrame(
        {"nid": ["s1", "s2", "a1", "a2", "a3"], "type": ["x", "y", "u", "u", "u"]}
    )
    edges = pd.DataFrame(
        [("a1", "s1"), ("a2", "s1"), ("a3", "s2")], columns=["s", "d"]
    )
    if engine == "cudf":
        graph = graphistry.nodes(cudf.from_pandas(nodes), "nid").edges(
            cudf.from_pandas(edges), "s", "d"
        )
    else:
        graph = graphistry.nodes(nodes, "nid").edges(edges, "s", "d")
    cases = [
        (
            "MATCH (a)-[]->(s) WITH s MATCH (b)-[]->(s) "
            "RETURN s.type AS t, count(*) AS c",
            [{"t": "x", "c": 4}, {"t": "y", "c": 1}],
        ),
        (
            "MATCH (a)-[]->(s) WITH DISTINCT s MATCH (b)-[]->(s) "
            "RETURN s.type AS t, count(*) AS c",
            [{"t": "x", "c": 2}, {"t": "y", "c": 1}],
        ),
    ]
    for query, expected in cases:
        result = graph.gfql(query, engine=engine)
        records = (
            result._nodes.to_dicts()
            if engine == "polars"
            else result._nodes.to_dict(orient="records")
        )
        assert sorted(records, key=lambda row: row["t"]) == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_scalar_reentry_empty_prefix_aggregation(engine: str) -> None:
    pytest.importorskip(engine, reason=f"{engine} unavailable")
    nodes = pd.DataFrame({"nid": ["s", "z"], "type": ["x", "z"]})
    edges = pd.DataFrame([("a", "s")], columns=["s", "d"])
    graph: Plottable = graphistry.nodes(nodes, "nid").edges(edges, "s", "d")
    grouped = graph.gfql(
        "MATCH (a)-[]->(s) WHERE s.type = 'missing' WITH s "
        "MATCH (b)-[]->(s) RETURN s.type AS t, count(*) AS c",
        engine=engine,
    )
    ungrouped = graph.gfql(
        "MATCH (a)-[]->(s) WHERE s.type = 'missing' WITH s "
        "MATCH (b)-[]->(s) RETURN count(*) AS c",
        engine=engine,
    )
    grouped_records = (
        grouped._nodes.to_dicts()
        if engine == "polars"
        else grouped._nodes.to_dict(orient="records")
    )
    ungrouped_records = (
        ungrouped._nodes.to_dicts()
        if engine == "polars"
        else ungrouped._nodes.to_dict(orient="records")
    )
    assert grouped_records == []
    assert ungrouped_records == [{"c": 0}]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_scalar_reentry_distinct_node_ids_share_group_property(engine: str) -> None:
    pytest.importorskip(engine, reason=f"{engine} unavailable")
    nodes = pd.DataFrame(
        {"nid": ["s1", "s2", "a1", "a2", "a3"], "type": ["x", "x", "u", "u", "u"]}
    )
    edges = pd.DataFrame(
        [("a1", "s1"), ("a2", "s1"), ("a3", "s2")], columns=["s", "d"]
    )
    graph: Plottable = graphistry.nodes(nodes, "nid").edges(edges, "s", "d")
    result = graph.gfql(
        "MATCH (a)-[]->(s) WITH s MATCH (b)-[]->(s) "
        "RETURN s.type AS t, count(*) AS c",
        engine=engine,
    )
    records = (
        result._nodes.to_dicts()
        if engine == "polars"
        else result._nodes.to_dict(orient="records")
    )
    assert records == [{"t": "x", "c": 5}]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_scalar_reentry_deferred_aggregation_respects_order_limit(engine: str) -> None:
    pytest.importorskip(engine, reason=f"{engine} unavailable")
    nodes = pd.DataFrame({"nid": ["s", "a1", "a2", "a3"], "type": ["x", "u", "u", "u"]})
    edges = pd.DataFrame([("a1", "s"), ("a2", "s"), ("a3", "s")], columns=["s", "d"])
    graph: Plottable = graphistry.nodes(nodes, "nid").edges(edges, "s", "d")
    result = graph.gfql(
        "MATCH (a)-[]->(s) WITH s MATCH (b)-[]->(s) "
        "RETURN s.type AS t, count(*) AS c ORDER BY c DESC LIMIT 1",
        engine=engine,
    )
    records = (
        result._nodes.to_dicts()
        if engine == "polars"
        else result._nodes.to_dict(orient="records")
    )
    assert records == [{"t": "x", "c": 9}]
