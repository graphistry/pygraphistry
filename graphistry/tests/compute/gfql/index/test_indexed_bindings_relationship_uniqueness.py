"""Indexed connected bindings honor Cypher relationship uniqueness (#2177).

A MATCH pattern may not bind one relationship to two of its hops. The indexed
bindings paths must return exactly the canonical rows for repeated-type, untyped,
mixed-direction and undirected multi-hop patterns, including self-loops and
parallel edges; distinct-type hops skip the reuse filter.
"""
import os
import random
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index import index_trace


def _engines() -> List[str]:
    engines = ["pandas"]
    try:
        import polars  # noqa: F401
        engines.append("polars")
    except ImportError:
        pass
    if os.environ.get("TEST_CUDF") == "1":
        engines.append("cudf")
    return engines


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str) -> Any:
    if engine == "polars":
        import polars as pl
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(result: Any) -> List[tuple]:
    frame = result._nodes
    if not hasattr(frame, "iloc") or hasattr(frame, "to_arrow"):
        frame = frame.to_pandas()
    return sorted(map(tuple, frame.astype(object).values.tolist()), key=repr)


PATTERNS = [
    "MATCH (a {k: 0})-[r1:K]->(b)-[r2:K]->(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[r1]->(b)-[r2]->(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[r1:K]->(b)<-[r2:K]-(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[r1:K]-(b)-[r2:K]-(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[r1]-(b)-[r2]-(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[:K]-(b)<-[:C]-(c) RETURN a.id AS a, b.id AS b, c.id AS c",
    "MATCH (a {k: 0})-[r1:K]-(b)<-[r2:C]-(c)-[r3:T]->(d) RETURN a.id AS a, b.id AS b, c.id AS c, d.id AS d",
]


def _random_graph(seed: int) -> Any:
    rng = random.Random(seed)
    ids = list(range(rng.randint(2, 7)))
    nodes = pd.DataFrame({"id": ids, "k": [rng.choice([0, 1]) for _ in ids]})
    m = rng.randint(0, 14)
    edges = pd.DataFrame({
        "s": pd.Series([rng.choice(ids) for _ in range(m)], dtype="int64"),
        "d": pd.Series([rng.choice(ids) for _ in range(m)], dtype="int64"),
        "type": pd.Series([rng.choice(["K", "C", "T"]) for _ in range(m)], dtype="object"),
    })
    return nodes, edges


@pytest.mark.parametrize("engine", _engines())
@pytest.mark.parametrize("query", PATTERNS)
@pytest.mark.parametrize("check_engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("indexed-kernel")),
])
def test_indexed_bindings_match_canonical_rows(engine: str, query: str, check_engagement: bool) -> None:
    served = 0
    for seed in range(40):
        nodes, edges = _random_graph(seed)
        g = _bind(nodes, edges, engine)
        canonical = _rows(g.gfql(query, engine=engine, index_policy="off"))
        indexed_graph = g.gfql_index_all(engine=engine)
        with index_trace() as steps:
            indexed = _rows(indexed_graph.gfql(query, engine=engine, index_policy="force"))
        served += any(s.get("seam") == "connected_bindings" and s.get("served") for s in steps)
        assert indexed == canonical, f"seed={seed}\nnodes={nodes.to_dict('records')}\nedges={edges.to_dict('records')}"
    if check_engagement:
        assert served, "the indexed bindings path never served; the comparison proves nothing"


@pytest.mark.parametrize("engine", _engines())
@pytest.mark.parametrize("check_engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("indexed-kernel")),
])
def test_default_policy_does_not_reuse_a_relationship(engine: str, check_engagement: bool) -> None:
    """Under the default ``use`` policy a selective seed is served by the index and stays exact."""
    rng = np.random.default_rng(0)
    n_nodes, n_edges = 2000, 20000
    edges = pd.concat([
        pd.DataFrame({"s": [0], "d": [5], "type": ["K"]}),
        pd.DataFrame({"s": rng.integers(1, n_nodes, n_edges), "d": rng.integers(1, n_nodes, n_edges), "type": "K"}),
    ], ignore_index=True)
    nodes = pd.DataFrame({"id": np.arange(n_nodes), "k": [0] + [1] * (n_nodes - 1)})
    query = "MATCH (a {k: 0})-[r1:K]->(b)<-[r2:K]-(c) RETURN a.id AS a, b.id AS b, c.id AS c"
    g = _bind(nodes, edges, engine)
    canonical = _rows(g.gfql(query, engine=engine, index_policy="off"))
    with index_trace() as steps:
        indexed = _rows(g.gfql_index_all(engine=engine).gfql(query, engine=engine, index_policy="use"))
    if check_engagement:
        assert any(s.get("seam") == "connected_bindings" and s.get("served") for s in steps)
    assert (0, 5, 0) not in indexed
    assert indexed == canonical
