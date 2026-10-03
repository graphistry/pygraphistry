"""The indexed bindings kernel accepts a membership seed on the node-id column — ``n({"id":
is_in([...])})`` or the Cypher ``WHERE a.id IN [...]`` that lowers to it — the way it already
accepts one integral id, so a multi-seed hop is served from the resident indexes instead of
running the unseeded pattern first.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index.bindings import _membership_seed_ids
from graphistry.compute.predicates.is_in import is_in


ENGINES = ["pandas", "polars", "cudf"]


def _require(engine: str) -> None:
    """Per-test gate, like test_indexed_bindings.py: polars is installed only in the polars lane, and a
    CPU-only box can import cuDF yet not run the index kernels (they JIT through cupy)."""
    if engine == "polars":
        pytest.importorskip("polars")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        cupy = pytest.importorskip("cupy")
        try:
            cudf.Series([1]).sum()
            cupy.arange(3).sum().item()
        except Exception:
            pytest.skip("cudf not runnable here")


KERNEL_DISPATCHED = {"pandas", "cudf"}  # _plan_indexed_middle hands the middle to the kernel on these engines


def _ids(result: Any, column: str) -> List[Any]:
    frame = result._nodes
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    return sorted(int(v) for v in frame[column].tolist())


def _native(frame: pd.DataFrame, engine: str) -> Any:
    """The engine's own frame type: an index is built for the frame it sees, so a cuDF arm must
    hold cuDF frames before CREATE GFQL INDEX, or the registry rightly treats the index as foreign."""
    if engine == "cudf":
        import cudf
        return cudf.from_pandas(frame)
    if engine == "polars":
        import polars as pl
        return pl.from_pandas(frame)
    return frame


def _graph(ids: List[Any], engine: str = "pandas") -> tuple:
    rng = np.random.default_rng(5)
    n = len(ids)
    nodes = pd.DataFrame({"id": ids, "kind": rng.choice(["person", "company"], n)})
    edges = pd.DataFrame({"src": [ids[i] for i in rng.integers(0, n, 15 * n)], "dst": [ids[i] for i in rng.integers(0, n, 15 * n)]})
    g = graphistry.edges(_native(edges, engine), "src", "dst").nodes(_native(nodes, engine), "id")
    g = g.gfql("CREATE GFQL INDEX FOR edge_out_adj", engine=engine).gfql("CREATE GFQL INDEX FOR node_id", engine=engine)
    return g, edges


def _served(report: Any) -> List[str]:
    return [step["seam"] for step in report["steps"] if step.get("served")]


@pytest.mark.route_engaged("indexed-kernel")
@pytest.mark.parametrize("engine", ENGINES)
def test_a_cypher_in_list_is_served_by_the_bindings_kernel(engine: str) -> None:
    _require(engine)
    g, edges = _graph(list(range(4_000)), engine)
    seeds = [3, 77, 1234, 3999]
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds} RETURN b"
    if engine in KERNEL_DISPATCHED:
        assert _served(g.gfql_explain(query, engine=engine)) == ["connected_bindings"]
    expected = sorted(edges[edges["src"].isin(seeds)]["dst"].tolist())
    assert _ids(g.gfql(query, engine=engine), "b.id") == expected
    assert _ids(g.gfql(query, engine=engine, index_policy="off"), "b.id") == expected


@pytest.mark.route_engaged("indexed-kernel")
@pytest.mark.parametrize("engine", ENGINES)
def test_two_hops_from_a_seed_set_are_served_too(engine: str) -> None:
    _require(engine)
    g, edges = _graph(list(range(4_000)), engine)
    seeds = [3, 77]
    query = f"MATCH (a)-[e1]->(b)-[e2]->(c) WHERE a.id IN {seeds} RETURN c"
    if engine in KERNEL_DISPATCHED:
        assert _served(g.gfql_explain(query, engine=engine)) == ["connected_bindings"]
    expected = sorted(edges[edges["src"].isin(seeds)][["dst"]].merge(edges, left_on="dst", right_on="src")["dst_y"].tolist())
    assert _ids(g.gfql(query, engine=engine), "c.id") == expected


@pytest.mark.route_engaged("indexed-kernel")
@pytest.mark.parametrize("engine", ENGINES)
def test_a_seed_set_covering_most_of_the_graph_takes_the_scan_and_still_agrees(engine: str) -> None:
    _require(engine)
    g, edges = _graph(list(range(4_000)), engine)
    seeds = list(range(0, 3_600))
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds} RETURN count(b) AS c"
    assert _served(g.gfql_explain(query, engine=engine)) == []
    frame = g.gfql(query, engine=engine)._nodes
    frame = frame.to_pandas() if hasattr(frame, "to_pandas") else frame
    assert int(frame["c"].iloc[0]) == int(edges["src"].isin(seeds).sum())


@pytest.mark.route_engaged("indexed-kernel")
def test_string_ids_decline_the_kernel_and_still_agree() -> None:
    ids = [f"n{i}" for i in range(2_000)]
    g, edges = _graph(ids)
    seeds = [ids[3], ids[77], ids[1999]]
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds!r} RETURN b"
    assert "connected_bindings" not in _served(g.gfql_explain(query))
    assert sorted(g.gfql(query)._nodes["b.id"].tolist()) == sorted(edges[edges["src"].isin(seeds)]["dst"].tolist())


def test_seed_ids_absent_from_the_graph_are_simply_unmatched() -> None:
    g, edges = _graph(list(range(4_000)))
    query = "MATCH (a)-[e]->(b) WHERE a.id IN [3, 999999, -7] RETURN b"
    assert _served(g.gfql_explain(query)) == ["connected_bindings"]
    assert _ids(g.gfql(query), "b.id") == sorted(edges[edges["src"] == 3]["dst"].tolist())


def test_a_label_beside_the_seed_set_still_goes_through_the_kernel() -> None:
    rng = np.random.default_rng(9)
    nodes = pd.DataFrame({"id": range(3_000), "type": rng.choice(["person", "company"], 3_000)})
    edges = pd.DataFrame({"src": rng.integers(0, 3_000, 30_000), "dst": rng.integers(0, 3_000, 30_000)})
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id").gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id")
    seeds = [3, 77, 1234, 2999]
    query = f"MATCH (a:person)-[e]->(b) WHERE a.id IN {seeds} RETURN b"
    assert _served(g.gfql_explain(query)) == ["connected_bindings"]
    persons = set(nodes[nodes["type"] == "person"]["id"]) & set(seeds)
    assert _ids(g.gfql(query), "b.id") == sorted(edges[edges["src"].isin(persons)]["dst"].tolist())


@pytest.mark.parametrize(
    "value,expected",
    [
        (is_in([3, 1, 3]), [1, 3]),
        ([5, 2], [2, 5]),
        ((7,), [7]),
        (is_in([1, True]), None),
        (is_in([1.5]), None),
        (is_in(["a"]), None),
        (7, None),
        ({"a": 1}, None),
    ],
    ids=["is_in dedups and sorts", "plain list", "tuple", "bool is not an id", "float is not an id", "str is not an id", "scalar is not a set", "dict is not a set"],
)
def test_membership_seed_ids(value: Any, expected: Any) -> None:
    assert _membership_seed_ids(value) == expected
