"""Low-latency contract for basic Cypher shapes on a wide node table.

Per engine and shape: (1) ``gfql_explain`` reports a fast path served, and (2) the Cypher
call costs at most ``K`` times a plain frame-op floor for the same lookup (an id mask on the
node table plus one join per hop), measured interleaved on the same graph. Ratio pins run
only for shapes the served pin expects to be served. Known coverage gaps are
``xfail(strict=True)`` so the fix that closes one flips the pin.
"""
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import graphistry

N_PERSON = 100_000
N_MESSAGE = 200_000
N_WIDE_OBJECT_COLUMNS = 27
ENGINES = ["pandas", "polars", "cudf"]
RATIO_LIMIT = 12.0
NATIVE_FLOOR_MS = 0.05

PERSON, MESSAGE = 123, N_PERSON + 456
CYPHER: Dict[str, str] = {
    "seeded_hop_props": (
        f"MATCH (m:Message {{id: {MESSAGE}}})-[:HAS_CREATOR]->(p:Person) "
        "RETURN p.id AS personId, p.firstName AS firstName"
    ),
    "seeded_hop_entity": f"MATCH (m:Message {{id: {MESSAGE}}})-[:HAS_CREATOR]->(p:Person) RETURN p",
    "node_only_props": (
        f"MATCH (p:Person {{id: {PERSON}}}) RETURN p.firstName AS firstName, p.lastName AS lastName"
    ),
    "seeded_hop_all_aliases": (
        f"MATCH (m:Message {{id: {MESSAGE}}})-[r:HAS_CREATOR]->(p:Person) "
        "RETURN m.lastName AS mln, r.type AS rt, p.firstName AS firstName"
    ),
}
# Shapes no fast path serves yet; strict so the fix that closes one flips the pin.
SERVED_GAPS: Dict[Tuple[str, str], str] = {}


def _wide_object_column_graph() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2029)
    persons = pd.DataFrame({
        "id": np.arange(N_PERSON), "type": "Person",
        "firstName": [f"f{i}" for i in range(N_PERSON)],
        "lastName": [f"l{i}" for i in range(N_PERSON)],
    })
    messages = pd.DataFrame({
        "id": np.arange(N_PERSON, N_PERSON + N_MESSAGE), "type": "Message",
        "firstName": None, "lastName": None,
    })
    nodes = pd.concat([persons, messages], ignore_index=True)
    for k in range(N_WIDE_OBJECT_COLUMNS):
        nodes[f"attr{k}"] = np.where(nodes.index % 3 == k % 3, None, "v")
    edges = pd.DataFrame({
        "src": np.arange(N_PERSON, N_PERSON + N_MESSAGE),
        "dst": rng.integers(0, N_PERSON, N_MESSAGE), "type": "HAS_CREATOR",
    })
    return nodes, edges


def _engine_frames(engine: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[Any, Any]:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return pl.from_pandas(nodes), pl.from_pandas(edges)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return nodes, edges


def _bind(engine: str) -> Any:
    nodes, edges = _engine_frames(engine, *_wide_object_column_graph())
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst").gfql_index_all(engine=engine)


def _floor_ops(engine: str, nodes: Any, edges: Any) -> Dict[str, Callable[[], Any]]:
    """Plain frame ops for each shape: id mask on the node table, one join per hop."""
    if engine == "polars":
        import polars as pl

        def hop() -> Any:
            e = edges.filter(pl.col("src") == MESSAGE)
            return nodes.join(e.select("dst"), left_on="id", right_on="dst")

        def hop_all() -> Any:  # properties of both endpoints and the edge: two lookups + the edge columns
            e = edges.filter(pl.col("src") == MESSAGE)
            seed = nodes.filter(pl.col("id") == MESSAGE).select(pl.col("id").alias("sid"), "lastName")
            return (e.join(seed, left_on="src", right_on="sid")
                    .join(nodes.select("id", "firstName"), left_on="dst", right_on="id")
                    .select("lastName", "type", "firstName"))

        return {
            "seeded_hop_props": lambda: hop().select("id", "firstName"),
            "seeded_hop_entity": hop,
            "node_only_props": lambda: nodes.filter(pl.col("id") == PERSON).select("firstName", "lastName"),
            "seeded_hop_all_aliases": hop_all,
        }

    def hop_df() -> Any:
        e = edges[edges["src"] == MESSAGE]
        return nodes.merge(e[["dst"]], left_on="id", right_on="dst")

    def hop_all_df() -> Any:  # properties of both endpoints and the edge: two lookups + the edge columns
        e = edges[edges["src"] == MESSAGE]
        seed = nodes[nodes["id"] == MESSAGE][["id", "lastName"]]
        both = e.merge(seed, left_on="src", right_on="id").merge(nodes[["id", "firstName"]], left_on="dst", right_on="id")
        return both[["lastName", "type", "firstName"]]

    return {
        "seeded_hop_props": lambda: hop_df()[["id", "firstName"]],
        "seeded_hop_entity": hop_df,
        "node_only_props": lambda: nodes[nodes["id"] == PERSON][["firstName", "lastName"]],
        "seeded_hop_all_aliases": hop_all_df,
    }


def _interleaved_best_ms(a: Callable[[], Any], b: Callable[[], Any], warm: int = 2,
                         runs: int = 5) -> Tuple[float, float]:
    for _ in range(warm):
        a()
        b()
    best_a = best_b = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        a()
        best_a = min(best_a, time.perf_counter() - t0)
        t0 = time.perf_counter()
        b()
        best_b = min(best_b, time.perf_counter() - t0)
    return best_a * 1000, best_b * 1000


def _served(g: Any, query: str, engine: str) -> bool:
    steps = g.gfql_explain(query, engine=engine).get("steps", [])
    return any(s.get("op") == "fast_path" and s.get("served") is True for s in steps)


def _cases(only_served: bool) -> List[Any]:
    out = []
    for engine in ENGINES:
        for shape in CYPHER:
            gap = SERVED_GAPS.get((engine, shape))
            if only_served and gap is not None:
                continue
            marks = [pytest.mark.xfail(strict=True, reason=gap)] if gap else []
            out.append(pytest.param(engine, shape, marks=marks, id=f"{engine}-{shape}"))
    return out


@pytest.fixture(scope="module")
def graphs() -> Dict[str, Any]:
    return {}


def _graph_for(graphs: Dict[str, Any], engine: str) -> Any:
    if engine not in graphs:
        graphs[engine] = _bind(engine)
    return graphs[engine]


@pytest.mark.parametrize("engine,shape", _cases(only_served=False))
def test_basic_shape_is_served_by_a_fast_path(graphs: Dict[str, Any], engine: str, shape: str) -> None:
    g = _graph_for(graphs, engine)
    assert _served(g, CYPHER[shape], engine), f"{engine}: {shape} fell off the fast path"


@pytest.mark.parametrize("engine,shape", _cases(only_served=True))
def test_served_shape_costs_a_bounded_multiple_of_plain_frame_ops(
    graphs: Dict[str, Any], engine: str, shape: str
) -> None:
    g = _graph_for(graphs, engine)
    floor = _floor_ops(engine, g._nodes, g._edges)[shape]
    cypher_ms, floor_ms = _interleaved_best_ms(lambda: g.gfql(CYPHER[shape], engine=engine), floor)
    ratio = cypher_ms / max(floor_ms, NATIVE_FLOOR_MS)
    print(f"[latency-contract] {engine} {shape}: cypher {cypher_ms:.2f} ms, floor {floor_ms:.2f} ms, {ratio:.1f}x")
    assert ratio < RATIO_LIMIT, (
        f"{engine}: {shape} costs {ratio:.0f}x the plain frame ops (limit {RATIO_LIMIT:g}x)"
    )
