"""The low-latency contract for basic GFQL shapes on a wide node table.

Two pins per shape, both machine-independent: (1) ``gfql_explain`` reports a fast path
SERVED, and (2) the Cypher form costs no more than ``K`` times the native chain form of the
same lookup on the same graph. The graph is wide (30 object columns) and large enough
(300k nodes) that any per-call scan of column VALUES shows up as a ratio blow-up; the
per-frame dtype memo keeps it flat. Shapes the fast paths do not cover yet are marked
``xfail(strict=True)`` so closing a gap flips the pin rather than passing silently.
"""
import time

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False
try:
    import cudf
    HAS_CUDF = True
except ImportError:  # pragma: no cover
    HAS_CUDF = False

N_PERSON = 100_000
N_MESSAGE = 200_000
ENGINES = ["pandas", "polars", "cudf"]
ENGINE_SKIPS = {"polars": not HAS_POLARS, "cudf": not HAS_CUDF}


def _wide_graph():
    rng = np.random.default_rng(2029)
    persons = pd.DataFrame({"id": np.arange(N_PERSON), "type": "Person",
                            "firstName": [f"f{i}" for i in range(N_PERSON)],
                            "lastName": [f"l{i}" for i in range(N_PERSON)]})
    messages = pd.DataFrame({"id": np.arange(N_PERSON, N_PERSON + N_MESSAGE), "type": "Message",
                             "firstName": None, "lastName": None})
    nodes = pd.concat([persons, messages], ignore_index=True)
    for k in range(27):  # wide object columns, some holding non-strings (the gate's slow case)
        nodes[f"attr{k}"] = np.where(nodes.index % 3 == k % 3, None, "v")
    edges = pd.DataFrame({"src": np.arange(N_PERSON, N_PERSON + N_MESSAGE),
                          "dst": rng.integers(0, N_PERSON, N_MESSAGE), "type": "HAS_CREATOR"})
    return nodes, edges


def _bind(engine):
    nodes, edges = _wide_graph()
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst").gfql_index_all(engine=engine)


def _best_ms(fn, warm=2, runs=5):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def _served(g, query, engine):
    steps = g.gfql_explain(query, engine=engine).get("steps", [])
    return any(s.get("op") == "fast_path" and s.get("served") is True for s in steps)


PERSON, MESSAGE = 123, N_PERSON + 456
SHAPES = {
    # name: (cypher, native chain, K)
    "seeded typed 1-hop + props": (
        f"MATCH (m:Message {{id: {MESSAGE}}})-[:HAS_CREATOR]->(p:Person) RETURN p.id AS personId, p.firstName AS firstName",
        [n({"id": MESSAGE}), e_forward({"type": "HAS_CREATOR"}), n({"type": "Person"})], 12.0),
    "seeded typed 1-hop + whole entity": (
        f"MATCH (m:Message {{id: {MESSAGE}}})-[:HAS_CREATOR]->(p:Person) RETURN p",
        [n({"id": MESSAGE}), e_forward({"type": "HAS_CREATOR"}), n({"type": "Person"})], 12.0),
    "node-only seed + props": (
        f"MATCH (p:Person {{id: {PERSON}}}) RETURN p.firstName AS firstName, p.lastName AS lastName",
        [n({"id": PERSON})], 12.0),
}

# Known contract gaps; strict so the fix that closes one flips the pin.
SERVED_GAPS = {
    (engine, "node-only seed + props"): "no fast path serves a node-only seeded lookup with projections"
    for engine in ENGINES
}
RATIO_GAPS = {
    ("pandas", "node-only seed + props"): "node-only projection runs the full chain path (~27x native)",
}


def _cases(gaps):
    out = []
    for engine in ENGINES:
        for shape in SHAPES:
            marks = []
            if ENGINE_SKIPS.get(engine):
                marks.append(pytest.mark.skipif(True, reason=f"{engine} not installed"))
            if (engine, shape) in gaps:
                marks.append(pytest.mark.xfail(strict=True, reason=gaps[(engine, shape)]))
            out.append(pytest.param(engine, shape, marks=marks, id=f"{engine}-{shape}"))
    return out


@pytest.fixture(scope="module")
def graphs():
    return {}


def _graph_for(graphs, engine):
    if engine not in graphs:
        graphs[engine] = _bind(engine)
    return graphs[engine]


@pytest.mark.parametrize("engine,shape", _cases(SERVED_GAPS))
def test_basic_shape_is_served_by_a_fast_path(graphs, engine, shape):
    cypher, _, _ = SHAPES[shape]
    g = _graph_for(graphs, engine)
    assert _served(g, cypher, engine), f"{engine}: {shape} fell off the fast path"


@pytest.mark.parametrize("engine,shape", _cases(RATIO_GAPS))
def test_basic_shape_costs_a_bounded_multiple_of_the_native_chain(graphs, engine, shape):
    cypher, native, k = SHAPES[shape]
    g = _graph_for(graphs, engine)
    cypher_ms = _best_ms(lambda: g.gfql(cypher, engine=engine))
    native_ms = _best_ms(lambda: g.gfql(native, engine=engine))
    ratio = cypher_ms / max(native_ms, 0.05)
    print(f"[latency-contract] {engine} {shape}: cypher {cypher_ms:.2f} ms, native {native_ms:.2f} ms, {ratio:.1f}x")
    assert ratio < k, f"{engine}: {shape} costs {ratio:.0f}x the native chain (limit {k}x)"
