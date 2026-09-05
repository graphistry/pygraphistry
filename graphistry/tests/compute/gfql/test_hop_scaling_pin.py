"""Scaling pin for undirected multi-hop (#2023 / #2026 family).

Wall-clock thresholds are flaky in CI, so this pins a RATIO: an undirected 2-hop chain
from hub seeds must cost no more than a fixed multiple of two plain frame joins that
traverse the same edges. A per-edge interpreter loop (the #2023 defect) blew this ratio
to ~50x; the engine-native rule keeps it under ~20x. The graph is small enough for
the core lanes; polars runs when installed.
"""
import time

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_undirected, n
from graphistry.compute.predicates.is_in import IsIn

#: Measured on the same graph: the 0.59.0 tree (per-edge loop, #2023) 47-52x on both engines;
#: the engine-native rule 20x (pandas) / 6x (polars). 30 sits between the two populations.
MAX_RATIO = 30.0
N_NODES = 200_000
N_EDGES = 1_500_000
SEEDS = 50


def _graph():
    # heavy-tailed degrees (hubs), so a 2-hop from the top seeds touches most edges,
    # which is the shape that exposed #2023 on LiveJournal
    rng = np.random.default_rng(2023)
    src = (rng.zipf(1.3, N_EDGES) - 1) % N_NODES
    dst = rng.integers(0, N_NODES, N_EDGES)
    edges = pd.DataFrame({"s": src, "d": dst})
    degree = pd.concat([edges["s"], edges["d"]]).value_counts()
    nodes = pd.DataFrame({"id": degree.index, "degree": degree.values})
    seeds = degree.index[:SEEDS].tolist()
    return edges, nodes, seeds


def _baseline_two_joins(edges: pd.DataFrame, seeds) -> int:
    """Two undirected frontier expansions as plain merges: the work floor for a 2-hop."""
    both = pd.concat([edges, edges.rename(columns={"s": "d", "d": "s"})], ignore_index=True)
    frontier = pd.DataFrame({"s": seeds})
    seen = set(seeds)
    for _ in range(2):
        nxt = both.merge(frontier, on="s")["d"].drop_duplicates()
        seen.update(nxt.tolist())
        frontier = pd.DataFrame({"s": nxt})
    return len(seen)


def _best_of(fn, runs=2):
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


try:
    import polars  # noqa: F401
    HAS_POLARS = True
except ImportError:  # pragma: no cover
    HAS_POLARS = False


@pytest.mark.parametrize("engine", [
    "pandas",
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
])
def test_undirected_two_hop_costs_a_bounded_multiple_of_two_joins(engine):
    edges, nodes, seeds = _graph()
    if engine == "polars":
        import polars as pl
        g = graphistry.edges(pl.from_pandas(edges), "s", "d").nodes(pl.from_pandas(nodes), "id")
    else:
        g = graphistry.edges(edges, "s", "d").nodes(nodes, "id")
    query = [n({"id": IsIn(options=seeds)}), e_undirected(to_fixed_point=False, hops=2), n()]
    g.gfql(query, engine=engine)  # warm
    hop_s = _best_of(lambda: g.gfql(query, engine=engine))
    base_s = _best_of(lambda: _baseline_two_joins(edges, seeds))
    ratio = hop_s / base_s
    print(f"[scaling-pin] {engine}: 2-hop {hop_s * 1000:.0f} ms, "
          f"two joins {base_s * 1000:.0f} ms, ratio {ratio:.1f}x")
    assert ratio < MAX_RATIO, (
        f"{engine}: 2-hop took {hop_s * 1000:.0f} ms = {ratio:.0f}x two plain joins "
        f"({base_s * 1000:.0f} ms); a per-edge interpreter loop is back (see #2023)")
