"""A Cypher hop that falls through to the row pipeline must reach the same resident adjacency
index a native hop does. The pipeline tags the edge frame with a per-edge identity column before
hopping; that derivation keeps every row in place, so the index migrates with it instead of
missing on frame identity and scanning.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import pytest

import graphistry


def _graph(ids: List[Any]) -> tuple:
    rng = np.random.default_rng(3)
    n = len(ids)
    nodes = pd.DataFrame({"id": ids, "kind": rng.choice(["person", "company"], n)})
    pick = rng.integers(0, n, 20 * n)
    edges = pd.DataFrame({"src": [ids[i] for i in pick], "dst": [ids[i] for i in rng.integers(0, n, 20 * n)]})
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    return g.gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id"), edges


def _hop_steps(report: Any) -> List[str]:
    return [step["path"] for step in report["steps"] if step.get("op") == "hop"]


def _served(report: Any) -> List[str]:
    return [step["seam"] for step in report["steps"] if step.get("served")]


@pytest.mark.parametrize(
    "ids,route",
    [(list(range(3_000)), "kernel"), ([f"n{i}" for i in range(3_000)], "hop")],
    ids=["int ids: the bindings kernel serves before any hop", "str ids: the kernel declines, the hop itself is indexed"],
)
def test_a_multi_seed_cypher_hop_is_served_by_the_resident_index(ids: List[Any], route: str) -> None:
    g, edges = _graph(ids)
    seeds = [ids[i] for i in (5, 77, 1234, 2999)]
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds!r} RETURN b"
    report = g.gfql_explain(query)
    assert report["used_index"] is True
    if route == "kernel":
        assert _served(report) == ["connected_bindings"] and _hop_steps(report) == []
    else:
        assert _hop_steps(report) == ["index"]
    served = sorted(g.gfql(query)._nodes["b.id"].tolist())
    assert served == sorted(g.gfql(query, index_policy="off")._nodes["b.id"].tolist())
    assert served == sorted(edges[edges["src"].isin(seeds)]["dst"].tolist())  # one b row per matched edge


def test_an_edge_frame_the_index_was_not_built_over_still_scans() -> None:
    g, edges = _graph(list(range(3_000)))
    permuted = edges.sort_values("src").reset_index(drop=True)
    rebound = g.edges(permuted, "src", "dst")
    query = "MATCH (a)-[e]->(b) WHERE a.id IN [5, 77, 1234] RETURN b"
    report = rebound.gfql_explain(query)
    assert report["used_index"] is False
    assert _hop_steps(report) == ["scan"]
    assert sorted(rebound.gfql(query)._nodes["b.id"].tolist()) == sorted(g.gfql(query)._nodes["b.id"].tolist())
