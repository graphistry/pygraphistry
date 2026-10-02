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


def _graph(ids: List[Any]) -> tuple:
    rng = np.random.default_rng(5)
    n = len(ids)
    nodes = pd.DataFrame({"id": ids, "kind": rng.choice(["person", "company"], n)})
    edges = pd.DataFrame({"src": [ids[i] for i in rng.integers(0, n, 15 * n)], "dst": [ids[i] for i in rng.integers(0, n, 15 * n)]})
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    return g.gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id"), edges


def _served(report: Any) -> List[str]:
    return [step["seam"] for step in report["steps"] if step.get("served")]


def test_a_cypher_in_list_is_served_by_the_bindings_kernel() -> None:
    g, edges = _graph(list(range(4_000)))
    seeds = [3, 77, 1234, 3999]
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds} RETURN b"
    assert _served(g.gfql_explain(query)) == ["connected_bindings"]
    assert sorted(g.gfql(query)._nodes["b.id"].tolist()) == sorted(edges[edges["src"].isin(seeds)]["dst"].tolist())
    assert sorted(g.gfql(query)._nodes["b.id"].tolist()) == sorted(g.gfql(query, index_policy="off")._nodes["b.id"].tolist())


def test_two_hops_from_a_seed_set_are_served_too() -> None:
    g, edges = _graph(list(range(4_000)))
    seeds = [3, 77]
    query = f"MATCH (a)-[e1]->(b)-[e2]->(c) WHERE a.id IN {seeds} RETURN c"
    assert _served(g.gfql_explain(query)) == ["connected_bindings"]
    expected = sorted(edges[edges["src"].isin(seeds)][["dst"]].merge(edges, left_on="dst", right_on="src")["dst_y"].tolist())
    assert sorted(g.gfql(query)._nodes["c.id"].tolist()) == expected


def test_a_seed_set_covering_most_of_the_graph_takes_the_scan_and_still_agrees() -> None:
    g, edges = _graph(list(range(4_000)))
    seeds = list(range(0, 3_600))
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds} RETURN count(b) AS c"
    assert _served(g.gfql_explain(query)) == []
    assert int(g.gfql(query)._nodes["c"].iloc[0]) == int(edges["src"].isin(seeds).sum())


def test_string_ids_decline_the_kernel_and_still_agree() -> None:
    ids = [f"n{i}" for i in range(2_000)]
    g, edges = _graph(ids)
    seeds = [ids[3], ids[77], ids[1999]]
    query = f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds!r} RETURN b"
    assert "connected_bindings" not in _served(g.gfql_explain(query))
    assert sorted(g.gfql(query)._nodes["b.id"].tolist()) == sorted(edges[edges["src"].isin(seeds)]["dst"].tolist())


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
