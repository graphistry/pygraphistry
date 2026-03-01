"""Shared helpers for df_executor profiling scripts."""

from __future__ import annotations

import random
from typing import List, Tuple

import pandas as pd

from graphistry.compute.ast import n, e_forward
from graphistry.compute.gfql.same_path_types import col, compare, WhereComparison


def make_dense_graph(n_nodes: int, n_edges: int, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)

    nodes = pd.DataFrame(
        {
            "id": list(range(n_nodes)),
            "v": list(range(n_nodes)),
        }
    )

    edge_rows = []
    for edge_id in range(n_edges):
        src = random.randint(0, n_nodes - 2)
        dst = random.randint(src + 1, n_nodes - 1)
        edge_rows.append({"src": src, "dst": dst, "eid": edge_id})

    edges = pd.DataFrame(edge_rows).drop_duplicates(subset=["src", "dst"])
    return nodes, edges


def simple_chain():
    return [n(name="a"), e_forward(name="e"), n(name="c")]


def multihop_chain():
    return [
        n({"id": 0}, name="a"),
        e_forward(min_hops=1, max_hops=3, name="e"),
        n(name="c"),
    ]


def simple_where() -> List[WhereComparison]:
    return [compare(col("a", "v"), "<", col("c", "v"))]
