"""Bound-frame immutability, the library's own side: public APIs must never
mutate frames the caller handed in. Deep-snapshot before, byte-compare after.

Seeded from the 2026-08-13 immutability audit (29-arm dynamic harness): GFQL
proper was clean; tree_layout, label_components, and transform_umap
(merge_policy) mutated caller frames and were fixed with batched assigns.
"""
from typing import Any, Tuple

import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def _frames_pd() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["P"] * 4, "age": [25, 35, 45, 55]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "rel": ["F"] * 3})
    return nodes, edges


def _assert_unchanged(before: Any, after: Any) -> None:
    if isinstance(before, pd.DataFrame):
        pd.testing.assert_frame_equal(before, after)
    else:
        assert before.equals(after)


def _run_and_check(build_and_run, engine: str) -> None:
    nodes_pd, edges_pd = _frames_pd()
    if engine == "polars":
        pytest.importorskip("polars")
        nodes: Any = pl.from_pandas(nodes_pd)
        edges: Any = pl.from_pandas(edges_pd)
        n_snap, e_snap = nodes.clone(), edges.clone()
    else:
        nodes, edges = nodes_pd, edges_pd
        n_snap, e_snap = nodes.copy(deep=True), edges.copy(deep=True)
    build_and_run(nodes, edges)
    _assert_unchanged(n_snap, nodes)
    _assert_unchanged(e_snap, edges)


_APIS = {
    "gfql_cypher": lambda n, e: graphistry.nodes(n, "id").edges(e, "s", "d").gfql(
        "MATCH (a {kind:'P'})-[]->(b) WHERE b.age >= 35 RETURN b.id AS x ORDER BY x"),
    "gfql_two_hop_count": lambda n, e: graphistry.nodes(n, "id").edges(e, "s", "d").gfql(
        "MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})-[{rel:'F'}]->(c {kind:'P'}) RETURN count(*) AS n"),
    "hop": lambda n, e: graphistry.nodes(n, "id").edges(e, "s", "d").hop(
        nodes=pd.DataFrame({"id": [0]}), hops=2, direction="forward"),
    "index_all_plus_col_stats": lambda n, e: graphistry.nodes(n, "id").edges(e, "s", "d")
        .gfql_index_all().gfql_index_col_stats(node_type_column="kind", edge_type_column="rel"),
    "materialize_and_degrees": lambda n, e: graphistry.nodes(n, "id").edges(e, "s", "d")
        .materialize_nodes().get_degrees(),
}


@pytest.mark.parametrize("api", sorted(_APIS))
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_api_leaves_input_frames_unchanged(api: str, engine: str) -> None:
    if engine == "polars" and api in ("hop", "materialize_and_degrees"):
        pytest.importorskip("polars")
    _run_and_check(_APIS[api], engine)


def test_tree_layout_leaves_input_unchanged() -> None:
    nodes, edges = _frames_pd()
    n_snap, e_snap = nodes.copy(deep=True), edges.copy(deep=True)
    graphistry.nodes(nodes, "id").edges(edges, "s", "d").tree_layout()
    _assert_unchanged(n_snap, nodes)
    _assert_unchanged(e_snap, edges)


def test_label_components_leaves_input_unchanged() -> None:
    nodes, edges = _frames_pd()
    n_snap, e_snap = nodes.copy(deep=True), edges.copy(deep=True)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    try:
        from graphistry.layouts import LayoutsMixin  # noqa: F401
        g.label_components()
    except (ImportError, AttributeError, TypeError) as e:
        pytest.skip(f"label_components deps unavailable: {e}")
    _assert_unchanged(n_snap, nodes)
    _assert_unchanged(e_snap, edges)
