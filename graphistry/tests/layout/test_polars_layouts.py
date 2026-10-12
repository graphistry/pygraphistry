"""Layouts on Polars-bound graphs and explicitly converted Cypher results (#1966).

Each layout runs on pandas and returns eager Polars frames whose rows and values match
the same layout on a pandas-bound graph; explicit ``engine='pandas'`` returns pandas.
"""
import os
from typing import Any, Callable, Dict

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")

try:
    import igraph  # noqa: F401
    has_igraph = True
except ImportError:
    has_igraph = False

EDGES = pd.DataFrame({"s": [0, 0, 1, 2], "d": [1, 2, 3, 4]})  # a tree, so tree_layout applies
NODES = pd.DataFrame({
    "id": [0, 1, 2, 3, 4],
    "cat": ["a", "b", "a", "c", "b"],
    "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    "t": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]),
    "lat": [10.0, 20.0, 30.0, 40.0, 50.0],
    "lon": [1.0, 2.0, 3.0, 4.0, 5.0],
    "part": [0, 0, 0, 1, 1],
})

LAYOUTS: Dict[str, Callable[..., Any]] = {
    "circle_layout": lambda g, **k: g.circle_layout(bounding_box=(0, 0, 10, 10), **k),
    "tree_layout": lambda g, **k: g.tree_layout(**k),
    "ring_categorical_layout": lambda g, **k: g.ring_categorical_layout("cat", **k),
    "ring_continuous_layout": lambda g, **k: g.ring_continuous_layout("v", **k),
    "time_ring_layout": lambda g, **k: g.time_ring_layout("t", **k),
    "mercator_layout": lambda g, **k: g.mercator_layout(**k),
}
IGRAPH_LAYOUTS: Dict[str, Callable[..., Any]] = {
    "modularity_weighted_layout": lambda g, **k: g.modularity_weighted_layout(community_col="part", **k),
    "group_in_a_box_layout": lambda g, **k: g.group_in_a_box_layout(partition_key="part", **k),
    "fa2_layout_cpu_fallback": lambda g, **k: g.fa2_layout(allow_cpu_fallback=True, **k),
}
DETERMINISTIC = set(LAYOUTS) | {"modularity_weighted_layout"}
NO_ENGINE_PARAM = {"mercator_layout"}
GPU_TESTS = os.environ.get("TEST_CUGRAPH") == "1"
CYPHER_NODE_QUERY = "MATCH (n) WHERE n.id <= 3 RETURN n"


def _graphs(lazy: bool) -> Any:
    def bind(n: Any, e: Any) -> Any:
        return graphistry.edges(e, "s", "d").nodes(n, "id").bind(point_latitude="lat", point_longitude="lon")

    n, e = pl.from_pandas(NODES), pl.from_pandas(EDGES)
    if lazy:
        n, e = n.lazy(), e.lazy()
    return bind(NODES, EDGES), bind(n, e)


def _cases() -> Any:
    for name, fn in LAYOUTS.items():
        yield pytest.param(name, fn, id=name)
    for name, fn in IGRAPH_LAYOUTS.items():
        yield pytest.param(name, fn, id=name, marks=pytest.mark.skipif(not has_igraph, reason="Requires igraph"))


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("name, layout", list(_cases()))
def test_layout_on_polars_returns_polars_matching_pandas(name: str, layout: Callable[..., Any], lazy: bool) -> None:
    g_pd, g_pl = _graphs(lazy)
    out = layout(g_pl)
    assert isinstance(out._nodes, pl.DataFrame) and isinstance(out._edges, pl.DataFrame)
    assert out._nodes.schema["id"] == pl.Int64
    assert out._edges.schema["s"] == pl.Int64 and out._edges.schema["d"] == pl.Int64
    ref = layout(g_pd)
    got_nodes = out._nodes.to_pandas()
    assert list(got_nodes.columns) == list(ref._nodes.columns)
    assert got_nodes["id"].tolist() == ref._nodes["id"].tolist()
    assert out._point_x == ref._point_x and out._point_y == ref._point_y
    if name in DETERMINISTIC:
        pd.testing.assert_frame_equal(got_nodes, ref._nodes.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(out._edges.to_pandas(), ref._edges.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("name, layout", [c for c in _cases() if c.values[0] not in NO_ENGINE_PARAM])
def test_layout_on_polars_honors_explicit_pandas_engine(name: str, layout: Callable[..., Any]) -> None:
    _, g_pl = _graphs(lazy=False)
    out = layout(g_pl, engine="pandas")
    assert isinstance(out._nodes, pd.DataFrame) and isinstance(out._edges, pd.DataFrame)


@pytest.mark.parametrize("name, layout", [c for c in _cases() if c.values[0] not in NO_ENGINE_PARAM])
def test_layout_on_polars_accepts_explicit_polars_engine(name: str, layout: Callable[..., Any]) -> None:
    _, g_pl = _graphs(lazy=False)
    out = layout(g_pl, engine="polars")
    assert isinstance(out._nodes, pl.DataFrame) and isinstance(out._edges, pl.DataFrame)


def test_fa2_layout_on_polars_without_gpu_engine_declines_like_pandas() -> None:
    g_pd, g_pl = _graphs(lazy=False)
    with pytest.raises(NotImplementedError, match="GPU-enabled engine"):
        g_pd.fa2_layout()
    with pytest.raises(NotImplementedError, match="GPU-enabled engine"):
        g_pl.fa2_layout()


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_tree_layout_keeps_legacy_ignored_cudf_keyword_on_polars(lazy: bool) -> None:
    g_pd, g_pl = _graphs(lazy)
    ref = g_pd.tree_layout(engine="cudf")
    out = g_pl.tree_layout(engine="cudf")
    assert isinstance(out._nodes, pd.DataFrame) and isinstance(out._edges, pd.DataFrame)
    pd.testing.assert_frame_equal(out._nodes, ref._nodes)
    pd.testing.assert_frame_equal(out._edges, ref._edges)


@pytest.mark.parametrize("layout", ["circle_layout", "tree_layout"])
def test_layout_preserves_nullable_large_edge_id(layout: str) -> None:
    big = 2**53 + 1
    nodes = pl.DataFrame({"id": [1, 2, 3]})
    edges = pl.DataFrame({
        "s": [1, 2], "d": [2, 3],
        "eid": pl.Series("eid", [big, None], dtype=pl.Int64),
    })
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d", edge="eid")
    kwargs = {"bounding_box": (0, 0, 10, 10)} if layout == "circle_layout" else {}
    out = getattr(graph, layout)(**kwargs)
    assert out._edges.schema["eid"] == pl.Int64
    assert out._edges["eid"].to_list() == [big, None]


def _cypher_selected_graph(engine: str, lazy: bool = False) -> Any:
    """Make topology explicit: Cypher RETURN is rows, not a matched-edge graph."""
    edges = pl.from_pandas(EDGES.assign(eid=[10, 11, 12, 13]))
    nodes = pl.from_pandas(NODES)
    source = graphistry.edges(edges.lazy() if lazy else edges, "s", "d", edge="eid").nodes(
        nodes.lazy() if lazy else nodes, "id"
    )
    rows = source.gfql(CYPHER_NODE_QUERY, engine=engine)
    assert isinstance(rows._nodes, pl.DataFrame)
    assert rows._edges is not None and len(rows._edges) == 0
    columns = {f"n.{column}": column for column in NODES.columns}
    assert set(columns) <= set(rows._nodes.columns)
    selected_nodes = rows._nodes.rename(columns)
    assert selected_nodes["id"].n_unique() == len(selected_nodes)
    ids = selected_nodes.select("id")
    selected_edges = (
        edges.join(ids.rename({"id": "s"}), on="s", how="semi")
        .join(ids.rename({"id": "d"}), on="d", how="semi")
        .sort("eid")
    )
    return graphistry.edges(selected_edges, "s", "d", edge="eid").nodes(selected_nodes, "id").bind(
        point_latitude="lat", point_longitude="lon"
    )


@pytest.mark.parametrize("engine", [
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.skipif(not GPU_TESTS, reason="Requires real RAPIDS GPU")),
])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager-input", "lazy-input"])
@pytest.mark.parametrize("name, layout", list(_cases()))
def test_cypher_polars_result_explicit_graph_conversion_runs_layouts(
    name: str, layout: Callable[..., Any], engine: str, lazy: bool,
) -> None:
    if engine == "polars-gpu":
        pytest.importorskip("cudf")
        pytest.importorskip("cudf_polars")
    selected = _cypher_selected_graph(engine, lazy)
    assert selected._node == "id" and selected._source == "s" and selected._destination == "d"
    assert selected._edge == "eid"
    assert selected._nodes.schema["id"] == pl.Int64
    assert selected._edges.schema["eid"] == pl.Int64
    assert selected._nodes["id"].to_list() == [0, 1, 2, 3]
    assert selected._edges["eid"].to_list() == [10, 11, 12]

    out = layout(selected)
    assert isinstance(out._nodes, pl.DataFrame) and isinstance(out._edges, pl.DataFrame)
    assert out._node == "id" and out._source == "s" and out._destination == "d"
    assert out._edge == "eid"
    assert sorted(out._nodes["id"].to_list()) == [0, 1, 2, 3]
    assert sorted(out._edges["eid"].to_list()) == [10, 11, 12]
    if name in DETERMINISTIC:
        ref_nodes = NODES.loc[NODES["id"] <= 3].reset_index(drop=True)
        ref_edges = EDGES.assign(eid=[10, 11, 12, 13]).iloc[:3].reset_index(drop=True)
        ref = graphistry.edges(ref_edges, "s", "d", edge="eid").nodes(ref_nodes, "id").bind(
            point_latitude="lat", point_longitude="lon"
        )
        expected = layout(ref)
        pd.testing.assert_frame_equal(out._nodes.to_pandas(), expected._nodes.reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(out._edges.to_pandas(), expected._edges.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("query_engine", [
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.skipif(not GPU_TESTS, reason="Requires real RAPIDS GPU")),
])
def test_cypher_result_graph_layout_backend_is_explicit(query_engine: str) -> None:
    if query_engine == "polars-gpu":
        pytest.importorskip("cudf")
        pytest.importorskip("cudf_polars")
    selected = _cypher_selected_graph(query_engine)
    kwargs = {"bounding_box": (0, 0, 10, 10)}
    default = selected.circle_layout(**kwargs)
    assert isinstance(default._nodes, pl.DataFrame)
    explicit_cpu = selected.circle_layout(engine="pandas", **kwargs)
    assert isinstance(explicit_cpu._nodes, pd.DataFrame)
    pd.testing.assert_frame_equal(default._nodes.to_pandas(), explicit_cpu._nodes, check_dtype=False)
    # Legacy tree keyword is intentionally pandas-only, even after a GPU query.
    assert isinstance(selected.tree_layout(engine="cudf")._nodes, pd.DataFrame)
    if query_engine == "polars-gpu":
        import cudf

        explicit_cudf = selected.circle_layout(engine="cudf", **kwargs)
        assert isinstance(explicit_cudf._nodes, cudf.DataFrame)
        assert sorted(explicit_cudf._nodes["id"].to_pandas().tolist()) == [0, 1, 2, 3]
        explicit_gpu_polars = selected.circle_layout(engine="polars-gpu", **kwargs)
        assert isinstance(explicit_gpu_polars._nodes, pl.DataFrame)
        assert sorted(explicit_gpu_polars._nodes["id"].to_list()) == [0, 1, 2, 3]


@pytest.mark.parametrize("query_engine", [
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.skipif(not GPU_TESTS, reason="Requires real RAPIDS GPU")),
])
def test_cypher_result_graph_conversion_keeps_large_nullable_ids(query_engine: str) -> None:
    if query_engine == "polars-gpu":
        pytest.importorskip("cudf")
        pytest.importorskip("cudf_polars")
    big = 2**53 + 1
    nodes = pl.DataFrame({"id": [big, big + 1, big + 2, big + 3], "keep": [True, True, True, False]})
    edges = pl.DataFrame({
        "s": [big, big + 1, big + 2], "d": [big + 1, big + 2, big + 3],
        "eid": pl.Series("eid", [big + 10, None, big + 11], dtype=pl.Int64),
    })
    source = graphistry.edges(edges, "s", "d", edge="eid").nodes(nodes, "id")
    rows = source.gfql("MATCH (n) WHERE n.keep = true RETURN n", engine=query_engine)
    assert rows._edges is not None and len(rows._edges) == 0
    selected_nodes = rows._nodes.rename({"n.id": "id", "n.keep": "keep"})
    ids = selected_nodes.select("id")
    selected_edges = edges.join(ids.rename({"id": "s"}), on="s", how="semi").join(
        ids.rename({"id": "d"}), on="d", how="semi"
    )
    selected = graphistry.edges(selected_edges, "s", "d", edge="eid").nodes(selected_nodes, "id")
    out = selected.circle_layout(bounding_box=(0, 0, 10, 10))
    assert out._nodes.schema["id"] == pl.Int64 and out._edges.schema["eid"] == pl.Int64
    assert sorted(out._nodes["id"].to_list()) == [big, big + 1, big + 2]
    assert out._edges["eid"].to_list() == [big + 10, None]
    assert out._node == "id" and out._source == "s" and out._destination == "d" and out._edge == "eid"


@pytest.mark.parametrize("query_engine", [
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.skipif(not GPU_TESTS, reason="Requires real RAPIDS GPU")),
])
def test_cypher_return_rows_do_not_implicitly_claim_matched_edges(query_engine: str) -> None:
    if query_engine == "polars-gpu":
        pytest.importorskip("cudf")
        pytest.importorskip("cudf_polars")
    source = graphistry.edges(pl.from_pandas(EDGES), "s", "d").nodes(pl.from_pandas(NODES), "id")
    rows = source.gfql("MATCH (a)-[e]->(b) RETURN b", engine=query_engine)
    assert isinstance(rows._nodes, pl.DataFrame)
    assert rows._edges is not None and len(rows._edges) == 0
    assert rows._source is None and rows._destination is None
    with pytest.raises(ValueError, match="Missing source/destination bindings"):
        rows.circle_layout(bounding_box=(0, 0, 10, 10))
