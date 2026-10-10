"""Layouts on Polars-bound graphs (#1966).

Each layout runs on pandas and returns eager Polars frames whose rows and values match
the same layout on a pandas-bound graph; explicit ``engine='pandas'`` returns pandas.
"""
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
