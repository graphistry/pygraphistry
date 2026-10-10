"""igraph and cuGraph methods on Polars-bound graphs (#2025).

They run on pandas (igraph) or cuDF (cuGraph) and return Polars frames whose rows,
values and identifier dtypes match the same call on a pandas-bound graph.
"""
import os
from typing import Any

import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")

try:
    import igraph  # noqa: F401
    has_igraph = True
except ImportError:
    has_igraph = False

test_cugraph = os.environ.get("TEST_CUGRAPH") == "1"

EDGES = pd.DataFrame({"s": ["a", "b", "c", "a", "d"], "d": ["b", "c", "a", "c", "a"], "w": [1.0, 2.0, 3.0, 4.0, 5.0]})
NODES = pd.DataFrame({"id": ["a", "b", "c", "d"], "kind": ["x", "y", "x", "y"]})


def _graphs(lazy: bool) -> Any:
    g_pd = graphistry.edges(EDGES, "s", "d").nodes(NODES, "id")
    n, e = pl.from_pandas(NODES), pl.from_pandas(EDGES)
    if lazy:
        n, e = n.lazy(), e.lazy()
    return g_pd, graphistry.edges(e, "s", "d").nodes(n, "id")


def _assert_matches_pandas(out_pl: Any, out_ref: Any) -> None:
    assert isinstance(out_pl._nodes, pl.DataFrame) and isinstance(out_pl._edges, pl.DataFrame)
    ref_nodes = out_ref._nodes.to_pandas() if hasattr(out_ref._nodes, "to_pandas") else out_ref._nodes
    ref_edges = out_ref._edges.to_pandas() if hasattr(out_ref._edges, "to_pandas") else out_ref._edges
    pd.testing.assert_frame_equal(out_pl._nodes.to_pandas(), ref_nodes.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(out_pl._edges.to_pandas(), ref_edges.reset_index(drop=True), check_dtype=False)
    assert out_pl._nodes.schema["id"] == pl.String
    assert out_pl._edges.schema["s"] == pl.String and out_pl._edges.schema["d"] == pl.String


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("alg", ["pagerank", "community_multilevel", "k_core"])
def test_compute_igraph_on_polars_matches_pandas(lazy: bool, alg: str) -> None:
    g_pd, g_pl = _graphs(lazy)
    directed = False if alg == "community_multilevel" else None
    _assert_matches_pandas(g_pl.compute_igraph(alg, directed=directed), g_pd.compute_igraph(alg, directed=directed))


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_layout_igraph_on_polars_matches_pandas(lazy: bool) -> None:
    g_pd, g_pl = _graphs(lazy)
    out = g_pl.layout_igraph("circle")
    _assert_matches_pandas(out, g_pd.layout_igraph("circle"))
    assert out._point_x == "x" and out._point_y == "y"


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
def test_compute_igraph_integer_ids_keep_polars_dtype() -> None:
    e = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]}, schema={"s": pl.Int32, "d": pl.Int32})
    n = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.Int32})
    out = graphistry.edges(e, "s", "d").nodes(n, "id").compute_igraph("pagerank")
    assert out._nodes.schema["id"] == pl.Int32
    assert out._edges.schema["s"] == pl.Int32 and out._edges.schema["d"] == pl.Int32
    assert out._nodes["id"].to_list() == [0, 1, 2]


@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize("method, arg", [("compute_cugraph", "pagerank"), ("layout_cugraph", "force_atlas2")])
def test_cugraph_on_polars_matches_pandas(lazy: bool, method: str, arg: str) -> None:
    g_pd, g_pl = _graphs(lazy)
    out_pl = getattr(g_pl, method)(arg)
    out_ref = getattr(g_pd, method)(arg)  # pandas input returns cuDF frames today
    assert isinstance(out_pl._nodes, pl.DataFrame)
    ref_nodes = out_ref._nodes.to_pandas().sort_values("id").reset_index(drop=True)
    got_nodes = out_pl._nodes.to_pandas().sort_values("id").reset_index(drop=True)
    assert list(got_nodes.columns) == list(ref_nodes.columns)
    assert got_nodes["id"].tolist() == ref_nodes["id"].tolist()
    if method == "compute_cugraph":
        pd.testing.assert_series_equal(got_nodes["pagerank"], ref_nodes["pagerank"], check_dtype=False, atol=1e-6)
