"""Known cross-engine divergences on DEGENERATE inputs, pinned as strict xfails.

Each of these is a real, pre-existing divergence (verified to reproduce on
masters predating the current release) on input the data model treats as
malformed -- dangling edge endpoints, duplicate node ids. They are pinned
executable so (a) the release notes can point at exactly what diverges, and
(b) the fix PR flips a strict xfail instead of rediscovering the shape.
"""
import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

polars_only = pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")


@polars_only
@pytest.mark.xfail(strict=True, reason="#1808: pandas endpoint gate is asymmetric")
def test_1808_dangling_destination_pandas_matches_polars():
    """#1808: an edge whose DESTINATION id has no node row is matched by
    pandas/cuDF and dropped by polars (a dangling SOURCE is dropped by all).
    polars is the Cypher-correct side -- `(a)-[]->(b)` binds b to a NODE.
    Flipping this xfail = making the pandas/cuDF endpoint gate symmetric and
    re-baselining parity fixtures that encoded the old count."""
    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 9]})  # 1 -> 9 dangles
    q = "MATCH (a)-[]->(b) RETURN count(*) AS c"
    c_pandas = (graphistry.nodes(nodes, "id").edges(edges, "s", "d")
                .gfql(q, engine="pandas")._nodes["c"].iloc[0])
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(
        pl.from_pandas(edges), "s", "d")
    c_polars = g_pl.gfql(q, engine="polars")._nodes.to_pandas()["c"].iloc[0]
    assert int(c_polars) == 1  # the Cypher-correct answer, already served
    assert int(c_pandas) == int(c_polars)


@polars_only
@pytest.mark.xfail(strict=True, reason="#1739: HAS_<Label> narrowing missing on polars; pandas fast-path answer is order-dependent")
def test_1739_has_label_aggregate_on_duplicate_ids_converges():
    """#1739: duplicate node id across Tag/Forum rows; the aggregating
    single-MATCH shape serves BOTH engines via the grouped-aggregate fast path,
    where pandas keeps one row per id by frame order (an accident, not the row
    pipeline's label narrowing) and polars keeps both. The converged answer is
    the label-narrowed one. Flipping this xfail = porting the
    _gfql_disambiguate_has_edge_destination_nodes gate (or the #1737 decline)
    to the shared fast path and the generic polars route."""
    nodes = pd.DataFrame({
        "id": [601, 400, 400],
        "label__Post": [True, False, False],
        "label__Tag": [False, True, False],
        "label__Forum": [False, False, True],
        "name": [None, "t4", "f4"],
    })
    edges = pd.DataFrame({"src": [601], "dst": [400], "type": ["HAS_TAG"]})
    q = ("MATCH (post:Post {id: 601})-[:HAS_TAG]->(tag) "
         "RETURN tag.name AS tagName, count(post) AS c")
    out_pd = (graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
              .gfql(q, engine="pandas")._nodes.to_dict("records"))
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(
        pl.from_pandas(edges), "src", "dst")
    out_pl = sorted(g_pl.gfql(q, engine="polars")._nodes.to_pandas().to_dict("records"),
                    key=str)
    narrowed = [{"tagName": "t4", "c": 1}]
    assert out_pd == narrowed
    assert out_pl == narrowed
