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


@polars_only
@pytest.mark.xfail(strict=True, reason="#1824: fast paths serve CPU work under engine='polars-gpu'")
def test_1824_polars_gpu_fast_path_serve_is_gpu_or_decline():
    """#1824: the fast paths run before the chain route establishes the GPU
    execution target, and their eager arms (dense kernel, eager twins) never
    collect -- so engine='polars-gpu' serves CPU work labeled as GPU. On a
    GPU-less box this query must NOT be answered by a fast path: the honest
    outcomes are a decline (chain route raises its install error) or a
    GPU-executed serve. dgx measured 86 polars-gpu tests green only via the
    mislabel, so the fix (per-arm GPU-or-decline) is next-cycle scoped; the
    collect routing + NIE-decline plumbing already landed with this pin.
    """
    from graphistry.tests.compute.gfql.engagement import fast_path_decisions

    nodes = pl.DataFrame({"id": [0, 1, 2], "kind": ["P"] * 3})
    edges = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0], "rel": ["F"] * 3})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = ("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})-[{rel:'F'}]->(c {kind:'P'}) "
         "RETURN count(*) AS n")
    try:
        decisions = fast_path_decisions(g, q, engine="polars-gpu")
    except (NotImplementedError, ImportError):
        return  # honest: GPU stack absent and nothing served on CPU
    assert not any(decisions.values()), (
        f"fast path served CPU work under polars-gpu: {decisions}")


@polars_only
@pytest.mark.xfail(strict=True, reason="#1888: no endpoint-closure contract; polars chain fast path skips the node semi-join")
def test_1888_unconstrained_chain_respects_endpoint_closure():
    """#1888 F-01: with nodes bound, a pattern edge must match only if BOTH
    endpoints resolve to node rows (the converged contract). The polars
    unconstrained chain fast path returns edges whose endpoints do not exist,
    and attaching a policy flips the value."""
    from graphistry.compute.ast import n, e_forward

    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0, 1, 5], "d": [1, 2, 6]})
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    out = g_pl.gfql([n(), e_forward(), n()], engine="polars")._edges
    n_edges = len(out.to_pandas() if hasattr(out, "to_pandas") else out)
    assert n_edges == 1, f"dangling edges matched: {n_edges} (contract: endpoint closure)"


@polars_only
@pytest.mark.xfail(strict=True, reason="#1888: polars entity projection and count(*) disagree on dangling edges")
def test_1888_projection_and_count_agree_on_dangling():
    """#1888 F-02: within ONE engine, the entity projection (RETURN a, b) and
    count(*) of the same pattern must agree. On dangling-dst graphs polars
    projects 2 rows but counts 1 (projection and aggregate lower through
    different pipelines with separate endpoint gates)."""
    nodes = pd.DataFrame({"id": [0, 1], "v": [10, 20]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    rows = g_pl.gfql("MATCH (a)-[r]->(b) RETURN a, b", engine="polars")._nodes
    n_rows = len(rows.to_pandas() if hasattr(rows, "to_pandas") else rows)
    cnt = g_pl.gfql("MATCH (a)-[]->(b) RETURN count(*) AS c", engine="polars")._nodes
    c = int((cnt.to_pandas() if hasattr(cnt, "to_pandas") else cnt)["c"].iloc[0])
    assert n_rows == c, f"entity projection {n_rows} vs count {c}"
