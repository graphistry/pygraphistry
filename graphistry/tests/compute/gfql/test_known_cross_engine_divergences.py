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
def test_1808_dangling_destination_pandas_matches_polars():
    """#1808 (fixed by #1888 endpoint closure): an edge whose DESTINATION id has
    no node row was matched by pandas/cuDF and dropped by polars (a dangling
    SOURCE was dropped by all). polars was the Cypher-correct side --
    `(a)-[]->(b)` binds b to a NODE. The pandas/cuDF endpoint gate is now
    symmetric: both dangling sides are dropped on every engine."""
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
def test_1888_unconstrained_chain_respects_endpoint_closure():
    """#1888 F-01 (fixed): with nodes bound, a pattern edge matches only if BOTH
    endpoints resolve to node rows (the converged contract). The polars
    unconstrained chain fast path now applies the node-universe semi-join on
    both endpoints, so attaching a policy no longer flips the value."""
    from graphistry.compute.ast import n, e_forward

    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0, 1, 5], "d": [1, 2, 6]})
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    out = g_pl.gfql([n(), e_forward(), n()], engine="polars")._edges
    n_edges = len(out.to_pandas() if hasattr(out, "to_pandas") else out)
    assert n_edges == 1, f"dangling edges matched: {n_edges} (contract: endpoint closure)"


@polars_only
def test_1888_projection_and_count_agree_on_dangling():
    """#1888 F-02 (fixed): within ONE engine, the projection and count(*) of the
    same pattern must agree; with endpoint closure both serve the closed answer.
    The projection is pinned via property columns (`RETURN a.v, b.v`) -- the
    whole-entity multi-alias form (`RETURN a, b`) is an honest NIE on the polars
    projector (separate gap, not an endpoint-gate divergence). Pinned on BOTH
    engines: #1808's pandas asymmetry made pandas project the dangling row."""
    nodes = pd.DataFrame({"id": [0, 1], "v": [10, 20]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
    g_pd = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    g_pl = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    for engine, g in [("pandas", g_pd), ("polars", g_pl)]:
        rows = g.gfql("MATCH (a)-[r]->(b) RETURN a.v AS av, b.v AS bv", engine=engine)._nodes
        n_rows = len(rows.to_pandas() if hasattr(rows, "to_pandas") else rows)
        cnt = g.gfql("MATCH (a)-[]->(b) RETURN count(*) AS c", engine=engine)._nodes
        c = int((cnt.to_pandas() if hasattr(cnt, "to_pandas") else cnt)["c"].iloc[0])
        assert n_rows == c == 1, f"{engine}: projection {n_rows} vs count {c} (closed answer: 1)"


# --- #1888 endpoint-closure invariant pins -------------------------------------------------
# THE CONTRACT (one rule, all surfaces/engines/policy states): when a node table is
# bound, a pattern edge matches only if BOTH endpoints resolve to node rows.
# Fixture (findings round-002 agent-01): nodes id=[0,1]; edges 0->1 valid, 1->2
# dangling-destination, 5->6 both-endpoints-missing. Closed answer everywhere: 1 edge.

_1888_NODES = pd.DataFrame({"id": [0, 1], "v": [10, 20]})
_1888_EDGES = pd.DataFrame({"s": [0, 1, 5], "d": [1, 2, 6]})
_1888_POLICY = {"preload": (lambda ctx: None)}  # disables fast paths / flips serving lane


def _1888_g(engine):
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(_1888_NODES), "id").edges(
            pl.from_pandas(_1888_EDGES), "s", "d")
    return graphistry.nodes(_1888_NODES, "id").edges(_1888_EDGES, "s", "d")


def _height(df):
    return len(df.to_pandas() if hasattr(df, "to_pandas") else df)


@pytest.mark.parametrize("engine", ["pandas", pytest.param("polars", marks=polars_only)])
def test_1888_chain_closed_answer_policy_invariant(engine):
    """Chain surface: 1 edge, endpoints ⊆ node table, policy on/off identical."""
    from graphistry.compute.ast import n, e_forward

    g = _1888_g(engine)
    ops = [n(), e_forward(), n()]
    out = g.gfql(ops, engine=engine)
    out_policy = g.gfql(ops, engine=engine, policy=_1888_POLICY)
    assert _height(out._edges) == 1
    assert _height(out_policy._edges) == 1
    nodes_df = out._nodes.to_pandas() if hasattr(out._nodes, "to_pandas") else out._nodes
    assert set(nodes_df["id"]) <= {0, 1}


@pytest.mark.parametrize("engine", ["pandas", pytest.param("polars", marks=polars_only)])
def test_1888_cypher_rows_and_count_closed_answer(engine):
    """Cypher surface: property projection rows == count(*) == 1 on the closed answer."""
    g = _1888_g(engine)
    rows = g.gfql("MATCH (a)-[r]->(b) RETURN a.v AS av, b.v AS bv", engine=engine)._nodes
    cnt = g.gfql("MATCH (a)-[]->(b) RETURN count(*) AS c", engine=engine)._nodes
    c = int((cnt.to_pandas() if hasattr(cnt, "to_pandas") else cnt)["c"].iloc[0])
    assert _height(rows) == 1
    assert c == 1


@pytest.mark.parametrize("engine", ["pandas", pytest.param("polars", marks=polars_only)])
def test_1888_hop_closed_answer_no_phantom_nodes(engine):
    """Hop surface (F-03): 1 edge, node set == the real node rows reached — no
    synthesized NaN-attribute phantom rows, no int64 -> float64 upcast."""
    g = _1888_g(engine)
    out = g.hop()
    assert _height(out._edges) == 1
    nodes_df = out._nodes.to_pandas() if hasattr(out._nodes, "to_pandas") else out._nodes
    assert sorted(nodes_df["id"].tolist()) == [0, 1]
    assert not nodes_df["v"].isna().any()
    assert str(nodes_df["v"].dtype) == "int64"


def test_optional_match_aggregate_keeps_unmatched_seeds_polars_and_pandas():
    """Cypher: OPTIONAL MATCH keeps unmatched rows with NULLs, so
    `RETURN p.name, count(x)` must include zero-count seeds. Fixed by routing
    the single-node-seed shape onto the connected optional-match left-join
    lowering (#1891); formerly the aggregate projection bypassed the
    validator gate and inner-joined on BOTH engines."""
    nodes = pd.DataFrame({"id": [0, 1], "name": ["alice", "bob"], "t": ["P", "P"]})
    edges = pd.DataFrame({"s": [0], "d": [1]})
    q = ("MATCH (p {t:'P'}) OPTIONAL MATCH (p)-[]->(x) "
         "RETURN p.name AS name, count(x) AS c ORDER BY name")
    out = (graphistry.nodes(nodes, "id").edges(edges, "s", "d")
           .gfql(q, engine="pandas")._nodes.to_dict("records"))
    assert out == [{"name": "alice", "c": 1}, {"name": "bob", "c": 0}], out


def test_with_carried_scalar_projects_next_to_aggregate():
    """The carry survives filtering (WHERE q.score > s answers) and plain
    projection; RETURN s, count(q) used to die with a raw KeyError on pandas
    (round-002 BUG-3) because the grouped-aggregate fast path let the carried
    output name collide with the edge source column ('s') and suffixed it
    away. The fast path now declines on that collision and the row pipeline
    serves it."""
    nodes = pd.DataFrame({"id": [0, 1], "kind": ["person", "person"], "score": [5, 9]})
    edges = pd.DataFrame({"s": [0], "d": [1], "t": ["KNOWS"]})
    q = ("MATCH (p {kind:'person'}) WITH p, p.score AS s "
         "MATCH (p)-[{t:'KNOWS'}]->(q) RETURN s, count(q) AS c")
    out = (graphistry.nodes(nodes, "id").edges(edges, "s", "d")
           .gfql(q, engine="pandas")._nodes.to_dict("records"))
    assert out == [{"s": 5, "c": 1}], out


@pytest.mark.xfail(strict=True, reason="round-002 BUG-4: ungrouped sum over an empty match is NULL; openCypher says 0")
def test_sum_over_empty_match_is_zero():
    nodes = pd.DataFrame({"id": [0], "v": [1], "t": ["A"]})
    edges = pd.DataFrame({"s": [], "d": []})
    q = "MATCH (a {t:'ZZZ'}) RETURN sum(a.v) AS s"
    out = (graphistry.nodes(nodes, "id").edges(edges, "s", "d")
           .gfql(q, engine="pandas")._nodes.to_dict("records"))
    assert out == [{"s": 0}], out


cudf_only = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("cudf") is None,
    reason="cudf lane requires a GPU box (--gpus all)")


@cudf_only
@pytest.mark.xfail(strict=True, reason="cuDF drops the sliced edge's source node row under an output hop window; pandas backfills it")
def test_output_hop_window_backfills_the_source_node_row_on_cudf():
    """Found by the #1895 hop.py boundary amplification, PRE-EXISTING (reproduces identically
    at that branch's merge-base, so #1888/#1895 did not cause it).

    With an output hop window, pandas returns edge (0,1) AND both endpoint node rows; cuDF
    returns the edge with only node 1 -- the source row is never backfilled, leaving an edge
    whose endpoint has no node row. Every engine should satisfy endpoint closure on its OUTPUT.
    Flipping this xfail = the cuDF epilogue backfilling like the pandas one.
    """
    import cudf

    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "v": [10, 20, 30, 40]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    seed = pd.DataFrame({"id": [0]})

    def _hop(mk, engine):
        g = graphistry.nodes(mk(nodes), "id").edges(mk(edges), "s", "d")
        out = g.hop(nodes=mk(seed), max_hops=4, output_max_hops=1,
                    direction="forward", engine=engine)
        to_pd = (lambda d: d.to_pandas()) if engine == "cudf" else (lambda d: d)
        return set(to_pd(out._edges)["s"]), set(to_pd(out._nodes)["id"])

    src_pandas, ids_pandas = _hop(lambda d: d, "pandas")
    src_cudf, ids_cudf = _hop(cudf.from_pandas, "cudf")
    assert src_pandas <= ids_pandas  # pandas is the correct side
    assert src_cudf <= ids_cudf, "cuDF left an edge source with no node row"
