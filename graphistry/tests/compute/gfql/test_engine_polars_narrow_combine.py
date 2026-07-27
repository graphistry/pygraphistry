"""The polars chain combine must be proportional to the TRAVERSAL RESULT, not to the graph.

Two graph-sized terms used to sit inside a combine that answers with a handful of rows:

1. ``_combine_edges`` ran the prev/next endpoint gates for EVERY step, including the node steps
   whose edge frame is ``g._edges.clear()`` — zero rows. The eager code skipped those; Track B
   lazified the step frames and ``.lazy()`` erased the height, so the skip silently went dead.
   The cost lands on the side that is NOT empty: for the first step ``prev_nodes`` is the whole
   node table, and polars builds the hash table on that side before discovering the probe side
   has no rows (measured: 6.99 ms per such join at N=2M).
2. The node rows were materialized in TWO passes over the node table — once for the ids the
   steps kept, once more for the edge endpoints the first pass missed — then concatenated.

These tests pin the BOUNDARY, not a wall clock (which goes flaky on a shared host):
  * the node universe must not enter the EDGE plan at all when the chain starts with a node step
    (test 1 exercises ``_combine_edges`` directly with a marker column no step frame carries), and
  * the node universe must be read AT MOST ONCE by the whole node plan (test 2).
Both are backed by semantics: pandas is the oracle for a differential matrix over multi-step,
undirected, multi-hop, fixed-point, aliased and ``rows(...)``-projected shapes, plus explicit
row-ORDER, endpoint-materialization and duplicate-node-id pins that the refactor could break
without changing any id set.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, rows

pl = pytest.importorskip("polars")

from graphistry.compute.gfql.lazy.engine.polars import chain as chain_mod  # noqa: E402
from graphistry.tests.compute.gfql.polars_test_utils import (  # noqa: E402
    graph_sig, to_pandas_any)


# --------------------------------------------------------------------------- fixtures

def _clean_frames():
    nodes = pd.DataFrame({"key": [0, 1, 2, 3, 4, 5],
                          "id": ["a", "b", "c", "d", "e", "f"],
                          "grp": [1, 2, 2, 1, 2, 1]})
    edges = pd.DataFrame({"s": [0, 0, 1, 2, 3, 4],
                          "d": [1, 2, 3, 3, 4, 5],
                          "type": ["K", "K", "L", "K", "K", "K"]})
    return nodes, edges


def _dup_key_frames():
    """The same node key twice — the case the combine's trailing unique() exists for."""
    nodes, edges = _clean_frames()
    nodes = pd.concat([nodes, nodes[nodes["key"].isin([1, 3])]], ignore_index=True)
    return nodes, edges


def _dangling_frames():
    """An edge whose destination is absent from the node table: the endpoint gate is NOT vacuous."""
    nodes, edges = _clean_frames()
    edges = pd.concat([edges, pd.DataFrame({"s": [0], "d": [999], "type": ["K"]})],
                      ignore_index=True)
    return nodes, edges


def _null_id_frames():
    nodes, edges = _clean_frames()
    nodes.loc[2, "id"] = None
    return nodes, edges


def _unsorted_frames():
    """Node/edge frames whose row order is NOT id order — the order the output must restore."""
    nodes, edges = _clean_frames()
    return (nodes.iloc[[4, 0, 5, 2, 1, 3]].reset_index(drop=True),
            edges.iloc[[3, 5, 0, 4, 1, 2]].reset_index(drop=True))


BUILDERS = {
    "clean": _clean_frames,
    "dup_keys": _dup_key_frames,
    "dangling": _dangling_frames,
    "null_ids": _null_id_frames,
    "unsorted": _unsorted_frames,
}

SHAPES = {
    "node_only": [n({"id": "a"}, name="m")],
    "fwd_typed": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p")],
    "rev_typed": [n({"id": "d"}, name="m"), e_reverse({"type": "K"}, name="r"), n(name="p")],
    "undirected": [n({"id": "a"}, name="m"), e_undirected({"type": "K"}, name="r"), n(name="p")],
    "multi_step": [n({"id": "a"}, name="m"), e_forward(name="r1"), n(name="mid"),
                   e_forward(name="r2"), n(name="p")],
    "multi_hop": [n({"id": "a"}, name="m"), e_forward(hops=2, name="r"), n(name="p")],
    "fixed_point": [n({"id": "a"}, name="m"), e_forward(to_fixed_point=True, name="r"),
                    n(name="p")],
    "mixed_single_multi": [n({"id": "a"}), e_forward(hops=2), n(), e_forward(), n(name="p")],
    "no_match": [n({"id": "zzz"}, name="m"), e_forward(name="r"), n(name="p")],
    "trailing_node_filter": [n({"grp": 1}, name="m"), e_forward(name="r"), n({"grp": 2}, name="p")],
}

ROWS_SHAPES = {
    "rows_nodes": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p"),
                   rows(table="nodes", source="p")],
    "rows_edges": [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p"),
                   rows(table="edges", source="r")],
}


def _pair(nodes_pd, edges_pd):
    g_pd = graphistry.edges(edges_pd, "s", "d").nodes(nodes_pd, "key")
    g_pl = graphistry.edges(pl.from_pandas(edges_pd), "s", "d").nodes(
        pl.from_pandas(nodes_pd), "key")
    return g_pd, g_pl


# --------------------------------------------------------------------------- structure

def _shim_step(nodes_df, edges_df):
    """A combine step built the way the chain builds one (through the eager->lazy shim, which is
    where the row count is captured), without standing up a whole Plottable."""
    g = graphistry.edges(edges_df, "s", "d", edge="eid").nodes(nodes_df, "key")
    return chain_mod._LazyShim.step(g)


def test_empty_edge_step_never_reaches_the_node_universe():
    """A step with zero edges contributes zero ids, so its endpoint gates must not be planned.

    The universe frame carries a marker column NO step frame has, so its presence anywhere in
    the optimized edge plan means the empty step's ``prev_nodes = g._nodes`` gate was built —
    the graph-sized hash build this change removes.
    """
    universe_nodes = pl.DataFrame({"key": [0, 1, 2, 3], "probe_universe_marker": [9, 9, 9, 9]})
    universe_edges = pl.DataFrame({"eid": [0, 1, 2], "s": [0, 1, 2], "d": [1, 2, 3]})
    step_nodes = pl.DataFrame({"key": [0, 1]})

    g_lz = chain_mod._LazyShim(universe_nodes.lazy(), universe_edges.lazy(),
                               "key", "s", "d", "eid")
    node_step = _shim_step(step_nodes, universe_edges.clear())      # ASTNode -> cleared edges
    edge_step = _shim_step(step_nodes, universe_edges.head(1))      # ASTEdge -> one real edge
    steps = [(n(), node_step), (e_forward(), edge_step), (n(), node_step)]

    out = chain_mod._combine_edges(g_lz, steps, steps)
    plan = out.explain(optimized=True)
    assert "probe_universe_marker" not in plan, (
        "the empty node step's endpoint gate was planned against the whole node table:\n" + plan)
    # ...and the combine still returns the right edge, so the skip is not a silent drop.
    assert out.collect().sort("eid")["eid"].to_list() == [0]


def test_a_step_of_unknown_height_is_planned_not_dropped():
    """The skip must key on KNOWN-empty, never on 'not known to be non-empty'.

    ``_LazyShim.step`` can only record the height when the step frame is still eager; a frame
    that arrives already lazy records nothing. Treating that as empty would silently drop real
    edges from the result — the failure mode a cardinality shortcut has to be safe against.
    """
    edges = pl.DataFrame({"eid": [0, 1, 2], "s": [0, 1, 2], "d": [1, 2, 3]})
    assert chain_mod._known_empty(edges) is False
    assert chain_mod._known_empty(edges.clear()) is True
    assert chain_mod._known_empty(edges.lazy()) is None, "a lazy frame cannot report a height"
    assert chain_mod._known_empty(None) is None

    g_lz = chain_mod._LazyShim(pl.DataFrame({"key": [0, 1, 2, 3]}).lazy(), edges.lazy(),
                               "key", "s", "d", "eid")
    step_nodes = pl.DataFrame({"key": [0, 1]})
    unknown = chain_mod._LazyShim(step_nodes.lazy(), edges.head(1).lazy(),
                                  None, None, None, None, edges_empty=None)
    steps = [(n(), _shim_step(step_nodes, edges.clear())), (e_forward(), unknown)]
    out = chain_mod._combine_edges(g_lz, steps, steps)
    assert out.collect()["eid"].to_list() == [0], \
        "a step with an unrecorded height was dropped from the edge union"


def _node_universe_scans(plan: str) -> int:
    """How many times the plan reads the chain's node-universe frame.

    The universe is the ONLY frame carrying the synthetic node-order column (the chain attaches
    it with ``with_row_index`` to restore input order), so counting the ``DF [...]`` headers that
    mention it counts reads of the node table specifically — step frames are separate materialized
    frames and never carry it.
    """
    return sum(1 for line in plan.splitlines()
               if line.strip().startswith("DF [") and "norder" in line)


def test_node_rows_are_materialized_in_one_pass_over_the_node_table():
    """The output node set is (step ids) UNION (surviving edge endpoints). Unioning the two ID
    sides first means one scan of the node table; materializing the step rows and then fetching
    the endpoint rows the first pass missed means two."""
    _, g_pl = _pair(*_clean_frames())
    captured = []
    import graphistry.compute.gfql.lazy as lazy_mod
    orig = lazy_mod.collect_all

    def spy(frames, *a, **k):
        captured.append(list(frames))
        return orig(frames, *a, **k)

    lazy_mod.collect_all = spy
    try:
        g_pl.chain([n({"id": "a"}, name="m"), e_forward(name="r"), n(name="p")], engine="polars")
    finally:
        lazy_mod.collect_all = orig

    assert captured, "the chain did not go through the collect-once combine"
    plans = [f.explain(optimized=True) for f in captured[-1]]
    worst = max(_node_universe_scans(p) for p in plans)
    assert worst <= 1, (
        f"the node table is read {worst}x for one traversal result; "
        "the combine should union the id sides and scan it once")


# --------------------------------------------------------------------------- semantics

@pytest.mark.parametrize("frames", sorted(BUILDERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_parity_with_pandas_oracle(frames, shape):
    g_pd, g_pl = _pair(*BUILDERS[frames]())
    chain = list(SHAPES[shape])
    try:
        expected = graph_sig(g_pd.chain(chain, engine="pandas"))
    except Exception as ex:  # a broken oracle is not a polars failure
        pytest.skip(f"pandas oracle raised {type(ex).__name__}")
    try:
        got = graph_sig(g_pl.chain(chain, engine="polars"))
    except NotImplementedError:
        pytest.skip("shape declined by the native polars chain")
    assert expected == got, f"polars diverged from the pandas oracle [{frames}/{shape}]"


@pytest.mark.parametrize("frames", sorted(BUILDERS))
@pytest.mark.parametrize("shape", sorted(ROWS_SHAPES))
def test_parity_with_pandas_oracle_for_rows_projections(frames, shape):
    """``rows(...)`` reads the combine's OUTPUT ORDER (its slicing is positional), so a combine
    that returns the right rows in the wrong order shows up here and nowhere else."""
    g_pd, g_pl = _pair(*BUILDERS[frames]())
    query = list(ROWS_SHAPES[shape])
    try:
        expected = g_pd.gfql(query, engine="pandas")
    except Exception as ex:
        pytest.skip(f"pandas oracle raised {type(ex).__name__}")
    try:
        got = g_pl.gfql(query, engine="polars")
    except NotImplementedError:
        pytest.skip("shape declined by the native polars chain")
    # Column ORDER, and whether the engine's own synthetic `__gfql_*` bookkeeping column
    # survives into the projection, already differ between the two row pipelines (pre-existing,
    # and not something the combine controls). Compare on a canonical column order over the
    # USER columns — ROW order is what this test is for and it is compared as-is.
    def _user_cols(df):
        df = to_pandas_any(df).reset_index(drop=True)
        return df[[c for c in sorted(df.columns) if not str(c).startswith("__gfql_")]]

    exp, act = _user_cols(expected._nodes), _user_cols(got._nodes)
    assert list(exp.columns) == list(act.columns), f"[{frames}/{shape}] column set diverged"
    pd.testing.assert_frame_equal(exp, act, check_dtype=False)


@pytest.mark.parametrize("frames", sorted(BUILDERS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_output_row_order_follows_the_input_frame_order(frames, shape):
    """The combine explicitly sorts back to input-frame order. Row SETS matching is not enough:
    ``graph_sig`` sorts rows, so an order regression would pass every parity test above."""
    _, g_pl = _pair(*BUILDERS[frames]())
    try:
        out = g_pl.chain(list(SHAPES[shape]), engine="polars")
    except NotImplementedError:
        pytest.skip("shape declined by the native polars chain")

    def _first_positions(keys):
        """key -> FIRST position in the input frame (duplicate node ids collapse to their first
        row, so first-occurrence is the position the output should be sorted by)."""
        pos: dict = {}
        for i, k in enumerate(keys):
            pos.setdefault(k, i)
        return pos

    node_pos_of = _first_positions(g_pl._nodes["key"].to_list())
    node_pos = [node_pos_of[k] for k in out._nodes["key"].to_list()]
    assert node_pos == sorted(node_pos), f"[{frames}/{shape}] node rows came back out of order"
    # edges have no user id column here; compare (s, d) positions against the input edge frame
    edge_pos_of = _first_positions(list(zip(g_pl._edges["s"].to_list(),
                                            g_pl._edges["d"].to_list())))
    edge_pos = [edge_pos_of[e] for e in zip(out._edges["s"].to_list(),
                                            out._edges["d"].to_list())]
    assert edge_pos == sorted(edge_pos), f"[{frames}/{shape}] edge rows came back out of order"


def test_endpoint_only_nodes_are_materialized_with_their_attributes():
    """A node reached ONLY as an endpoint of a surviving edge must still come back, with its real
    columns — the safety net the second node-table pass used to provide, now folded into the id
    union. Driven at the helper, deliberately: across 6300 generated chain executions I could not
    build a graph/shape where the step ids MISS an endpoint, so an end-to-end test of this would
    pass with the endpoint side removed entirely (verified — it does). The helper takes the two id
    sides as arguments, so here the case can be constructed.
    """
    all_nodes = pl.DataFrame({"key": [0, 1, 2], "grp": [7, 8, 9]}).lazy()
    step_ids = pl.DataFrame({"key": [0]}).lazy()             # the steps kept node 0 only
    endpoints = pl.DataFrame({"key": [0, 1]}).lazy()         # a surviving edge 0 -> 1
    got = chain_mod._materialize_node_rows(all_nodes, step_ids, endpoints, "key").collect()
    # sorted: a semi-join does NOT preserve left-frame order, which is why the chain restores
    # it with an explicit sort afterwards. Row identity is what this test is about.
    assert got.sort("key").rows() == [(0, 7), (1, 8)], \
        "the endpoint-only node was dropped or lost its attributes"


def test_materialize_node_rows_dedups_rows_but_not_key_sides():
    """The two dedups in the helper are different and only one is needed.

    Duplicates on either ID side are inert (both are ``how="semi"`` key sides), while a node
    table carrying the same id twice must still collapse to ONE row — those rows go on to feed
    ``how="left"`` alias joins, where a duplicate key multiplies every matching row.
    """
    all_nodes = pl.DataFrame({"key": [0, 1, 1, 2], "tag": ["x", "one", "one", "z"]}).lazy()
    step_ids = pl.DataFrame({"key": [1, 1, 1]}).lazy()
    endpoints = pl.DataFrame({"key": [1, 2, 2]}).lazy()
    got = chain_mod._materialize_node_rows(all_nodes, step_ids, endpoints, "key").collect()
    assert got.sort("key").rows() == [(1, "one"), (2, "z")], \
        "duplicate keys multiplied rows, or the wrong rows came back"


def test_duplicate_node_ids_still_collapse_to_one_row():
    """The combine's trailing ``unique(subset=[node])`` is load-bearing: the node rows feed
    ``how="left"`` alias joins downstream, where a duplicated key multiplies rows.

    WHICH duplicate survives is left to the pandas oracle rather than asserted directly — the
    semi-join feeding the dedup does not preserve node-frame order, so 'the first row' is not a
    property this engine guarantees on its own (it is stable in practice, and unchanged by this
    change: 400 dup-key combos A/B, identical full frames).
    """
    nodes = pd.DataFrame({"key": [0, 1, 1, 2], "id": ["a", "b", "b", "c"],
                          "tag": ["x", "dup", "dup", "z"]})
    edges = pd.DataFrame({"s": [0], "d": [1], "type": ["K"]})
    g_pd, g_pl = _pair(nodes, edges)
    chain = [n({"id": "a"}, name="m"), e_forward(name="r"), n(name="p")]
    out = g_pl.chain(chain, engine="polars")
    got = to_pandas_any(out._nodes)
    assert (got["key"] == 1).sum() == 1, "duplicate node ids multiplied the output"
    assert graph_sig(g_pd.chain(chain, engine="pandas")) == graph_sig(out)


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
def test_output_row_order_survives_a_frame_big_enough_to_parallelize(streaming):
    """Order at fixture scale proves little: polars' joins happen to come back in left order on a
    handful of rows and only reorder once the hash join actually runs in parallel. Use a frame
    large enough to reorder, shuffled so input order is not id order, and pin that the combine's
    explicit sorts put both output frames back into input-frame order.

    Parametrized over the collect engine because the IN-MEMORY engine hides the edge sort: with
    `final_edges.sort(EORD)` deleted, in-memory still returns EORD-ordered rows at every size
    probed, so an in-memory-only test would call that sort dead code. Under STREAMING it does
    not, and a trailing rows(limit=)/skip would then slice the wrong rows. Both engines here,
    so neither sort can be removed on the strength of the other's silence."""
    from graphistry.compute.gfql.lazy import set_cpu_streaming
    size = 60_000
    rng = list(range(size))
    order = rng[1::2] + rng[0::2][::-1]                       # deterministic shuffle
    nodes = pd.DataFrame({"key": order, "grp": [k % 3 for k in order]})
    edges = pd.DataFrame({"s": [(k * 7) % size for k in order],
                          "d": [(k * 13 + 1) % size for k in order],
                          "type": ["K" if k % 2 else "L" for k in order]})
    _, g_pl = _pair(nodes, edges)
    set_cpu_streaming(streaming)
    try:
        out = g_pl.chain([n({"grp": 1}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p")],
                         engine="polars")
    finally:
        set_cpu_streaming(None)
    assert out._nodes.height > 1000 and out._edges.height > 1000, "fixture stopped being big"

    node_rank = {k: i for i, k in enumerate(nodes["key"].tolist())}
    node_pos = [node_rank[k] for k in out._nodes["key"].to_list()]
    assert node_pos == sorted(node_pos), "node rows came back out of input-frame order"

    edge_rank: dict = {}
    for i, e in enumerate(zip(edges["s"].tolist(), edges["d"].tolist())):
        edge_rank.setdefault(e, i)
    edge_pos = [edge_rank[e] for e in zip(out._edges["s"].to_list(), out._edges["d"].to_list())]
    assert edge_pos == sorted(edge_pos), "edge rows came back out of input-frame order"


def test_endpoint_gate_still_excludes_a_dangling_edge():
    """Non-vacuity: the gates that survive must still do their job. Skipping the EMPTY steps
    must not be mistaken for skipping the gates."""
    g_pd, g_pl = _pair(*_dangling_frames())
    chain = [n({"id": "a"}, name="m"), e_forward({"type": "K"}, name="r"), n(name="p")]
    out = g_pl.chain(chain, engine="polars")
    assert 999 not in out._nodes["key"].to_list(), "dangling endpoint leaked into the nodes"
    assert graph_sig(g_pd.chain(chain, engine="pandas")) == graph_sig(out)


def test_chain_with_no_edge_steps_returns_no_edges():
    """Every step empty -> the union has no frames at all. That branch must still produce an
    empty, correctly-typed edge frame rather than the whole edge table."""
    g_pd, g_pl = _pair(*_clean_frames())
    chain = [n({"grp": 1}, name="m")]
    out = g_pl.chain(chain, engine="polars")
    assert out._edges.height == 0
    assert graph_sig(g_pd.chain(chain, engine="pandas")) == graph_sig(out)
