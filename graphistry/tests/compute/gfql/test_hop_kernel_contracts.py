"""Direct pins for the hop-kernel helpers #1895 extracted, so their NAMES are load-bearing.

A helper named ``_keep_edges_with_both_endpoints_resolvable`` only carries meaning if
something fails when it stops doing that. Each helper below gets a pin naming its rule, plus
the two behaviour-adjacent changes the #1895 remediation made:

  * ``.implode()`` on the endpoint membership test (the bare ``is_in(<Series>)`` form is
    deprecated in polars 1.x) -- pinned VALUE-IDENTICAL on the shapes where the two forms
    could plausibly differ: nulls among the ids, nulls among the endpoints, and empty.
  * the unconditional trailing de-dup in ``hop()``'s endpoint backfill, which replaced a
    de-dup that used to live only on the else-branch -- pinned on ``to_fixed_point``, the
    only shape where deleting it changes an answer (round-5 mutation audit; every bounded
    hop is de-duped again by the output-window epilogue).
  * the undirected seed hop-label STRIP the ``_ensure_node_hop_col()`` call exists to serve.
  * ``_with_every_lazy_input_collected``: the polars closure kernel calls ``get_column``,
    which a LazyFrame does not have.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_concat
from graphistry.compute.hop import (
    _endpoint_ids_without_node_rows, _reached_node_ids,
)

from .polars_test_utils import edge_pair_set, node_id_set, to_pandas_any

pl = pytest.importorskip("polars")

from graphistry.compute.gfql.lazy.engine.polars.hop_eager import (  # noqa: E402
    _ids_an_endpoint_may_resolve_to, _keep_edges_with_both_endpoints_resolvable,
)

ALL_ENGINES = ["pandas", "polars", "cudf", "polars-gpu"]


def _require_engine(engine: str) -> None:
    if engine == "cudf":
        pytest.importorskip("cudf", reason="cudf engine lane requires a GPU box (--gpus all)")
    if engine == "polars-gpu":
        pytest.importorskip("cudf", reason="polars-gpu lane requires a GPU box (--gpus all)")
        import importlib.util
        if importlib.util.find_spec("cudf_polars") is None:
            pytest.skip("polars-gpu lane requires cudf_polars (RAPIDS 26.02+ image)")


def _frame(pdf: pd.DataFrame, engine: str):
    if engine in ("polars", "polars-gpu"):
        return pl.from_pandas(pdf)
    if engine == "cudf":
        import cudf
        return cudf.from_pandas(pdf)
    return pdf


# --- _keep_edges_with_both_endpoints_resolvable: the name IS the rule -----------------------

_EDGES = pl.DataFrame({"s": [0, 1, 2, 8, 5], "d": [1, 2, 7, 0, 6]},
                      schema={"s": pl.Int64, "d": pl.Int64})


def test_keep_edges_drops_an_edge_when_either_endpoint_is_unresolvable():
    ids = pl.Series("id", [0, 1, 2], dtype=pl.Int64)
    kept = _keep_edges_with_both_endpoints_resolvable(_EDGES, "s", "d", pl.Int64, ids)
    assert set(zip(kept["s"].to_list(), kept["d"].to_list())) == {(0, 1), (1, 2)}


def test_keep_edges_keeps_everything_when_every_id_resolves():
    ids = pl.Series("id", [0, 1, 2, 5, 6, 7, 8], dtype=pl.Int64)
    kept = _keep_edges_with_both_endpoints_resolvable(_EDGES, "s", "d", pl.Int64, ids)
    assert kept.height == _EDGES.height


def test_keep_edges_drops_everything_when_no_id_resolves():
    ids = pl.Series("id", [], dtype=pl.Int64)
    kept = _keep_edges_with_both_endpoints_resolvable(_EDGES, "s", "d", pl.Int64, ids)
    assert kept.height == 0


@pytest.mark.parametrize("label,ids_vals,edge_s,edge_d", [
    ("plain", [0, 1, 2], [0, 1, 2, 8, 5], [1, 2, 7, 0, 6]),
    ("null_among_ids", [0, 1, None], [0, 1, 2, 8, 5], [1, 2, 7, 0, 6]),
    ("null_among_endpoints", [0, 1, 2], [0, None, 2], [1, 2, None]),
    ("empty_ids", [], [0, 1], [1, 2]),
    ("empty_edges", [0, 1], [], []),
])
def test_implode_membership_is_value_identical_to_the_deprecated_bare_form(
    label, ids_vals, edge_s, edge_d
):
    """The switch to ``.implode()`` was a deprecation fix, NOT a semantics change. Nulls are
    where a membership test is most likely to differ, so they are pinned explicitly."""
    import warnings

    ids = pl.Series("id", ids_vals, dtype=pl.Int64)
    edges = pl.DataFrame({"s": edge_s, "d": edge_d}, schema={"s": pl.Int64, "d": pl.Int64})

    new = _keep_edges_with_both_endpoints_resolvable(edges, "s", "d", pl.Int64, ids)
    with warnings.catch_warnings():  # the bare form is the deprecated one under comparison
        warnings.simplefilter("ignore", DeprecationWarning)
        old = edges.filter(pl.col("s").cast(pl.Int64).is_in(ids)
                           & pl.col("d").cast(pl.Int64).is_in(ids))
    assert new.rows() == old.rows(), label


def test_endpoint_membership_does_not_emit_a_polars_deprecation_warning():
    """The bare ``is_in(<Series>)`` form warns on polars 1.x; ``.implode()`` must not."""
    import warnings

    ids = pl.Series("id", [0, 1, 2], dtype=pl.Int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _keep_edges_with_both_endpoints_resolvable(_EDGES, "s", "d", pl.Int64, ids)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == [], [str(w.message) for w in deprecations]


# --- _ids_an_endpoint_may_resolve_to: node table, widened by any target wavefront -----------

def test_endpoint_universe_is_the_node_table_when_there_is_no_target_wavefront():
    nodes = pl.DataFrame({"id": [0, 1, 2]}, schema={"id": pl.Int64})
    got = _ids_an_endpoint_may_resolve_to(nodes, None, "id")
    assert sorted(got.to_list()) == [0, 1, 2]


def test_endpoint_universe_is_widened_by_the_target_wavefront():
    nodes = pl.DataFrame({"id": [0, 1]}, schema={"id": pl.Int64})
    wavefront = pl.DataFrame({"id": [7, 8]}, schema={"id": pl.Int64})
    got = _ids_an_endpoint_may_resolve_to(nodes, wavefront, "id")
    assert sorted(got.to_list()) == [0, 1, 7, 8]


def test_endpoint_universe_ignores_a_wavefront_without_the_node_column():
    nodes = pl.DataFrame({"id": [0, 1]}, schema={"id": pl.Int64})
    wavefront = pl.DataFrame({"other": [7]}, schema={"other": pl.Int64})
    got = _ids_an_endpoint_may_resolve_to(nodes, wavefront, "id")
    assert sorted(got.to_list()) == [0, 1]


# --- _endpoint_ids_without_node_rows / _reached_node_ids ----------------

# _endpoint_ids_without_node_rows is a pandas-IDIOM helper (.rename(columns=), .isin,
# .drop_duplicates) reached only from the pandas/cuDF hop; polars has its own kernel.
PANDAS_IDIOM_ENGINES = ["pandas", "cudf"]


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_no_endpoint_ids_are_unbacked_when_the_node_table_covers_them(engine):
    _require_engine(engine)
    eng = {"pandas": Engine.PANDAS, "cudf": Engine.CUDF}[engine]
    g = (graphistry
         .nodes(_frame(pd.DataFrame({"id": [0, 1, 2], "v": [1, 2, 3]}), engine), "id")
         .edges(_frame(pd.DataFrame({"s": [0, 1], "d": [1, 2]}), engine), "s", "d"))
    assert len(_endpoint_ids_without_node_rows(g, df_concat(eng))) == 0


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_an_unbacked_endpoint_id_is_reported(engine):
    _require_engine(engine)
    eng = {"pandas": Engine.PANDAS, "cudf": Engine.CUDF}[engine]
    g = (graphistry
         .nodes(_frame(pd.DataFrame({"id": [0], "v": [1]}), engine), "id")
         .edges(_frame(pd.DataFrame({"s": [0], "d": [9]}), engine), "s", "d"))
    missing = to_pandas_any(_endpoint_ids_without_node_rows(g, df_concat(eng)))
    assert missing["id"].tolist() == [9]


@pytest.mark.parametrize("matches,expected", [
    (None, set()),
    (pd.DataFrame({"id": []}), set()),
    (pd.DataFrame({"id": [1, 2, 2]}), {1, 2}),
])
def test_reached_node_ids(matches, expected):
    assert _reached_node_ids(matches, "id") == expected


# --- the unconditional trailing de-dup in hop()'s endpoint backfill -------------------------

def _dup_node_graph(engine):
    return (graphistry
            .nodes(_frame(pd.DataFrame({"id": [0, 0, 1], "v": [1, 1, 2]}), engine), "id")
            .edges(_frame(pd.DataFrame({"s": [0], "d": [1]}), engine), "s", "d"))


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_duplicate_node_rows_are_deduped_under_to_fixed_point(engine):
    """to_fixed_point is the ONLY shape where the backfill's trailing de-dup is load-bearing.

    Round-5 mutation audit: deleting it changes nothing on any bounded hop, because
    ``output_max_hops`` defaults to ``max_hops`` and the output-window epilogue de-dups on
    its way out. ``to_fixed_point=True`` leaves max_hops None, that epilogue never runs, and
    the duplicate id rows reach the caller. Oracle: node ids {0,0,1} + edge 0->1, so a
    forward walk keeps rows for 0 and 1 -- ONE row each, whatever the input multiplicity."""
    _require_engine(engine)
    out = _dup_node_graph(engine).hop(to_fixed_point=True, direction="forward", engine=engine)
    assert to_pandas_any(out._nodes)["id"].tolist() == [0, 1]


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_duplicate_node_rows_are_deduped_under_to_fixed_point_seeded(engine):
    """Seeded twin of the above: the seed 0 is the duplicated id, so a surviving duplicate
    would double the SEED row specifically."""
    _require_engine(engine)
    out = _dup_node_graph(engine).hop(
        nodes=_frame(pd.DataFrame({"id": [0]}), engine), to_fixed_point=True,
        direction="forward", engine=engine)
    assert to_pandas_any(out._nodes)["id"].tolist() == [0, 1]


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_duplicate_node_rows_are_deduped_when_no_endpoint_is_missing(engine):
    """CONTROL (passes with the backfill de-dup deleted -- the output-window epilogue de-dups
    this shape). Kept because it pins the OUTPUT property on the no-backfill path."""
    _require_engine(engine)
    out = _dup_node_graph(engine).hop(engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert nodes_pdf["id"].tolist() == sorted(set(nodes_pdf["id"].tolist()))
    assert node_id_set(out) == {0, 1}


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_duplicate_node_rows_are_deduped_when_an_endpoint_is_backfilled(engine):
    """CONTROL, concat side: wavefront mode leaves an endpoint unbacked, so the concat branch
    runs. Also de-duped by the output-window epilogue, so this pins the property, not the site."""
    _require_engine(engine)
    out = _dup_node_graph(engine).hop(
        nodes=_frame(pd.DataFrame({"id": [0]}), engine), hops=1,
        direction="forward", return_as_wave_front=True, engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert len(nodes_pdf["id"].tolist()) == len(set(nodes_pdf["id"].tolist())), (
        f"duplicate node rows survived the backfill: {nodes_pdf['id'].tolist()}")


@pytest.mark.xfail(strict=True, reason=(
    "PRE-EXISTING cross-engine divergence (reproduces at 86013f4, before the #1895 "
    "remediation): the pandas hop de-dups its output node table by id, the polars hop "
    "does not, so duplicate node rows survive on polars. Not introduced here; pinned "
    "executable so the fix flips an xfail."))
@pytest.mark.parametrize("engine", ["polars"])
def test_duplicate_node_rows_are_deduped_on_polars_too(engine):
    out = _dup_node_graph(engine).hop(engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert nodes_pdf["id"].tolist() == sorted(set(nodes_pdf["id"].tolist()))


# --- the undirected seed label STRIP, not just the column it needs -------------------------
#
# #1895 added ``_ensure_node_hop_col()`` above this strip because the strip crashed on cuDF
# when the column was absent. That pins the column EXISTS; these pin what the strip WRITES.
# Fixture: path 0-1-2-3, seeded undirected. An undirected walk re-enters a seed over its own
# departure edge, so every seed would otherwise carry a hop distance; a seed is at distance
# 0 from itself and hop labels are "distance travelled", so a seed's label is NULL.
#
#   seed [0], hops 2 : 1 is one edge away, 2 is two.  {0: NULL, 1: 1, 2: 2}   (3 unreached)
#   seed [1], hops 2 : 0 and 2 are one edge away, 3 is two. {1: NULL, 0: 1, 2: 1, 3: 2}
#   seed [0,3], hops 2: 1 is one from 0, 2 is one from 3.   {0: NULL, 3: NULL, 1: 1, 2: 1}

_SEED_LABEL_ORACLE = [
    ([0], {0: None, 1: 1, 2: 2}),
    ([1], {1: None, 0: 1, 2: 1, 3: 2}),
    ([0, 3], {0: None, 3: None, 1: 1, 2: 1}),
]


def _labelled_undirected_hop(engine, seeds):
    g = (graphistry
         .nodes(_frame(pd.DataFrame({"id": [0, 1, 2, 3], "v": [10, 20, 30, 40]}), engine), "id")
         .edges(_frame(pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]}), engine), "s", "d"))
    return g.hop(nodes=_frame(pd.DataFrame({"id": seeds}), engine), hops=2,
                 direction="undirected", label_node_hops="nh", engine=engine)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("seeds,want", _SEED_LABEL_ORACLE)
def test_undirected_hop_labels_leave_the_seed_unlabelled(engine, seeds, want):
    _require_engine(engine)
    got = to_pandas_any(_labelled_undirected_hop(engine, seeds)._nodes)
    labels = {int(r.id): (None if pd.isna(r.nh) else int(r.nh)) for r in got.itertuples()}
    assert labels == want


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_labelled_undirected_hop_output_still_backs_every_edge_endpoint(engine):
    """The closure contract read on the OUTPUT: every endpoint of a surviving edge has a node
    row. Input here is fully closed, so nothing may be dropped."""
    _require_engine(engine)
    out = _labelled_undirected_hop(engine, [0])
    edges = to_pandas_any(out._edges)
    endpoints = set(edges["s"].tolist()) | set(edges["d"].tolist())
    assert endpoints <= node_id_set(out), (
        f"edge endpoints with no node row: {sorted(endpoints - node_id_set(out))}")


# --- polars hop accepts LazyFrame inputs (_with_every_lazy_input_collected) -----------------

@pytest.mark.parametrize("lazy_nodes,lazy_edges", [(True, False), (False, True), (True, True)])
def test_polars_hop_accepts_lazy_graph_frames(lazy_nodes, lazy_edges):
    """A LazyFrame is an accepted INPUT format, and #1895's endpoint-closure kernel calls
    ``get_column`` on the node frame -- which a LazyFrame does not have. Without the entry
    normalization this raises AttributeError instead of answering.

    Oracle: nodes {0,1,2}, edge 0->1 and 1->2, one forward hop from 0 => edge (0,1), nodes {0,1}.
    """
    nodes = pl.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
    edges = pl.DataFrame({"s": [0, 1], "d": [1, 2]})
    g = (graphistry
         .nodes(nodes.lazy() if lazy_nodes else nodes, "id")
         .edges(edges.lazy() if lazy_edges else edges, "s", "d"))
    out = g.hop(nodes=pl.DataFrame({"id": [0]}), hops=1, direction="forward", engine="polars")
    assert node_id_set(out) == {0, 1}
    assert edge_pair_set(out) == {(0, 1)}


# --- frame-identity cache keys: strong ref + `is`, never id() ------------------------------

def test_row_pipeline_cache_is_keyed_on_a_strong_ref_not_a_recyclable_id():
    """``id()`` alone is not a safe cache key: CPython reuses the address of a freed object, so
    a dead frame's id can validate a cache built for a different frame. The key is the frame
    OBJECT (compared with ``is``), which also pins it alive for as long as it is cached."""
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

    cache_for = RowPipelineMixin._gfql_native_shortest_path_cache
    g = (graphistry.nodes(pd.DataFrame({"id": [0, 1]}), "id")
         .edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d"))
    cache = cache_for(g)
    assert cache is not None
    assert cache["__edges_strong_ref__"] is g._edges

    same = cache_for(g)
    assert same is cache, "same bound frame must reuse the cache"

    rebound = g.edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    fresh = cache_for(rebound)
    assert fresh is not cache, "a DIFFERENT edge frame must not hit the previous cache"
    assert fresh["__edges_strong_ref__"] is rebound._edges


def test_nan_free_frame_id_cache_evicts_on_gc_so_a_recycled_id_cannot_hit():
    """The NaN-probe cache DOES key on ``id()``, so it carries a weakref.finalize that evicts
    the id when the frame dies -- otherwise a recycled address would be a stale hit."""
    import gc

    from graphistry.compute.gfql.lazy.engine.polars import nan_clean

    df = pl.DataFrame({"x": [1.0, 2.0]})
    nan_clean._mark_pl_nan_clean(df)
    frame_id = id(df)
    assert frame_id in nan_clean._nan_free_frame_id_cache

    del df
    gc.collect()
    assert frame_id not in nan_clean._nan_free_frame_id_cache, (
        "a dead frame's id survived in the cache; a recycled id would be a false hit")


# The adjacency-index re-point that chain.py performs across ``with_row_index`` is pinned at
# the level where it is observable -- ``test_index.py::test_rebind_edges_revalidates_after_
# shallow_augmentation`` and ``::test_rebind_edges_drops_index_on_fingerprint_mismatch``.
# No integration pin is added here: removing the chain's re-point call does NOT change any
# observable result or index trace in the shapes reachable today (measured), so an
# integration test would assert nothing.


# --- #1798: tracked undirected hop must not drop self-loop/seed node rows on cuDF -----------
#
# Fixture: nodes 0..3 kind [b,a,a,b]; edges (2,2)x3, (0,3)x2, (3,1), (1,1). Seeded at the
# 'a' nodes {1,2}, one undirected hop touches 5 edge rows -- (3,1), (1,1), (2,2)x3 -- and
# nodes {1,2,3}. Any track_hops arm leaves seeds hop-label NULL, and cuDF's non-Kleene
# NULL | True stayed NULL, so the output-window epilogue dropped BOTH self-loop seeds.

_SELF_LOOP_NODES = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["b", "a", "a", "b"]})
_SELF_LOOP_EDGES = pd.DataFrame(
    [(2, 2), (0, 3), (2, 2), (3, 1), (0, 3), (2, 2), (1, 1)], columns=["s", "d"])
_SELF_LOOP_EDGE_ORACLE = [(1, 1), (2, 2), (2, 2), (2, 2), (3, 1)]

_TRACKED_UNDIRECTED_ARMS = [
    ("min1max1_label_nodes", dict(min_hops=1, max_hops=1, label_node_hops="nh")),
    ("min1max1_label_edges", dict(min_hops=1, max_hops=1, label_edge_hops="eh")),
    ("output_window", dict(hops=1, output_min_hops=1, output_max_hops=1)),
]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("arm,kwargs", _TRACKED_UNDIRECTED_ARMS, ids=[a for a, _ in _TRACKED_UNDIRECTED_ARMS])
def test_tracked_undirected_hop_keeps_self_loop_seed_rows(engine, arm, kwargs):
    _require_engine(engine)
    g = (graphistry
         .nodes(_frame(_SELF_LOOP_NODES, engine), "id")
         .edges(_frame(_SELF_LOOP_EDGES, engine), "s", "d"))
    out = g.hop(nodes=_frame(pd.DataFrame({"id": [1, 2]}), engine),
                direction="undirected", engine=engine, **kwargs)
    edges = to_pandas_any(out._edges)
    assert sorted(map(tuple, edges[["s", "d"]].itertuples(index=False))) == _SELF_LOOP_EDGE_ORACLE
    assert node_id_set(out) == {1, 2, 3}, "self-loop seed node rows must survive tracking"


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_seeded_undirected_degenerate_varlen_counts_self_loops(engine):
    """#1798 end to end: MATCH (a {kind:'a'})-[*1..1]-(b) on the self-loop fixture. Hand
    enumeration: seed 1 via (3,1) and its self-loop (1,1); seed 2 via (2,2)x3 -- self-loops
    counted once per edge row (Neo4j relationship semantics) = 5. cuDF returned 1."""
    _require_engine(engine)
    g = (graphistry
         .nodes(_frame(_SELF_LOOP_NODES, engine), "id")
         .edges(_frame(_SELF_LOOP_EDGES, engine), "s", "d"))
    out = g.gfql("MATCH (a {kind:'a'})-[*1..1]-(b) RETURN count(*) AS c", engine=engine)
    assert int(to_pandas_any(out._nodes)["c"].iloc[0]) == 5


# --- #1940: internal __gfqlhop__ tracking columns must never reach user output --------------

_LEAK_ARMS = [
    ("min_hops2", dict(min_hops=2, hops=2)),
    ("label_nodes_only", dict(label_node_hops="nh", hops=2)),
    ("label_edges_only", dict(label_edge_hops="eh", hops=2)),
    ("output_min", dict(output_min_hops=1, hops=2)),
    ("output_max", dict(output_max_hops=1, hops=2)),
    ("output_window_both", dict(output_min_hops=1, output_max_hops=2, hops=2)),
    ("label_seeds", dict(label_node_hops="nh", label_seeds=True, hops=2)),
    ("wavefront_window", dict(output_min_hops=1, hops=2, return_as_wave_front=True)),
    ("tfp_label_nodes", dict(to_fixed_point=True, label_node_hops="nh")),
    ("tfp_output_min", dict(to_fixed_point=True, output_min_hops=1)),
]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("arm,kwargs", _LEAK_ARMS, ids=[a for a, _ in _LEAK_ARMS])
def test_no_internal_gfqlhop_column_reaches_user_output(engine, direction, arm, kwargs):
    """Every track_hops arm allocates internal `__gfqlhop_*__` label columns; several output
    paths re-added them after the mid-function drop (#1940). Requested label columns stay;
    internal ones never escape, on any direction x window x label x tfp arm."""
    _require_engine(engine)
    ndf = pd.DataFrame({"id": [0, 1, 2], "val": [10, 11, 12]})
    edf = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
    g = (graphistry
         .nodes(_frame(ndf, engine), "id")
         .edges(_frame(edf, engine), "s", "d"))
    out = g.hop(nodes=_frame(pd.DataFrame({"id": [0]}), engine),
                direction=direction, engine=engine, **kwargs)
    leaked_node_cols = [c for c in out._nodes.columns if str(c).startswith("__gfqlhop_")]
    leaked_edge_cols = [c for c in out._edges.columns if str(c).startswith("__gfqlhop_")]
    assert leaked_node_cols == [] and leaked_edge_cols == [], (
        f"internal tracking columns leaked: nodes={leaked_node_cols} edges={leaked_edge_cols}")
    if "label_node_hops" in kwargs:
        assert "nh" in out._nodes.columns, "requested node label column must remain"
    if "label_edge_hops" in kwargs:
        assert "eh" in out._edges.columns, "requested edge label column must remain"
