"""Direct pins for the hop-kernel helpers #1895 extracted, so their NAMES are load-bearing.

A helper named ``_keep_edges_with_both_endpoints_resolvable`` only carries meaning if
something fails when it stops doing that. Each helper below gets a pin naming its rule, plus
the two behaviour-adjacent changes the #1895 remediation made:

  * ``.implode()`` on the endpoint membership test (the bare ``is_in(<Series>)`` form is
    deprecated in polars 1.x) -- pinned VALUE-IDENTICAL on the shapes where the two forms
    could plausibly differ: nulls among the ids, nulls among the endpoints, and empty.
  * the unconditional trailing de-dup in ``hop()``'s endpoint backfill, which replaced a
    de-dup that used to live only on the else-branch.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_concat
from graphistry.compute.hop import (
    _endpoint_ids_without_node_rows, _reached_node_ids,
)

from .polars_test_utils import node_id_set, to_pandas_any

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
def test_duplicate_node_rows_are_deduped_when_no_endpoint_is_missing(engine):
    """POSITIVE: nothing to backfill, so this exercises the path that used to carry the
    de-dup only on its else-branch. Making the de-dup unconditional must preserve it."""
    _require_engine(engine)
    out = _dup_node_graph(engine).hop(engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert nodes_pdf["id"].tolist() == sorted(set(nodes_pdf["id"].tolist()))
    assert node_id_set(out) == {0, 1}


@pytest.mark.parametrize("engine", PANDAS_IDIOM_ENGINES)
def test_duplicate_node_rows_are_deduped_when_an_endpoint_is_backfilled(engine):
    """The other side: wavefront mode leaves an endpoint unbacked, so the concat branch runs.
    The de-dup must apply to the CONCATENATED frame, not just the untouched one."""
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
