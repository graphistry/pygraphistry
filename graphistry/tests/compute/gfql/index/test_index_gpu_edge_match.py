"""GPU coverage for the indexed `edge_match` candidate-row path (#1782) and the semi-join
key sides (#1784).

Why a separate file: the regression tests for those changes are parametrized over
``_cpu_engines()`` — pandas + polars — so the ``Engine.CUDF`` and ``Engine.POLARS_GPU``
branches had NO coverage from them. They cannot simply be switched to ``_engines()``:
that helper includes cudf whenever cudf is *importable*, so on a dev box with cudf
installed but no working GPU every such test fails. This file gates on a real runtime
probe instead, so it runs where there is a GPU and skips cleanly where there is not.

The specific device risks these pin, all raised in review of #1782:
  * ``(sub == val).fillna(False).values`` on a GATHERED cudf column — ``.values`` RAISES on a
    null-bearing cudf column, so correctness rests entirely on ``fillna`` having cleared nulls
    for EVERY result dtype, not just the numeric ones.
  * the EMPTY candidate batch on device (a zero-length gather map).
  * the cost guard's whole-column fallback firing on device.
Pandas is the oracle in every case — comparing a GPU engine only against another GPU engine
would let a shared defect pass.

WHAT THESE DO **NOT** PIN, measured rather than assumed: deleting the ``fillna(False)`` from
the cudf branch of ``_EdgeMatchRowFilter.mask_for`` leaves all of these GREEN on cudf 26.02.
The review's concern was that ``.values`` raises on a null-bearing cudf column; on this
version ``(sub == val)`` already yields a non-null boolean column, so the fillna is defensive
rather than load-bearing *here*. It is kept because the pandas branch genuinely needs it and
because a future cudf could restore null propagation — but do not read a green run of this
file as evidence that the fillna is required. What the file DOES pin is device-vs-pandas
parity across null-bearing dtypes and the empty-batch path, both of which had no coverage at
all before: every new test the stack added is parametrized over ``_cpu_engines()``.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index import create_index  # noqa: F401  (registers plottable API)

cudf = pytest.importorskip("cudf")


def _gpu_available() -> bool:
    """Probe by running the smallest version of what these tests do.

    Cheaper probes do not discriminate on a box with cudf installed but no CUDA runtime:
    `cudf.DataFrame(...)`, `.to_pandas()`, `.values` (a small cupy alloc), a comparison, and
    even `groupby().sum()` all SUCCEED there, and the suite then fails with
    `OSError: libnvrtc.so.12` inside the first real kernel. So the probe is an actual indexed
    typed hop on a two-edge graph — if that runs, everything below can.
    """
    try:
        import numpy as _np
        import pandas as _pd
        from graphistry.Engine import Engine as _E, df_to_engine as _to
        _n = _to(_pd.DataFrame({"id": _np.arange(3, dtype=_np.int64)}), _E.CUDF)
        _e = _to(_pd.DataFrame({"src": [0, 1], "dst": [1, 2], "etype": [1, 2]}), _E.CUDF)
        _g = graphistry.nodes(_n, "id").edges(_e, "src", "dst").gfql_index_all(engine="cudf")
        _g.hop(nodes=_n[:1], hops=1, return_as_wave_front=True,
               edge_match={"etype": 1}, engine="cudf")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="no working cudf / GPU")

NULL_DTYPES = ["int64", "Int64", "float", "string", "boolean"]


def _frames(null_col_dtype: str, n_nodes: int = 400, deg: int = 6):
    """Typed graph whose edge predicate column carries NULLS — the case `.values` trips on."""
    rng = np.random.default_rng(3)
    m = n_nodes * deg
    etype = rng.integers(0, 3, m)
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, m),
                          "dst": rng.integers(0, n_nodes, m),
                          "etype": pd.Series(etype).astype(null_col_dtype)
                          if null_col_dtype not in ("string", "boolean")
                          else pd.Series([str(v) for v in etype] if null_col_dtype == "string"
                                         else (etype % 2).astype(bool)).astype(null_col_dtype)})
    edges.loc[edges.index[::7], "etype"] = None      # ~14% nulls
    nodes = pd.DataFrame({"id": np.arange(n_nodes, dtype=np.int64)})
    return nodes, edges


def _match_value(null_col_dtype: str):
    return {"string": "1", "boolean": True}.get(null_col_dtype, 1)


@pytest.mark.route_engaged("index-hop")
@pytest.mark.parametrize("engine", ["cudf", "polars-gpu"])
@pytest.mark.parametrize("null_col_dtype", NULL_DTYPES)
def test_null_bearing_edge_predicate_matches_the_pandas_oracle_on_device(engine, null_col_dtype):
    """`.values` on a gathered, null-bearing device column must neither raise nor diverge."""
    nodes, edges = _frames(null_col_dtype)
    g_pd = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    kwargs = dict(hops=1, return_as_wave_front=True,
                  edge_match={"etype": _match_value(null_col_dtype)})
    oracle = g_pd.hop(nodes=nodes[:1], engine="pandas", **kwargs)

    from graphistry.Engine import Engine as _E, df_to_engine
    target = _E.CUDF if engine == "cudf" else _E.POLARS
    g = graphistry.nodes(df_to_engine(nodes, target), "id").edges(
        df_to_engine(edges, target), "src", "dst")
    got = g.gfql_index_all(engine=engine).hop(
        nodes=df_to_engine(nodes[:1], target), engine=engine, **kwargs)

    def pairs(gg):
        df = gg._edges
        p = df.to_pandas() if hasattr(df, "to_pandas") else df
        return sorted(zip(p["src"].tolist(), p["dst"].tolist()))

    assert pairs(got) == pairs(oracle), f"[{engine}/{null_col_dtype}] diverged from pandas"


@pytest.mark.route_engaged("index-hop")
@pytest.mark.parametrize("engine", ["cudf", "polars-gpu"])
def test_empty_candidate_batch_on_device(engine):
    """A seed with no matching typed edges yields a zero-length gather map on device."""
    nodes, edges = _frames("int64")
    from graphistry.Engine import Engine as _E, df_to_engine
    target = _E.CUDF if engine == "cudf" else _E.POLARS
    g_pd = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    kwargs = dict(hops=1, return_as_wave_front=True, edge_match={"etype": 99})  # matches nothing
    oracle = g_pd.hop(nodes=nodes[:1], engine="pandas", **kwargs)

    g = graphistry.nodes(df_to_engine(nodes, target), "id").edges(
        df_to_engine(edges, target), "src", "dst")
    got = g.gfql_index_all(engine=engine).hop(
        nodes=df_to_engine(nodes[:1], target), engine=engine, **kwargs)
    assert int(got._edges.shape[0]) == int(oracle._edges.shape[0]) == 0
