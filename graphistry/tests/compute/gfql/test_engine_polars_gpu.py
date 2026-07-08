"""Differential: engine='polars-gpu' == engine='polars'. The GPU mode (cudf_polars) runs the same
ops but collects the hot joins on GPU; it MUST match CPU Polars (itself parity-gated vs pandas).
Reuses the cypher conformance corpus + core traversals; skips when no GPU / cudf_polars.
See plans/gfql-polars-engine (GPU engine)."""
import pandas as pd
import pytest

import graphistry  # noqa: F401  (ensures plottable methods are registered)
from graphistry.compute.ast import n, e_forward
from graphistry.compute.predicates.numeric import gt

pl = pytest.importorskip("polars")

from graphistry.tests.compute.gfql.test_engine_polars_cypher_conformance import (  # noqa: E402
    CORPUS,
    _graph,
)


def _gpu_available() -> bool:
    try:
        pl.DataFrame({"a": [1, 2]}).lazy().filter(pl.col("a") > 0).collect(
            engine=pl.GPUEngine(raise_on_fail=True)
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="no cudf_polars / GPU available")


def _norm(df):
    p = df.to_pandas() if hasattr(df, "to_pandas") else df
    return p.astype(str).sort_values(list(p.columns)).reset_index(drop=True)


def _assert_nodes_parity(cpu, gpu):
    pd.testing.assert_frame_equal(_norm(cpu._nodes), _norm(gpu._nodes), check_dtype=False)


@pytest.mark.parametrize("query", CORPUS)
def test_gpu_cypher_parity(query):
    g = _graph(seed=3, n=24)
    _assert_nodes_parity(g.gfql(query, engine="polars"), g.gfql(query, engine="polars-gpu"))


@pytest.mark.parametrize("ops_name", ["hop1", "hop2", "filt_hop", "rev_hop"])
def test_gpu_chain_parity(ops_name):
    from graphistry.compute.ast import e_reverse
    g = _graph(seed=5, n=40)
    ops = {
        "hop1": [n(), e_forward(), n()],
        "hop2": [n(), e_forward(), n(), e_forward(), n()],
        "filt_hop": [n({"val": gt(50)}), e_forward(), n()],
        "rev_hop": [n(), e_reverse(), n()],
    }[ops_name]
    cpu = g.gfql(ops, engine="polars")
    gpu = g.gfql(ops, engine="polars-gpu")
    _assert_nodes_parity(cpu, gpu)
    pd.testing.assert_frame_equal(_norm(cpu._edges), _norm(gpu._edges), check_dtype=False)


@pytest.mark.parametrize("executor", ["in-memory", "streaming"])
def test_gpu_executor_modes_parity(executor):
    """Both GPU executors (in-memory default + streaming opt-in) must match CPU Polars. Locks in
    the streaming executor on a REAL GPU — otherwise only mock-wiring-covered
    (test_engine_polars_chain) + manual dgx runs. Skipped in CI (no GPU); dgx GPU lane."""
    from graphistry.compute.gfql import lazy
    g = _graph(seed=7, n=60)
    ops = [n(), e_forward(), n()]
    cpu = g.gfql(ops, engine="polars")
    try:
        lazy.set_gpu_executor(executor)
        assert lazy.gpu_executor() == executor
        gpu = g.gfql(ops, engine="polars-gpu")
    finally:
        lazy.set_gpu_executor(None)
    _assert_nodes_parity(cpu, gpu)
    pd.testing.assert_frame_equal(_norm(cpu._edges), _norm(gpu._edges), check_dtype=False)


def test_polars_gpu_engine_enum_is_explicit_only():
    # AUTO must never resolve to the GPU engine — opt-in only
    from graphistry.Engine import Engine, resolve_engine, EngineAbstract
    assert Engine.POLARS_GPU.value == "polars-gpu"
    assert resolve_engine(EngineAbstract.AUTO) != Engine.POLARS_GPU
    assert resolve_engine("polars-gpu") == Engine.POLARS_GPU


@pytest.mark.parametrize("target", ["polars", "polars-gpu"])
def test_cudf_to_polars_via_arrow_preserves_dtypes_and_nulls(target):
    # cuDF converts to polars via Arrow (native interchange), NOT a cuDF->pandas->polars
    # double-convert — the pandas detour is lossy (nullable Int64 degrades to Float64,
    # null -> NaN); Arrow preserves dtypes and nulls
    cudf = pytest.importorskip("cudf")
    from graphistry.Engine import Engine, df_to_engine

    gdf = cudf.from_pandas(pd.DataFrame({
        "a": pd.array([1, 2, None], dtype="Int64"),
        "flag": pd.array([True, None, False], dtype="boolean"),
        "s": ["x", "y", None],
    }))
    out = df_to_engine(gdf, Engine(target))
    assert isinstance(out, pl.DataFrame)
    assert out.schema["a"] == pl.Int64  # NOT Float64 (the pandas-detour regression)
    assert out.schema["flag"] == pl.Boolean
    assert out.schema["s"] == pl.String
    assert out["a"].to_list() == [1, 2, None]
    assert out["flag"].to_list() == [True, None, False]
    assert out["s"].to_list() == ["x", "y", None]
