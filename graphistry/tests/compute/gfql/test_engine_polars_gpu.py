"""Differential: engine='polars-gpu' == engine='polars'.

The GPU execution mode of the native Polars engine (cudf_polars) runs the same
ops but collects the hot joins on GPU; it MUST produce identical results to CPU
Polars (which is itself parity-gated vs pandas). Reuses the cypher conformance
corpus + core traversals. Skips when no GPU / cudf_polars is available.
See plans/gfql-polars-engine (GPU engine, stacked on the CPU engine).
"""
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


def test_polars_gpu_engine_enum_is_explicit_only():
    # AUTO must never resolve to the GPU engine — opt-in only.
    from graphistry.Engine import Engine, resolve_engine, EngineAbstract
    assert Engine.POLARS_GPU.value == "polars-gpu"
    assert resolve_engine(EngineAbstract.AUTO) != Engine.POLARS_GPU
    assert resolve_engine("polars-gpu") == Engine.POLARS_GPU
