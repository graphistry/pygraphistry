"""PHASE 12: off-engine ``call()`` modality policy (``call_mode`` auto/strict) under polars.

An analytic ``call()`` with no native polars impl (hypergraph / umap / compute_cugraph /
prune_self_edges / ...) runs OFF-ENGINE on pandas (polars) / cuDF (polars-gpu) by default
(``call_mode='auto'``) and its result is coerced back to polars losslessly (Arrow), warning once
per (process, function). ``call_mode='strict'`` declines with ``NotImplementedError``. Traversal/
row ops stay parity-or-NIE (tested elsewhere). This is the CPU lane (polars); GPU parity for
cugraph / polars-gpu (incl. the GPU-or-error decline) lives on the dgx GPU conformance lane.
"""
import warnings
import pandas as pd
import pytest

from graphistry.compute.ast import call, let
from graphistry.compute.calls import hypergraph
from graphistry.tests.test_compute import CGFull
from graphistry.compute.gfql.lazy import call_mode, set_call_mode

pytest.importorskip("polars")


@pytest.fixture(autouse=True)
def _reset_call_mode():
    """Isolate the process-global call_mode override + warn-once tracker per test."""
    from graphistry.compute.gfql.call import executor as _ex
    set_call_mode(None)
    _ex._OFFENGINE_BRIDGE_WARNED.clear()
    yield
    set_call_mode(None)
    _ex._OFFENGINE_BRIDGE_WARNED.clear()


# --------------------------------------------------------------------------- config surface
def test_call_mode_default_auto(monkeypatch):
    monkeypatch.delenv("GFQL_POLARS_CALL_MODE", raising=False)
    set_call_mode(None)
    assert call_mode() == "auto"


def test_call_mode_env_strict(monkeypatch):
    monkeypatch.setenv("GFQL_POLARS_CALL_MODE", "strict")
    set_call_mode(None)
    assert call_mode() == "strict"


def test_call_mode_env_invalid_falls_back_auto(monkeypatch):
    monkeypatch.setenv("GFQL_POLARS_CALL_MODE", "banana")
    set_call_mode(None)
    assert call_mode() == "auto"


def test_call_mode_python_override_beats_env(monkeypatch):
    monkeypatch.setenv("GFQL_POLARS_CALL_MODE", "strict")
    set_call_mode("auto")
    assert call_mode() == "auto"
    set_call_mode(None)  # reset override -> env wins again
    assert call_mode() == "strict"


def test_set_call_mode_invalid_raises():
    with pytest.raises(ValueError):
        set_call_mode("nope")


# --------------------------------------------------------------------------- fixtures / helpers
def _selfedge_graph():
    # (1,1) and (2,2) are self-loops that prune_self_edges must drop. Explicit nodes so the
    # let() DAG surface does not pre-materialize nodes (materialize_nodes has a separate
    # pandas-only gap under polars for edges-only graphs — tracked as a PHASE 12 follow-up).
    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({"s": [0, 1, 2, 2], "d": [1, 1, 2, 0]})
    return CGFull().nodes(nodes, "id").edges(edges, "s", "d")


def _events_graph():
    events = pd.DataFrame({
        "user": ["alice", "bob", "alice", "carol"],
        "product": ["laptop", "phone", "tablet", "laptop"],
        "type": ["person", "person", "person", "person"],
    })
    return CGFull().nodes(events)


def _val_sig(df):
    """Order-insensitive value signature of a frame (polars or pandas)."""
    if df is None:
        return None
    pdf = df.to_pandas() if hasattr(df, "to_pandas") and "polars" in type(df).__module__ else df
    pdf = pdf.reindex(sorted(pdf.columns), axis=1)
    obj = pdf.astype(object).where(pdf.notna(), None)
    rows = sorted(tuple(str(v) for v in r) for r in obj.to_numpy().tolist())
    return (tuple(sorted(str(c) for c in pdf.columns)), tuple(rows))


# --------------------------------------------------------------------------- regressions
def test_materialize_nodes_polars_edges_only_let_dag():
    """Regression (P13.6-fix): a let() DAG on an EDGES-ONLY graph under engine='polars'
    pre-materializes nodes (chain_let), which crashed when ComputeMixin.materialize_nodes used
    pandas-only `.drop_duplicates()`/`.reset_index()` on a polars Series. Now polars-aware
    (`.unique(maintain_order=True)`). Must match the pandas oracle."""
    edges = pd.DataFrame({"s": [0, 1, 2, 2], "d": [1, 1, 2, 0]})
    g = CGFull().edges(edges, "s", "d")   # NO explicit nodes -> triggers materialize_nodes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = g.gfql(let({"a": call("prune_self_edges")}), engine="polars")
    oracle = g.gfql(let({"a": call("prune_self_edges")}), engine="pandas")
    assert _val_sig(out._edges) == _val_sig(oracle._edges)


# --------------------------------------------------------------------------- auto bridge (default)
def test_prune_self_edges_auto_bridges_and_matches_pandas_oracle():
    """call_mode='auto' (default): a non-native analytic runs off-engine and its result is
    coerced back to polars, byte-parity with the pandas oracle (NO-CHEATING: bridge OK, wrong
    answer NOT). Exercises the polars CHAIN _run_calls_polars -> execute_call delegation."""
    g = _selfedge_graph()
    oracle = g.gfql([call("prune_self_edges")], engine="pandas")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = g.gfql([call("prune_self_edges")], engine="polars")
    assert "polars" in type(out._edges).__module__, "off-engine result not coerced back to polars"
    assert _val_sig(out._edges) == _val_sig(oracle._edges)


def test_prune_self_edges_chain_vs_dag_consistent():
    """Both the chain and the let()/DAG surfaces must bridge identically (no surface may decline
    where the other bridges)."""
    g = _selfedge_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain_out = g.gfql([call("prune_self_edges")], engine="polars")
        dag_out = g.gfql(let({"a": call("prune_self_edges")}), engine="polars")
    assert _val_sig(chain_out._edges) == _val_sig(dag_out._edges)


def test_hypergraph_auto_bridges_under_polars():
    """A schema-changer analytic (hypergraph) bridges off-engine and returns polars frames."""
    g = _events_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = g.gfql(hypergraph(entity_types=["user", "product"], direct=True), engine="polars")
    assert out._nodes is not None and out._edges is not None
    assert "polars" in type(out._nodes).__module__
    assert "polars" in type(out._edges).__module__
    assert len(out._nodes) > 0 and len(out._edges) > 0


# --------------------------------------------------------------------------- strict decline
def test_prune_self_edges_strict_declines():
    """call_mode='strict' declines the off-engine bridge with NotImplementedError (no hidden
    modality switch — for benchmark integrity / a hard memory ceiling)."""
    g = _selfedge_graph()
    set_call_mode("strict")
    with pytest.raises(NotImplementedError):
        g.gfql([call("prune_self_edges")], engine="polars")


def test_hypergraph_strict_declines_dag():
    g = _events_graph()
    set_call_mode("strict")
    with pytest.raises(NotImplementedError):
        g.gfql(let({"a": hypergraph(entity_types=["user", "product"], direct=True)}), engine="polars")


# --------------------------------------------------------------------------- warn once
def test_offengine_bridge_warns_once_per_function():
    g = _selfedge_graph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.gfql([call("prune_self_edges")], engine="polars")
        g.gfql([call("prune_self_edges")], engine="polars")
    bridge = [w for w in caught
              if issubclass(w.category, RuntimeWarning) and "off-engine" in str(w.message)]
    assert len(bridge) == 1, f"expected one off-engine warning per function, got {len(bridge)}"
