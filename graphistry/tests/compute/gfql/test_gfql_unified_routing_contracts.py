"""Routing contracts in gfql_unified that used to be stated as source comments (#1895 review).

Each test here replaces a comment the reviewer asked to be turned into a test:

  * fast paths always run on the CPU execution target, whatever engine was requested (#1824)
  * a NotImplementedError from a fast path on the CPU target is a real error, not a decline
  * a policied AUTO query is routed to pandas by ``resolve_engine`` -- NOT by a frame-shape
    check, because a frame-shape check lets MIXED frames slip past a denying policy
  * the AUTO polars-native decline serves via pandas with COERCED frames
"""
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, EngineAbstract
from graphistry.compute import gfql_unified
from graphistry.compute.gfql_unified import (
    _fast_path_execution_target_ignoring_requested_engine,
    _policied_auto_serves_via_pandas_until_the_polars_route_emits_hooks as _policied_auto_to_pandas,
)

pl = pytest.importorskip("polars")

NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
EDGES = pd.DataFrame({"s": [0, 1], "d": [1, 2]})


def _polars_graph():
    return (graphistry
            .nodes(pl.from_pandas(NODES), "id")
            .edges(pl.from_pandas(EDGES), "s", "d"))


# --- fast-path execution target -----------------------------------------------------------

@pytest.mark.parametrize("engine", [
    "pandas", "polars", "polars-gpu", "cudf", "auto",
    Engine.POLARS_GPU, EngineAbstract.AUTO,
])
def test_fast_paths_target_cpu_whatever_engine_was_requested(engine):
    """The whole point of the name: the REQUESTED engine does not move the target. Flipping
    this to GPU without making each arm GPU-or-decline is the #1824 regression."""
    from graphistry.compute.gfql.lazy import ExecutionTarget
    assert _fast_path_execution_target_ignoring_requested_engine(engine) is ExecutionTarget.CPU


def test_fast_path_body_actually_runs_under_the_cpu_target(monkeypatch):
    """Not just the constant -- the fast-path call is really wrapped in that target_mode."""
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget

    seen = []

    def _record(*args, **kwargs):
        seen.append(active_target())
        return None  # decline, so the chain route answers

    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", _record)
    _polars_graph().gfql(
        "MATCH (a)-[]->(b) RETURN b.v AS v, count(*) AS c", engine="polars")
    assert seen, "the grouped-aggregate fast path was never consulted"
    assert all(t == ExecutionTarget.CPU for t in seen), seen


def test_cpu_fast_path_not_implemented_error_is_not_swallowed(monkeypatch):
    """On the CPU target an NIE is a real bug and must surface. Only the GPU target may treat
    it as 'plan not executable here, fall back'."""
    def _boom(*args, **kwargs):
        raise NotImplementedError("fast path exploded")

    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", _boom)
    with pytest.raises(NotImplementedError, match="fast path exploded"):
        _polars_graph().gfql(
            "MATCH (a)-[]->(b) RETURN b.v AS v, count(*) AS c", engine="polars")


# --- the policied-AUTO -> pandas predicate --------------------------------------------------

_DENY_ALL = {"preload": (lambda ctx: None)}


def _graph(nodes_polars: bool, edges_polars: bool):
    nodes = pl.from_pandas(NODES) if nodes_polars else NODES
    edges = pl.from_pandas(EDGES) if edges_polars else EDGES
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("engine,policy,nodes_pl,edges_pl,expected", [
    (EngineAbstract.AUTO, _DENY_ALL, True, True, True),      # all-polars + policy -> pandas
    ("auto", _DENY_ALL, True, True, True),                   # string form of AUTO
    (EngineAbstract.AUTO, None, True, True, False),          # no policy -> no reroute
    (EngineAbstract.AUTO, _DENY_ALL, False, False, False),   # pandas already
    ("polars", _DENY_ALL, True, True, False),                # explicit engine is not AUTO
])
def test_policied_auto_predicate(engine, policy, nodes_pl, edges_pl, expected):
    assert _policied_auto_to_pandas(engine, policy, _graph(nodes_pl, edges_pl)) is expected


def test_mixed_frames_with_a_policy_are_still_routed_to_pandas():
    """THE reason the predicate is resolve_engine and not a frame-shape check: with polars
    edges and pandas nodes a shape check ("both frames polars?") answers False and the query
    would run the polars route with the policy hooks never firing. resolve_engine says POLARS
    for this graph, so the guard still catches it."""
    from graphistry.Engine import resolve_engine

    mixed = _graph(nodes_polars=False, edges_polars=True)
    with pytest.warns(UserWarning, match="same type"):
        assert resolve_engine(EngineAbstract.AUTO, mixed) == Engine.POLARS
    with pytest.warns(UserWarning, match="same type"):
        assert _policied_auto_to_pandas(EngineAbstract.AUTO, _DENY_ALL, mixed) is True


# --- the AUTO polars-native decline -> pandas with coerced frames ----------------------------

def test_auto_polars_decline_reruns_on_pandas_with_coerced_frames(monkeypatch):
    """Pins both halves of the fallback that used to be a comment: the retry pins engine=PANDAS
    (not AUTO, which would re-resolve these polars frames straight back to POLARS and re-raise
    the same NIE), and it COERCES the frames first because the pandas executors are pandas-idiom.
    """
    real_gfql = gfql_unified.gfql
    calls = []

    def _spy(self, query, *args, **kwargs):
        calls.append((kwargs.get("engine"), type(self._edges).__module__.split(".")[0]))
        if len(calls) == 1:
            # first hop is the AUTO wrapper itself; let it through
            return real_gfql(self, query, *args, **kwargs)
        if kwargs.get("engine") == Engine.POLARS.value:
            raise NotImplementedError("polars route declines")
        return real_gfql(self, query, *args, **kwargs)

    monkeypatch.setattr(gfql_unified, "gfql", _spy)
    out = _spy(_polars_graph(), "MATCH (a)-[]->(b) RETURN b.v AS v", engine=EngineAbstract.AUTO)

    engines = [e for e, _ in calls]
    assert Engine.POLARS.value in engines, f"polars was never attempted: {calls}"
    assert Engine.PANDAS.value in engines, f"never fell back to explicit pandas: {calls}"
    assert EngineAbstract.AUTO not in engines[1:], "fallback re-entered on AUTO, not pandas"
    pandas_call_frames = [mod for e, mod in calls if e == Engine.PANDAS.value]
    assert pandas_call_frames and all(m == "pandas" for m in pandas_call_frames), (
        f"pandas executors were handed non-pandas frames: {calls}")
    assert out is not None
