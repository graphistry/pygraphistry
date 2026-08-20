"""Routing contracts in gfql_unified that used to be stated as source comments (#1895 review).

Each test here replaces a comment the reviewer asked to be turned into a test:

  * a fast path runs on the GPU execution target exactly when ``polars-gpu`` was requested,
    and on CPU for every other engine (#1824)
  * a NotImplementedError from a fast path on the CPU target is a real error, not a decline
  * a policied AUTO query is routed to pandas by ``resolve_engine`` -- NOT by a frame-shape
    check, because a frame-shape check lets MIXED frames slip past a denying policy
  * the AUTO polars-native decline serves via pandas with COERCED frames
"""
import importlib.util

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, EngineAbstract
from graphistry.compute import gfql_unified
from graphistry.compute.gfql_unified import (
    _fast_path_execution_target,
    _policied_auto_serves_via_pandas_until_the_polars_route_emits_hooks as _policied_auto_to_pandas,
)

pl = pytest.importorskip("polars")

#: Without the RAPIDS stack a ``polars-gpu`` query still consults the fast paths (they run
#: before chain dispatch), then reports the missing install once it reaches the generic route.
HAS_CUDF_POLARS = importlib.util.find_spec("cudf_polars") is not None

NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
EDGES = pd.DataFrame({"s": [0, 1], "d": [1, 2]})


def _polars_graph():
    return (graphistry
            .nodes(pl.from_pandas(NODES), "id")
            .edges(pl.from_pandas(EDGES), "s", "d"))


#: A connected comma-pattern that routes through ``_apply_connected_match_join``'s two-star
#: arms and on into the FUSED lazy lane, whose single collect carries the engine label.
Q_TWO_STAR = (
    "MATCH (p {node_type:'Person'})-[{rel:'HAS_INTEREST'}]->(i {node_type:'Interest'}), "
    "(p)-[{rel:'LIVES_IN'}]->(c {node_type:'City'}) "
    "WHERE toLower(i.interest) = 'fine dining' AND p.age >= 20 AND p.age <= 40 "
    "RETURN c.city AS city, count(p) AS n ORDER BY n DESC, city ASC")


def _two_star_graph():
    nodes = pl.DataFrame({
        "node_id": [1, 2, 3, 4, 5, 6, 7],
        "node_type": ["Person", "Person", "Person", "Interest", "Interest", "City", "City"],
        "age": [25, 30, 55, None, None, None, None],
        "interest": [None, None, None, "Fine Dining", "tennis", None, None],
        "city": [None, None, None, None, None, "London", "Paris"],
    })
    edges = pl.DataFrame({
        "src": [1, 2, 3, 1, 2, 3],
        "dst": [4, 4, 5, 6, 6, 7],
        "rel": ["HAS_INTEREST"] * 3 + ["LIVES_IN"] * 3,
    })
    return graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")


def _run_tolerating_absent_rapids(graph, query, engine):
    try:
        return graph.gfql(query, engine=engine)
    except ImportError:
        if engine == "polars-gpu" and not HAS_CUDF_POLARS:
            return None
        raise


# --- fast-path execution target -----------------------------------------------------------

Q_GROUPED = "MATCH (a)-[]->(b) RETURN b.v AS v, count(*) AS c"

#: The two ``_apply_connected_match_join`` fast-path arms, in consultation order.
TWO_STAR_ARMS = [
    "_connected_join_two_star_fast_grouped_count",
    "_connected_join_two_star_fast_rows",
]


def _isolate_two_star_arm(monkeypatch, arm):
    """Make ``arm`` the arm under test: an earlier arm serves Q_TWO_STAR and would
    short-circuit it, so decline the earlier arms."""
    for earlier in TWO_STAR_ARMS[:TWO_STAR_ARMS.index(arm)]:
        monkeypatch.setattr(gfql_unified, earlier, lambda *a, **k: None)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf", "auto", EngineAbstract.AUTO])
def test_fast_path_target_is_cpu_for_every_engine_that_is_not_polars_gpu(engine):
    from graphistry.compute.gfql.lazy import ExecutionTarget
    assert _fast_path_execution_target(engine) is ExecutionTarget.CPU


@pytest.mark.parametrize("engine", ["polars-gpu", Engine.POLARS_GPU])
def test_fast_path_target_is_gpu_for_an_explicitly_requested_polars_gpu(engine):
    """An explicit ``polars-gpu`` must reach the GPU collect target; serving it on CPU and
    labelling the result GPU is the whole defect."""
    from graphistry.compute.gfql.lazy import ExecutionTarget
    assert _fast_path_execution_target(engine) is ExecutionTarget.GPU


def _record_target(seen):
    def _record(*args, **kwargs):
        from graphistry.compute.gfql.lazy import active_target
        seen.append(active_target())
        return None  # decline, so the generic route answers
    return _record


@pytest.mark.parametrize("engine,expected", [("polars", "CPU"), ("auto", "CPU"), ("polars-gpu", "GPU")])
def test_grouped_aggregate_fast_path_body_runs_under_the_requested_engines_target(
        monkeypatch, engine, expected):
    """Not just the constant -- the fast-path call is really wrapped in that target_mode."""
    from graphistry.compute.gfql.lazy import ExecutionTarget

    seen: list = []
    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", _record_target(seen))
    _run_tolerating_absent_rapids(_polars_graph(), Q_GROUPED, engine)
    assert seen, "the grouped-aggregate fast path was never consulted"
    assert all(t is getattr(ExecutionTarget, expected) for t in seen), seen


@pytest.mark.parametrize("arm", TWO_STAR_ARMS)
@pytest.mark.parametrize("engine,expected", [("polars", "CPU"), ("auto", "CPU"), ("polars-gpu", "GPU")])
def test_connected_join_two_star_fast_path_runs_under_the_requested_engines_target(
        monkeypatch, arm, engine, expected):
    """The connected-join arms are a SEPARATE call site from the same_path ``_try_fast`` arms;
    wrapping only one of the two leaves half the OLAP surface collecting on the wrong target."""
    from graphistry.compute.gfql.lazy import ExecutionTarget

    seen: list = []
    _isolate_two_star_arm(monkeypatch, arm)
    monkeypatch.setattr(gfql_unified, arm, _record_target(seen))
    _run_tolerating_absent_rapids(_two_star_graph(), Q_TWO_STAR, engine)
    assert seen, f"{arm} was never consulted"
    assert all(t is getattr(ExecutionTarget, expected) for t in seen), seen


@pytest.mark.parametrize("engine", ["polars", "auto"])
def test_cpu_fast_path_not_implemented_error_is_not_swallowed(monkeypatch, engine):
    """On the CPU target an NIE is a real bug and must surface. Only the GPU target may treat
    it as 'plan not executable here, fall back'."""
    def _boom(*args, **kwargs):
        raise NotImplementedError("fast path exploded")

    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", _boom)
    with pytest.raises(NotImplementedError, match="fast path exploded"):
        _polars_graph().gfql(Q_GROUPED, engine=engine)


@pytest.mark.parametrize("arm", TWO_STAR_ARMS)
@pytest.mark.parametrize("engine", ["polars", "auto"])
def test_cpu_connected_join_two_star_not_implemented_error_is_not_swallowed(
        monkeypatch, arm, engine):
    def _boom(*args, **kwargs):
        raise NotImplementedError("two star exploded")

    _isolate_two_star_arm(monkeypatch, arm)
    monkeypatch.setattr(gfql_unified, arm, _boom)
    with pytest.raises(NotImplementedError, match="two star exploded"):
        _two_star_graph().gfql(Q_TWO_STAR, engine=engine)


def _assert_declines_rather_than_raising(graph, query, marker, expected):
    """A non-GPU-executable fast path is a DECLINE: the generic route -- itself GPU-or-raise --
    answers with the same values. Without RAPIDS installed the generic route reports the missing
    install instead, which still proves the fast path's NIE did not escape as the answer."""
    try:
        out = graph.gfql(query, engine="polars-gpu")
    except NotImplementedError as ex:
        if marker in str(ex):
            pytest.fail(f"fast-path decline escaped to the caller as a raise: {ex}")
        raise
    except ImportError:
        assert not HAS_CUDF_POLARS
        return
    assert out._nodes.to_dicts() == expected


def test_gpu_fast_path_not_implemented_error_declines_to_the_generic_route(monkeypatch):
    marker = "plan is not GPU-executable"

    def _boom(*args, **kwargs):
        raise NotImplementedError(marker)

    expected = _polars_graph().gfql(Q_GROUPED, engine="polars")._nodes.to_dicts()
    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", _boom)
    _assert_declines_rather_than_raising(_polars_graph(), Q_GROUPED, marker, expected)


@pytest.mark.parametrize("arm", TWO_STAR_ARMS)
def test_gpu_connected_join_two_star_not_implemented_error_declines_to_the_generic_route(
        monkeypatch, arm):
    marker = "plan is not GPU-executable"

    def _boom(*args, **kwargs):
        raise NotImplementedError(marker)

    expected = _two_star_graph().gfql(Q_TWO_STAR, engine="polars")._nodes.to_dicts()
    _isolate_two_star_arm(monkeypatch, arm)
    monkeypatch.setattr(gfql_unified, arm, _boom)
    _assert_declines_rather_than_raising(_two_star_graph(), Q_TWO_STAR, marker, expected)


# --- the fused OLAP lane on a real GPU ------------------------------------------------------

def _fused_lane_collect_engines(query, engine):
    """Engine object of every collect issued from inside the fused two-star lane."""
    import traceback
    seen = []
    original = pl.LazyFrame.collect

    def _spy(self, *args, **kwargs):
        if any(f.name == "_connected_join_two_star_fused_polars"
               for f in traceback.extract_stack()):
            seen.append(type(kwargs.get("engine")).__name__)
        return original(self, *args, **kwargs)

    pl.LazyFrame.collect = _spy
    try:
        out = _two_star_graph().gfql(query, engine=engine)
    finally:
        pl.LazyFrame.collect = original
    return seen, out._nodes.to_dicts()


@pytest.mark.skipif(not HAS_CUDF_POLARS, reason="needs the RAPIDS cudf_polars stack")
def test_fused_two_star_lane_collects_on_the_gpu_engine_when_polars_gpu_requested():
    """The fused lane's collect carries a GPUEngine, and its values equal the CPU lane's."""
    gpu_engines, gpu_rows = _fused_lane_collect_engines(Q_TWO_STAR, "polars-gpu")
    cpu_engines, cpu_rows = _fused_lane_collect_engines(Q_TWO_STAR, "polars")
    assert "GPUEngine" in gpu_engines, gpu_engines
    assert "GPUEngine" not in cpu_engines, cpu_engines
    assert gpu_rows == cpu_rows


@pytest.mark.parametrize("engine", ["polars", "auto"])
def test_fused_two_star_lane_never_reaches_the_gpu_engine_off_polars_gpu(engine):
    engines, _ = _fused_lane_collect_engines(Q_TWO_STAR, engine)
    assert engines, "the fused two-star lane never collected"
    assert "GPUEngine" not in engines, engines


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
