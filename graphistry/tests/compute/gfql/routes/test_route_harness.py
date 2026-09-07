"""Route harness: every registered shape is tried against every chain route whose admission
predicate admits it. Three pins per cell: the route SERVES (its lane answers; a lane that
declines an admitted shape is recorded as an expected failure, the attenuation ledger), the
answer matches the same engine's general path (all routes off) on node/edge values, and its
node/edge sets match the pandas general path (the cross-engine oracle). Filed divergences are
strict expected failures keyed by their tag, so they flip when fixed.
"""
import math
import os
from typing import Callable, Dict, List, NamedTuple, Tuple

import pandas as pd
import pytest

import graphistry.compute.chain as chain_mod
import graphistry.compute.gfql.lazy.engine.polars.chain as pchain
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTObject
from graphistry.compute.chain_specializations.admission import native_fast_path_admits
from graphistry.compute.gfql.lazy.engine.polars.chain_specializations.admission import (
    polars_plain_single_hop_admits, polars_seeded_lane_admits,
)
from graphistry.tests.compute.gfql.routes.registry import REGISTRY, Shape, graph_for
from graphistry.tests.compute.gfql.routes.switch import ROUTES as ALL_ROUTES, routes_off

import graphistry.tests.compute.gfql.routes.corpus  # noqa: F401  registers routes.corpus
import graphistry.tests.compute.test_chain  # noqa: F401  registers test_chain.*
import graphistry.tests.compute.test_chain_alias_column_collision  # noqa: F401  registers collision.*


class Route(NamedTuple):
    name: str
    engines: Tuple[str, ...]
    admits: Callable[[List[ASTObject], str], bool]
    lane: Tuple[object, str]
    indexed: bool


ROUTES = [
    Route("native-fast", ("pandas", "cudf"),
          lambda ops, engine: native_fast_path_admits(ops, Engine(engine), None) is not None,
          (chain_mod, "_try_chain_fast_path"), False),
    Route("polars-plain", ("polars",),
          lambda ops, engine: polars_plain_single_hop_admits(ops, None) is not None,
          (pchain, "_plain_single_hop_polars"), False),
    Route("polars-seeded", ("polars",),
          lambda ops, engine: polars_seeded_lane_admits(ops),
          (pchain, "_try_seeded_chain_polars"), True),
]

KNOWN: Dict[Tuple[str, str], str] = {}  # (route, tag) -> issue: strict xfail until it lands


class Case(NamedTuple):
    route: Route
    engine: str
    shape: Shape

    @property
    def id(self) -> str:
        return f"{self.route.name}/{self.engine}/{self.shape.name}"


def _cases() -> List[Case]:
    out = []
    for shape in REGISTRY.values():
        for route in ROUTES:
            for engine in route.engines:
                try:
                    admitted = route.admits(shape.build(), engine)
                except Exception:
                    admitted = False
                if admitted:
                    out.append(Case(route, engine, shape))
    return out


CASES = _cases()


def _topd(df):
    if df is None:
        return None
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def _canon(df) -> Tuple[Tuple[str, ...], List[Tuple]]:
    df = _topd(df)
    if df is None:
        return ((), [])
    cols = tuple(sorted(df.columns))
    rows = []
    for row in df[list(cols)].itertuples(index=False, name=None):
        rows.append(tuple(None if (isinstance(v, float) and math.isnan(v)) or v is pd.NA or v is pd.NaT else v for v in row))
    rows.sort(key=repr)
    return cols, rows


def _sig(res, frames) -> Tuple[List, List]:
    nn, ee = _topd(res._nodes), _topd(res._edges)
    nodes = sorted(nn[frames.node].tolist()) if nn is not None else []
    edges = sorted(map(tuple, ee[[frames.src, frames.dst]].values.tolist())) if ee is not None and len(ee) else []
    return nodes, edges


def _served(case: Case, monkeypatch):
    mod, name = case.route.lane
    real = getattr(mod, name)
    calls = {"served": 0}

    def spy(*a, **k):
        out = real(*a, **k)
        calls["served"] += out is not None
        return out
    monkeypatch.setattr(mod, name, spy)
    return calls


def _skip_unavailable(engine: str) -> None:
    if engine == "cudf":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cuDF lane runs with TEST_CUDF=1")
        pytest.importorskip("cudf")
    if engine == "polars":
        pytest.importorskip("polars")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_admitted_shape_is_served_and_matches_the_general_path(case: Case, request, monkeypatch):
    _skip_unavailable(case.engine)
    for tag in case.shape.tags:
        if (case.route.name, tag) in KNOWN:
            request.applymarker(pytest.mark.xfail(strict=True, reason=KNOWN[(case.route.name, tag)]))
    g = graph_for(case.shape, case.engine, indexed=case.route.indexed)
    calls = _served(case, monkeypatch)
    try:
        served = g.gfql(case.shape.build(), engine=case.engine)
    except Exception as served_exc:
        with routes_off(ALL_ROUTES):
            with pytest.raises(type(served_exc)):
                g.gfql(case.shape.build(), engine=case.engine)
        return
    with routes_off(ALL_ROUTES):
        general = g.gfql(case.shape.build(), engine=case.engine)
        oracle = _sig(graph_for(case.shape, "pandas").gfql(case.shape.build(), engine="pandas"), case.shape.frames)
    assert _canon(served._nodes) == _canon(general._nodes), f"{case.id}: node rows differ from the general path"
    assert _canon(served._edges) == _canon(general._edges), f"{case.id}: edge rows differ from the general path"
    assert _sig(served, case.shape.frames) == oracle, f"{case.id}: node/edge sets differ from the pandas general path"
    if calls["served"] == 0:
        pytest.xfail(f"{case.id}: admitted by the predicate, declined by the lane body (attenuation ledger)")


@pytest.mark.route_engaged("native-fast", "polars-plain", "polars-seeded")
def test_every_route_serves_most_of_what_it_admits(monkeypatch):
    """A lane that declines most admitted shapes has a predicate that no longer describes it."""
    per_route: Dict[str, List[int]] = {}
    for case in CASES:
        if case.engine != ("polars" if case.route.name.startswith("polars") else "pandas"):
            continue
        if case.engine == "polars":
            pytest.importorskip("polars")
        g = graph_for(case.shape, case.engine, indexed=case.route.indexed)
        calls = _served(case, monkeypatch)
        try:
            g.gfql(case.shape.build(), engine=case.engine)
        except Exception:
            continue
        per_route.setdefault(case.route.name, []).append(calls["served"] > 0)
    for route, served in per_route.items():
        assert sum(served) * 2 >= len(served), f"{route}: served {sum(served)} of {len(served)} admitted shapes"


def test_every_route_has_admitted_shapes():
    covered = {(c.route.name, c.engine) for c in CASES}
    for route in ROUTES:
        for engine in route.engines:
            assert (route.name, engine) in covered, f"{route.name}/{engine} admits no registered shape"
