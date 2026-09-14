"""AUTO must resolve the INDEX engine and the QUERY engine to the same place.

There are two AUTO gates -- the query router in ``gfql_unified`` and
``resolve_index_engine`` -- and their DISAGREEMENT is the #1767 failure class: an
index built for one engine meets a query routed to another, every hop declines,
and the default path silently runs at the scan floor while answers stay correct.
So the property under test is AGREEMENT per frame combination, both sides.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.gfql.index.api import resolve_index_engine

pl = pytest.importorskip("polars")


def _g(nodes_kind: str, edges_kind: str):
    n = pd.DataFrame({"id": [0, 1, 2, 3]})
    e = pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]})
    conv = {"pandas": lambda d: d, "polars": pl.from_pandas,
            "lazy": lambda d: pl.from_pandas(d).lazy()}
    return graphistry.nodes(conv[nodes_kind](n), "id").edges(conv[edges_kind](e), "src", "dst")


@pytest.mark.parametrize("nodes_kind,edges_kind,expected", [
    ("pandas", "pandas", Engine.PANDAS),   # negative: pandas stays pandas
    ("polars", "polars", Engine.POLARS),   # positive: the whole point
    ("lazy",   "polars", Engine.PANDAS),   # negative: lazy nodes cannot be gathered
    ("polars", "lazy",   Engine.PANDAS),   # negative: lazy edges likewise
    ("pandas", "polars", Engine.PANDAS),   # negative: mixed keeps legacy
    ("polars", "pandas", Engine.PANDAS),
])
def test_index_engine_per_frame_combination(nodes_kind, edges_kind, expected) -> None:
    assert resolve_index_engine("auto", _g(nodes_kind, edges_kind)) is expected


def test_explicit_engine_always_wins() -> None:
    g = _g("polars", "polars")
    assert resolve_index_engine("pandas", g) is Engine.PANDAS
    assert resolve_index_engine(Engine.PANDAS, g) is Engine.PANDAS


@pytest.mark.parametrize("nodes_kind,edges_kind", [
    ("pandas", "pandas"), ("polars", "polars"),
    ("lazy", "polars"), ("pandas", "polars"),
])
def test_auto_index_and_auto_query_agree_end_to_end(nodes_kind, edges_kind) -> None:
    """The agreement property itself, observed rather than asserted structurally:
    after an AUTO index build, an AUTO query's result frames come back in the
    SAME engine family the index was built for -- because create_index coerces
    the frames it indexes, the query router then sees those frames, and the two
    gates land together. If either gate drifts, this catches the mismatch."""
    g = _g(nodes_kind, edges_kind).gfql_index_all()  # AUTO build
    built = resolve_index_engine("auto", g)
    out = g.gfql("MATCH (a {id: 0})-[]->(b) RETURN b.id AS x")  # AUTO query
    mod = type(out._nodes).__module__
    if built is Engine.POLARS:
        assert "polars" in mod
    else:
        assert "pandas" in mod


def test_cudf_frames_are_never_silently_rerouted_by_the_index_gate() -> None:
    """The index gate must not touch cudf resolutions: the query side may route
    cudf->polars-gpu when a GPU is genuinely usable, and whether the INDEX should
    follow is an open design question (#1843 thread) -- but the index gate
    unilaterally deciding either way would create the mismatch silently. Pin that
    it stays source-native so the question stays visible."""
    cudf = pytest.importorskip("cudf")
    n = cudf.DataFrame({"id": [0, 1, 2]})
    e = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
    g = graphistry.nodes(n, "id").edges(e, "src", "dst")
    assert resolve_index_engine("auto", g) is Engine.CUDF
