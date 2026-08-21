"""Whole-entity endpoint projection preserves relationship-match bags."""
from __future__ import annotations

import typing

import pandas as pd
import pytest
from typing_extensions import Literal

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.Plottable import Plottable
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason, to_pandas_any

_GFQLEngine = Literal["pandas", "polars", "cudf", "polars-gpu"]
_ENGINES: typing.Tuple[_GFQLEngine, ...] = ("pandas", "polars", "cudf", "polars-gpu")
_VARIABLE_LENGTH_ENGINES: typing.Tuple[_GFQLEngine, ...] = ("pandas", "polars")

_NODES = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5], "name": ["Ann", "Bob", "Cat", "Dan", "Eve"]}
)
_EDGES = pd.DataFrame({"s": [1, 1, 2, 3], "d": [2, 3, 3, 4]})
_PARALLEL_NODES = pd.DataFrame(
    {"id": [1, 2, 3], "name": ["Ann", "Bob", "Cat"]}
)
_PARALLEL_EDGES = pd.DataFrame({"s": [1, 1, 2], "d": [2, 2, 3]})


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, engine: _GFQLEngine) -> Plottable:
    resolved_engine = Engine(engine)
    return graphistry.nodes(df_to_engine(nodes, resolved_engine), "id").edges(
        df_to_engine(edges, resolved_engine), "s", "d"
    )


def _smoke(engine: _GFQLEngine) -> Plottable:
    return _bind(_NODES.iloc[:2], _EDGES.iloc[:1], engine).gfql(
        "MATCH (n) RETURN n.id AS id", engine=engine
    )

def _require_engine(engine: _GFQLEngine) -> None:
    skip_reason = engine_skip_reason(engine, lambda: _smoke(engine))
    if skip_reason is not None:
        pytest.skip(skip_reason)



def _run(query: str, engine: _GFQLEngine, *, parallel: bool = False) -> pd.DataFrame:
    _require_engine(engine)
    graph = (
        _bind(_PARALLEL_NODES, _PARALLEL_EDGES, engine)
        if parallel
        else _bind(_NODES, _EDGES, engine)
    )
    result_frame = graph.gfql(query, engine=engine)._nodes
    pandas_frame = to_pandas_any(result_frame)
    assert isinstance(pandas_frame, pd.DataFrame)
    return pandas_frame.reset_index(drop=True)


def _bag(df: pd.DataFrame, column: str) -> typing.List[typing.Optional[int]]:
    numeric_values = pd.to_numeric(df[column], errors="coerce")
    values = [
        None if pd.isna(value) else int(value)
        for value in numeric_values.tolist()
    ]
    return sorted(values, key=lambda value: (value is None, value))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query,column,expected", [
    ("MATCH (a)-->(b) RETURN b", "b.id", [2, 3, 3, 4]),
    ("MATCH (a)-->(b) RETURN a", "a.id", [1, 1, 2, 3]),
    ("MATCH (a)-->(b) RETURN b AS n", "n.id", [2, 3, 3, 4]),
    ("MATCH (a)-->(b)-->(c) RETURN c", "c.id", [3, 4, 4]),
], ids=["dst", "src", "aliased", "two_hop_dst"])
def test_whole_entity_endpoint_projection_keeps_bag(query: str, column: str, expected: typing.Sequence[int], engine: _GFQLEngine) -> None:
    assert _bag(_run(query, engine), column) == expected


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query,column,expected", [
    ("MATCH (a)-->(b) RETURN b", "b.id", [2, 2, 3]),
    ("MATCH (a)-->(b) RETURN a", "a.id", [1, 1, 2]),
], ids=["dst", "src"])
def test_whole_entity_projection_counts_parallel_edges(query: str, column: str, expected: typing.Sequence[int], engine: _GFQLEngine) -> None:
    assert _bag(_run(query, engine, parallel=True), column) == expected


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_projection_with_sibling_property_output(engine: _GFQLEngine) -> None:
    df = _run("MATCH (a)-->(b) RETURN b, b.id AS x", engine)
    assert _bag(df, "b.id") == [2, 3, 3, 4]
    assert _bag(df, "x") == [2, 3, 3, 4]


@pytest.mark.parametrize("engine", _ENGINES)
def test_multi_alias_whole_entity_projection_renders(engine: _GFQLEngine) -> None:
    df = _run("MATCH (a)-->(b) RETURN a, b", engine)
    got = sorted((int(r["a.id"]), int(r["b.id"])) for r in df.to_dict("records"))
    assert got == [(1, 2), (1, 3), (2, 3), (3, 4)]


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_projection_carries_every_field(engine: _GFQLEngine) -> None:
    df = _run("MATCH (a)-->(b) RETURN b", engine)
    got = sorted((int(r["b.id"]), str(r["b.name"])) for r in df.to_dict("records"))
    assert got == [(2, "Bob"), (3, "Cat"), (3, "Cat"), (4, "Dan")]


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_projection_ordered_bag(engine: _GFQLEngine) -> None:
    df = _run("MATCH (a)-->(b) RETURN b ORDER BY b.id", engine)
    assert [int(v) for v in df["b.id"]] == [2, 3, 3, 4]




@pytest.mark.parametrize("engine", _ENGINES)
def test_distinct_whole_entity_still_dedupes(engine: _GFQLEngine) -> None:
    assert _bag(_run("MATCH (a)-->(b) RETURN DISTINCT b", engine), "b.id") == [2, 3, 4]


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_projection_without_relationship_unchanged(engine: _GFQLEngine) -> None:
    assert _bag(_run("MATCH (a) RETURN a", engine), "a.id") == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_projection_after_where(engine: _GFQLEngine) -> None:
    assert _bag(_run("MATCH (a)-->(b) WHERE b.id >= 3 RETURN a", engine), "a.id") == [1, 2, 3]


@pytest.mark.parametrize("engine", _ENGINES)
def test_property_projection_bag_unchanged(engine: _GFQLEngine) -> None:
    assert _bag(_run("MATCH (a)-->(b) RETURN b.id AS x", engine), "x") == [2, 3, 3, 4]


@pytest.mark.parametrize("engine", _VARIABLE_LENGTH_ENGINES)
def test_variable_length_whole_entity_projection_unchanged(engine: _GFQLEngine) -> None:
    edges = pd.DataFrame({"s": ["p0", "p1", "p2", "p1"], "d": ["p1", "p2", "p4", "p0"]})
    _require_engine(engine)
    resolved_engine = Engine(engine)
    graph = graphistry.edges(df_to_engine(edges, resolved_engine), "s", "d").materialize_nodes(
        engine=engine
    )
    result_frame = graph.gfql("MATCH (a {id: 'p0'})-[*1..2]-(b) RETURN b", engine=engine)._nodes
    pandas_frame = to_pandas_any(result_frame)
    assert isinstance(pandas_frame, pd.DataFrame)
    assert sorted(pandas_frame["b.id"].tolist()) == ["p1", "p2"]


@pytest.mark.parametrize("engine", _ENGINES)
def test_whole_entity_carry_into_reentry_unchanged(engine: _GFQLEngine) -> None:
    df = _run(
        "MATCH (a)-->(c) WITH a AS p OPTIONAL MATCH (p)-->(z) RETURN p.id AS pid, z.id AS zid",
        engine,
    )
    assert len(df) == 4
