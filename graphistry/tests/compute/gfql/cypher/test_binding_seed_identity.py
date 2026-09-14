"""Generic binding rows use node ids as identities."""
from __future__ import annotations

import typing

import pandas as pd
import pytest
from typing_extensions import Literal

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.Plottable import Plottable
from graphistry.tests.compute.gfql.engagement import assert_fast_path
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason, to_pandas_any


_GFQLEngine = Literal["pandas", "polars", "cudf", "polars-gpu"]
_ENGINES: typing.Tuple[_GFQLEngine, ...] = ("pandas", "polars", "cudf", "polars-gpu")


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, engine: _GFQLEngine) -> Plottable:
    resolved_engine = Engine(engine)
    return graphistry.nodes(df_to_engine(nodes, resolved_engine), "id").edges(
        df_to_engine(edges, resolved_engine), "s", "d"
    )


def _smoke(engine: _GFQLEngine) -> Plottable:
    nodes = pd.DataFrame({"id": [1, 2]})
    edges = pd.DataFrame({"s": [1], "d": [2]})
    return _bind(nodes, edges, engine).gfql("MATCH (n) RETURN n.id AS id", engine=engine)


def _require_engine(engine: _GFQLEngine) -> None:
    skip_reason = engine_skip_reason(engine, lambda: _smoke(engine))
    if skip_reason is not None:
        pytest.skip(skip_reason)


def _run_generic(engine: _GFQLEngine, *, parallel_edge: bool) -> typing.Mapping[str, int]:
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2],
        "kind": ["P", "P", "P", "P", "C", "C", "C", "C", "P", "P"],
        "city": [None, None, None, None, "LA", "NY", "SF", "LA", None, None],
    })
    edges = pd.DataFrame({
        "s": [1, 2, 3, 4, 1, 2, 3, 4, 1],
        "d": [5, 5, 6, 7, 8, 6, 8, 8, 5],
    })
    if parallel_edge:
        edges = pd.concat(
            [edges, pd.DataFrame({"s": [1], "d": [5]})],
            ignore_index=True,
        )

    _require_engine(engine)
    graph = _bind(nodes, edges, engine)
    query = (
        "MATCH (p {kind:'P'})-->(c {kind:'C'}) "
        "RETURN c.city AS city, count(*) AS n ORDER BY city ASC SKIP 0"
    )
    assert_fast_path(
        graph, query, "single_hop_grouped_aggregate", served=False, engine=engine
    )
    result_frame = graph.gfql(query, engine=engine)._nodes
    pandas_frame = to_pandas_any(result_frame)
    assert isinstance(pandas_frame, pd.DataFrame)
    return {str(row.city): int(row.n) for row in pandas_frame.itertuples(index=False)}


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize(
    "parallel_edge,expected",
    [
        (False, {"LA": 6, "NY": 2, "SF": 1}),
        (True, {"LA": 7, "NY": 2, "SF": 1}),
    ],
)
def test_generic_binding_seed_ids_are_identities(
    engine: _GFQLEngine,
    parallel_edge: bool,
    expected: typing.Mapping[str, int],
) -> None:
    assert _run_generic(engine, parallel_edge=parallel_edge) == expected
