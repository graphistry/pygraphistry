"""Generic binding rows use node ids as identities.

Duplicate physical node rows do not add nodes or relationships. Match rows come
from relationship rows, including parallel relationships.
"""
from typing import Dict

import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql_unified as gfql_unified


try:
    import cudf

    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False


try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


ENGINES = [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(not HAS_CUDF, reason="cudf not installed")),
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
]


def _run_generic(engine: str, *, parallel_edge: bool) -> Dict[str, int]:
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

    if engine == "cudf":
        graph = graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(edges), "s", "d"
        )
    elif engine == "polars":
        graph = graphistry.nodes(pl.from_pandas(nodes), "id").edges(
            pl.from_pandas(edges), "s", "d"
        )
    else:
        graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    result = graph.gfql(
        "MATCH (p {kind:'P'})-->(c {kind:'C'}) "
        "RETURN c.city AS city, count(*) AS n ORDER BY city ASC",
        engine=engine,
    )._nodes
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    return {str(row.city): int(row.n) for row in result.itertuples(index=False)}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "parallel_edge,expected",
    [
        (False, {"LA": 6, "NY": 2, "SF": 1}),
        (True, {"LA": 7, "NY": 2, "SF": 1}),
    ],
)
def test_generic_binding_seed_ids_are_identities(
    engine: str, parallel_edge: bool, expected: Dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gfql_unified, "_execute_single_hop_grouped_aggregate_fast_path", lambda *args, **kwargs: None
    )
    assert _run_generic(engine, parallel_edge=parallel_edge) == expected
