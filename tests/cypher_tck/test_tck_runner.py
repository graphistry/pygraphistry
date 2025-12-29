import os
from typing import Any, Iterable, Sequence

import pandas as pd
import pytest

from graphistry.embed_utils import check_cudf
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.scenarios import SCENARIOS


_HAS_CUDF, _ = check_cudf()
_TEST_CUDF = os.environ.get("TEST_CUDF", "0") == "1"


def _df_from_records(records: Sequence[dict], required_cols: Iterable[str]) -> pd.DataFrame:
    if records:
        df = pd.DataFrame(records)
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    return pd.DataFrame(columns=list(required_cols))


def _normalize_labels(value: Any) -> Sequence[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _expand_label_columns(nodes_df: pd.DataFrame, label_col: str = "labels") -> pd.DataFrame:
    if label_col not in nodes_df.columns:
        return nodes_df
    normalized = [_normalize_labels(value) for value in nodes_df[label_col].tolist()]
    all_labels = sorted({label for labels in normalized for label in labels})
    for label in all_labels:
        nodes_df[f"label__{label}"] = [label in labels for labels in normalized]
    return nodes_df


def _build_graph(fixture: GraphFixture) -> Any:
    g = CGFull()
    nodes_df = _df_from_records(fixture.nodes, fixture.node_columns)
    nodes_df = _expand_label_columns(nodes_df)
    g = g.nodes(nodes_df, fixture.node_id)
    edges_df = _df_from_records(fixture.edges, fixture.edge_columns)
    g = g.edges(edges_df, fixture.src, fixture.dst, edge=fixture.edge_id)
    return g


def _to_pandas(df: Any) -> Any:
    if df is None:
        return None
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _ids_from_df(df: Any, id_col: str) -> set:
    if df is None:
        return set()
    pdf = _to_pandas(df)
    if pdf is None or id_col not in pdf.columns:
        return set()
    return set(pdf[id_col])


def _alias_nodes(df: Any, id_col: str, alias: str) -> set:
    if df is None:
        return set()
    pdf = _to_pandas(df)
    if pdf is None or alias not in pdf.columns:
        return set()
    return set(pdf.loc[pdf[alias].astype(bool), id_col])


def _assert_ids(
    expected: Expected,
    oracle_nodes: set,
    oracle_edges: set,
    actual_nodes: set,
    actual_edges: set,
) -> None:
    if expected.node_ids is not None:
        assert set(expected.node_ids) == oracle_nodes
        assert set(expected.node_ids) == actual_nodes
    else:
        assert oracle_nodes == actual_nodes

    if expected.edge_ids is not None:
        assert set(expected.edge_ids) == oracle_edges
        assert set(expected.edge_ids) == actual_edges
    else:
        assert oracle_edges == actual_edges


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.key)
def test_cypher_tck_scenario(scenario: Scenario) -> None:
    if scenario.status == "skip":
        pytest.skip(scenario.reason or "skipped")
    if scenario.status == "xfail":
        pytest.xfail(scenario.reason or "expected failure")

    assert scenario.gfql is not None

    g = _build_graph(scenario.graph)
    oracle = enumerate_chain(g, scenario.gfql, caps=OracleCaps(max_nodes=100, max_edges=100))

    oracle_nodes = _ids_from_df(oracle.nodes, g._node)
    oracle_edges = _ids_from_df(oracle.edges, g._edge)

    pandas_result = g.gfql(scenario.gfql, engine="pandas")
    pandas_nodes = _ids_from_df(pandas_result._nodes, g._node)
    pandas_edges = _ids_from_df(pandas_result._edges, g._edge)

    if scenario.return_alias:
        oracle_nodes = set(oracle.tags.get(scenario.return_alias, set()))
        pandas_nodes = _alias_nodes(pandas_result._nodes, g._node, scenario.return_alias)

    _assert_ids(scenario.expected, oracle_nodes, oracle_edges, pandas_nodes, pandas_edges)

    if _TEST_CUDF and _HAS_CUDF:
        cudf_result = g.gfql(scenario.gfql, engine="cudf")
        cudf_nodes = _ids_from_df(cudf_result._nodes, g._node)
        cudf_edges = _ids_from_df(cudf_result._edges, g._edge)
        if scenario.return_alias:
            cudf_nodes = _alias_nodes(cudf_result._nodes, g._node, scenario.return_alias)
        _assert_ids(scenario.expected, oracle_nodes, oracle_edges, cudf_nodes, cudf_edges)
