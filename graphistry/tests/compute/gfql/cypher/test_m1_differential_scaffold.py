from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, cast

import pandas as pd
import pytest

from graphistry.tests.test_compute import CGFull


class _CypherTestGraph(CGFull):
    _dgl_graph = None

    def search_graph(
        self,
        query: str,
        scale: float = 0.5,
        top_n: int = 100,
        thresh: float = 5000,
        broader: bool = False,
        inplace: bool = False,
    ):
        raise NotImplementedError

    def search(self, query: str, cols=None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10):
        raise NotImplementedError

    def embed(self, relation: str, *args, **kwargs):
        raise NotImplementedError


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    return cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


@dataclass(frozen=True)
class _DiffCase:
    name: str
    graph_factory: Callable[[], _CypherTestGraph]
    query: str
    expected_rows: list[dict[str, object]]
    params: Optional[Mapping[str, object]] = None


def _mk_ic1_two_independent_optional_arms_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["m1", "m2", "a1", "b2"],
                "label__M": [True, True, False, False],
                "label__A": [False, False, True, False],
                "label__B": [False, False, False, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["m1", "m2"],
                "d": ["a1", "b2"],
                "type": ["T1", "T2"],
            }
        ),
    )


def _mk_with_boundary_binding_row_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame(
            {
                "id": ["tagA", "tagB", "post1", "post2", "post3"],
                "label__Tag": [True, True, False, False, False],
                "label__Post": [False, False, True, True, True],
                "tagId": [1, 2, None, None, None],
                "name": ["topicA", "topicB", None, None, None],
            }
        ),
        pd.DataFrame(
            {
                "s": ["post1", "post2", "post3"],
                "d": ["tagA", "tagA", "tagB"],
                "type": ["HAS_TAG", "HAS_TAG", "HAS_TAG"],
            }
        ),
    )


_IC1_EXPECTED_ROWS = [
    {"mid": "m1", "aid": "a1", "bid": "no-b"},
    {"mid": "m2", "aid": "no-a", "bid": "b2"},
]

_DIFF_CASES: tuple[_DiffCase, ...] = (
    _DiffCase(
        name="ic1-independent-optional-arms",
        graph_factory=_mk_ic1_two_independent_optional_arms_graph,
        query=(
            "MATCH (m:M) "
            "OPTIONAL MATCH (m)-[:T1]->(a:A) "
            "OPTIONAL MATCH (m)-[:T2]->(b:B) "
            "RETURN m.id AS mid, "
            "CASE a WHEN null THEN 'no-a' ELSE a.id END AS aid, "
            "CASE b WHEN null THEN 'no-b' ELSE b.id END AS bid "
            "ORDER BY mid, aid, bid"
        ),
        expected_rows=_IC1_EXPECTED_ROWS,
    ),
    _DiffCase(
        name="ic1-independent-optional-arms-reversed-order",
        graph_factory=_mk_ic1_two_independent_optional_arms_graph,
        query=(
            "MATCH (m:M) "
            "OPTIONAL MATCH (m)-[:T2]->(b:B) "
            "OPTIONAL MATCH (m)-[:T1]->(a:A) "
            "RETURN m.id AS mid, "
            "CASE a WHEN null THEN 'no-a' ELSE a.id END AS aid, "
            "CASE b WHEN null THEN 'no-b' ELSE b.id END AS bid "
            "ORDER BY mid, aid, bid"
        ),
        expected_rows=_IC1_EXPECTED_ROWS,
    ),
    _DiffCase(
        name="with-boundary-binding-row-regression",
        graph_factory=_mk_with_boundary_binding_row_graph,
        query=(
            "MATCH (t:Tag) "
            "WITH t.tagId AS knownTagId "
            "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
            "RETURN post.id AS pid "
            "ORDER BY pid"
        ),
        expected_rows=[
            {"pid": "post1"},
            {"pid": "post2"},
            {"pid": "post3"},
        ],
    ),
)
_CASE_BY_NAME = {case.name: case for case in _DIFF_CASES}


def _run_legacy(case: _DiffCase) -> list[dict[str, object]]:
    g = case.graph_factory()
    result = g.gfql(case.query, params=case.params)
    assert result._nodes is not None
    # Keep cast annotation as a string for Python 3.8 runtime compatibility.
    return cast("list[dict[str, object]]", result._nodes.to_dict(orient="records"))


def _run_binder_prepass_scaffold(case: _DiffCase) -> list[dict[str, object]]:
    # TODO: Route this through the binder-prepass entrypoint once available.
    # Keep candidate path deterministic for now by delegating to legacy.
    return _run_legacy(case)


@pytest.mark.parametrize("case", _DIFF_CASES, ids=[case.name for case in _DIFF_CASES])
def test_diff_corpus_legacy_baseline(case: _DiffCase) -> None:
    assert _run_legacy(case) == case.expected_rows


@pytest.mark.parametrize("case", _DIFF_CASES, ids=[case.name for case in _DIFF_CASES])
def test_diff_corpus_legacy_vs_candidate(case: _DiffCase) -> None:
    assert _run_binder_prepass_scaffold(case) == _run_legacy(case)


@pytest.mark.xfail(
    reason="TODO: Assert semantic_table.null_extended_from for IC1 once binder-prepass execution path is wired",
    strict=False,
)
def test_trust_placeholder_ic1_null_extended_from_semantics() -> None:
    # Placeholder trust-but-verify target for future binder semantic assertions.
    case = _CASE_BY_NAME["ic1-independent-optional-arms"]
    assert _run_legacy(case) == case.expected_rows
    pytest.xfail("Awaiting binder semantic table exposure in execution harness")


@pytest.mark.xfail(
    reason="TODO: Assert binding-row lineage through WITH boundary once binder-prepass path is wired",
    strict=False,
)
def test_trust_placeholder_with_boundary_binding_rows() -> None:
    # Placeholder trust-but-verify target for future binding-row lineage assertions.
    case = _CASE_BY_NAME["with-boundary-binding-row-regression"]
    assert _run_legacy(case) == case.expected_rows
    pytest.xfail("Awaiting binder-path binding-row lineage checks")
