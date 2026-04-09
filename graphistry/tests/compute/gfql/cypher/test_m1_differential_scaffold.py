from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, cast

import pandas as pd
import pytest

from graphistry.compute.ast import ASTCall, ASTObject
from graphistry.compute.gfql.cypher.api import CompiledCypherQuery, compile_cypher
from graphistry.compute.gfql.cypher.lowering import _alias_target, _return_references_optional_only_alias, lower_match_clause
from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundVariable, ScopeFrame, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import ScalarType
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


_IC1_EXPECTED_ROWS: list[dict[str, object]] = [
    {"mid": "m1", "aid": "a1", "bid": "no-b"},
    {"mid": "m2", "aid": "no-a", "bid": "b2"},
]

_WITH_BOUNDARY_EXPECTED_ROWS: list[dict[str, object]] = [
    {"pid": "post1"},
    {"pid": "post2"},
    {"pid": "post3"},
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
        expected_rows=_WITH_BOUNDARY_EXPECTED_ROWS,
    ),
)
_CASE_BY_NAME = {case.name: case for case in _DIFF_CASES}


def _run_legacy(case: _DiffCase) -> list[dict[str, object]]:
    g = case.graph_factory()
    result = g.gfql(case.query, params=case.params)
    assert result._nodes is not None
    # Keep cast annotation as a string for Python 3.8 runtime compatibility.
    return cast("list[dict[str, object]]", result._nodes.to_dict(orient="records"))


def _run_binder_prepass_scaffold(
    case: _DiffCase,
    *,
    monkeypatch: pytest.MonkeyPatch,
    bound_ir: Optional[BoundIR] = None,
) -> list[dict[str, object]]:
    calls: list[tuple[object, PlanContext]] = []

    def _fake_bind(self: FrontendBinder, ast: object, ctx: PlanContext) -> BoundIR:
        _ = self
        calls.append((ast, ctx))
        return bound_ir if bound_ir is not None else BoundIR()

    monkeypatch.setattr(FrontendBinder, "bind", _fake_bind)
    rows = _run_legacy(case)
    assert len(calls) >= 1
    return rows


def _ic1_bound_ir_with_null_extensions() -> BoundIR:
    return BoundIR(
        semantic_table=SemanticTable(
            variables={
                "m": BoundVariable(
                    name="m",
                    logical_type=ScalarType(kind="node"),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                ),
                "a": BoundVariable(
                    name="a",
                    logical_type=ScalarType(kind="node"),
                    nullable=True,
                    null_extended_from=frozenset({"m"}),
                    entity_kind="node",
                ),
                "b": BoundVariable(
                    name="b",
                    logical_type=ScalarType(kind="node"),
                    nullable=True,
                    null_extended_from=frozenset({"m"}),
                    entity_kind="node",
                ),
            }
        ),
        scope_stack=(
            [
                ScopeFrame(
                    visible_vars=frozenset({"m", "a", "b"}),
                    schema=RowSchema(),
                    origin_clause="RETURN",
                )
            ]
        ),
    )


@pytest.mark.parametrize("case", _DIFF_CASES, ids=[case.name for case in _DIFF_CASES])
def test_diff_corpus_legacy_baseline(case: _DiffCase) -> None:
    assert _run_legacy(case) == case.expected_rows


@pytest.mark.parametrize("case", _DIFF_CASES, ids=[case.name for case in _DIFF_CASES])
def test_diff_corpus_legacy_vs_candidate(case: _DiffCase, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_binder_prepass_scaffold(case, monkeypatch=monkeypatch) == _run_legacy(case)


def test_trust_ic1_null_extended_from_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _CASE_BY_NAME["ic1-independent-optional-arms"]
    bound_ir = _ic1_bound_ir_with_null_extensions()

    # Candidate path: binder prepass is actively invoked during execution.
    assert _run_binder_prepass_scaffold(case, monkeypatch=monkeypatch, bound_ir=bound_ir) == case.expected_rows

    # Trust-but-verify: IC1 optional aliases are recognized as null-extended.
    parsed = cast(CypherQuery, parse_cypher(case.query))
    lowered_alias_targets: dict[str, ASTObject] = {}
    for clause in parsed.matches:
        lowered_alias_targets.update(_alias_target(lower_match_clause(clause)))
    nullable_aliases = {
        alias
        for alias, var in bound_ir.semantic_table.variables.items()
        if var.nullable or bool(var.null_extended_from)
    }
    assert {"a", "b"} <= nullable_aliases
    assert _return_references_optional_only_alias(
        parsed,
        alias_targets=lowered_alias_targets,
        bound_nullable_aliases=nullable_aliases,
    ) is True


def test_trust_with_boundary_binding_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _CASE_BY_NAME["with-boundary-binding-row-regression"]
    # Candidate path execution remains parity-stable.
    assert _run_binder_prepass_scaffold(case, monkeypatch=monkeypatch) == case.expected_rows

    # Trust-but-verify: compiled chain keeps bindings-row lineage through WITH boundary.
    compiled = cast(CompiledCypherQuery, compile_cypher(case.query))
    calls = [cast(ASTCall, op) for op in compiled.chain.chain if isinstance(op, ASTCall)]
    row_calls = [call for call in calls if "table" in call.params]
    assert len(row_calls) >= 1
    assert "binding_ops" in row_calls[0].params
    # Keep cast annotation as a string for Python 3.8 runtime compatibility.
    binding_ops = cast("list[dict[str, object]]", row_calls[0].params["binding_ops"])
    assert any(op.get("name") == "post" for op in binding_ops if isinstance(op, dict))
    assert any(op.get("name") == "x" for op in binding_ops if isinstance(op, dict))
