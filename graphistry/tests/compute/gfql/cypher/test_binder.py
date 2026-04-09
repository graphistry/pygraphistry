import pytest

from graphistry.compute.gfql.cypher.api import compile_cypher, cypher_to_gfql
from graphistry.compute.gfql.cypher.ast import CypherGraphQuery, CypherUnionQuery
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR
from graphistry.compute.gfql.ir.compilation import PlanContext
from typing import Any, List, Tuple


def test_cypher_frontend_binder_module_imports() -> None:
    binder = FrontendBinder()
    assert isinstance(binder, FrontendBinder)


def test_cypher_frontend_binder_returns_bound_ir_placeholder() -> None:
    ast = parse_cypher("RETURN 1 AS x")
    bound = FrontendBinder().bind(ast, PlanContext())
    assert isinstance(bound, BoundIR)


def _capture_binder_calls(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[object, PlanContext]]:
    calls: List[Tuple[object, PlanContext]] = []

    def _fake_bind(self: FrontendBinder, ast: object, ctx: PlanContext) -> BoundIR:
        calls.append((ast, ctx))
        return BoundIR()

    monkeypatch.setattr(FrontendBinder, "bind", _fake_bind)
    return calls


def test_compile_cypher_invokes_binder_prepass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("RETURN 1 AS x")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][1], PlanContext)


def test_cypher_to_gfql_invokes_binder_prepass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    chain = cypher_to_gfql("RETURN 1 AS x")
    assert chain is not None
    assert len(calls) >= 1


def test_compile_cypher_union_invokes_binder_with_union_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("RETURN 1 AS x UNION RETURN 2 AS x")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][0], CypherUnionQuery)


def test_compile_cypher_graph_constructor_invokes_binder_with_graph_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("GRAPH { MATCH (a)-[r]->(b) }")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][0], CypherGraphQuery)


def test_binder_does_not_mutate_input_ast() -> None:
    ast = parse_cypher("MATCH (n) RETURN n")
    before = repr(ast)
    bound = FrontendBinder().bind(ast, PlanContext())
    assert isinstance(bound, BoundIR)
    assert repr(ast) == before
