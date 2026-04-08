import pytest

from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR
from graphistry.compute.gfql.ir.compilation import PlanContext


def test_cypher_frontend_binder_module_imports() -> None:
    binder = FrontendBinder()
    assert isinstance(binder, FrontendBinder)


def test_cypher_frontend_binder_returns_bound_ir_placeholder() -> None:
    ast = parse_cypher("RETURN 1 AS x")
    bound = FrontendBinder().bind(ast, PlanContext())
    assert isinstance(bound, BoundIR)


def test_compile_cypher_invokes_binder_prepass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _fake_bind(self: FrontendBinder, ast: object, ctx: PlanContext) -> BoundIR:
        calls.append((ast, ctx))
        return BoundIR()

    monkeypatch.setattr(FrontendBinder, "bind", _fake_bind)
    compiled = compile_cypher("RETURN 1 AS x")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][1], PlanContext)
