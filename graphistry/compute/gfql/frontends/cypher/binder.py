"""Cypher frontend binder skeleton."""
from __future__ import annotations

from typing import Union

from graphistry.compute.gfql.cypher.ast import CypherGraphQuery, CypherQuery, CypherUnionQuery
from graphistry.compute.gfql.ir.bound_ir import BoundIR
from graphistry.compute.gfql.ir.compilation import PlanContext

CypherAST = Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]


class FrontendBinder:
    """Typed skeleton for M1 Binder extraction."""

    def bind(self, ast: CypherAST, ctx: PlanContext) -> BoundIR:
        """Bind frontend AST into frontend-neutral IR.

        M1 PR-1 intentionally ships only the hook surface; semantic binding
        behavior is deferred to follow-on PRs.
        """
        _ = (ast, ctx)
        return BoundIR()

