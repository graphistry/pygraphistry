"""Cypher frontend binder interface."""
from __future__ import annotations

from typing import Union

from graphistry.compute.gfql.cypher.ast import CypherGraphQuery, CypherQuery, CypherUnionQuery
from graphistry.compute.gfql.ir.bound_ir import BoundIR
from graphistry.compute.gfql.ir.compilation import PlanContext

CypherAST = Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]


class FrontendBinder:
    """Typed binder interface for Cypher frontend ASTs."""

    def bind(self, ast: CypherAST, ctx: PlanContext) -> BoundIR:
        """Bind frontend AST into frontend-neutral IR.

        This initial version intentionally provides the integration surface
        only; semantic binding behavior is added in later changes.
        """
        _ = (ast, ctx)
        return BoundIR()
