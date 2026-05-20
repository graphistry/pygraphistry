from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Mapping, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    PropertyAccessExpr,
    _rebuild_expr_node,
    parse_expr,
)

from .ast import (
    BooleanExpr,
    CypherQuery,
    ExpressionText,
    MatchClause,
    NodePattern,
    OrderByClause,
    ReturnClause,
    WhereClause,
)
from ._boolean_expr_text import boolean_expr_to_text
from .expression_text import render_expr_node as _render_expr_node
from .shortest_path_aliases import (
    _ShortestPathAliasSpec,
    _match_pattern_alias_kinds,
    _shortest_path_alias_specs,
)


def _unsupported(message: str, *, field: str, value: Any, line: int, column: int) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion="Use a subset currently supported by the local Cypher compiler.",
        line=line,
        column=column,
        language="cypher",
    )


def _rewrite_shortest_path_expr_node(
    node: ExprNode,
    *,
    specs: Mapping[str, _ShortestPathAliasSpec],
    field: str,
    value: str,
    line: int,
    column: int,
) -> ExprNode:
    def _hop_expr(spec: _ShortestPathAliasSpec) -> ExprNode:
        if spec.end_alias is not None:
            return PropertyAccessExpr(Identifier(spec.end_alias), spec.hop_column)
        return Identifier(spec.hop_column)

    if isinstance(node, IsNullOp) and isinstance(node.value, Identifier) and node.value.name in specs:
        return IsNullOp(_hop_expr(specs[node.value.name]), negated=node.negated)
    if isinstance(node, Identifier):
        if node.name in specs:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher shortestPath() aliases currently support only length(path) and path IS NULL in the local compiler",
                field=field,
                value=value,
                suggestion="Use length(path), path IS NULL, or CASE expressions built from those forms.",
                line=line,
                column=column,
                language="cypher",
            )
        return node
    if (
        isinstance(node, FunctionCall)
        and node.name == "length"
        and len(node.args) == 1
        and isinstance(node.args[0], Identifier)
        and node.args[0].name in specs
    ):
        return _hop_expr(specs[node.args[0].name])
    return _rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_shortest_path_expr_node(
            child,
            specs=specs,
            field=field,
            value=value,
            line=line,
            column=column,
        ),
        error_context="shortestPath rewrite",
    )


def _rewrite_shortest_path_expr_text(
    expr: ExpressionText,
    *,
    specs: Mapping[str, _ShortestPathAliasSpec],
    field: str,
) -> ExpressionText:
    if not specs:
        return expr
    node = parse_expr(expr.text)
    rewritten = _rewrite_shortest_path_expr_node(
        node,
        specs=specs,
        field=field,
        value=expr.text,
        line=expr.span.line,
        column=expr.span.column,
    )
    return ExpressionText(text=_render_expr_node(rewritten), span=expr.span)


def _rewrite_shortest_path_query(query: CypherQuery) -> CypherQuery:
    specs = _shortest_path_alias_specs(query)
    if not specs:
        return query

    rewritten_matches: List[MatchClause] = []
    for clause in query.matches:
        pattern_kinds = _match_pattern_alias_kinds(clause)
        if len(clause.patterns) <= 1 or "shortestPath" not in pattern_kinds:
            rewritten_matches.append(clause)
            continue
        if clause.optional:
            raise _unsupported(
                "Cypher shortestPath() inside OPTIONAL MATCH is not yet supported in the local compiler",
                field="match",
                value="shortestPath",
                line=clause.span.line,
                column=clause.span.column,
            )
        for idx, (pattern, kind) in enumerate(zip(clause.patterns, pattern_kinds)):
            if kind == "shortestPath":
                rewritten_matches.append(
                    MatchClause(
                        patterns=(pattern,),
                        span=clause.span,
                        optional=False,
                        pattern_aliases=((clause.pattern_aliases[idx] if idx < len(clause.pattern_aliases) else None),),
                        pattern_alias_kinds=("shortestPath",),
                    )
                )
                continue
            if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
                raise _unsupported(
                    "Cypher shortestPath() currently supports only node-only seed patterns plus the final shortestPath pattern in the local compiler",
                    field="match",
                    value=None,
                    line=clause.span.line,
                    column=clause.span.column,
                )
            rewritten_matches.append(
                MatchClause(
                    patterns=(pattern,),
                    span=clause.span,
                    optional=False,
                    pattern_aliases=(None,),
                    pattern_alias_kinds=("pattern",),
                )
            )

    def _rewrite_clause(clause: ReturnClause) -> ReturnClause:
        return replace(
            clause,
            items=tuple(
                replace(
                    item,
                    expression=_rewrite_shortest_path_expr_text(
                        item.expression,
                        specs=specs,
                        field=clause.kind,
                    ),
                )
                for item in clause.items
            ),
        )

    def _rewrite_order(order_by: Optional[OrderByClause]) -> Optional[OrderByClause]:
        if order_by is None:
            return None
        return replace(
            order_by,
            items=tuple(
                replace(
                    item,
                    expression=_rewrite_shortest_path_expr_text(
                        item.expression,
                        specs=specs,
                        field="order_by",
                    ),
                )
                for item in order_by.items
            ),
        )

    def _rewrite_where(where: Optional[WhereClause]) -> Optional[WhereClause]:
        # #1213 sub-PR C: read from ``expr_tree`` via Option B (text round-trip
        # + single-atom resynthesis).  Incidentally fixes the latent text/tree
        # staleness that the prior ``replace(where, expr=rewrite(where.expr, ...))``
        # pattern left behind.  Sub-PR D+E dropped the ``expr=`` arg here.
        if where is None or where.expr_tree is None:
            return where
        synthesized = ExpressionText(text=boolean_expr_to_text(where.expr_tree), span=where.span)
        rewritten = _rewrite_shortest_path_expr_text(synthesized, specs=specs, field="where")
        new_tree = BooleanExpr(
            op="atom",
            span=rewritten.span,
            atom_text=rewritten.text,
            atom_span=rewritten.span,
        )
        return replace(where, expr_tree=new_tree)

    return replace(
        query,
        matches=tuple(rewritten_matches),
        where=_rewrite_where(query.where),
        with_stages=tuple(
            replace(
                stage,
                clause=_rewrite_clause(stage.clause),
                where=None if stage.where is None else _rewrite_shortest_path_expr_text(
                    stage.where,
                    specs=specs,
                    field="with.where",
                ),
                order_by=_rewrite_order(stage.order_by),
            )
            for stage in query.with_stages
        ),
        return_=_rewrite_clause(query.return_),
        order_by=_rewrite_order(query.order_by),
        reentry_wheres=tuple(_rewrite_where(where) for where in query.reentry_wheres),
    )


class ASTNormalizer:
    """Owns frontend-AST rewrites that must remain behavior-preserving."""

    def rewrite_shortest_path(self, query: CypherQuery) -> CypherQuery:
        return _rewrite_shortest_path_query(query)

    def normalize(self, query: CypherQuery) -> CypherQuery:
        # Keep normalization as a single composition point for frontend AST rewrites.
        return self.rewrite_shortest_path(query)


__all__ = ["ASTNormalizer"]
