from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast
from typing_extensions import Literal

from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    PropertyAccessExpr,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    UnaryOp,
    Wildcard,
    parse_expr,
)

from .ast import (
    BooleanExpr,
    CypherQuery,
    ExpressionText,
    MatchClause,
    NodePattern,
    OrderByClause,
    PathPatternKind,
    PatternElement,
    RelationshipPattern,
    ReturnClause,
    WhereClause,
    WherePatternPredicate,
)
from ._boolean_expr_text import boolean_expr_to_text


@dataclass(frozen=True)
class _ShortestPathAliasSpec:
    alias: str
    hop_column: str
    pattern: Tuple[PatternElement, ...]
    start_alias: Optional[str]
    end_alias: Optional[str]


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


def _is_variable_length_relationship_pattern(relationship: RelationshipPattern) -> bool:
    return (
        relationship.min_hops is not None
        or relationship.max_hops is not None
        or relationship.to_fixed_point
    )


def _match_pattern_alias_kinds(clause: MatchClause) -> Tuple[PathPatternKind, ...]:
    if clause.pattern_alias_kinds:
        return cast(Tuple[PathPatternKind, ...], clause.pattern_alias_kinds)
    return tuple("pattern" for _ in clause.patterns)


def _shortest_path_alias_specs(query: CypherQuery) -> Dict[str, _ShortestPathAliasSpec]:
    out: Dict[str, _ShortestPathAliasSpec] = {}
    for clause in query.matches + query.reentry_matches:
        pattern_aliases = clause.pattern_aliases or tuple(None for _ in clause.patterns)
        pattern_kinds = _match_pattern_alias_kinds(clause)
        for alias, pattern, kind in zip(pattern_aliases, clause.patterns, pattern_kinds):
            if alias is None or kind != "shortestPath":
                continue
            relationships = [element for element in pattern if isinstance(element, RelationshipPattern)]
            if len(relationships) != 1 or len(pattern) != 3:
                raise _unsupported(
                    "Cypher shortestPath() currently supports only single-relationship path patterns in the local compiler",
                    field="match",
                    value=alias,
                    line=clause.span.line,
                    column=clause.span.column,
                )
            relationship = relationships[0]
            if not _is_variable_length_relationship_pattern(relationship):
                raise _unsupported(
                    "Cypher shortestPath() requires a variable-length relationship pattern in the local compiler",
                    field="match",
                    value=alias,
                    line=clause.span.line,
                    column=clause.span.column,
                )
            start_alias = pattern[0].variable if isinstance(pattern[0], NodePattern) else None
            end_alias = pattern[-1].variable if isinstance(pattern[-1], NodePattern) else None
            out[alias] = _ShortestPathAliasSpec(
                alias=alias,
                hop_column=f"__cypher_shortest_path_hops__{alias}",
                pattern=pattern,
                start_alias=start_alias,
                end_alias=end_alias,
            )
    return out


def _cypher_literal_expr_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "null"
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_cypher_literal_expr_text(item) for item in value) + "]"
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            key_txt = str(key)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_txt):
                rendered_key = key_txt
            else:
                rendered_key = "'" + key_txt.replace("\\", "\\\\").replace("'", "\\'") + "'"
            parts.append(f"{rendered_key}: {_cypher_literal_expr_text(item)}")
        return "{" + ", ".join(parts) + "}"
    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher parameter value is outside the currently supported literal subset",
        field="params",
        value=type(value).__name__,
        suggestion="Use null, booleans, numbers, strings, lists, or maps as parameter values.",
        language="cypher",
    )


def _render_expr_node(node: ExprNode) -> str:
    if isinstance(node, Identifier):
        return cast(str, node.name)
    if isinstance(node, ExprLiteral):
        return _cypher_literal_expr_text(node.value)
    if isinstance(node, UnaryOp):
        operand = _render_expr_node(node.operand)
        if node.op == "not":
            return f"(NOT {operand})"
        return f"({node.op}{operand})"
    if isinstance(node, BinaryOp):
        left = _render_expr_node(node.left)
        right = _render_expr_node(node.right)
        if node.op in {"and", "or", "xor", "in"}:
            op_txt = node.op.upper()
        elif node.op == "starts_with":
            op_txt = "STARTS WITH"
        elif node.op == "ends_with":
            op_txt = "ENDS WITH"
        elif node.op == "contains":
            op_txt = "CONTAINS"
        else:
            op_txt = node.op
        return f"({left} {op_txt} {right})"
    if isinstance(node, IsNullOp):
        suffix = "IS NOT NULL" if node.negated else "IS NULL"
        return f"({_render_expr_node(node.value)} {suffix})"
    if isinstance(node, FunctionCall):
        args = ", ".join(_render_expr_node(arg) for arg in node.args)
        if node.distinct:
            args = f"DISTINCT {args}"
        return f"{node.name}({args})"
    if isinstance(node, Wildcard):
        return "*"
    if isinstance(node, CaseWhen):
        return (
            "CASE WHEN "
            f"{_render_expr_node(node.condition)} THEN {_render_expr_node(node.when_true)} "
            f"ELSE {_render_expr_node(node.when_false)} END"
        )
    if isinstance(node, QuantifierExpr):
        return (
            f"{node.fn.upper()}({node.var} IN {_render_expr_node(node.source)} "
            f"WHERE {_render_expr_node(node.predicate)})"
        )
    if isinstance(node, ListComprehension):
        rendered = f"[{node.var} IN {_render_expr_node(node.source)}"
        if node.predicate is not None:
            rendered += f" WHERE {_render_expr_node(node.predicate)}"
        if node.projection is not None:
            rendered += f" | {_render_expr_node(node.projection)}"
        return rendered + "]"
    if isinstance(node, ListLiteral):
        return "[" + ", ".join(_render_expr_node(item) for item in node.items) + "]"
    if isinstance(node, MapLiteral):
        parts: List[str] = []
        for key, value in node.items:
            rendered_key = key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else _cypher_literal_expr_text(key)
            parts.append(f"{rendered_key}: {_render_expr_node(value)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, SubscriptExpr):
        return f"{_render_expr_node(node.value)}[{_render_expr_node(node.key)}]"
    if isinstance(node, SliceExpr):
        start = "" if node.start is None else _render_expr_node(node.start)
        stop = "" if node.stop is None else _render_expr_node(node.stop)
        return f"{_render_expr_node(node.value)}[{start}..{stop}]"
    if isinstance(node, PropertyAccessExpr):
        return f"{_render_expr_node(node.value)}.{node.property}"
    raise TypeError(f"Unsupported expression node type for rendering: {type(node).__name__}")


def _rebuild_expr_node(
    node: ExprNode,
    *,
    rewrite: Callable[[ExprNode], ExprNode],
    error_context: str,
) -> ExprNode:
    if isinstance(node, (Identifier, ExprLiteral, Wildcard)):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, rewrite(node.operand))
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, rewrite(node.left), rewrite(node.right))
    if isinstance(node, IsNullOp):
        return IsNullOp(rewrite(node.value), negated=node.negated)
    if isinstance(node, FunctionCall):
        return FunctionCall(node.name, tuple(rewrite(arg) for arg in node.args), distinct=node.distinct)
    if isinstance(node, CaseWhen):
        return CaseWhen(
            rewrite(node.condition),
            rewrite(node.when_true),
            rewrite(node.when_false),
        )
    if isinstance(node, QuantifierExpr):
        return QuantifierExpr(
            node.fn,
            node.var,
            rewrite(node.source),
            rewrite(node.predicate),
        )
    if isinstance(node, ListComprehension):
        return ListComprehension(
            node.var,
            rewrite(node.source),
            predicate=None if node.predicate is None else rewrite(node.predicate),
            projection=None if node.projection is None else rewrite(node.projection),
        )
    if isinstance(node, ListLiteral):
        return ListLiteral(tuple(rewrite(item) for item in node.items))
    if isinstance(node, MapLiteral):
        return MapLiteral(tuple((key, rewrite(value)) for key, value in node.items))
    if isinstance(node, SubscriptExpr):
        return SubscriptExpr(rewrite(node.value), rewrite(node.key))
    if isinstance(node, SliceExpr):
        return SliceExpr(
            rewrite(node.value),
            None if node.start is None else rewrite(node.start),
            None if node.stop is None else rewrite(node.stop),
        )
    if isinstance(node, PropertyAccessExpr):
        return PropertyAccessExpr(rewrite(node.value), node.property)
    raise TypeError(f"Unsupported expression node in {error_context}: {type(node).__name__}")


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
        # #1213 sub-PR C: read from ``expr_tree`` (single source of truth post-#1214)
        # via Option B (text round-trip + single-atom resynthesis).  Keeps the
        # ``(expr is None) == (expr_tree is None)`` invariant intact and
        # incidentally fixes the latent text/tree staleness that the prior
        # ``replace(where, expr=rewrite(where.expr, ...))`` left behind.
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
        return replace(where, expr=rewritten, expr_tree=new_tree)

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


def _rewrite_where_pattern_predicates_to_matches(query: CypherQuery) -> CypherQuery:
    if query.where is None or not query.where.predicates:
        return query
    pattern_preds = [predicate for predicate in query.where.predicates if isinstance(predicate, WherePatternPredicate)]
    if not pattern_preds:
        return query
    first = pattern_preds[0]
    if len(pattern_preds) > 1:
        raise _unsupported(
            "Cypher WHERE currently supports one positive pattern predicate at a time",
            field="where",
            value=len(pattern_preds),
            line=first.span.line,
            column=first.span.column,
        )
    if len(first.pattern) < 3:
        raise _unsupported(
            "Cypher WHERE pattern predicates must include a relationship",
            field="where",
            value=None,
            line=first.span.line,
            column=first.span.column,
        )
    bound_aliases = {
        cast(str, element.variable)
        for clause in query.matches
        for pattern in clause.patterns
        for element in pattern
        if getattr(element, "variable", None) is not None
    }
    introduced_aliases = sorted(
        cast(str, element.variable)
        for element in first.pattern
        if getattr(element, "variable", None) is not None and cast(str, element.variable) not in bound_aliases
    )
    if introduced_aliases:
        raise _unsupported(
            "Cypher WHERE pattern predicates cannot introduce new aliases in this phase",
            field="where",
            value=introduced_aliases,
            line=first.span.line,
            column=first.span.column,
        )

    remaining = tuple(predicate for predicate in query.where.predicates if not isinstance(predicate, WherePatternPredicate))
    remaining_where = None
    if remaining or query.where.expr_tree is not None:
        remaining_where = WhereClause(
            predicates=cast(Any, remaining),
            expr=query.where.expr,  # constructor passthrough; sub-PR D drops this arg
            expr_tree=query.where.expr_tree,
            span=query.where.span,
        )
    extra_match = MatchClause(
        patterns=(first.pattern,),
        span=first.span,
        optional=False,
        pattern_aliases=(None,),
        pattern_alias_kinds=("pattern",),
    )
    return replace(query, matches=query.matches + (extra_match,), where=remaining_where)


class ASTNormalizer:
    """Owns frontend-AST rewrites that must remain behavior-preserving."""

    def rewrite_shortest_path(self, query: CypherQuery) -> CypherQuery:
        return _rewrite_shortest_path_query(query)

    def rewrite_where_pattern_predicates(self, query: CypherQuery) -> CypherQuery:
        return _rewrite_where_pattern_predicates_to_matches(query)

    def normalize(self, query: CypherQuery) -> CypherQuery:
        query = self.rewrite_shortest_path(query)
        query = self.rewrite_where_pattern_predicates(query)
        return query


__all__ = ["ASTNormalizer"]
