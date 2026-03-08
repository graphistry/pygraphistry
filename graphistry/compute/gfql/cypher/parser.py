from __future__ import annotations

import ast as pyast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Type, Union, cast

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    CypherQuery,
    ExpressionText,
    MatchClause,
    NodePattern,
    ParameterRef,
    PatternElement,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    SourceSpan,
    WhereClause,
    WherePredicate,
)


_GRAMMAR = r"""
?start: query

query: match_clause where_clause? return_clause SEMI?

match_clause: "MATCH"i pattern
pattern: node_pattern (relationship_pattern node_pattern)*

node_pattern: "(" variable? labels? properties? ")"
labels: label+
label: ":" NAME

relationship_pattern: rel_forward
                    | rel_reverse
                    | rel_undirected

rel_forward: "-" "[" variable? rel_types? properties? "]" "->"
rel_reverse: "<-" "[" variable? rel_types? properties? "]" "-"
rel_undirected: "-" "[" variable? rel_types? properties? "]" "-"

rel_types: rel_type+
rel_type: ":" NAME

variable: NAME

properties: "{" [property_entry ("," property_entry)*] "}"
property_entry: NAME ":" value

where_clause: "WHERE"i where_predicate ("AND"i where_predicate)*
where_predicate: property_ref COMP_OP where_rhs -> cmp_where
               | property_ref "IS"i "NULL"i -> is_null_where
               | property_ref "IS"i "NOT"i "NULL"i -> is_not_null_where
where_rhs: property_ref
         | value

return_clause: "RETURN"i distinct? return_item ("," return_item)*
distinct: "DISTINCT"i
return_item: return_expr alias?
return_expr: qualified_name
alias: "AS"i NAME

qualified_name: NAME ("." NAME)*
property_ref: NAME "." NAME

?value: parameter
      | literal

parameter: "$" NAME

literal: "NULL"i    -> null_lit
       | "TRUE"i    -> true_lit
       | "FALSE"i   -> false_lit
       | NUMBER     -> number_lit
       | STRING     -> string_lit

COMP_OP: "=" | "<>" | "!=" | "<=" | "<" | ">=" | ">"

SEMI: ";"
NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?/
STRING : /'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/
LINE_COMMENT: /--[^\n]*/
BLOCK_COMMENT: /\/\*[\s\S]*?\*\//
%import common.WS
%ignore WS
%ignore LINE_COMMENT
%ignore BLOCK_COMMENT
"""


class _ParserLike:
    def parse(self, text: str) -> object:
        raise NotImplementedError


class _TransformerLike(Protocol):
    def transform(self, tree: object) -> object:
        ...


def _lark_imports() -> Tuple[type, type, Type[Exception], Any]:
    try:
        from lark import Lark, Transformer, v_args
        from lark.exceptions import LarkError
        return Lark, Transformer, LarkError, v_args
    except Exception as exc:
        raise ImportError(
            "Lark is required for Cypher parsing. Install dependency 'lark'."
        ) from exc


def _span_from_meta(meta: Any) -> SourceSpan:
    return SourceSpan(
        line=int(getattr(meta, "line", 1)),
        column=int(getattr(meta, "column", 1)),
        end_line=int(getattr(meta, "end_line", getattr(meta, "line", 1))),
        end_column=int(getattr(meta, "end_column", getattr(meta, "column", 1))),
        start_pos=int(getattr(meta, "start_pos", 0)),
        end_pos=int(getattr(meta, "end_pos", 0)),
    )


def _parse_string_token(token: str) -> str:
    try:
        value = pyast.literal_eval(token)
    except Exception as exc:
        raise ValueError("Invalid string literal") from exc
    if not isinstance(value, str):
        raise ValueError("Invalid string literal")
    return value


def _parse_number_token(token: str) -> Union[int, float]:
    if any(ch in token for ch in (".", "e", "E")):
        return float(token)
    return int(token)


@dataclass(frozen=True)
class _ExpressionSlice:
    text: str
    span: SourceSpan


@lru_cache(maxsize=1)
def _parser() -> _ParserLike:
    Lark, _, _, _ = _lark_imports()
    parser = Lark(
        _GRAMMAR,
        start="start",
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )
    return cast(_ParserLike, parser)


def _to_syntax_error(message: str, *, line: Optional[int] = None, column: Optional[int] = None, **extra: Any) -> GFQLSyntaxError:
    return GFQLSyntaxError(
        ErrorCode.E107,
        message,
        suggestion="Check Cypher clause structure and punctuation.",
        line=line,
        column=column,
        language="cypher",
        **extra,
    )


def _build_transformer(source: str) -> _TransformerLike:
    _, Transformer, _, v_args = _lark_imports()
    op_map = {
        "=": "==",
        "!=": "!=",
        "<>": "!=",
        "<": "<",
        "<=": "<=",
        ">": ">",
        ">=": ">=",
    }

    @v_args(meta=True)  # type: ignore[misc]
    class _CypherAstBuilder(Transformer):  # type: ignore[valid-type,misc]
        def _slice(self, span: SourceSpan) -> str:
            return source[span.start_pos:span.end_pos]

        def variable(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid variable reference", line=meta.line, column=meta.column)
            return str(items[0])

        def label(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid node label", line=meta.line, column=meta.column)
            return str(items[0])

        def labels(self, _meta: Any, items: Sequence[Any]) -> Tuple[str, ...]:
            return tuple(str(item) for item in items)

        def rel_type(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid relationship type", line=meta.line, column=meta.column)
            return str(items[0])

        def rel_types(self, _meta: Any, items: Sequence[Any]) -> Tuple[str, ...]:
            return tuple(str(item) for item in items)

        def parameter(self, meta: Any, items: Sequence[Any]) -> ParameterRef:
            if len(items) != 1:
                raise _to_syntax_error("Invalid parameter reference", line=meta.line, column=meta.column)
            return ParameterRef(name=str(items[0]), span=_span_from_meta(meta))

        def null_lit(self, _meta: Any, _items: Sequence[Any]) -> None:
            return None

        def true_lit(self, _meta: Any, _items: Sequence[Any]) -> bool:
            return True

        def false_lit(self, _meta: Any, _items: Sequence[Any]) -> bool:
            return False

        def number_lit(self, meta: Any, items: Sequence[Any]) -> Union[int, float]:
            if len(items) != 1:
                raise _to_syntax_error("Invalid numeric literal", line=meta.line, column=meta.column)
            return _parse_number_token(str(items[0]))

        def string_lit(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid string literal", line=meta.line, column=meta.column)
            try:
                return _parse_string_token(str(items[0]))
            except ValueError as exc:
                raise _to_syntax_error(str(exc), line=meta.line, column=meta.column) from exc

        def property_entry(self, meta: Any, items: Sequence[Any]) -> PropertyEntry:
            if len(items) != 2:
                raise _to_syntax_error("Invalid property entry", line=meta.line, column=meta.column)
            key = str(items[0])
            value = cast(CypherLiteral, items[1])
            return PropertyEntry(key=key, value=value, span=_span_from_meta(meta))

        def properties(self, _meta: Any, items: Sequence[Any]) -> Tuple[PropertyEntry, ...]:
            return tuple(cast(PropertyEntry, item) for item in items)

        def node_pattern(self, meta: Any, items: Sequence[Any]) -> NodePattern:
            variable: Optional[str] = None
            labels: Tuple[str, ...] = ()
            properties: Tuple[PropertyEntry, ...] = ()
            for item in items:
                if isinstance(item, str):
                    variable = item
                elif isinstance(item, tuple) and all(isinstance(v, str) for v in item):
                    labels = cast(Tuple[str, ...], item)
                elif isinstance(item, tuple):
                    properties = cast(Tuple[PropertyEntry, ...], item)
            return NodePattern(
                variable=variable,
                labels=labels,
                properties=properties,
                span=_span_from_meta(meta),
            )

        def _relationship(
            self, meta: Any, items: Sequence[Any], *, direction: str
        ) -> RelationshipPattern:
            variable: Optional[str] = None
            rel_types: Tuple[str, ...] = ()
            properties: Tuple[PropertyEntry, ...] = ()
            for item in items:
                if isinstance(item, str):
                    variable = item
                elif isinstance(item, tuple) and all(isinstance(v, str) for v in item):
                    rel_types = cast(Tuple[str, ...], item)
                elif isinstance(item, tuple):
                    properties = cast(Tuple[PropertyEntry, ...], item)
            return RelationshipPattern(
                direction=cast(Any, direction),
                variable=variable,
                types=rel_types,
                properties=properties,
                span=_span_from_meta(meta),
            )

        def rel_forward(self, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, items, direction="forward")

        def rel_reverse(self, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, items, direction="reverse")

        def rel_undirected(self, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, items, direction="undirected")

        def relationship_pattern(self, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            if len(items) != 1 or not isinstance(items[0], RelationshipPattern):
                raise _to_syntax_error("Invalid relationship pattern", line=meta.line, column=meta.column)
            return cast(RelationshipPattern, items[0])

        def pattern(self, _meta: Any, items: Sequence[Any]) -> Tuple[PatternElement, ...]:
            return tuple(cast(PatternElement, item) for item in items)

        def match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) != 1:
                raise _to_syntax_error("Cypher MATCH clause must contain exactly one linear pattern", line=meta.line, column=meta.column)
            pattern = cast(Tuple[PatternElement, ...], items[0])
            return MatchClause(pattern=pattern, span=_span_from_meta(meta))

        def distinct(self, _meta: Any, _items: Sequence[Any]) -> bool:
            return True

        def qualified_name(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def property_ref(self, meta: Any, items: Sequence[Any]) -> PropertyRef:
            if len(items) != 2:
                raise _to_syntax_error("Invalid property reference", line=meta.line, column=meta.column)
            return PropertyRef(alias=str(items[0]), property=str(items[1]), span=_span_from_meta(meta))

        def where_rhs(self, _meta: Any, items: Sequence[Any]) -> object:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE right-hand side")
            return items[0]

        def cmp_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 3:
                raise _to_syntax_error("Invalid WHERE comparison", line=meta.line, column=meta.column)
            left = cast(PropertyRef, items[0])
            op_token = str(items[1])
            right = cast(object, items[2])
            op = op_map.get(op_token)
            if op is None:
                raise _to_syntax_error("Unsupported WHERE comparison operator", line=meta.line, column=meta.column)
            return WherePredicate(
                left=left,
                op=cast(Any, op),
                right=cast(Any, right),
                span=_span_from_meta(meta),
            )

        def is_null_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE IS NULL predicate", line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op="is_null",
                right=None,
                span=_span_from_meta(meta),
            )

        def is_not_null_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE IS NOT NULL predicate", line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op="is_not_null",
                right=None,
                span=_span_from_meta(meta),
            )

        def where_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            predicates = tuple(cast(WherePredicate, item) for item in items)
            if len(predicates) == 0:
                raise _to_syntax_error("WHERE clause cannot be empty", line=meta.line, column=meta.column)
            return WhereClause(predicates=predicates, span=_span_from_meta(meta))

        def return_expr(self, _meta: Any, items: Sequence[Any]) -> _ExpressionSlice:
            if len(items) != 1:
                raise _to_syntax_error("Invalid RETURN expression")
            return cast(_ExpressionSlice, items[0])

        def alias(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid RETURN alias", line=meta.line, column=meta.column)
            return str(items[0])

        def return_item(self, meta: Any, items: Sequence[Any]) -> ReturnItem:
            if len(items) == 0:
                raise _to_syntax_error("RETURN item cannot be empty", line=meta.line, column=meta.column)
            expr = cast(_ExpressionSlice, items[0])
            alias = cast(Optional[str], items[1] if len(items) > 1 else None)
            return ReturnItem(
                expression=ExpressionText(text=expr.text, span=expr.span),
                alias=alias,
                span=_span_from_meta(meta),
            )

        def return_clause(self, meta: Any, items: Sequence[Any]) -> ReturnClause:
            distinct = False
            return_items: List[ReturnItem] = []
            for item in items:
                if isinstance(item, bool):
                    distinct = item
                else:
                    return_items.append(cast(ReturnItem, item))
            if len(return_items) == 0:
                raise _to_syntax_error("RETURN clause must project at least one item", line=meta.line, column=meta.column)
            return ReturnClause(
                items=tuple(return_items),
                distinct=distinct,
                span=_span_from_meta(meta),
            )

        def query(self, meta: Any, items: Sequence[Any]) -> CypherQuery:
            if len(items) < 2:
                raise _to_syntax_error("Cypher query must contain MATCH and RETURN clauses", line=meta.line, column=meta.column)
            trailing_semicolon = any(str(item) == ";" for item in items)
            match_clause = cast(MatchClause, items[0])
            where_clause = cast(Optional[WhereClause], items[1] if len(items) > 2 and isinstance(items[1], WhereClause) else None)
            return_clause = cast(ReturnClause, items[2] if where_clause is not None else items[1])
            return CypherQuery(
                match=match_clause,
                where=where_clause,
                return_=return_clause,
                trailing_semicolon=trailing_semicolon,
                span=_span_from_meta(meta),
            )

    return cast(_TransformerLike, _CypherAstBuilder())


def parse_cypher(query: str) -> CypherQuery:
    if not isinstance(query, str) or query.strip() == "":
        raise _to_syntax_error("Cypher query must be a non-empty string")

    parser = _parser()
    transformer = _build_transformer(query)
    try:
        tree = parser.parse(query)
        node = transformer.transform(tree)
    except Exception as exc:
        _, _, LarkError, _ = _lark_imports()
        if isinstance(exc, GFQLSyntaxError):
            raise
        if isinstance(exc, LarkError):
            line = getattr(exc, "line", None)
            column = getattr(exc, "column", None)
            raise _to_syntax_error("Invalid Cypher query syntax", line=line, column=column) from exc
        raise _to_syntax_error("Invalid Cypher query syntax") from exc

    if not isinstance(node, CypherQuery):
        raise _to_syntax_error("Cypher parser did not produce a query")
    return node
