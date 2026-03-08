from __future__ import annotations

import ast as pyast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Type, Union, cast

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
from graphistry.compute.gfql.cypher.ast import (
    CypherLiteral,
    CypherPageValue,
    CypherQuery,
    LimitClause,
    LabelRef,
    OrderByClause,
    OrderItem,
    ExpressionText,
    MatchClause,
    NodePattern,
    ParameterRef,
    PatternElement,
    ProjectionStage,
    PropertyRef,
    PropertyEntry,
    RelationshipPattern,
    ReturnClause,
    ReturnItem,
    SkipClause,
    SourceSpan,
    UnwindClause,
    WhereClause,
    WherePredicate,
)


_GRAMMAR = r"""
?start: query

query: match_clause+ where_clause? unwind_clause* stage+ SEMI?
     | unwind_clause+ stage+ SEMI?
     | stage+ SEMI?

stage: with_stage
     | return_stage

with_stage: with_clause with_where_clause? order_by_clause? skip_clause? limit_clause?
return_stage: return_clause order_by_clause? skip_clause? limit_clause?

match_clause: "MATCH"i pattern ("," pattern)*
pattern: node_pattern (relationship_pattern node_pattern)*

node_pattern: "(" variable? labels? properties? ")"
labels: label+
label: ":" NAME

relationship_pattern: rel_forward
                    | rel_reverse
                    | rel_undirected
                    | rel_forward_simple
                    | rel_reverse_simple
                    | rel_undirected_simple
                    | rel_bidirectional_simple

rel_forward: "-" "[" variable? rel_types? properties? "]" "->"
rel_reverse: "<-" "[" variable? rel_types? properties? "]" "-"
rel_undirected: "-" "[" variable? rel_types? properties? "]" "-"
rel_forward_simple: REL_FWD_SIMPLE
rel_reverse_simple: REL_REV_SIMPLE
rel_undirected_simple: REL_UNDIR_SIMPLE
rel_bidirectional_simple: REL_BIDIR_SIMPLE

rel_types: ":" NAME ("|" ":"? NAME)*

variable: NAME

properties: "{" [property_entry ("," property_entry)*] "}"
property_entry: NAME ":" value

where_clause: "WHERE"i where_predicate ("AND"i where_predicate)*
where_predicate: property_ref COMP_OP where_rhs -> cmp_where
               | property_ref "IS"i "NULL"i -> is_null_where
               | property_ref "IS"i "NOT"i "NULL"i -> is_not_null_where
               | variable labels -> has_labels_where
where_rhs: property_ref
         | value

unwind_clause: "UNWIND"i unwind_expr "AS"i NAME

with_clause: "WITH"i distinct? return_item ("," return_item)*
with_where_clause: "WHERE"i expr
return_clause: "RETURN"i distinct? return_item ("," return_item)*
distinct: "DISTINCT"i
return_item: return_expr alias?
return_expr: expr
alias: "AS"i NAME

order_by_clause: "ORDER"i "BY"i order_item ("," order_item)*
order_item: order_expr order_direction?
order_direction: "ASC"i  -> asc_order
               | "ASCENDING"i -> asc_order
               | "DESC"i -> desc_order
               | "DESCENDING"i -> desc_order

skip_clause: "SKIP"i page_value
limit_clause: "LIMIT"i page_value
page_value: parameter
          | INT

qualified_name: NAME ("." NAME)*
property_ref: NAME "." NAME

unwind_expr: expr
order_expr: expr

?expr: or_expr

?or_expr: xor_expr
        | or_expr "OR"i xor_expr            -> or_op

?xor_expr: and_expr
         | xor_expr "XOR"i and_expr         -> xor_op

?and_expr: not_expr
         | and_expr "AND"i not_expr         -> and_op

?not_expr: "NOT"i not_expr                  -> not_op
         | predicate

?predicate: comparable
          | comparable COMP_OP comparable   -> cmp_op

?comparable: additive
           | additive "IS"i "NULL"i          -> expr_is_null
           | additive "IS"i "NOT"i "NULL"i   -> expr_is_not_null
           | additive "IN"i additive         -> in_op
           | additive "CONTAINS"i additive   -> contains_op
           | additive "STARTS"i "WITH"i additive -> starts_with_op
           | additive "ENDS"i "WITH"i additive -> ends_with_op

?additive: multiplicative
         | additive "+" multiplicative      -> add_op
         | additive MINUS multiplicative    -> sub_op

?multiplicative: unary
               | multiplicative "*" unary   -> mul_op
               | multiplicative "/" unary   -> div_op
               | multiplicative "%" unary   -> mod_op

?unary: "+" unary                           -> uplus
      | MINUS unary                         -> uminus
      | postfix

?postfix: primary
        | postfix "[" subscript_key "]"     -> subscript

?primary: parameter
        | literal
        | qualified_name
        | function_call
        | quantifier_expr
        | list_comprehension
        | list_literal
        | map_literal
        | "(" expr ")"                      -> grouped_expr

?subscript_key: expr                        -> subscript_index
              | expr ".." expr              -> subscript_slice_between
              | expr ".."                   -> subscript_slice_from
              | ".." expr                   -> subscript_slice_to
              | ".."                        -> subscript_slice_all

function_call: NAME "(" [func_args] ")"
?func_args: distinct_func_args
         | regular_func_args
regular_func_args: func_arg ("," func_arg)*
distinct_func_args: "DISTINCT"i func_arg
?func_arg: expr
         | "*"                              -> star_arg

list_literal: "[" [expr_list] "]"
expr_list: expr ("," expr)*

map_literal: "{" [map_entries] "}"
map_entries: map_entry ("," map_entry)*
map_entry: map_key ":" expr
map_key: NAME                               -> map_key_name
       | STRING                             -> map_key_string

quantifier_expr: "ANY"i "(" NAME "IN"i expr "WHERE"i expr ")"       -> any_quant
               | "ALL"i "(" NAME "IN"i expr "WHERE"i expr ")"       -> all_quant
               | "NONE"i "(" NAME "IN"i expr "WHERE"i expr ")"      -> none_quant
               | "SINGLE"i "(" NAME "IN"i expr "WHERE"i expr ")"    -> single_quant

list_comprehension: "[" NAME "IN"i expr "]"                          -> lc_source
                  | "[" NAME "IN"i expr "|" expr "]"                 -> lc_projection
                  | "[" NAME "IN"i expr "WHERE"i expr "]"            -> lc_where
                  | "[" NAME "IN"i expr "WHERE"i expr "|" expr "]"   -> lc_where_projection

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
MINUS: /-(?!-)/
NAME: /(?!(?i:MATCH|RETURN|WITH|ORDER|BY|SKIP|LIMIT|UNWIND|WHERE|AS|ASC|ASCENDING|DESC|DESCENDING|AND|OR|XOR|NOT|IN|IS|NULL|TRUE|FALSE|CONTAINS|STARTS|ENDS|ANY|ALL|NONE|SINGLE)\b)[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /[+-]?(?:0[xX][0-9A-Fa-f]+|0[oO][0-7]+|(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?))/
INT: /[0-9]+/
STRING : /'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/
REL_FWD_SIMPLE: /-->/
REL_REV_SIMPLE: /<--/
REL_UNDIR_SIMPLE: /--/
REL_BIDIR_SIMPLE: /<-->/
LINE_COMMENT: /\/\/[^\n]*/
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
    sign = -1 if token.startswith("-") else 1
    body = token[1:] if token[:1] in {"+", "-"} else token
    lowered = body.lower()
    if lowered.startswith("0x"):
        return sign * int(lowered[2:], 16)
    if lowered.startswith("0o"):
        return sign * int(lowered[2:], 8)
    if any(ch in body for ch in (".", "e", "E")):
        value = float(token)
        return 0.0 if value == 0.0 else value
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

        def rel_forward_simple(self, meta: Any, _items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, (), direction="forward")

        def rel_reverse_simple(self, meta: Any, _items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, (), direction="reverse")

        def rel_undirected_simple(self, meta: Any, _items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, (), direction="undirected")

        def rel_bidirectional_simple(self, meta: Any, _items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, (), direction="undirected")

        def relationship_pattern(self, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            if len(items) != 1 or not isinstance(items[0], RelationshipPattern):
                raise _to_syntax_error("Invalid relationship pattern", line=meta.line, column=meta.column)
            return cast(RelationshipPattern, items[0])

        def pattern(self, _meta: Any, items: Sequence[Any]) -> Tuple[PatternElement, ...]:
            return tuple(cast(PatternElement, item) for item in items)

        def match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) < 1:
                raise _to_syntax_error("Cypher MATCH clause cannot be empty", line=meta.line, column=meta.column)
            patterns = tuple(cast(Tuple[PatternElement, ...], item) for item in items)
            return MatchClause(patterns=patterns, span=_span_from_meta(meta))

        def distinct(self, _meta: Any, _items: Sequence[Any]) -> bool:
            return True

        def return_kw(self, _meta: Any, _items: Sequence[Any]) -> str:
            return "return"

        def with_kw(self, _meta: Any, _items: Sequence[Any]) -> str:
            return "with"

        def qualified_name(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def return_expr(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def order_expr(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def unwind_expr(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
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

        def has_labels_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 2:
                raise _to_syntax_error("Invalid WHERE label predicate", line=meta.line, column=meta.column)
            alias = str(items[0])
            labels = cast(Tuple[str, ...], items[1])
            return WherePredicate(
                left=LabelRef(alias=alias, labels=labels, span=_span_from_meta(meta)),
                op="has_labels",
                right=None,
                span=_span_from_meta(meta),
            )

        def where_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            predicates = tuple(cast(WherePredicate, item) for item in items)
            if len(predicates) == 0:
                raise _to_syntax_error("WHERE clause cannot be empty", line=meta.line, column=meta.column)
            return WhereClause(predicates=predicates, span=_span_from_meta(meta))

        def unwind_clause(self, meta: Any, items: Sequence[Any]) -> UnwindClause:
            if len(items) != 2:
                raise _to_syntax_error("Invalid UNWIND clause", line=meta.line, column=meta.column)
            expr = cast(_ExpressionSlice, items[0])
            alias = str(items[1])
            if alias == "":
                raise _to_syntax_error("UNWIND alias must be non-empty", line=meta.line, column=meta.column)
            return UnwindClause(
                expression=ExpressionText(text=expr.text, span=expr.span),
                alias=alias,
                span=_span_from_meta(meta),
            )

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

        def _projection_clause(
            self,
            *,
            meta: Any,
            items: Sequence[Any],
            kind: str,
        ) -> ReturnClause:
            distinct = False
            return_items: List[ReturnItem] = []
            for item in items:
                if isinstance(item, bool):
                    distinct = item
                else:
                    return_items.append(cast(ReturnItem, item))
            if len(return_items) == 0:
                raise _to_syntax_error(
                    f"{kind.upper()} clause must project at least one item",
                    line=meta.line,
                    column=meta.column,
                )
            return ReturnClause(
                items=tuple(return_items),
                distinct=distinct,
                kind=cast(Any, kind),
                span=_span_from_meta(meta),
            )

        def with_clause(self, meta: Any, items: Sequence[Any]) -> ReturnClause:
            return self._projection_clause(meta=meta, items=items, kind="with")

        def with_where_clause(self, meta: Any, _items: Sequence[Any]) -> ExpressionText:
            span = _span_from_meta(meta)
            return ExpressionText(text=self._slice(span)[len("WHERE"):].strip(), span=span)

        def return_clause(self, meta: Any, items: Sequence[Any]) -> ReturnClause:
            return self._projection_clause(meta=meta, items=items, kind="return")

        def asc_order(self, _meta: Any, _items: Sequence[Any]) -> str:
            return "asc"

        def desc_order(self, _meta: Any, _items: Sequence[Any]) -> str:
            return "desc"

        def order_item(self, meta: Any, items: Sequence[Any]) -> OrderItem:
            if len(items) == 0:
                raise _to_syntax_error("ORDER BY item cannot be empty", line=meta.line, column=meta.column)
            expr = cast(_ExpressionSlice, items[0])
            direction = cast(str, items[1] if len(items) > 1 else "asc")
            return OrderItem(
                expression=ExpressionText(text=expr.text, span=expr.span),
                direction=cast(Any, direction),
                span=_span_from_meta(meta),
            )

        def order_by_clause(self, meta: Any, items: Sequence[Any]) -> OrderByClause:
            order_items = tuple(cast(OrderItem, item) for item in items)
            if len(order_items) == 0:
                raise _to_syntax_error("ORDER BY clause cannot be empty", line=meta.line, column=meta.column)
            return OrderByClause(items=order_items, span=_span_from_meta(meta))

        def page_value(self, meta: Any, items: Sequence[Any]) -> CypherPageValue:
            if len(items) != 1:
                raise _to_syntax_error("Invalid page value", line=meta.line, column=meta.column)
            item = items[0]
            if isinstance(item, ParameterRef):
                return item
            return int(str(item))

        def skip_clause(self, meta: Any, items: Sequence[Any]) -> SkipClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid SKIP clause", line=meta.line, column=meta.column)
            return SkipClause(value=cast(CypherPageValue, items[0]), span=_span_from_meta(meta))

        def limit_clause(self, meta: Any, items: Sequence[Any]) -> LimitClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid LIMIT clause", line=meta.line, column=meta.column)
            return LimitClause(value=cast(CypherPageValue, items[0]), span=_span_from_meta(meta))

        def _projection_stage(self, meta: Any, items: Sequence[Any], *, expected_kind: str) -> ProjectionStage:
            clause: Optional[ReturnClause] = None
            where_expr: Optional[ExpressionText] = None
            order_by_clause: Optional[OrderByClause] = None
            skip_clause: Optional[SkipClause] = None
            limit_clause: Optional[LimitClause] = None
            for item in items:
                if isinstance(item, ReturnClause):
                    clause = item
                elif isinstance(item, ExpressionText):
                    where_expr = item
                elif isinstance(item, OrderByClause):
                    order_by_clause = item
                elif isinstance(item, SkipClause):
                    skip_clause = item
                elif isinstance(item, LimitClause):
                    limit_clause = item
            if clause is None or clause.kind != expected_kind:
                raise _to_syntax_error(
                    f"Invalid {expected_kind.upper()} stage",
                    line=meta.line,
                    column=meta.column,
                )
            return ProjectionStage(
                clause=clause,
                where=where_expr,
                order_by=order_by_clause,
                skip=skip_clause,
                limit=limit_clause,
                span=_span_from_meta(meta),
            )

        def with_stage(self, meta: Any, items: Sequence[Any]) -> ProjectionStage:
            return self._projection_stage(meta, items, expected_kind="with")

        def return_stage(self, meta: Any, items: Sequence[Any]) -> ProjectionStage:
            return self._projection_stage(meta, items, expected_kind="return")

        def stage(self, _meta: Any, items: Sequence[Any]) -> ProjectionStage:
            if len(items) != 1 or not isinstance(items[0], ProjectionStage):
                raise _to_syntax_error("Invalid projection stage")
            return items[0]

        def query(self, meta: Any, items: Sequence[Any]) -> CypherQuery:
            trailing_semicolon = any(str(item) == ";" for item in items)
            match_clauses: List[MatchClause] = []
            where_clause: Optional[WhereClause] = None
            unwind_clauses: List[UnwindClause] = []
            stages: List[ProjectionStage] = []
            for item in items:
                if isinstance(item, MatchClause):
                    match_clauses.append(item)
                elif isinstance(item, WhereClause):
                    where_clause = item
                elif isinstance(item, UnwindClause):
                    unwind_clauses.append(item)
                elif isinstance(item, ProjectionStage):
                    stages.append(item)
            if len(stages) == 0:
                raise _to_syntax_error(
                    "Cypher query must contain a RETURN/WITH clause",
                    line=meta.line,
                    column=meta.column,
                )
            return_stage: Optional[ProjectionStage] = None
            with_stages: List[ProjectionStage] = []
            for idx, stage in enumerate(stages):
                if stage.clause.kind == "return":
                    if idx != len(stages) - 1 or return_stage is not None:
                        raise _to_syntax_error(
                            "Cypher RETURN must be the final projection stage in the local compiler",
                            line=stage.span.line,
                            column=stage.span.column,
                        )
                    return_stage = stage
                elif idx != len(stages) - 1:
                    with_stages.append(stage)
            final_stage = return_stage or stages[-1]
            if where_clause is not None and not match_clauses:
                raise _to_syntax_error(
                    "Cypher WHERE is currently only supported after MATCH in the local compiler",
                    line=where_clause.span.line,
                    column=where_clause.span.column,
                )
            return CypherQuery(
                matches=tuple(match_clauses),
                where=where_clause,
                unwinds=tuple(unwind_clauses),
                with_stages=tuple(with_stages),
                return_=final_stage.clause,
                order_by=final_stage.order_by,
                skip=final_stage.skip,
                limit=final_stage.limit,
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
