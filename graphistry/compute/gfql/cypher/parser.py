from __future__ import annotations

import ast as pyast
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union, cast

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import (
    CallClause,
    CypherLiteral,
    CypherPageValue,
    CypherQuery,
    CypherUnionQuery,
    CypherYieldItem,
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
    WherePatternPredicate,
    WherePredicate,
)


_GRAMMAR = r"""
?start: union_query

union_query: query_body (union_op query_body)* SEMI?
query_body: query_item+
union_op: "UNION"i "ALL"i       -> union_all
        | "UNION"i              -> union_distinct

query_item: match_clause
          | where_clause
          | call_clause
          | unwind_clause
          | stage

stage: with_stage
     | return_stage

with_stage: with_clause with_where_clause? order_by_clause? skip_clause? limit_clause?
return_stage: return_clause order_by_clause? skip_clause? limit_clause?

match_clause: "MATCH"i match_item ("," match_item)*              -> match_clause
            | "OPTIONAL"i "MATCH"i match_item ("," match_item)*  -> optional_match_clause
match_item: pattern
          | NAME "=" pattern               -> bound_pattern
pattern: node_pattern (relationship_pattern node_pattern)*

node_pattern: "(" variable? labels? properties? ")"
labels: label+
label: ":" LABEL_NAME

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

rel_types: ":" LABEL_NAME ("|" ":"? LABEL_NAME)*

variable: NAME

properties: "{" [property_entry ("," property_entry)*] "}"
property_entry: NAME ":" value

where_clause: "WHERE"i WHERE_PATTERN -> where_pattern_only_clause
            | "WHERE"i where_predicates
            | "WHERE"i expr                -> generic_where_clause
where_predicates: where_predicate ("AND"i where_predicate)*
where_predicate: property_ref COMP_OP where_rhs -> cmp_where
               | property_ref "IS"i "NULL"i -> is_null_where
               | property_ref "IS"i "NOT"i "NULL"i -> is_not_null_where
               | property_ref "CONTAINS"i where_rhs -> contains_where
               | property_ref "STARTS"i "WITH"i where_rhs -> starts_with_where
               | property_ref "ENDS"i "WITH"i where_rhs -> ends_with_where
               | variable labels -> has_labels_where
where_rhs: property_ref
         | value

unwind_clause: "UNWIND"i unwind_expr "AS"i NAME
call_clause: "CALL"i qualified_name call_invocation? yield_clause?
call_invocation: "(" [call_args] ")"
call_args: call_arg ("," call_arg)*
call_arg: expr
yield_clause: "YIELD"i yield_item ("," yield_item)*
yield_item: NAME alias?

with_clause: "WITH"i distinct? return_item ("," return_item)*
with_where_clause: "WHERE"i expr
return_clause: "RETURN"i distinct? return_item ("," return_item)*
distinct: "DISTINCT"i
return_item: return_expr alias?
return_expr: "*"                              -> projection_star
           | label_predicate_expr
           | expr
label_predicate_expr: "(" NAME labels ")"    -> grouped_label_predicate
bare_label_predicate_expr: NAME labels       -> bare_label_predicate
alias: "AS"i NAME

order_by_clause: "ORDER"i "BY"i order_item ("," order_item)*
order_item: order_expr order_direction?
order_direction: "ASC"i  -> asc_order
               | "ASCENDING"i -> asc_order
               | "DESC"i -> desc_order
               | "DESCENDING"i -> desc_order

skip_clause: "SKIP"i expr
limit_clause: "LIMIT"i expr

qualified_name: NAME ("." NAME)*
property_ref.2: NAME "." NAME

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
          | comparable COMP_OP comparable COMP_OP comparable -> chained_cmp
          | comparable COMP_OP comparable   -> cmp_op
          | bare_label_predicate_expr

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
        | postfix "." NAME                  -> property_access

?primary: parameter
        | literal
        | function_call
        | qualified_name
        | case_expr
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

function_call: qualified_name "(" [func_args] ")"
?func_args: distinct_func_args
         | regular_func_args
regular_func_args: func_arg ("," func_arg)*
distinct_func_args: "DISTINCT"i func_arg
?func_arg: expr
         | "*"                              -> star_arg

case_expr: searched_case_expr
         | simple_case_expr
searched_case_expr: "CASE"i case_when+ case_else? "END"i
simple_case_expr: "CASE"i expr case_when+ case_else? "END"i
case_when: "WHEN"i expr "THEN"i expr
case_else: "ELSE"i expr

list_literal: "[" [expr_list] "]"
expr_list: expr ("," expr)*

map_literal: "{" [map_entries] "}"
map_entries: map_entry ("," map_entry)*
map_entry: map_key ":" expr
map_key: MAP_KEY_NAME                       -> map_key_name
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
NAME: /(?!(?i:MATCH|RETURN|WITH|ORDER|BY|SKIP|LIMIT|UNWIND|WHERE|AS|ASC|ASCENDING|DESC|DESCENDING|AND|OR|XOR|NOT|IN|IS|NULL|TRUE|FALSE|CONTAINS|STARTS|ENDS|ANY|ALL|NONE|SINGLE|CASE|WHEN|THEN|ELSE|END)\b)[A-Za-z_][A-Za-z0-9_]*/
MAP_KEY_NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /[+-]?(?:0[xX][0-9A-Fa-f]+|0[oO][0-7]+|(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?))/
INT: /[0-9]+/
STRING : /'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/
WHERE_PATTERN: /\([^)\n]*\)\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\)(?:\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\))*?(?:\s+AND\s+\([^)\n]*\)\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\)(?:\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\))*)*/
REL_FWD_SIMPLE: /-->/
REL_REV_SIMPLE: /<--/
REL_UNDIR_SIMPLE: /--/
REL_BIDIR_SIMPLE: /<-->/
LABEL_NAME: /[A-Za-z_][A-Za-z0-9_]*/
LINE_COMMENT: /\/\/[^\n]*/
BLOCK_COMMENT: /\/\*[\s\S]*?\*\//
%import common.WS
%ignore WS
%ignore LINE_COMMENT
%ignore BLOCK_COMMENT
"""

_BARE_LABEL_PREDICATE_RE = re.compile(r"^(?P<alias>[A-Za-z_][A-Za-z0-9_]*)((?::[A-Za-z_][A-Za-z0-9_]*)+)$")
_WHERE_PATTERN_ITEM_RE = re.compile(
    r"\([^)\n]*\)\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\)"
    r"(?:\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\))*"
)


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


@lru_cache(maxsize=1)
def _pattern_parser() -> _ParserLike:
    Lark, _, _, _ = _lark_imports()
    parser = Lark(
        _GRAMMAR,
        start="pattern",
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


def _to_unsupported(message: str, *, line: Optional[int] = None, column: Optional[int] = None, **extra: Any) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        suggestion="Use a subset currently supported by the local Cypher compiler.",
        line=line,
        column=column,
        language="cypher",
        **extra,
    )


_VARIABLE_REL_PATTERN_RE = re.compile(
    r"(?:<-\s*\[[^\]\n]*\*[^\]\n]*\]\s*-)|(?:-\s*\[[^\]\n]*\*[^\]\n]*\]\s*->)|(?:-\s*\[[^\]\n]*\*[^\]\n]*\]\s*-)"
)


def _line_and_column_from_offset(source: str, offset: int) -> Tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    column = offset + 1 if last_newline < 0 else offset - last_newline
    return line, column


def _find_variable_length_relationship_pattern(source: str) -> Optional[Tuple[str, int, int]]:
    in_single_quote = False
    escape = False
    segment_start = 0
    for idx, ch in enumerate(source):
        if in_single_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_single_quote = False
                segment_start = idx + 1
            continue
        if ch == "'":
            match = _VARIABLE_REL_PATTERN_RE.search(source, segment_start, idx)
            if match is not None:
                line, column = _line_and_column_from_offset(source, match.start())
                return match.group(0), line, column
            in_single_quote = True
            continue
    match = _VARIABLE_REL_PATTERN_RE.search(source, segment_start)
    if match is None:
        return None
    line, column = _line_and_column_from_offset(source, match.start())
    return match.group(0), line, column


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

        def bound_pattern(self, _meta: Any, items: Sequence[Any]) -> Tuple[PatternElement, ...]:
            if len(items) != 2:
                raise _to_syntax_error("Invalid bound MATCH pattern")
            return cast(Tuple[PatternElement, ...], items[1])

        def match_item(self, _meta: Any, items: Sequence[Any]) -> Tuple[PatternElement, ...]:
            if len(items) != 1:
                raise _to_syntax_error("Invalid MATCH pattern")
            return cast(Tuple[PatternElement, ...], items[0])

        def match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) < 1:
                raise _to_syntax_error("Cypher MATCH clause cannot be empty", line=meta.line, column=meta.column)
            patterns = tuple(cast(Tuple[PatternElement, ...], item) for item in items)
            return MatchClause(patterns=patterns, span=_span_from_meta(meta))

        def optional_match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) < 1:
                raise _to_syntax_error("Cypher OPTIONAL MATCH clause cannot be empty", line=meta.line, column=meta.column)
            patterns = tuple(cast(Tuple[PatternElement, ...], item) for item in items)
            return MatchClause(patterns=patterns, span=_span_from_meta(meta), optional=True)

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

        def contains_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 2:
                raise _to_syntax_error("Invalid WHERE CONTAINS predicate", line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op="contains",
                right=cast(Any, items[1]),
                span=_span_from_meta(meta),
            )

        def starts_with_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 2:
                raise _to_syntax_error("Invalid WHERE STARTS WITH predicate", line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op="starts_with",
                right=cast(Any, items[1]),
                span=_span_from_meta(meta),
            )

        def ends_with_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            if len(items) != 2:
                raise _to_syntax_error("Invalid WHERE ENDS WITH predicate", line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op="ends_with",
                right=cast(Any, items[1]),
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

        def where_predicates(self, _meta: Any, items: Sequence[Any]) -> Tuple[WherePredicate, ...]:
            return tuple(items)

        def where_pattern_only_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE pattern predicate", line=meta.line, column=meta.column)
            pattern_text = str(items[0]).strip()
            span = _span_from_meta(meta)
            pattern_items = [match.group(0).strip() for match in _WHERE_PATTERN_ITEM_RE.finditer(pattern_text)]
            if len(pattern_items) != 1:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE currently supports one positive pattern predicate at a time",
                    field="where",
                    value=pattern_text,
                    suggestion="Use a single positive relationship existence pattern in WHERE for the local compiler subset.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                )
            try:
                pattern_tree = _pattern_parser().parse(pattern_items[0])
                pattern_node = _build_transformer(pattern_items[0]).transform(pattern_tree)
            except Exception as exc:
                _, _, LarkError, _ = _lark_imports()
                if isinstance(exc, (GFQLSyntaxError, GFQLValidationError)):
                    raise
                if isinstance(exc, LarkError):
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "Cypher WHERE pattern predicate is outside the currently supported local subset",
                        field="where",
                        value=pattern_text,
                        suggestion="Use a single positive fixed-length relationship existence pattern in WHERE.",
                        line=span.line,
                        column=span.column,
                        language="cypher",
                    ) from exc
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE pattern predicate is outside the currently supported local subset",
                    field="where",
                    value=pattern_text,
                    suggestion="Use a single positive fixed-length relationship existence pattern in WHERE.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                ) from exc
            if not isinstance(pattern_node, tuple):
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE pattern predicate is outside the currently supported local subset",
                    field="where",
                    value=pattern_text,
                    suggestion="Use a single positive fixed-length relationship existence pattern in WHERE.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                )
            return WhereClause(
                predicates=(WherePatternPredicate(pattern=cast(Tuple[PatternElement, ...], pattern_node), span=span),),
                expr=None,
                span=span,
            )

        def where_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            if len(items) != 1:
                raise _to_syntax_error("WHERE clause cannot be empty", line=meta.line, column=meta.column)
            if isinstance(items[0], _ExpressionSlice):
                expr = cast(_ExpressionSlice, items[0])
                return WhereClause(predicates=(), expr=ExpressionText(text=expr.text, span=expr.span), span=_span_from_meta(meta))
            predicates = cast(Tuple[WherePredicate, ...], items[0])
            if len(predicates) == 0:
                raise _to_syntax_error("WHERE clause cannot be empty", line=meta.line, column=meta.column)
            return WhereClause(predicates=cast(Any, predicates), expr=None, span=_span_from_meta(meta))

        def generic_where_clause(self, meta: Any, _items: Sequence[Any]) -> WhereClause:
            span = _span_from_meta(meta)
            expr = self._slice(span)[len("WHERE"):].strip()
            bare_label_match = _BARE_LABEL_PREDICATE_RE.fullmatch(expr)
            if bare_label_match is not None:
                labels = tuple(label for label in bare_label_match.group(2).split(":") if label)
                return WhereClause(
                    predicates=(
                        WherePredicate(
                            left=LabelRef(alias=bare_label_match.group("alias"), labels=labels, span=span),
                            op="has_labels",
                            right=None,
                            span=span,
                        ),
                    ),
                    expr=None,
                    span=span,
                )
            return WhereClause(predicates=(), expr=ExpressionText(text=expr, span=span), span=span)

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

        def call_arg(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def call_args(self, _meta: Any, items: Sequence[Any]) -> Tuple[_ExpressionSlice, ...]:
            return tuple(cast(_ExpressionSlice, item) for item in items)

        def call_invocation(self, _meta: Any, items: Sequence[Any]) -> Tuple[_ExpressionSlice, ...]:
            if len(items) == 0:
                return ()
            if len(items) != 1 or not isinstance(items[0], tuple):
                raise _to_syntax_error("Invalid CALL invocation")
            return cast(Tuple[_ExpressionSlice, ...], items[0])

        def yield_item(self, meta: Any, items: Sequence[Any]) -> CypherYieldItem:
            if len(items) == 0:
                raise _to_syntax_error("YIELD item cannot be empty", line=meta.line, column=meta.column)
            return CypherYieldItem(
                name=str(items[0]),
                alias=cast(Optional[str], items[1] if len(items) > 1 else None),
                span=_span_from_meta(meta),
            )

        def yield_clause(self, _meta: Any, items: Sequence[Any]) -> Tuple[CypherYieldItem, ...]:
            return tuple(cast(CypherYieldItem, item) for item in items)

        def call_clause(self, meta: Any, items: Sequence[Any]) -> CallClause:
            if len(items) == 0:
                raise _to_syntax_error("CALL clause cannot be empty", line=meta.line, column=meta.column)
            procedure_item = cast(_ExpressionSlice, items[0])
            call_args: Tuple[_ExpressionSlice, ...] = ()
            yield_items: Tuple[CypherYieldItem, ...] = ()
            for item in items[1:]:
                if isinstance(item, tuple) and all(isinstance(arg, _ExpressionSlice) for arg in item):
                    call_args = cast(Tuple[_ExpressionSlice, ...], item)
                elif isinstance(item, tuple) and all(isinstance(yield_item, CypherYieldItem) for yield_item in item):
                    yield_items = cast(Tuple[CypherYieldItem, ...], item)
            return CallClause(
                procedure=procedure_item.text,
                args=tuple(
                    ExpressionText(text=arg.text, span=arg.span)
                    for arg in call_args
                ),
                yield_items=yield_items,
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

        def projection_star(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text="*", span=span)

        def grouped_label_predicate(self, meta: Any, items: Sequence[Any]) -> _ExpressionSlice:
            if len(items) != 2:
                raise _to_syntax_error("Invalid label predicate expression", line=meta.line, column=meta.column)
            labels = cast(Sequence[str], items[1])
            if len(labels) == 0:
                raise _to_syntax_error("Label predicate must reference at least one label", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span), span=span)

        def bare_label_predicate(self, meta: Any, items: Sequence[Any]) -> _ExpressionSlice:
            if len(items) != 2:
                raise _to_syntax_error("Invalid label predicate expression", line=meta.line, column=meta.column)
            labels = cast(Sequence[str], items[1])
            if len(labels) == 0:
                raise _to_syntax_error("Label predicate must reference at least one label", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span), span=span)

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

        def skip_clause(self, meta: Any, items: Sequence[Any]) -> SkipClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid SKIP clause", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return SkipClause(
                value=ExpressionText(text=self._slice(span)[len("SKIP"):].strip(), span=span),
                span=span,
            )

        def limit_clause(self, meta: Any, items: Sequence[Any]) -> LimitClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid LIMIT clause", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return LimitClause(
                value=ExpressionText(text=self._slice(span)[len("LIMIT"):].strip(), span=span),
                span=span,
            )

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

        def query_item(self, meta: Any, items: Sequence[Any]) -> Any:
            if len(items) != 1:
                raise _to_syntax_error("Invalid query item", line=meta.line, column=meta.column)
            return items[0]

        def query_body(self, meta: Any, items: Sequence[Any]) -> CypherQuery:
            trailing_semicolon = any(str(item) == ";" for item in items)
            match_clauses: List[MatchClause] = []
            reentry_match_clauses: List[MatchClause] = []
            where_clause: Optional[WhereClause] = None
            reentry_where_clause: Optional[WhereClause] = None
            call_clause: Optional[CallClause] = None
            unwind_clauses: List[UnwindClause] = []
            stages: List[ProjectionStage] = []
            ordered_row_items: List[Union[ProjectionStage, UnwindClause]] = []
            seen_stage = False
            for item in items:
                if isinstance(item, MatchClause):
                    if seen_stage:
                        reentry_match_clauses.append(item)
                    else:
                        match_clauses.append(item)
                elif isinstance(item, WhereClause):
                    if call_clause is not None:
                        raise _to_syntax_error(
                            "Cypher WHERE is not supported with CALL in the local compiler; use YIELD/RETURN row expressions instead",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if reentry_match_clauses:
                        if reentry_where_clause is not None:
                            raise _to_syntax_error(
                                "Cypher only supports one WHERE clause after post-WITH MATCH in the local compiler",
                                line=item.span.line,
                                column=item.span.column,
                            )
                        reentry_where_clause = item
                    else:
                        where_clause = item
                elif isinstance(item, CallClause):
                    if call_clause is not None:
                        raise _to_syntax_error(
                            "Cypher only supports one CALL clause per query in the local compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if match_clauses or reentry_match_clauses or stages or unwind_clauses:
                        raise _to_syntax_error(
                            "Cypher CALL is currently only supported as the first clause in standalone or row-only local queries",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    call_clause = item
                elif isinstance(item, UnwindClause):
                    if reentry_match_clauses:
                        raise _to_syntax_error(
                            "Cypher UNWIND after post-WITH MATCH is not yet supported in the local compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if match_clauses and seen_stage:
                        raise _to_unsupported(
                            "Cypher UNWIND after WITH/RETURN is not yet supported once MATCH has introduced graph aliases",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    unwind_clauses.append(item)
                    ordered_row_items.append(item)
                elif isinstance(item, ProjectionStage):
                    if call_clause is not None and reentry_match_clauses:
                        raise _to_syntax_error(
                            "Cypher CALL with MATCH re-entry is not yet supported in the local compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if reentry_match_clauses and item.clause.kind != "return":
                        raise _to_syntax_error(
                            "Cypher WITH after post-WITH MATCH is not yet supported in the local compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    stages.append(item)
                    ordered_row_items.append(item)
                    seen_stage = True
            if len(stages) == 0 and call_clause is None:
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
            if reentry_match_clauses:
                if len(stages) != 2 or stages[0].clause.kind != "with" or stages[-1].clause.kind != "return":
                    first_match = reentry_match_clauses[0]
                    raise _to_syntax_error(
                        "Cypher MATCH after WITH is only supported for a single MATCH ... WITH ... MATCH ... RETURN shape in the local compiler",
                        line=first_match.span.line,
                        column=first_match.span.column,
                    )
            final_stage: Optional[ProjectionStage] = return_stage or (stages[-1] if stages else None)
            if where_clause is not None and not match_clauses:
                raise _to_syntax_error(
                    "Cypher WHERE is currently only supported after MATCH in the local compiler",
                    line=where_clause.span.line,
                    column=where_clause.span.column,
                )
            row_sequence: Tuple[Union[ProjectionStage, UnwindClause], ...] = ()
            if call_clause is not None and len(stages) == 0:
                synthetic_item = ReturnItem(
                    expression=ExpressionText(text="*", span=call_clause.span),
                    alias=None,
                    span=call_clause.span,
                )
                synthetic_clause = ReturnClause(
                    items=(synthetic_item,),
                    distinct=False,
                    kind="return",
                    span=call_clause.span,
                )
                synthetic_stage = ProjectionStage(
                    clause=synthetic_clause,
                    where=None,
                    order_by=None,
                    skip=None,
                    limit=None,
                    span=call_clause.span,
                )
                final_stage = synthetic_stage
                ordered_row_items.append(synthetic_stage)
            if final_stage is None:
                raise _to_syntax_error(
                    "Cypher query must contain a RETURN/WITH clause or a supported CALL",
                    line=meta.line,
                    column=meta.column,
                )
            if not match_clauses and where_clause is None:
                row_sequence = tuple(ordered_row_items)
            return CypherQuery(
                matches=tuple(match_clauses),
                where=where_clause,
                call=call_clause,
                unwinds=tuple(unwind_clauses),
                with_stages=tuple(with_stages),
                return_=final_stage.clause,
                order_by=final_stage.order_by,
                skip=final_stage.skip,
                limit=final_stage.limit,
                row_sequence=row_sequence,
                trailing_semicolon=trailing_semicolon,
                span=_span_from_meta(meta),
                reentry_matches=tuple(reentry_match_clauses),
                reentry_where=reentry_where_clause,
            )

        def union_all(self, meta: Any, _items: Sequence[Any]) -> str:
            if meta.empty:
                return "all"
            return "all"

        def union_distinct(self, meta: Any, _items: Sequence[Any]) -> str:
            if meta.empty:
                return "distinct"
            return "distinct"

        def union_query(self, meta: Any, items: Sequence[Any]) -> Union[CypherQuery, CypherUnionQuery]:
            trailing_semicolon = any(str(item) == ";" for item in items)
            branches: List[CypherQuery] = []
            union_kinds: List[str] = []
            for item in items:
                if isinstance(item, CypherQuery):
                    branches.append(item)
                elif isinstance(item, str) and item in {"distinct", "all"}:
                    union_kinds.append(item)
            if not branches:
                raise _to_syntax_error(
                    "Cypher query must contain a RETURN/WITH clause",
                    line=meta.line,
                    column=meta.column,
                )
            if len(branches) == 1:
                branch = branches[0]
                if trailing_semicolon and not branch.trailing_semicolon:
                    return CypherQuery(
                        matches=branch.matches,
                        where=branch.where,
                        call=branch.call,
                        unwinds=branch.unwinds,
                        with_stages=branch.with_stages,
                        return_=branch.return_,
                        order_by=branch.order_by,
                        skip=branch.skip,
                        limit=branch.limit,
                        row_sequence=branch.row_sequence,
                        trailing_semicolon=True,
                        span=branch.span,
                        reentry_matches=branch.reentry_matches,
                        reentry_where=branch.reentry_where,
                    )
                return branch
            if len(union_kinds) != len(branches) - 1:
                raise _to_syntax_error("Invalid UNION query", line=meta.line, column=meta.column)
            union_kind_set = set(union_kinds)
            if len(union_kind_set) != 1:
                raise _to_syntax_error(
                    "Mixing UNION and UNION ALL is not supported in the local compiler",
                    line=meta.line,
                    column=meta.column,
                )
            union_kind = cast(Union[Literal["distinct"], Literal["all"]], union_kinds[0])
            return CypherUnionQuery(
                branches=tuple(branches),
                union_kind=union_kind,
                trailing_semicolon=trailing_semicolon,
                span=_span_from_meta(meta),
            )

    return cast(_TransformerLike, _CypherAstBuilder())


def parse_cypher(query: str) -> Union[CypherQuery, CypherUnionQuery]:
    """Parse supported local Cypher text into a typed AST.

    The returned AST preserves the clause structure needed by the local GFQL
    compiler, including unions and row-pipeline stages.

    :param query: Local Cypher text to parse.
    :returns: A parsed ``CypherQuery`` or ``CypherUnionQuery``.
    :raises GFQLSyntaxError: If the query is not valid within the supported
        local Cypher grammar.
    """
    if not isinstance(query, str) or query.strip() == "":
        raise _to_syntax_error("Cypher query must be a non-empty string")
    variable_length_pattern = _find_variable_length_relationship_pattern(query)
    if variable_length_pattern is not None:
        pattern_text, line, column = variable_length_pattern
        raise _to_unsupported(
            "Cypher variable-length relationship patterns are not yet supported in the local compiler",
            line=line,
            column=column,
            field="match",
            value=pattern_text,
        )

    parser = _parser()
    transformer = _build_transformer(query)
    try:
        tree = parser.parse(query)
        node = transformer.transform(tree)
    except Exception as exc:
        _, _, LarkError, _ = _lark_imports()
        orig_exc = getattr(exc, "orig_exc", None)
        if isinstance(orig_exc, (GFQLSyntaxError, GFQLValidationError)):
            raise orig_exc
        if isinstance(exc, (GFQLSyntaxError, GFQLValidationError)):
            raise
        if isinstance(exc, LarkError):
            err_line = cast(Optional[int], getattr(exc, "line", None))
            err_column = cast(Optional[int], getattr(exc, "column", None))
            raise _to_syntax_error("Invalid Cypher query syntax", line=err_line, column=err_column) from exc
        raise _to_syntax_error("Invalid Cypher query syntax") from exc

    if not isinstance(node, (CypherQuery, CypherUnionQuery)):
        raise _to_syntax_error("Cypher parser did not produce a query")
    return node
