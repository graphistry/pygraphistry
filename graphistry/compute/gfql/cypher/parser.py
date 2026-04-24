from __future__ import annotations

import ast as pyast
from dataclasses import dataclass, replace
from functools import lru_cache
import re
from typing import Any, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union, cast

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.compute.gfql.expr_split import split_top_level_and
from graphistry.compute.gfql.cypher.ast import (
    BooleanExpr,
    CallClause,
    CypherGraphQuery,
    CypherLiteral,
    CypherPropertyValue,
    CypherPageValue,
    CypherQuery,
    CypherUnionQuery,
    CypherYieldItem,
    GraphBinding,
    GraphConstructor,
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
    UseClause,
    WhereClause,
    WherePatternPredicate,
    WherePredicate,
)


_GRAMMAR = r"""
?start: graph_query

graph_query: graph_binding* union_query          -> graph_query_with_bindings
           | graph_binding* graph_constructor     -> graph_query_standalone

graph_binding: "GRAPH"i NAME "=" graph_constructor

graph_constructor: "GRAPH"i "{" graph_constructor_body "}"

graph_constructor_body: graph_constructor_item+

graph_constructor_item: match_clause
                      | where_clause
                      | call_clause
                      | use_clause

use_clause: "USE"i NAME

union_query: query_body (union_op query_body)* SEMI?
query_body: query_item+
union_op: "UNION"i "ALL"i       -> union_all
        | "UNION"i              -> union_distinct

query_item: match_clause
          | where_clause
          | call_clause
          | unwind_clause
          | use_clause
          | stage

stage: with_stage
     | return_stage

with_stage: with_clause with_where_clause? order_by_clause? skip_clause? limit_clause?
return_stage: return_clause order_by_clause? skip_clause? limit_clause?

match_clause: "MATCH"i match_item ("," match_item)*              -> match_clause
            | "OPTIONAL"i "MATCH"i match_item ("," match_item)*  -> optional_match_clause
match_item: pattern
          | NAME "=" pattern               -> bound_pattern
          | NAME "=" "shortestPath"i "(" pattern ")"          -> shortest_path_pattern
          | NAME "=" "allShortestPaths"i "(" pattern ")"      -> all_shortest_paths_pattern
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

rel_forward: "-" "[" variable? rel_types? rel_range? properties? "]" "->"
rel_reverse: "<-" "[" variable? rel_types? rel_range? properties? "]" "-"
rel_undirected: "-" "[" variable? rel_types? rel_range? properties? "]" "-"
rel_forward_simple: REL_FWD_SIMPLE
rel_reverse_simple: REL_REV_SIMPLE
rel_undirected_simple: REL_UNDIR_SIMPLE
rel_bidirectional_simple: REL_BIDIR_SIMPLE

rel_types: ":" LABEL_NAME ("|" ":"? LABEL_NAME)*
rel_range: "*" INT ".." INT      -> rel_range_bounded
         | "*" INT ".."          -> rel_range_open_max
         | "*" INT               -> rel_range_exact
         | "*"                   -> rel_range_fixed

variable: NAME

properties: "{" [property_entry ("," property_entry)*] "}"
property_entry: NAME ":" expr

where_clause: "WHERE"i WHERE_PATTERN "AND"i expr -> where_pattern_and_expr_clause
            | "WHERE"i expr "AND"i WHERE_PATTERN -> expr_and_where_pattern_clause
            | "WHERE"i WHERE_PATTERN -> where_pattern_only_clause
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
_RESERVED_IDENTIFIER_GRAPH = "graph"
_WHERE_PATTERN_ITEM_RE = re.compile(
    r"\([^)\n]*\)\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\)"
    r"(?:\s*(?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)\s*\([^)\n]*\))*"
)
_WHERE_PATTERN_SEQUENCE_RE = re.compile(
    rf"(?:{_WHERE_PATTERN_ITEM_RE.pattern})(?:\s+AND\s+(?:{_WHERE_PATTERN_ITEM_RE.pattern}))*",
    re.IGNORECASE,
)
_WHERE_PATTERN_THEN_EXPR_RE = re.compile(
    rf"^(?P<pattern>{_WHERE_PATTERN_SEQUENCE_RE.pattern})\s+AND\s+(?P<expr>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_WHERE_EXPR_THEN_PATTERN_RE = re.compile(
    rf"^(?P<expr>.+)\s+AND\s+(?P<pattern>{_WHERE_PATTERN_SEQUENCE_RE.pattern})$",
    re.IGNORECASE | re.DOTALL,
)
_WHERE_CLAUSE_BODY_RE = re.compile(
    r"\bWHERE\b(?P<body>.*?)(?=\bRETURN\b|\bWITH\b|\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b|\bUNWIND\b|\bCALL\b|\bMATCH\b|\bOPTIONAL\s+MATCH\b|\bUNION\b|;|$)",
    re.IGNORECASE | re.DOTALL,
)
_BOOLEAN_KEYWORD_RE = re.compile(r"\b(?:AND|OR|XOR|NOT)\b", re.IGNORECASE)


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


@dataclass(frozen=True)
class _BoundPattern:
    alias: str
    pattern: Tuple[PatternElement, ...]
    kind: Literal["pattern", "shortestPath", "allShortestPaths"] = "pattern"


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
        suggestion="Use a subset currently supported by the GFQL Cypher compiler.",
        line=line,
        column=column,
        language="cypher",
        **extra,
    )


def _line_and_column_from_offset(source: str, offset: int) -> Tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    last_newline = source.rfind("\n", 0, offset)
    column = offset + 1 if last_newline < 0 else offset - last_newline
    return line, column


def _mixed_where_pattern_expr_error(source: str) -> Optional[GFQLValidationError]:
    for match in _WHERE_CLAUSE_BODY_RE.finditer(source):
        body = match.group("body").strip()
        if body == "":
            continue
        if _WHERE_PATTERN_ITEM_RE.search(body) is None:
            continue
        if _WHERE_PATTERN_SEQUENCE_RE.fullmatch(body) is not None:
            continue
        if _BOOLEAN_KEYWORD_RE.search(body) is None:
            continue
        line, column = _line_and_column_from_offset(source, match.start("body"))
        return _to_unsupported(
            "Cypher WHERE pattern predicates cannot yet be mixed with generic row expressions",
            line=line,
            column=column,
            field="where",
            value=body,
        )
    return None


def _canonicalize_where_single_pattern_and_expr(source: str) -> Optional[str]:
    for match in _WHERE_CLAUSE_BODY_RE.finditer(source):
        body = match.group("body").strip()
        terms = split_top_level_and(body)
        if len(terms) <= 1:
            continue
        pattern_indices = [idx for idx, term in enumerate(terms) if _WHERE_PATTERN_SEQUENCE_RE.fullmatch(term) is not None]
        if len(pattern_indices) != 1:
            continue
        pattern_index = pattern_indices[0]
        pattern_text = terms[pattern_index].strip()
        expr_terms = [term for idx, term in enumerate(terms) if idx != pattern_index]
        if not expr_terms:
            continue
        canonical_body = f"{pattern_text} AND {' AND '.join(expr_terms)}"
        if canonical_body == body:
            continue
        return f"{source[:match.start('body')]}{canonical_body}{source[match.end('body'):]}"
    return None


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
            raw_value = items[1]
            if isinstance(raw_value, (ParameterRef, type(None), bool, int, float, str)):
                value = cast(CypherPropertyValue, raw_value)
            elif isinstance(raw_value, PropertyRef):
                value = ExpressionText(
                    text=f"{raw_value.alias}.{raw_value.property}",
                    span=raw_value.span,
                )
            elif isinstance(raw_value, _ExpressionSlice):
                value = ExpressionText(text=raw_value.text, span=raw_value.span)
            else:
                span = _span_from_meta(meta)
                entry_text = self._slice(span)
                value = ExpressionText(
                    text=entry_text.split(":", 1)[1].strip(),
                    span=span,
                )
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
            min_hops: Optional[int] = None
            max_hops: Optional[int] = None
            to_fixed_point = False
            for item in items:
                if isinstance(item, str):
                    variable = item
                elif isinstance(item, dict):
                    min_hops = cast(Optional[int], item.get("min_hops"))
                    max_hops = cast(Optional[int], item.get("max_hops"))
                    to_fixed_point = bool(item.get("to_fixed_point", False))
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
                min_hops=min_hops,
                max_hops=max_hops,
                to_fixed_point=to_fixed_point,
            )

        def _rel_hops(self, meta: Any, token: Any) -> int:
            try:
                value = int(str(token))
            except Exception as exc:
                raise _to_syntax_error("Invalid relationship range bound", line=meta.line, column=meta.column) from exc
            if value < 0:
                raise _to_syntax_error(
                    "Cypher relationship range bounds must be non-negative",
                    line=meta.line,
                    column=meta.column,
                )
            return value

        def rel_range_exact(self, meta: Any, items: Sequence[Any]) -> dict[str, Any]:
            if len(items) != 1:
                raise _to_syntax_error("Invalid relationship range", line=meta.line, column=meta.column)
            hops = self._rel_hops(meta, items[0])
            if hops == 0:
                raise _to_unsupported(
                    "Cypher exact zero-hop relationship patterns (*0) are not supported",
                    line=meta.line,
                    column=meta.column,
                    field="match",
                    value=self._slice(_span_from_meta(meta)),
                )
            return {"min_hops": hops, "max_hops": hops, "to_fixed_point": False}

        def rel_range_bounded(self, meta: Any, items: Sequence[Any]) -> dict[str, Any]:
            if len(items) != 2:
                raise _to_syntax_error("Invalid relationship range", line=meta.line, column=meta.column)
            min_hops = self._rel_hops(meta, items[0])
            max_hops = self._rel_hops(meta, items[1])
            if min_hops > max_hops:
                raise _to_unsupported(
                    "Cypher relationship ranges require lower bound <= upper bound",
                    line=meta.line,
                    column=meta.column,
                    field="match",
                    value=self._slice(_span_from_meta(meta)),
                )
            return {"min_hops": min_hops, "max_hops": max_hops, "to_fixed_point": False}

        def rel_range_open_max(self, meta: Any, items: Sequence[Any]) -> dict[str, Any]:
            if len(items) != 1:
                raise _to_syntax_error("Invalid relationship range", line=meta.line, column=meta.column)
            try:
                value = int(str(items[0]))
            except Exception as exc:
                raise _to_syntax_error("Invalid relationship range bound", line=meta.line, column=meta.column) from exc
            if value < 0:
                raise _to_unsupported(
                    "Cypher negative-hop relationship ranges are not supported",
                    line=meta.line,
                    column=meta.column,
                    field="match",
                    value=self._slice(_span_from_meta(meta)),
                )
            return {"min_hops": value, "max_hops": None, "to_fixed_point": True}

        def rel_range_fixed(self, meta: Any, _items: Sequence[Any]) -> dict[str, Any]:
            return {"min_hops": None, "max_hops": None, "to_fixed_point": True}

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

        def bound_pattern(self, _meta: Any, items: Sequence[Any]) -> _BoundPattern:
            if len(items) != 2:
                raise _to_syntax_error("Invalid bound MATCH pattern")
            return _BoundPattern(alias=str(items[0]), pattern=cast(Tuple[PatternElement, ...], items[1]))

        def shortest_path_pattern(self, meta: Any, items: Sequence[Any]) -> _BoundPattern:
            if len(items) != 2:
                raise _to_syntax_error("Invalid shortestPath() MATCH pattern")
            return _BoundPattern(
                alias=str(items[0]),
                pattern=cast(Tuple[PatternElement, ...], items[1]),
                kind="shortestPath",
            )

        def all_shortest_paths_pattern(self, meta: Any, items: Sequence[Any]) -> _BoundPattern:
            raise _to_unsupported(
                "allShortestPaths() is not yet supported in the local Cypher compiler",
                line=meta.line,
                column=meta.column,
                field="match",
                value="allShortestPaths",
            )

        def match_item(self, _meta: Any, items: Sequence[Any]) -> Union[Tuple[PatternElement, ...], _BoundPattern]:
            if len(items) != 1:
                raise _to_syntax_error("Invalid MATCH pattern")
            return cast(Union[Tuple[PatternElement, ...], _BoundPattern], items[0])

        def match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) < 1:
                raise _to_syntax_error("Cypher MATCH clause cannot be empty", line=meta.line, column=meta.column)
            patterns: List[Tuple[PatternElement, ...]] = []
            pattern_aliases: List[Optional[str]] = []
            pattern_alias_kinds: List[Literal["pattern", "shortestPath", "allShortestPaths"]] = []
            for item in items:
                if isinstance(item, _BoundPattern):
                    patterns.append(item.pattern)
                    pattern_aliases.append(item.alias)
                    pattern_alias_kinds.append(item.kind)
                else:
                    patterns.append(cast(Tuple[PatternElement, ...], item))
                    pattern_aliases.append(None)
                    pattern_alias_kinds.append("pattern")
            return MatchClause(
                patterns=tuple(patterns),
                span=_span_from_meta(meta),
                pattern_aliases=tuple(pattern_aliases),
                pattern_alias_kinds=tuple(pattern_alias_kinds),
            )

        def optional_match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            if len(items) < 1:
                raise _to_syntax_error("Cypher OPTIONAL MATCH clause cannot be empty", line=meta.line, column=meta.column)
            patterns: List[Tuple[PatternElement, ...]] = []
            pattern_aliases: List[Optional[str]] = []
            pattern_alias_kinds: List[Literal["pattern", "shortestPath", "allShortestPaths"]] = []
            for item in items:
                if isinstance(item, _BoundPattern):
                    patterns.append(item.pattern)
                    pattern_aliases.append(item.alias)
                    pattern_alias_kinds.append(item.kind)
                else:
                    patterns.append(cast(Tuple[PatternElement, ...], item))
                    pattern_aliases.append(None)
                    pattern_alias_kinds.append("pattern")
            return MatchClause(
                patterns=tuple(patterns),
                span=_span_from_meta(meta),
                optional=True,
                pattern_aliases=tuple(pattern_aliases),
                pattern_alias_kinds=tuple(pattern_alias_kinds),
            )

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

        def _parse_where_pattern_predicate_text(self, pattern_text: str, span: SourceSpan) -> WherePatternPredicate:
            pattern_items = [match.group(0).strip() for match in _WHERE_PATTERN_ITEM_RE.finditer(pattern_text)]
            if len(pattern_items) != 1:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE currently supports one positive pattern predicate at a time",
                    field="where",
                    value=pattern_text,
                    suggestion="Use a single positive relationship existence pattern in WHERE for the current GFQL Cypher subset.",
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
                        "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
                        field="where",
                        value=pattern_text,
                        suggestion="Use a single positive fixed-length relationship existence pattern in WHERE.",
                        line=span.line,
                        column=span.column,
                        language="cypher",
                    ) from exc
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
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
                    "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
                    field="where",
                    value=pattern_text,
                    suggestion="Use a single positive fixed-length relationship existence pattern in WHERE.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                )
            return WherePatternPredicate(pattern=cast(Tuple[PatternElement, ...], pattern_node), span=span)

        def _mixed_where_clause(
            self,
            *,
            pattern_text: str,
            expr_text: str,
            span: SourceSpan,
        ) -> WhereClause:
            if expr_text.strip() == "":
                raise _to_syntax_error("Invalid WHERE clause", line=span.line, column=span.column)
            # Mixed-clause handlers (``where_pattern_and_expr_clause`` and
            # ``expr_and_where_pattern_clause``) reconstruct ``expr_text``
            # from the source slice and do not capture Lark's structural
            # expression tree — so ``expr_tree`` stays unset here even when
            # ``expr_text`` contains AND/OR/XOR/NOT operators.  Out of scope
            # for issue #1200 slice 1 (text-only path; tracked for a later
            # slice alongside full binder migration).
            return WhereClause(
                predicates=(self._parse_where_pattern_predicate_text(pattern_text, span),),
                expr=ExpressionText(text=expr_text.strip(), span=span),
                span=span,
            )

        def where_pattern_only_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE pattern predicate", line=meta.line, column=meta.column)
            pattern_text = str(items[0]).strip()
            span = _span_from_meta(meta)
            return WhereClause(
                predicates=(self._parse_where_pattern_predicate_text(pattern_text, span),),
                expr=None,
                span=span,
            )

        def where_pattern_and_expr_clause(self, meta: Any, _items: Sequence[Any]) -> WhereClause:
            span = _span_from_meta(meta)
            body = self._slice(span)[len("WHERE"):].strip()
            match = _WHERE_PATTERN_THEN_EXPR_RE.fullmatch(body)
            if match is None:
                raise _to_syntax_error("Invalid WHERE clause", line=meta.line, column=meta.column)
            return self._mixed_where_clause(
                pattern_text=match.group("pattern").strip(),
                expr_text=match.group("expr").strip(),
                span=span,
            )

        def expr_and_where_pattern_clause(self, meta: Any, _items: Sequence[Any]) -> WhereClause:
            span = _span_from_meta(meta)
            body = self._slice(span)[len("WHERE"):].strip()
            match = _WHERE_EXPR_THEN_PATTERN_RE.fullmatch(body)
            if match is None:
                raise _to_syntax_error("Invalid WHERE clause", line=meta.line, column=meta.column)
            return self._mixed_where_clause(
                pattern_text=match.group("pattern").strip(),
                expr_text=match.group("expr").strip(),
                span=span,
            )

        def _wrap_as_boolean_atom(self, operand: Any, enclosing_meta: Any) -> BooleanExpr:
            """Coerce a parsed expression operand into a ``BooleanExpr`` atom.

            Recursion bottom-out for ``and_op`` / ``or_op`` / ``xor_op`` /
            ``not_op``.  ``BooleanExpr`` passes through.  Lark ``Tree`` and
            ``_ExpressionSlice`` operands carry their own span, so we use
            it to extract the source slice precisely.

            **Known limitation — primitive literal atoms.**  Literal
            transformers (``true_lit`` / ``false_lit`` / ``null_lit`` /
            ``number_lit``) return raw Python values without span info.
            When such a value reaches us as a boolean-operator operand
            (``WHERE true AND false``), we cannot recover the original
            source text for that specific operand; we approximate with
            the enclosing operator's span and ``str(operand)`` (which
            produces Python-style text like ``"True"`` not Cypher-style
            ``"true"``).  No current consumer reads ``atom_text`` on
            literal atoms — the binder is not wired to ``expr_tree`` in
            this slice.  Accuracy for this path is a follow-up concern
            tracked in issue #1200; if/when literal transformers gain
            span-carrying wrappers, this fallback can be removed.
            """
            if isinstance(operand, BooleanExpr):
                return operand
            if isinstance(operand, _ExpressionSlice):
                span = operand.span
                text = operand.text
            else:
                operand_meta = getattr(operand, "meta", None)
                if operand_meta is None:
                    # Primitive literal — see docstring caveat.
                    span = _span_from_meta(enclosing_meta)
                    text = str(operand)
                else:
                    span = _span_from_meta(operand_meta)
                    text = self._slice(span)
            return BooleanExpr(
                op="atom",
                span=span,
                atom_text=text,
                atom_span=span,
            )

        def _boolean_binary(
            self,
            op: Literal["and", "or", "xor"],
            meta: Any,
            items: Sequence[Any],
        ) -> BooleanExpr:
            if len(items) != 2:
                raise _to_syntax_error(
                    f"{op.upper()} expression requires two operands",
                    line=meta.line, column=meta.column,
                )
            return BooleanExpr(
                op=op,
                span=_span_from_meta(meta),
                left=self._wrap_as_boolean_atom(items[0], meta),
                right=self._wrap_as_boolean_atom(items[1], meta),
            )

        def grouped_expr(self, _meta: Any, items: Sequence[Any]) -> Any:
            # ``(expr)`` — passthrough so a parenthesized BooleanExpr bubbles
            # up unchanged to enclosing ``and_op`` / ``or_op`` / ``xor_op`` /
            # ``not_op`` handlers.  Without this, Lark would wrap the parens
            # as a ``grouped_expr`` Tree and our ``_wrap_as_boolean_atom``
            # would collapse the inner structure into a text atom, losing
            # the nested boolean tree that this PR exists to preserve.
            return items[0] if len(items) == 1 else items

        def and_op(self, meta: Any, items: Sequence[Any]) -> BooleanExpr:
            return self._boolean_binary("and", meta, items)

        def or_op(self, meta: Any, items: Sequence[Any]) -> BooleanExpr:
            return self._boolean_binary("or", meta, items)

        def xor_op(self, meta: Any, items: Sequence[Any]) -> BooleanExpr:
            return self._boolean_binary("xor", meta, items)

        def not_op(self, meta: Any, items: Sequence[Any]) -> BooleanExpr:
            if len(items) != 1:
                raise _to_syntax_error(
                    "NOT expression requires one operand",
                    line=meta.line, column=meta.column,
                )
            return BooleanExpr(
                op="not",
                span=_span_from_meta(meta),
                left=self._wrap_as_boolean_atom(items[0], meta),
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

        def generic_where_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            span = _span_from_meta(meta)
            expr = self._slice(span)[len("WHERE"):].strip()
            # Capture Lark's structural expression tree when available so
            # downstream consumers can walk ``BooleanExpr`` instead of
            # re-parsing ``ExpressionText``.  Only ``and_op``/``or_op``/
            # ``xor_op``/``not_op`` produce ``BooleanExpr`` today; atomic
            # WHERE expressions (single predicate) still route here with
            # a non-BooleanExpr operand, in which case ``expr_tree``
            # remains ``None`` — no behavior change for those callers.
            expr_tree: Optional[BooleanExpr] = (
                cast(BooleanExpr, items[0])
                if items and isinstance(items[0], BooleanExpr)
                else None
            )
            # Lark's ambiguity resolution prefers the generic `expr` path over
            # the structured `where_predicates` rule, so AND-joined bare label
            # predicates ("n:Admin AND n:Active") land here rather than in
            # `where_clause`.  Detect and lift them to structured predicates so
            # the binder can perform label narrowing without regex.  Any part
            # that is not a bare label predicate causes a conservative fallback
            # to raw expr (no narrowing).
            #
            # Split on top-level AND using the shared helper (quote/bracket/
            # paren/backtick-aware, case-insensitive).  An empty result means
            # malformed input — fall back to the whole expression as a single
            # term so the label-match check runs uniformly.
            parts = split_top_level_and(expr) or (expr,)
            predicates: List[WherePredicate] = []
            for part in parts:
                # `fullmatch` anchoring is load-bearing for security: it prevents
                # fragments of string literals (which contain quotes/spaces) from
                # matching as bare label predicates.  Relaxing to `match` would
                # re-introduce the false-positive that motivated issue #1125.
                m = _BARE_LABEL_PREDICATE_RE.fullmatch(part.strip())
                if m is None:
                    predicates = []
                    break
                labels = tuple(label for label in m.group(2).split(":") if label)
                predicates.append(
                    WherePredicate(
                        left=LabelRef(alias=m.group("alias"), labels=labels, span=span),
                        op="has_labels",
                        right=None,
                        span=span,
                    )
                )
            if predicates:
                return WhereClause(predicates=tuple(predicates), expr=None, span=span)
            return WhereClause(
                predicates=(),
                expr=ExpressionText(text=expr, span=span),
                expr_tree=expr_tree,
                span=span,
            )

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
            reentry_where_clauses: List[Optional[WhereClause]] = []
            reentry_where_pending_with_idx: Optional[int] = None
            call_clause: Optional[CallClause] = None
            unwind_clauses: List[UnwindClause] = []
            reentry_unwind_clauses: List[UnwindClause] = []
            staged_graph_unwind_span: Optional[SourceSpan] = None
            use_clause_node: Optional[UseClause] = None
            stages: List[ProjectionStage] = []
            ordered_row_items: List[Union[ProjectionStage, UnwindClause]] = []
            seen_stage = False
            for item in items:
                if isinstance(item, UseClause):
                    if use_clause_node is not None:
                        raise _to_syntax_error(
                            "Only one USE clause is allowed per query scope",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if match_clauses or stages or call_clause is not None:
                        raise _to_syntax_error(
                            "USE must appear before MATCH/RETURN/CALL clauses in the query body",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    use_clause_node = item
                elif isinstance(item, MatchClause):
                    if reentry_where_pending_with_idx is not None:
                        raise _to_syntax_error(
                            "Cypher MATCH after post-WITH WHERE is not yet supported in the current GFQL Cypher compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if reentry_unwind_clauses and len(reentry_match_clauses) >= 2:
                        raise _to_syntax_error(
                            "Cypher MATCH after post-WITH MATCH UNWIND is not yet supported in the current GFQL Cypher compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if seen_stage:
                        reentry_match_clauses.append(item)
                        reentry_where_clauses.append(None)
                    else:
                        match_clauses.append(item)
                elif isinstance(item, WhereClause):
                    if call_clause is not None:
                        raise _to_syntax_error(
                            "Cypher WHERE is not supported with CALL in the current GFQL Cypher compiler; use YIELD/RETURN row expressions instead",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if reentry_match_clauses:
                        if not reentry_where_clauses:
                            raise _to_syntax_error(
                                "Cypher WHERE after post-WITH MATCH is not yet supported in the current GFQL Cypher compiler",
                                line=item.span.line,
                                column=item.span.column,
                            )
                        if reentry_where_clauses[-1] is not None:
                            raise _to_syntax_error(
                                "Cypher only supports one WHERE clause per post-WITH MATCH stage in the current GFQL Cypher compiler",
                                line=item.span.line,
                                column=item.span.column,
                            )
                        reentry_where_clauses[-1] = item
                        reentry_where_pending_with_idx = len(reentry_where_clauses) - 1
                    else:
                        # Associate the WHERE with its preceding MATCH clause
                        # so that MATCH ... WHERE ... OPTIONAL MATCH ... WHERE ...
                        # correctly scopes each predicate to its own clause.
                        if match_clauses and match_clauses[-1].where is None:
                            match_clauses[-1] = replace(match_clauses[-1], where=item)
                        where_clause = item
                elif isinstance(item, CallClause):
                    if call_clause is not None:
                        raise _to_syntax_error(
                            "Cypher only supports one CALL clause per query in the current GFQL Cypher compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if match_clauses or reentry_match_clauses or stages or unwind_clauses:
                        raise _to_syntax_error(
                            "Cypher CALL is currently only supported as the first clause in standalone or row-only GFQL Cypher queries",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    call_clause = item
                elif isinstance(item, UnwindClause):
                    if reentry_match_clauses:
                        if reentry_unwind_clauses:
                            raise _to_syntax_error(
                                "Cypher only supports one UNWIND after post-WITH MATCH in the current GFQL Cypher compiler",
                                line=item.span.line,
                                column=item.span.column,
                            )
                        reentry_unwind_clauses.append(item)
                        continue
                    if match_clauses and seen_stage:
                        staged_graph_unwind_span = item.span
                    unwind_clauses.append(item)
                    ordered_row_items.append(item)
                elif isinstance(item, ProjectionStage):
                    if call_clause is not None and reentry_match_clauses:
                        raise _to_syntax_error(
                            "Cypher CALL with MATCH re-entry is not yet supported in the current GFQL Cypher compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    if reentry_match_clauses and item.clause.kind != "return" and item.clause.kind != "with":
                        raise _to_syntax_error(
                            "Cypher WITH after post-WITH MATCH is not yet supported in the current GFQL Cypher compiler",
                            line=item.span.line,
                            column=item.span.column,
                        )
                    stages.append(item)
                    ordered_row_items.append(item)
                    seen_stage = True
                    if reentry_where_pending_with_idx is not None and item.clause.kind == "with":
                        reentry_where_pending_with_idx = None
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
                            "Cypher RETURN must be the final projection stage in the current GFQL Cypher compiler",
                            line=stage.span.line,
                            column=stage.span.column,
                        )
                    return_stage = stage
                elif idx != len(stages) - 1:
                    with_stages.append(stage)
            if reentry_match_clauses:
                if (
                    stages[0].clause.kind != "with"
                    or stages[-1].clause.kind != "return"
                    or any(stage.clause.kind != "with" for stage in stages[:-1])
                    or len(with_stages) < len(reentry_match_clauses)
                    or len(with_stages) > len(reentry_match_clauses) + 1
                ):
                    first_match = reentry_match_clauses[0]
                    raise _to_syntax_error(
                        "Cypher MATCH after WITH is only supported for alternating MATCH ... WITH ... MATCH ... [WITH ... MATCH ...] ... [WITH] RETURN read shapes in the current GFQL Cypher compiler",
                        line=first_match.span.line,
                        column=first_match.span.column,
                    )
            elif staged_graph_unwind_span is not None:
                raise _to_unsupported(
                    "Cypher UNWIND after WITH/RETURN is not yet supported once MATCH has introduced graph aliases",
                    line=staged_graph_unwind_span.line,
                    column=staged_graph_unwind_span.column,
                )
            final_stage: Optional[ProjectionStage] = return_stage or (stages[-1] if stages else None)
            if where_clause is not None and not match_clauses:
                raise _to_syntax_error(
                    "Cypher WHERE is currently only supported after MATCH in the current GFQL Cypher compiler",
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
                reentry_wheres=tuple(reentry_where_clauses),
                reentry_unwinds=tuple(reentry_unwind_clauses),
                use=use_clause_node,
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
                        reentry_wheres=branch.reentry_wheres,
                        reentry_unwinds=branch.reentry_unwinds,
                    )
                return branch
            if len(union_kinds) != len(branches) - 1:
                raise _to_syntax_error("Invalid UNION query", line=meta.line, column=meta.column)
            union_kind_set = set(union_kinds)
            if len(union_kind_set) != 1:
                raise _to_syntax_error(
                    "Mixing UNION and UNION ALL is not supported in the current GFQL Cypher compiler",
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

        def use_clause(self, meta: Any, items: Sequence[Any]) -> UseClause:
            name = str(items[0])
            return UseClause(ref=name, span=_span_from_meta(meta))

        def graph_constructor_item(self, meta: Any, items: Sequence[Any]) -> Any:
            if len(items) != 1:
                raise _to_syntax_error("Invalid graph constructor item", line=meta.line, column=meta.column)
            return items[0]

        def graph_constructor_body(self, meta: Any, items: Sequence[Any]) -> Any:
            return list(items)

        def graph_constructor(self, meta: Any, items: Sequence[Any]) -> GraphConstructor:
            span = _span_from_meta(meta)
            body_items = items[0] if items and isinstance(items[0], list) else list(items)
            matches: List[MatchClause] = []
            where: Optional[WhereClause] = None
            use: Optional[UseClause] = None
            call: Optional[CallClause] = None
            for item in body_items:
                if isinstance(item, MatchClause):
                    if call is not None:
                        raise _to_syntax_error(
                            "MATCH and CALL cannot be combined inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    matches.append(item)
                elif isinstance(item, WhereClause):
                    if where is not None:
                        raise _to_syntax_error(
                            "Only one WHERE clause is allowed inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    where = item
                elif isinstance(item, CallClause):
                    if call is not None:
                        raise _to_syntax_error(
                            "Only one CALL clause is allowed inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    if matches:
                        raise _to_syntax_error(
                            "MATCH and CALL cannot be combined inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    if where is not None:
                        raise _to_syntax_error(
                            "WHERE is not allowed with CALL inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    call = item
                elif isinstance(item, UseClause):
                    if use is not None:
                        raise _to_syntax_error(
                            "Only one USE clause is allowed inside a graph constructor",
                            line=item.span.line, column=item.span.column,
                        )
                    use = item
                else:
                    raise _to_syntax_error(
                        "Only MATCH, WHERE, CALL, and USE clauses are allowed inside a graph constructor",
                        line=span.line, column=span.column,
                    )
            if not matches and call is None:
                raise _to_syntax_error(
                    "Graph constructor must contain at least one MATCH or CALL clause",
                    line=span.line, column=span.column,
                )
            return GraphConstructor(
                matches=tuple(matches),
                where=where,
                use=use,
                span=span,
                call=call,
            )

        def graph_binding(self, meta: Any, items: Sequence[Any]) -> GraphBinding:
            name = str(items[0])
            constructor = items[1]
            return GraphBinding(name=name, constructor=constructor, span=_span_from_meta(meta))

        def graph_query_with_bindings(
            self, meta: Any, items: Sequence[Any]
        ) -> Union[CypherQuery, CypherUnionQuery]:
            bindings: List[GraphBinding] = []
            query: Optional[Union[CypherQuery, CypherUnionQuery]] = None
            for item in items:
                if isinstance(item, GraphBinding):
                    bindings.append(item)
                elif isinstance(item, (CypherQuery, CypherUnionQuery)):
                    query = item
            if query is None:
                raise _to_syntax_error(
                    "Graph query must contain a query body after graph bindings",
                    line=meta.line, column=meta.column,
                )
            if not bindings and not isinstance(query, CypherUnionQuery):
                # Check for USE in the query body (already parsed as part of query_item)
                return query
            if bindings:
                if isinstance(query, CypherUnionQuery):
                    raise _to_unsupported(
                        "GRAPH bindings with UNION queries are not yet supported",
                        line=meta.line, column=meta.column,
                    )
                return CypherQuery(
                    matches=query.matches,
                    where=query.where,
                    call=query.call,
                    unwinds=query.unwinds,
                    with_stages=query.with_stages,
                    return_=query.return_,
                    order_by=query.order_by,
                    skip=query.skip,
                    limit=query.limit,
                    row_sequence=query.row_sequence,
                    trailing_semicolon=query.trailing_semicolon,
                    span=query.span,
                    reentry_matches=query.reentry_matches,
                    reentry_wheres=query.reentry_wheres,
                    reentry_unwinds=query.reentry_unwinds,
                    graph_bindings=tuple(bindings),
                    use=query.use,
                )
            return query

        def graph_query_standalone(
            self, meta: Any, items: Sequence[Any]
        ) -> CypherGraphQuery:
            bindings: List[GraphBinding] = []
            constructor: Optional[GraphConstructor] = None
            for item in items:
                if isinstance(item, GraphBinding):
                    bindings.append(item)
                elif isinstance(item, GraphConstructor):
                    constructor = item
            if constructor is None:
                raise _to_syntax_error(
                    "Standalone graph query must end with a GRAPH { } constructor",
                    line=meta.line, column=meta.column,
                )
            return CypherGraphQuery(
                graph_bindings=tuple(bindings),
                constructor=constructor,
                trailing_semicolon=False,
                span=_span_from_meta(meta),
            )

    return cast(_TransformerLike, _CypherAstBuilder())


_PATTERN_EXISTENCE_RE = re.compile(
    r"""
    (?:not\s*)\(\s*\(\s*\w+\s*\)\s*-\s*\[  # not((a)-[
    |
    not\s+exists\s*\{                        # not exists {
    |
    exists\s*\{                              # exists {
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _check_unsupported_syntax_patterns(query: str) -> None:
    """Detect known-but-unsupported Cypher syntax and raise a clear error."""
    if _PATTERN_EXISTENCE_RE.search(query):
        raise _to_unsupported(
            "Pattern existence expressions (e.g., not((a)-[:R]-(b)) or exists { ... }) "
            "are not yet supported in the local Cypher compiler",
            field="expression",
            value="pattern_existence",
        )


def parse_cypher(query: str) -> Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]:
    """Parse supported Cypher text into the typed AST used by GFQL's Cypher compiler.

    The returned AST preserves the clause structure needed by the current GFQL
    Cypher compiler, including unions and row-pipeline stages.

    :param query: Cypher text to parse.
    :returns: A parsed ``CypherQuery`` or ``CypherUnionQuery``.
    :raises GFQLSyntaxError: If the query is not valid within GFQL's current
        supported Cypher grammar.
    """
    if not isinstance(query, str) or query.strip() == "":
        raise _to_syntax_error("Cypher query must be a non-empty string")

    # Pre-parse detection of known-but-unsupported Cypher forms
    _check_unsupported_syntax_patterns(query)

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
            canonical_query = _canonicalize_where_single_pattern_and_expr(query)
            if canonical_query is not None and canonical_query != query:
                canonical_transformer = _build_transformer(canonical_query)
                tree = parser.parse(canonical_query)
                node = canonical_transformer.transform(tree)
                if not isinstance(node, (CypherQuery, CypherUnionQuery, CypherGraphQuery)):
                    raise _to_syntax_error("Cypher parser did not produce a query")
                return node
            mixed_where_error = _mixed_where_pattern_expr_error(query)
            if mixed_where_error is not None:
                raise mixed_where_error from exc
            err_line = cast(Optional[int], getattr(exc, "line", None))
            err_column = cast(Optional[int], getattr(exc, "column", None))
            raise _to_syntax_error("Invalid Cypher query syntax", line=err_line, column=err_column) from exc
        raise _to_syntax_error("Invalid Cypher query syntax") from exc

    if not isinstance(node, (CypherQuery, CypherUnionQuery, CypherGraphQuery)):
        raise _to_syntax_error("Cypher parser did not produce a query")
    _validate_reserved_identifiers(node)
    if isinstance(node, (CypherQuery, CypherGraphQuery)):
        _validate_graph_bindings(node)
    return node


def _validate_graph_bindings(node: Union[CypherQuery, CypherGraphQuery]) -> None:
    """Validate graph binding names: no duplicates, no forward/circular refs."""
    if isinstance(node, CypherGraphQuery):
        bindings = node.graph_bindings
    else:
        bindings = node.graph_bindings

    seen: dict[str, GraphBinding] = {}
    for binding in bindings:
        lower_name = binding.name.lower()
        if lower_name in seen:
            raise GFQLValidationError(
                ErrorCode.E150,
                f"Duplicate graph binding name '{binding.name}'. Each GRAPH binding must have a unique name.",
            )
        if binding.constructor.use is not None:
            ref_lower = binding.constructor.use.ref.lower()
            if ref_lower == lower_name:
                raise GFQLValidationError(
                    ErrorCode.E153,
                    f"Graph binding '{binding.name}' cannot USE itself.",
                )
            if ref_lower not in seen:
                raise GFQLValidationError(
                    ErrorCode.E152,
                    f"USE refers to graph '{binding.constructor.use.ref}' which has not been defined yet. "
                    f"GRAPH bindings must appear before the USE that references them.",
                )
        seen[lower_name] = binding

    # Validate USE in the final query/constructor references a known binding
    if isinstance(node, CypherGraphQuery):
        if node.constructor.use is not None:
            ref_lower = node.constructor.use.ref.lower()
            if ref_lower not in seen:
                raise GFQLValidationError(
                    ErrorCode.E151,
                    f"USE refers to graph '{node.constructor.use.ref}' which has not been defined. "
                    f"Define it with GRAPH {node.constructor.use.ref} = GRAPH {{ ... }} before USE.",
                )
    elif isinstance(node, CypherQuery) and node.use is not None:
        ref_lower = node.use.ref.lower()
        if ref_lower not in seen:
            raise GFQLValidationError(
                ErrorCode.E151,
                f"USE refers to graph '{node.use.ref}' which has not been defined. "
                f"Define it with GRAPH {node.use.ref} = GRAPH {{ ... }} before USE.",
            )


def _validate_reserved_identifiers(node: Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]) -> None:
    if isinstance(node, CypherUnionQuery):
        for branch in node.branches:
            _validate_reserved_identifiers(branch)
        return
    if isinstance(node, CypherGraphQuery):
        # Graph constructors have their own identifier space; skip RETURN/WITH checks
        return

    def _check_identifier(name: Optional[str], *, line: int, column: int) -> None:
        if name is not None and name.lower() == _RESERVED_IDENTIFIER_GRAPH:
            raise _to_syntax_error(
                "Cypher identifier GRAPH is reserved in the local compiler",
                line=line,
                column=column,
            )

    for match in (*node.matches, *node.reentry_matches):
        for alias in match.pattern_aliases:
            _check_identifier(alias, line=match.span.line, column=match.span.column)
        for pattern in match.patterns:
            for element in pattern:
                if isinstance(element, NodePattern):
                    _check_identifier(element.variable, line=element.span.line, column=element.span.column)
                elif isinstance(element, RelationshipPattern):
                    _check_identifier(element.variable, line=element.span.line, column=element.span.column)

    for unwind in node.unwinds:
        _check_identifier(unwind.alias, line=unwind.span.line, column=unwind.span.column)

    for stage in (*node.with_stages, ProjectionStage(
        clause=node.return_,
        where=None,
        order_by=node.order_by,
        skip=node.skip,
        limit=node.limit,
        span=node.return_.span,
    ),):
        for item in stage.clause.items:
            _check_identifier(item.alias, line=item.span.line, column=item.span.column)

    if node.call is not None:
        for yield_item in node.call.yield_items:
            _check_identifier(yield_item.alias, line=yield_item.span.line, column=yield_item.span.column)
