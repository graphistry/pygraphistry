from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Type, Union, cast
from typing_extensions import TypeAlias, TypeGuard

from graphistry.compute.gfql.string_literals import parse_cypher_string_token
from graphistry.compute.gfql.language_defs import (
    GFQL_ALLOWED_BINARY_OPS,
    GFQL_ALLOWED_FUNCTIONS,
    GFQL_ALLOWED_QUANTIFIERS,
    GFQL_ALLOWED_UNARY_OPS,
    GFQL_COMPARISON_GRAMMAR_ALTS,
    GFQL_STRING_PREDICATE_OPS,
)


DEFAULT_ALLOWED_FUNCTIONS: FrozenSet[str] = GFQL_ALLOWED_FUNCTIONS
DEFAULT_ALLOWED_BINARY_OPS: FrozenSet[str] = GFQL_ALLOWED_BINARY_OPS
DEFAULT_ALLOWED_UNARY_OPS: FrozenSet[str] = GFQL_ALLOWED_UNARY_OPS
DEFAULT_ALLOWED_QUANTIFIERS: FrozenSet[str] = GFQL_ALLOWED_QUANTIFIERS

_LiteralValue: TypeAlias = Union[None, bool, int, float, str]


@dataclass(frozen=True)
class Identifier:
    name: str


@dataclass(frozen=True)
class Literal:
    value: _LiteralValue


@dataclass(frozen=True)
class UnaryOp:
    op: str
    operand: "ExprNode"


@dataclass(frozen=True)
class BinaryOp:
    op: str
    left: "ExprNode"
    right: "ExprNode"


@dataclass(frozen=True)
class IsNullOp:
    value: "ExprNode"
    negated: bool = False


@dataclass(frozen=True)
class FunctionCall:
    name: str
    args: Tuple["ExprNode", ...]
    distinct: bool = False


@dataclass(frozen=True)
class Wildcard:
    pass


@dataclass(frozen=True)
class CaseWhen:
    condition: "ExprNode"
    when_true: "ExprNode"
    when_false: "ExprNode"


@dataclass(frozen=True)
class QuantifierExpr:
    fn: str
    var: str
    source: "ExprNode"
    predicate: "ExprNode"


@dataclass(frozen=True)
class ListComprehension:
    var: str
    source: "ExprNode"
    predicate: Optional["ExprNode"] = None
    projection: Optional["ExprNode"] = None


@dataclass(frozen=True)
class ListLiteral:
    items: Tuple["ExprNode", ...]


@dataclass(frozen=True)
class MapLiteral:
    items: Tuple[Tuple[str, "ExprNode"], ...]


@dataclass(frozen=True)
class SubscriptExpr:
    value: "ExprNode"
    key: "ExprNode"


@dataclass(frozen=True)
class SliceExpr:
    value: "ExprNode"
    start: Optional["ExprNode"]
    stop: Optional["ExprNode"]


@dataclass(frozen=True)
class PropertyAccessExpr:
    value: "ExprNode"
    property: str


ExprNode = Union[
    Identifier,
    Literal,
    UnaryOp,
    BinaryOp,
    IsNullOp,
    FunctionCall,
    Wildcard,
    CaseWhen,
    QuantifierExpr,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    SubscriptExpr,
    SliceExpr,
    PropertyAccessExpr,
]

_EXPR_NODE_TYPES = (
    Identifier,
    Literal,
    UnaryOp,
    BinaryOp,
    IsNullOp,
    FunctionCall,
    Wildcard,
    CaseWhen,
    QuantifierExpr,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    SubscriptExpr,
    SliceExpr,
    PropertyAccessExpr,
)

ExprVisitor = Callable[[ExprNode], None]


@dataclass(frozen=True)
class _FunctionArgs:
    args: Tuple[ExprNode, ...]
    distinct: bool = False


@dataclass(frozen=True)
class _CaseArm:
    when_expr: ExprNode
    then_expr: ExprNode


_MapEntry: TypeAlias = Tuple[str, ExprNode]
_SubscriptIndex: TypeAlias = _MapEntry
_SubscriptSlice: TypeAlias = Tuple[str, Optional[ExprNode], Optional[ExprNode]]
_AstItem: TypeAlias = Union[ExprNode, _FunctionArgs, _CaseArm, List[ExprNode], List[_MapEntry], _MapEntry, _SubscriptSlice]
_ExprFieldValue: TypeAlias = Union[_LiteralValue, ExprNode, Optional[ExprNode], _MapEntry, Tuple[ExprNode, ...], Tuple[_MapEntry, ...]]


class _LarkTree(Protocol):
    pass


class _LarkToken(Protocol):
    type: str
    value: str


_TransformItem: TypeAlias = Union[_AstItem, _LarkToken]
_TransformItems: TypeAlias = Sequence[_TransformItem]


class _LarkParser(Protocol):
    def parse(self, text: str) -> _LarkTree:
        ...


class _LarkTransformer(Protocol):
    def transform(self, tree: _LarkTree) -> _TransformItem:
        ...


class GFQLExprParseError(ValueError):
    def __init__(self, message: str, *, line: Optional[int] = None, column: Optional[int] = None):
        self.line = line
        self.column = column
        suffix = ""
        if line is not None and column is not None:
            suffix = f" (line {line}, column {column})"
        super().__init__(f"{message}{suffix}")


_GRAMMAR = r"""
?start: expr

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
          | comparable COMP_OP comparable      -> cmp_op

?comparable: additive
           | additive "IS"i "NULL"i             -> is_null
           | additive "IS"i "NOT"i "NULL"i      -> is_not_null
           | additive "IN"i additive            -> in_op
           | additive "CONTAINS"i additive      -> contains_op
           | additive "STARTS"i "WITH"i additive -> starts_with_op
           | additive "ENDS"i "WITH"i additive  -> ends_with_op

?additive: multiplicative
         | additive "+" multiplicative   -> add_op
         | additive MINUS multiplicative -> sub_op

?multiplicative: unary
               | multiplicative "*" unary -> mul_op
               | multiplicative "/" unary -> div_op
               | multiplicative "%" unary -> mod_op

?unary: "+" unary                        -> uplus
      | MINUS unary                      -> uminus
      | postfix

?postfix: primary
        | postfix "[" subscript_key "]"  -> subscript
        | postfix "." NAME               -> property_access

?subscript_key: expr                     -> subscript_index
              | expr ".." expr           -> subscript_slice_between
              | expr ".."                -> subscript_slice_from
              | ".." expr                -> subscript_slice_to
              | ".."                     -> subscript_slice_all

?primary: literal
        | function_call
        | identifier
        | case_expr
        | quantifier_expr
        | list_comprehension
        | list_literal
        | map_literal
        | "(" expr ")"                   -> grouped

list_literal: "[" [expr_list] "]"
expr_list: expr ("," expr)*

map_literal: "{" [map_entries] "}"
map_entries: map_entry ("," map_entry)*
map_entry: map_key ":" expr
map_key: MAP_KEY_NAME                    -> map_key_name
       | STRING                          -> map_key_string

function_call: identifier "(" func_args ")"
?func_args: distinct_func_args
         | regular_func_args
regular_func_args: func_arg ("," func_arg)*
distinct_func_args: "DISTINCT"i func_arg
?func_arg: expr
         | "*"                           -> star_arg
identifier: NAME ("." NAME)*

case_expr: searched_case_expr
         | simple_case_expr
searched_case_expr: "CASE"i case_when+ case_else? "END"i
simple_case_expr: "CASE"i expr case_when+ case_else? "END"i
case_when: "WHEN"i expr "THEN"i expr
case_else: "ELSE"i expr

quantifier_expr: "ANY"i "(" NAME "IN"i expr "WHERE"i expr ")"       -> any_quant
               | "ALL"i "(" NAME "IN"i expr "WHERE"i expr ")"       -> all_quant
               | "NONE"i "(" NAME "IN"i expr "WHERE"i expr ")"      -> none_quant
               | "SINGLE"i "(" NAME "IN"i expr "WHERE"i expr ")"    -> single_quant

list_comprehension: "[" NAME "IN"i expr "]"                                 -> lc_source
                  | "[" NAME "IN"i expr "|" expr "]"                        -> lc_projection
                  | "[" NAME "IN"i expr "WHERE"i expr "]"                   -> lc_where
                  | "[" NAME "IN"i expr "WHERE"i expr "|" expr "]"          -> lc_where_projection

literal: "NULL"i                            -> null_lit
       | "TRUE"i                            -> true_lit
       | "FALSE"i                           -> false_lit
       | NUMBER                          -> number_lit
       | STRING                          -> string_lit

COMP_OP: __GFQL_COMPARISON_GRAMMAR_ALTS__
MINUS: /-(?!-)/
NAME: /(?!(?i:AND|OR|XOR|NOT|IN|IS|NULL|CASE|WHEN|THEN|ELSE|END|CONTAINS|STARTS|WITH|ENDS|ANY|ALL|NONE|SINGLE)\b)[A-Za-z_][A-Za-z0-9_]*/
MAP_KEY_NAME: /[A-Za-z_][A-Za-z0-9_]*/
NUMBER: /[+-]?(?:0[xX][0-9A-Fa-f]+|0[oO][0-7]+|(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?))/
STRING : /'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/
LINE_COMMENT: /--[^\n]*/
BLOCK_COMMENT: /\/\*[\s\S]*?\*\//
%import common.WS
%ignore WS
"""

_GRAMMAR = _GRAMMAR.replace("__GFQL_COMPARISON_GRAMMAR_ALTS__", GFQL_COMPARISON_GRAMMAR_ALTS)

def _lark_imports() -> Tuple[type, type, Type[Exception]]:
    try:
        from lark import Lark, Transformer
        from lark.exceptions import LarkError
        return Lark, Transformer, LarkError
    except Exception as exc:
        raise ImportError(
            "Lark is required for GFQL expression parsing. Install dependency 'lark'."
        ) from exc


@lru_cache(maxsize=1)
def _parser() -> _LarkParser:
    Lark, _, _ = _lark_imports()
    return cast(_LarkParser, Lark(_GRAMMAR, start="start", parser="lalr", maybe_placeholders=False))


def _parse_string_token(token: str) -> str:
    try:
        value = parse_cypher_string_token(token)
    except Exception as exc:
        raise GFQLExprParseError("Invalid string literal") from exc
    return value


def _parse_number_token(token: str) -> Union[int, float]:
    sign = -1 if token.startswith("-") else 1
    body = token[1:] if token[:1] in {"+", "-"} else token
    lowered = body.lower()
    if lowered.startswith("0x"):
        return sign * int(lowered[2:], 16)
    if lowered.startswith("0o"):
        return sign * int(lowered[2:], 8)
    if any(c in body for c in (".", "e", "E")):
        value = float(token)
        return 0.0 if value == 0.0 else value
    return int(token)


def _build_transformer() -> _LarkTransformer:
    _, Transformer, _ = _lark_imports()

    class _RuleHost(Protocol):
        def _quantifier_expr(self, fn: str, items: _TransformItems) -> QuantifierExpr:
            ...

        def _list_comprehension(
            self, items: _TransformItems, *, has_where: bool, has_projection: bool
        ) -> ListComprehension:
            ...

    def _is_token(value: _TransformItem) -> TypeGuard[_LarkToken]:
        return hasattr(value, "type") and hasattr(value, "value")

    def _strip_tokens(items: _TransformItems) -> List[_AstItem]:
        stripped: List[_AstItem] = []
        for item in items:
            if _is_token(item):
                continue
            stripped.append(cast(_AstItem, item))
        return stripped

    def _unary_rule(op: str) -> Callable[[_RuleHost, _TransformItems], UnaryOp]:
        def _rule(self: _RuleHost, items: _TransformItems) -> UnaryOp:
            return UnaryOp(op=op, operand=cast(ExprNode, _strip_tokens(items)[0]))
        return _rule

    def _binary_rule(op: str) -> Callable[[_RuleHost, _TransformItems], BinaryOp]:
        def _rule(self: _RuleHost, items: _TransformItems) -> BinaryOp:
            stripped = _strip_tokens(items)
            return BinaryOp(op=op, left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))
        return _rule

    def _quantifier_rule(fn: str) -> Callable[[_RuleHost, _TransformItems], QuantifierExpr]:
        def _rule(self: _RuleHost, items: _TransformItems) -> QuantifierExpr:
            return self._quantifier_expr(fn, items)
        return _rule

    def _list_comprehension_rule(
        *, has_where: bool, has_projection: bool
    ) -> Callable[[_RuleHost, _TransformItems], ListComprehension]:
        def _rule(self: _RuleHost, items: _TransformItems) -> ListComprehension:
            return self._list_comprehension(items, has_where=has_where, has_projection=has_projection)
        return _rule

    def _slice_rule(
        start_index: Optional[int], stop_index: Optional[int]
    ) -> Callable[[_RuleHost, _TransformItems], _SubscriptSlice]:
        def _rule(self: _RuleHost, items: _TransformItems) -> _SubscriptSlice:
            stripped = _strip_tokens(items)
            start = None if start_index is None else cast(ExprNode, stripped[start_index])
            stop = None if stop_index is None else cast(ExprNode, stripped[stop_index])
            return ("slice", start, stop)
        return _rule

    class _AstBuilder(Transformer):  # type: ignore[valid-type,misc]
        def grouped(self, items: _TransformItems) -> ExprNode:
            return cast(ExprNode, _strip_tokens(items)[0])

        def expr_list(self, items: _TransformItems) -> List[ExprNode]:
            return [cast(ExprNode, i) for i in _strip_tokens(items)]

        def null_lit(self, _: _TransformItems) -> Literal:
            return Literal(None)

        def true_lit(self, _: _TransformItems) -> Literal:
            return Literal(True)

        def false_lit(self, _: _TransformItems) -> Literal:
            return Literal(False)

        def number_lit(self, items: _TransformItems) -> Literal:
            if len(items) != 1:
                raise GFQLExprParseError("Invalid numeric literal")
            return Literal(_parse_number_token(str(items[0])))

        def string_lit(self, items: _TransformItems) -> Literal:
            if len(items) != 1:
                raise GFQLExprParseError("Invalid string literal")
            return Literal(_parse_string_token(str(items[0])))

        def identifier(self, items: _TransformItems) -> Identifier:
            names = [str(i) for i in items if _is_token(i) and str(getattr(i, "type", "")) == "NAME"]
            if len(names) == 0:
                raise GFQLExprParseError("Invalid identifier")
            return Identifier(".".join(names))

        def regular_func_args(self, items: _TransformItems) -> _FunctionArgs:
            return _FunctionArgs(tuple(cast(ExprNode, i) for i in _strip_tokens(items)))

        def distinct_func_args(self, items: _TransformItems) -> _FunctionArgs:
            stripped = _strip_tokens(items)
            if len(stripped) != 1:
                raise GFQLExprParseError("Invalid DISTINCT function argument")
            return _FunctionArgs((cast(ExprNode, stripped[0]),), distinct=True)

        def star_arg(self, _: _TransformItems) -> Wildcard:
            return Wildcard()

        def function_call(self, items: _TransformItems) -> FunctionCall:
            fn = ""
            args: Tuple[ExprNode, ...] = ()
            distinct = False
            for item in items:
                if _is_token(item):
                    continue
                if isinstance(item, Identifier) and fn == "":
                    fn = item.name.lower()
                    continue
                if isinstance(item, _FunctionArgs):
                    args = item.args
                    distinct = item.distinct
                else:
                    args = (cast(ExprNode, item),)
            if fn == "":
                raise GFQLExprParseError("Invalid function call")
            return FunctionCall(fn, args, distinct=distinct)

        def case_when(self, items: _TransformItems) -> _CaseArm:
            stripped = _strip_tokens(items)
            if len(stripped) != 2:
                raise GFQLExprParseError("Invalid CASE arm")
            return _CaseArm(
                when_expr=cast(ExprNode, stripped[0]),
                then_expr=cast(ExprNode, stripped[1]),
            )

        def case_else(self, items: _TransformItems) -> ExprNode:
            stripped = _strip_tokens(items)
            if len(stripped) != 1:
                raise GFQLExprParseError("Invalid CASE ELSE clause")
            return cast(ExprNode, stripped[0])

        def _fold_case_arms(
            self,
            arms: Sequence[_CaseArm],
            *,
            base_expr: Optional[ExprNode] = None,
            else_expr: Optional[ExprNode] = None,
        ) -> CaseWhen:
            if len(arms) == 0:
                raise GFQLExprParseError("CASE requires at least one WHEN branch")
            node: ExprNode = Literal(None) if else_expr is None else else_expr
            for arm in reversed(tuple(arms)):
                condition = arm.when_expr
                if base_expr is not None:
                    condition = FunctionCall("__cypher_case_eq__", (base_expr, condition))
                node = CaseWhen(
                    condition=condition,
                    when_true=arm.then_expr,
                    when_false=node,
                )
            if not isinstance(node, CaseWhen):
                raise GFQLExprParseError("Invalid CASE expression")
            return node

        def searched_case_expr(self, items: _TransformItems) -> CaseWhen:
            arms: List[_CaseArm] = []
            else_expr: Optional[ExprNode] = None
            for item in _strip_tokens(items):
                if isinstance(item, _CaseArm):
                    arms.append(item)
                else:
                    else_expr = cast(ExprNode, item)
            return self._fold_case_arms(arms, else_expr=else_expr)

        def simple_case_expr(self, items: _TransformItems) -> CaseWhen:
            stripped = _strip_tokens(items)
            if len(stripped) < 2:
                raise GFQLExprParseError("Invalid CASE expression")
            base_expr = cast(ExprNode, stripped[0])
            arms: List[_CaseArm] = []
            else_expr: Optional[ExprNode] = None
            for item in stripped[1:]:
                if isinstance(item, _CaseArm):
                    arms.append(item)
                else:
                    else_expr = cast(ExprNode, item)
            return self._fold_case_arms(arms, base_expr=base_expr, else_expr=else_expr)

        def case_expr(self, items: _TransformItems) -> CaseWhen:
            stripped = _strip_tokens(items)
            if len(stripped) != 1 or not isinstance(stripped[0], CaseWhen):
                raise GFQLExprParseError("Invalid CASE expression")
            return cast(CaseWhen, stripped[0])

        def _quantifier_expr(self, fn: str, items: _TransformItems) -> QuantifierExpr:
            var = ""
            expr_nodes: List[ExprNode] = []
            for item in items:
                if _is_token(item):
                    if str(getattr(item, "type", "")) == "NAME" and var == "":
                        var = str(item)
                    continue
                expr_nodes.append(cast(ExprNode, item))
            if var == "" or len(expr_nodes) != 2:
                raise GFQLExprParseError("Invalid quantifier expression")
            source = expr_nodes[0]
            predicate = expr_nodes[1]
            return QuantifierExpr(fn=fn, var=var, source=source, predicate=predicate)

        any_quant = _quantifier_rule("any")
        all_quant = _quantifier_rule("all")
        none_quant = _quantifier_rule("none")
        single_quant = _quantifier_rule("single")

        def _list_comprehension(
            self, items: _TransformItems, *, has_where: bool, has_projection: bool
        ) -> ListComprehension:
            var_name = ""
            expr_nodes: List[ExprNode] = []
            for item in items:
                if _is_token(item):
                    if str(getattr(item, "type", "")) == "NAME" and var_name == "":
                        var_name = str(item)
                    continue
                expr_nodes.append(cast(ExprNode, item))

            if var_name == "" or len(expr_nodes) < 1:
                raise GFQLExprParseError("Invalid list comprehension")
            source = expr_nodes[0]
            predicate: Optional[ExprNode] = None
            projection: Optional[ExprNode] = None

            idx = 1
            if has_where:
                if len(expr_nodes) < 2:
                    raise GFQLExprParseError("Invalid list comprehension WHERE clause")
                predicate = expr_nodes[idx]
                idx += 1
            if has_projection:
                if len(expr_nodes) <= idx:
                    raise GFQLExprParseError("Invalid list comprehension projection")
                projection = expr_nodes[idx]

            return ListComprehension(
                var=var_name,
                source=source,
                predicate=predicate,
                projection=projection,
            )

        lc_source = _list_comprehension_rule(has_where=False, has_projection=False)
        lc_projection = _list_comprehension_rule(has_where=False, has_projection=True)
        lc_where = _list_comprehension_rule(has_where=True, has_projection=False)
        lc_where_projection = _list_comprehension_rule(has_where=True, has_projection=True)

        def list_literal(self, items: _TransformItems) -> ListLiteral:
            if len(items) == 0:
                return ListLiteral(())
            return ListLiteral(tuple(cast(List[ExprNode], items[0])))

        def map_key_name(self, items: _TransformItems) -> str:
            if len(items) != 1:
                raise GFQLExprParseError("Invalid map key")
            return str(items[0])

        def map_key_string(self, items: _TransformItems) -> str:
            if len(items) != 1:
                raise GFQLExprParseError("Invalid map key")
            return _parse_string_token(str(items[0]))

        def map_entry(self, items: _TransformItems) -> _MapEntry:
            stripped = _strip_tokens(items)
            return (str(stripped[0]), cast(ExprNode, stripped[1]))

        def map_entries(self, items: _TransformItems) -> List[_MapEntry]:
            return [cast(_MapEntry, i) for i in _strip_tokens(items)]

        def map_literal(self, items: _TransformItems) -> MapLiteral:
            stripped = _strip_tokens(items)
            if len(stripped) == 0:
                return MapLiteral(())
            return MapLiteral(tuple(cast(List[Tuple[str, ExprNode]], stripped[0])))

        def subscript_index(self, items: _TransformItems) -> _SubscriptIndex:
            return ("index", cast(ExprNode, _strip_tokens(items)[0]))

        subscript_slice_between = _slice_rule(0, 1)
        subscript_slice_from = _slice_rule(0, None)
        subscript_slice_to = _slice_rule(None, 0)
        subscript_slice_all = _slice_rule(None, None)

        def subscript(self, items: _TransformItems) -> ExprNode:
            stripped = _strip_tokens(items)
            value = cast(ExprNode, stripped[0])
            sub = cast(Union[_SubscriptIndex, _SubscriptSlice], stripped[1])
            if sub[0] == "index":
                index = cast(_SubscriptIndex, sub)
                return SubscriptExpr(value=value, key=index[1])
            span = cast(_SubscriptSlice, sub)
            return SliceExpr(
                value=value,
                start=span[1],
                stop=span[2],
            )

        def property_access(self, items: _TransformItems) -> PropertyAccessExpr:
            stripped = _strip_tokens(items)
            if len(stripped) != 1:
                raise GFQLExprParseError("Invalid property access")
            value = cast(ExprNode, stripped[0])
            names = [str(i) for i in items if _is_token(i) and str(getattr(i, "type", "")) == "NAME"]
            if len(names) == 0:
                raise GFQLExprParseError("Invalid property access")
            return PropertyAccessExpr(value=value, property=names[-1])

        uplus = _unary_rule("+")
        uminus = _unary_rule("-")
        not_op = _unary_rule("not")

        or_op = _binary_rule("or")
        xor_op = _binary_rule("xor")
        and_op = _binary_rule("and")
        add_op = _binary_rule("+")
        sub_op = _binary_rule("-")
        mul_op = _binary_rule("*")
        div_op = _binary_rule("/")
        mod_op = _binary_rule("%")

        def cmp_op(self, items: _TransformItems) -> BinaryOp:
            op = ""
            for item in items:
                if _is_token(item) and str(getattr(item, "type", "")) == "COMP_OP":
                    op = str(item).lower()
                    break
            stripped = _strip_tokens(items)
            if op == "":
                raise GFQLExprParseError("Missing comparison operator")
            return BinaryOp(op=op, left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))

        in_op = _binary_rule("in")
        contains_op = _binary_rule("contains")
        starts_with_op = _binary_rule("starts_with")
        ends_with_op = _binary_rule("ends_with")

        def is_null(self, items: _TransformItems) -> IsNullOp:
            return IsNullOp(value=cast(ExprNode, _strip_tokens(items)[0]), negated=False)

        def is_not_null(self, items: _TransformItems) -> IsNullOp:
            return IsNullOp(value=cast(ExprNode, _strip_tokens(items)[0]), negated=True)

    return cast(_LarkTransformer, _AstBuilder())


def _rebuild_expr_node(
    node: ExprNode,
    *,
    rewrite: Callable[[ExprNode], ExprNode],
    error_context: str,
) -> ExprNode:
    if isinstance(node, (Identifier, Literal, Wildcard)):
        return node
    if isinstance(node, _EXPR_NODE_TYPES):
        return cast(
            ExprNode,
            replace(
                node,
                **{  # type: ignore[arg-type]
                    field.name: _rewrite_expr_value(cast(_ExprFieldValue, getattr(node, field.name)), rewrite)
                    for field in fields(node)
                },
            ),
        )
    raise TypeError(f"Unsupported expression node type for {error_context}: {type(node).__name__}")


def _rewrite_expr_value(value: _ExprFieldValue, rewrite: Callable[[ExprNode], ExprNode]) -> _ExprFieldValue:
    if isinstance(value, _EXPR_NODE_TYPES):
        return rewrite(value)
    if isinstance(value, tuple):
        return cast(
            _ExprFieldValue,
            tuple(_rewrite_expr_value(cast(_ExprFieldValue, item), rewrite) for item in value),
        )
    return value


def _normalize_dotted_identifiers(node: ExprNode) -> ExprNode:
    if isinstance(node, Identifier):
        parts = node.name.split(".")
        if len(parts) <= 1:
            return node
        out: ExprNode = Identifier(parts[0])
        for prop in parts[1:]:
            out = PropertyAccessExpr(value=out, property=prop)
        return out
    return _rebuild_expr_node(
        node,
        rewrite=_normalize_dotted_identifiers,
        error_context="dotted identifier normalization",
    )


def parse_expr(expr: str) -> ExprNode:
    if not isinstance(expr, str) or expr.strip() == "":
        raise GFQLExprParseError("Expression must be a non-empty string")

    parser = _parser()
    transformer = _build_transformer()
    try:
        tree = parser.parse(expr)
        node = _normalize_dotted_identifiers(cast(ExprNode, transformer.transform(tree)))
    except Exception as exc:
        _, _, LarkError = _lark_imports()
        if isinstance(exc, LarkError):
            line = getattr(exc, "line", None)
            column = getattr(exc, "column", None)
            raise GFQLExprParseError("Invalid GFQL expression", line=line, column=column) from exc
        if isinstance(exc, GFQLExprParseError):
            raise
        raise GFQLExprParseError("Invalid GFQL expression") from exc

    if not isinstance(node, _EXPR_NODE_TYPES):
        raise GFQLExprParseError("Invalid GFQL expression AST")
    return cast(ExprNode, node)


def is_expr_node(node: object) -> bool:
    return isinstance(node, _EXPR_NODE_TYPES)


def iter_expr_children(node: ExprNode) -> Tuple[ExprNode, ...]:
    if isinstance(node, (Identifier, Literal, Wildcard)):
        return ()
    if isinstance(node, _EXPR_NODE_TYPES):
        return tuple(
            child
            for field in fields(node)
            for child in _iter_expr_values(getattr(node, field.name))
        )
    return ()


def _iter_expr_values(value: _ExprFieldValue) -> Iterable[ExprNode]:
    if isinstance(value, _EXPR_NODE_TYPES):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _iter_expr_values(cast(_ExprFieldValue, item))


def walk_expr_nodes(
    node: ExprNode,
    *,
    enter: Optional[ExprVisitor] = None,
    leave: Optional[ExprVisitor] = None,
) -> None:
    def _walk(current: ExprNode) -> None:
        if enter is not None:
            enter(current)
        for child in iter_expr_children(current):
            _walk(child)
        if leave is not None:
            leave(current)

    _walk(node)


def collect_identifiers(node: ExprNode) -> Set[str]:
    out: Set[str] = set()

    def _enter(n: ExprNode) -> None:
        if isinstance(n, Identifier):
            out.add(n.name)

    def _leave(n: ExprNode) -> None:
        if isinstance(n, (QuantifierExpr, ListComprehension)):
            bound_prefix = f"{n.var}."
            out.difference_update(
                {name for name in out if name == n.var or name.startswith(bound_prefix)}
            )

    walk_expr_nodes(node, enter=_enter, leave=_leave)
    return out


def find_unsupported_functions(node: ExprNode, allowed: FrozenSet[str] = DEFAULT_ALLOWED_FUNCTIONS) -> Set[str]:
    bad: Set[str] = set()

    def _enter(n: ExprNode) -> None:
        if isinstance(n, FunctionCall) and n.name not in allowed:
            bad.add(n.name)

    walk_expr_nodes(node, enter=_enter)
    return bad


def validate_expr_capabilities(
    node: ExprNode,
    *,
    allowed_functions: FrozenSet[str] = DEFAULT_ALLOWED_FUNCTIONS,
    allowed_binary_ops: FrozenSet[str] = DEFAULT_ALLOWED_BINARY_OPS,
    allowed_unary_ops: FrozenSet[str] = DEFAULT_ALLOWED_UNARY_OPS,
    allowed_quantifiers: FrozenSet[str] = DEFAULT_ALLOWED_QUANTIFIERS,
) -> List[str]:
    errors: List[str] = []

    def _properties_arg_supported(arg: ExprNode) -> bool:
        if isinstance(arg, (Identifier, MapLiteral)):
            return True
        if isinstance(arg, Literal):
            return arg.value is None
        return False

    def _enter(n: ExprNode) -> None:
        if isinstance(n, (Identifier, Literal)):
            return
        if isinstance(n, Wildcard):
            errors.append("unsupported wildcard: *")
            return
        if isinstance(n, UnaryOp):
            if n.op.lower() not in allowed_unary_ops:
                errors.append(f"unsupported unary op: {n.op}")
            return
        if isinstance(n, BinaryOp):
            op = n.op.lower()
            if op not in allowed_binary_ops:
                errors.append(f"unsupported binary op: {n.op}")
            if op in GFQL_STRING_PREDICATE_OPS and not isinstance(n.right, Literal):
                errors.append(f"string predicate rhs must be literal scalar: {n.op}")
            return
        if isinstance(n, IsNullOp):
            return
        if isinstance(n, FunctionCall):
            if n.name not in allowed_functions:
                errors.append(f"unsupported function: {n.name}")
                return
            if n.name == "properties":
                if len(n.args) != 1:
                    errors.append("properties() requires exactly one argument")
                    return
                if not _properties_arg_supported(n.args[0]):
                    errors.append("properties() requires a node, relationship, map, or null argument")
            return
        if isinstance(n, CaseWhen):
            return
        if isinstance(n, QuantifierExpr):
            if n.fn not in allowed_quantifiers:
                errors.append(f"unsupported quantifier: {n.fn}")
            return
        if isinstance(n, (ListComprehension, ListLiteral, MapLiteral, SubscriptExpr, SliceExpr, PropertyAccessExpr)):
            return
        errors.append(f"unsupported node type: {type(n).__name__}")

    walk_expr_nodes(node, enter=_enter)
    return errors
