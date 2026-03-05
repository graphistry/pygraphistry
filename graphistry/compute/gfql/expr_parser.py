from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast


DEFAULT_ALLOWED_FUNCTIONS: FrozenSet[str] = frozenset(
    {
        "size",
        "abs",
        "toboolean",
        "tostring",
        "coalesce",
        "sign",
        "head",
        "tail",
        "reverse",
        "nodes",
        "relationships",
        "any",
        "all",
        "none",
        "single",
    }
)

DEFAULT_ALLOWED_BINARY_OPS: FrozenSet[str] = frozenset(
    {
        "or",
        "and",
        "=",
        "!=",
        "<>",
        "<",
        "<=",
        ">",
        ">=",
        "in",
        "contains",
        "starts_with",
        "ends_with",
        "+",
        "-",
        "*",
        "/",
        "%",
    }
)

DEFAULT_ALLOWED_UNARY_OPS: FrozenSet[str] = frozenset({"+", "-", "not"})

DEFAULT_ALLOWED_QUANTIFIERS: FrozenSet[str] = frozenset({"any", "all", "none", "single"})


@dataclass(frozen=True)
class Identifier:
    name: str


@dataclass(frozen=True)
class Literal:
    value: Any


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


ExprNode = Union[
    Identifier,
    Literal,
    UnaryOp,
    BinaryOp,
    IsNullOp,
    FunctionCall,
    CaseWhen,
    QuantifierExpr,
    ListComprehension,
    ListLiteral,
    MapLiteral,
    SubscriptExpr,
    SliceExpr,
]


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

?or_expr: and_expr
        | or_expr OR and_expr            -> or_op

?and_expr: not_expr
         | and_expr AND not_expr         -> and_op

?not_expr: NOT not_expr                  -> not_op
         | predicate

?predicate: additive
          | additive COMP_OP additive    -> cmp_op
          | additive IS NULL             -> is_null
          | additive IS NOT NULL         -> is_not_null
          | additive IN additive         -> in_op
          | additive CONTAINS additive   -> contains_op
          | additive STARTS WITH additive -> starts_with_op
          | additive ENDS WITH additive  -> ends_with_op

?additive: multiplicative
         | additive "+" multiplicative   -> add_op
         | additive "-" multiplicative   -> sub_op

?multiplicative: unary
               | multiplicative "*" unary -> mul_op
               | multiplicative "/" unary -> div_op
               | multiplicative "%" unary -> mod_op

?unary: "+" unary                        -> uplus
      | "-" unary                        -> uminus
      | postfix

?postfix: primary
        | postfix "[" subscript_key "]"  -> subscript

?subscript_key: expr                     -> subscript_index
              | [expr] ".." [expr]       -> subscript_slice

?primary: literal
        | identifier
        | function_call
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
map_key: NAME                            -> map_key_name
       | STRING                          -> map_key_string

function_call: NAME "(" [expr_list] ")"
identifier: NAME ("." NAME)*

case_expr: CASE WHEN expr THEN expr ELSE expr END

var_name: NAME

quantifier_expr: QUANTIFIER "(" var_name IN expr WHERE expr ")"

list_comprehension: "[" var_name IN expr [WHERE expr] ["|" expr] "]"

literal: NULL                            -> null_lit
       | TRUE                            -> true_lit
       | FALSE                           -> false_lit
       | NUMBER                          -> number_lit
       | STRING                          -> string_lit

COMP_OP: "<=" | ">=" | "<>" | "!=" | "=" | "<" | ">"

AND: /(?i:AND)/
OR: /(?i:OR)/
NOT: /(?i:NOT)/
IS: /(?i:IS)/
NULL: /(?i:NULL)/
IN: /(?i:IN)/
CONTAINS: /(?i:CONTAINS)/
STARTS: /(?i:STARTS)/
ENDS: /(?i:ENDS)/
WITH: /(?i:WITH)/
CASE: /(?i:CASE)/
WHEN: /(?i:WHEN)/
THEN: /(?i:THEN)/
ELSE: /(?i:ELSE)/
END: /(?i:END)/
WHERE: /(?i:WHERE)/
QUANTIFIER: /(?i:ANY|ALL|NONE|SINGLE)/
TRUE: /(?i:TRUE)/
FALSE: /(?i:FALSE)/

%import common.CNAME -> NAME
%import common.SIGNED_NUMBER -> NUMBER
STRING : /'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"/
%import common.WS
%ignore WS
"""


def _lark_imports() -> Tuple[Any, Any, Any]:
    try:
        from lark import Lark, Transformer
        from lark.exceptions import LarkError
        return Lark, Transformer, LarkError
    except Exception as exc:
        raise ImportError(
            "Lark is required for GFQL expression parsing. Install dependency 'lark'."
        ) from exc


@lru_cache(maxsize=1)
def _parser() -> Any:
    Lark, _, _ = _lark_imports()
    return Lark(_GRAMMAR, start="start", parser="lalr", maybe_placeholders=False)


def _parse_string_token(token: str) -> str:
    if len(token) < 2 or token[0] != token[-1] or token[0] not in {"'", '"'}:
        raise GFQLExprParseError("Invalid string literal")
    quote = token[0]
    out: List[str] = []
    i = 1
    end = len(token) - 1
    while i < end:
        ch = token[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= end:
            raise GFQLExprParseError("Invalid string literal")
        nxt = token[i + 1]
        if nxt == quote or nxt == "\\":
            out.append(nxt)
        elif nxt == "n":
            out.append("\n")
        elif nxt == "r":
            out.append("\r")
        elif nxt == "t":
            out.append("\t")
        elif nxt == "b":
            out.append("\b")
        elif nxt == "f":
            out.append("\f")
        else:
            # Keep unknown escapes permissive for Cypher-like inputs.
            out.append(nxt)
        i += 2
    return "".join(out)


def _parse_number_token(token: str) -> Union[int, float]:
    if any(c in token for c in (".", "e", "E")):
        return float(token)
    return int(token)


def _build_transformer() -> Any:
    _, Transformer, _ = _lark_imports()

    def _is_token(value: Any) -> bool:
        return hasattr(value, "type") and hasattr(value, "value")

    def _strip_tokens(items: Sequence[Any]) -> List[Any]:
        return [item for item in items if not _is_token(item)]

    class _AstBuilder(Transformer):  # type: ignore[valid-type,misc]
        def __init__(self) -> None:
            super().__init__(visit_tokens=True)

        def var_name(self, items: Sequence[Any]) -> str:
            return str(items[0])

        def grouped(self, items: Sequence[Any]) -> ExprNode:
            return cast(ExprNode, _strip_tokens(items)[0])

        def expr_list(self, items: Sequence[Any]) -> List[ExprNode]:
            return [cast(ExprNode, i) for i in _strip_tokens(items)]

        def null_lit(self, _: Sequence[Any]) -> Literal:
            return Literal(None)

        def true_lit(self, _: Sequence[Any]) -> Literal:
            return Literal(True)

        def false_lit(self, _: Sequence[Any]) -> Literal:
            return Literal(False)

        def number_lit(self, items: Sequence[Any]) -> Literal:
            return Literal(_parse_number_token(str(_strip_tokens(items)[0])))

        def string_lit(self, items: Sequence[Any]) -> Literal:
            return Literal(_parse_string_token(str(_strip_tokens(items)[0])))

        def identifier(self, items: Sequence[Any]) -> Identifier:
            names = [str(i) for i in _strip_tokens(items)]
            return Identifier(".".join(names))

        def function_call(self, items: Sequence[Any]) -> FunctionCall:
            stripped = _strip_tokens(items)
            fn = str(stripped[0]).lower()
            args: Tuple[ExprNode, ...] = ()
            if len(stripped) > 1:
                args = tuple(cast(List[ExprNode], stripped[1]))
            return FunctionCall(fn, args)

        def case_expr(self, items: Sequence[Any]) -> CaseWhen:
            stripped = _strip_tokens(items)
            return CaseWhen(
                condition=cast(ExprNode, stripped[0]),
                when_true=cast(ExprNode, stripped[1]),
                when_false=cast(ExprNode, stripped[2]),
            )

        def quantifier_expr(self, items: Sequence[Any]) -> QuantifierExpr:
            stripped = _strip_tokens(items)
            fn = str(stripped[0]).lower()
            var = str(stripped[1])
            source = cast(ExprNode, stripped[2])
            predicate = cast(ExprNode, stripped[3])
            return QuantifierExpr(fn=fn, var=var, source=source, predicate=predicate)

        def list_comprehension(self, items: Sequence[Any]) -> ListComprehension:
            stripped = _strip_tokens(items)
            if len(stripped) < 2:
                raise GFQLExprParseError("Invalid list comprehension")
            var_name = str(stripped[0])
            source = cast(ExprNode, stripped[1])
            predicate: Optional[ExprNode] = None
            projection: Optional[ExprNode] = None

            has_where = any(str(item).upper() == "WHERE" for item in items if _is_token(item))
            if has_where:
                if len(stripped) >= 3:
                    predicate = cast(ExprNode, stripped[2])
                if len(stripped) >= 4:
                    projection = cast(ExprNode, stripped[3])
            elif len(stripped) >= 3:
                projection = cast(ExprNode, stripped[2])

            return ListComprehension(
                var=var_name,
                source=source,
                predicate=predicate,
                projection=projection,
            )

        def list_literal(self, items: Sequence[Any]) -> ListLiteral:
            if len(items) == 0:
                return ListLiteral(())
            return ListLiteral(tuple(cast(List[ExprNode], items[0])))

        def map_key_name(self, items: Sequence[Any]) -> str:
            return str(_strip_tokens(items)[0])

        def map_key_string(self, items: Sequence[Any]) -> str:
            return _parse_string_token(str(_strip_tokens(items)[0]))

        def map_entry(self, items: Sequence[Any]) -> Tuple[str, ExprNode]:
            stripped = _strip_tokens(items)
            return (str(stripped[0]), cast(ExprNode, stripped[1]))

        def map_entries(self, items: Sequence[Any]) -> List[Tuple[str, ExprNode]]:
            return [cast(Tuple[str, ExprNode], i) for i in _strip_tokens(items)]

        def map_literal(self, items: Sequence[Any]) -> MapLiteral:
            stripped = _strip_tokens(items)
            if len(stripped) == 0:
                return MapLiteral(())
            return MapLiteral(tuple(cast(List[Tuple[str, ExprNode]], stripped[0])))

        def subscript_index(self, items: Sequence[Any]) -> Tuple[str, ExprNode]:
            return ("index", cast(ExprNode, _strip_tokens(items)[0]))

        def subscript_slice(self, items: Sequence[Any]) -> Tuple[str, Optional[ExprNode], Optional[ExprNode]]:
            stripped = _strip_tokens(items)
            if len(stripped) == 0:
                return ("slice", None, None)
            if len(stripped) == 1:
                return ("slice", cast(ExprNode, stripped[0]), None)
            return ("slice", cast(Optional[ExprNode], stripped[0]), cast(Optional[ExprNode], stripped[1]))

        def subscript(self, items: Sequence[Any]) -> ExprNode:
            stripped = _strip_tokens(items)
            value = cast(ExprNode, stripped[0])
            sub = cast(Tuple[Any, ...], stripped[1])
            if sub[0] == "index":
                return SubscriptExpr(value=value, key=cast(ExprNode, sub[1]))
            return SliceExpr(
                value=value,
                start=cast(Optional[ExprNode], sub[1]),
                stop=cast(Optional[ExprNode], sub[2]),
            )

        def uplus(self, items: Sequence[Any]) -> UnaryOp:
            return UnaryOp(op="+", operand=cast(ExprNode, _strip_tokens(items)[0]))

        def uminus(self, items: Sequence[Any]) -> UnaryOp:
            return UnaryOp(op="-", operand=cast(ExprNode, _strip_tokens(items)[0]))

        def not_op(self, items: Sequence[Any]) -> UnaryOp:
            return UnaryOp(op="not", operand=cast(ExprNode, _strip_tokens(items)[0]))

        def _bin(self, op: str, items: Sequence[Any]) -> BinaryOp:
            stripped = _strip_tokens(items)
            return BinaryOp(op=op, left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))

        def or_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("or", items)

        def and_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("and", items)

        def add_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("+", items)

        def sub_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("-", items)

        def mul_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("*", items)

        def div_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("/", items)

        def mod_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("%", items)

        def cmp_op(self, items: Sequence[Any]) -> BinaryOp:
            op = ""
            for item in items:
                if _is_token(item) and str(getattr(item, "type", "")) == "COMP_OP":
                    op = str(item).lower()
                    break
            stripped = _strip_tokens(items)
            if op == "":
                raise GFQLExprParseError("Missing comparison operator")
            return BinaryOp(op=op, left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))

        def in_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("in", items)

        def contains_op(self, items: Sequence[Any]) -> BinaryOp:
            return self._bin("contains", items)

        def starts_with_op(self, items: Sequence[Any]) -> BinaryOp:
            stripped = _strip_tokens(items)
            return BinaryOp(op="starts_with", left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))

        def ends_with_op(self, items: Sequence[Any]) -> BinaryOp:
            stripped = _strip_tokens(items)
            return BinaryOp(op="ends_with", left=cast(ExprNode, stripped[0]), right=cast(ExprNode, stripped[1]))

        def is_null(self, items: Sequence[Any]) -> IsNullOp:
            return IsNullOp(value=cast(ExprNode, _strip_tokens(items)[0]), negated=False)

        def is_not_null(self, items: Sequence[Any]) -> IsNullOp:
            return IsNullOp(value=cast(ExprNode, _strip_tokens(items)[0]), negated=True)

    return _AstBuilder()


def parse_expr(expr: str) -> ExprNode:
    if not isinstance(expr, str) or expr.strip() == "":
        raise GFQLExprParseError("Expression must be a non-empty string")

    parser = _parser()
    transformer = _build_transformer()
    try:
        tree = parser.parse(expr)
        node = transformer.transform(tree)
    except Exception as exc:
        _, _, LarkError = _lark_imports()
        if isinstance(exc, LarkError):
            line = getattr(exc, "line", None)
            column = getattr(exc, "column", None)
            raise GFQLExprParseError("Invalid GFQL expression", line=line, column=column) from exc
        if isinstance(exc, GFQLExprParseError):
            raise
        raise GFQLExprParseError("Invalid GFQL expression") from exc

    if not isinstance(
        node,
        (
            Identifier,
            Literal,
            UnaryOp,
            BinaryOp,
            IsNullOp,
            FunctionCall,
            CaseWhen,
            QuantifierExpr,
            ListComprehension,
            ListLiteral,
            MapLiteral,
            SubscriptExpr,
            SliceExpr,
        ),
    ):
        raise GFQLExprParseError("Invalid GFQL expression AST")
    return cast(ExprNode, node)


def collect_identifiers(node: ExprNode) -> Set[str]:
    out: Set[str] = set()

    def _walk(n: ExprNode) -> None:
        if isinstance(n, Identifier):
            out.add(n.name)
            return
        if isinstance(n, Literal):
            return
        if isinstance(n, UnaryOp):
            _walk(n.operand)
            return
        if isinstance(n, BinaryOp):
            _walk(n.left)
            _walk(n.right)
            return
        if isinstance(n, IsNullOp):
            _walk(n.value)
            return
        if isinstance(n, FunctionCall):
            for arg in n.args:
                _walk(arg)
            return
        if isinstance(n, CaseWhen):
            _walk(n.condition)
            _walk(n.when_true)
            _walk(n.when_false)
            return
        if isinstance(n, QuantifierExpr):
            _walk(n.source)
            _walk(n.predicate)
            out.discard(n.var)
            return
        if isinstance(n, ListComprehension):
            _walk(n.source)
            if n.predicate is not None:
                _walk(n.predicate)
            if n.projection is not None:
                _walk(n.projection)
            out.discard(n.var)
            return
        if isinstance(n, ListLiteral):
            for i in n.items:
                _walk(i)
            return
        if isinstance(n, MapLiteral):
            for _, v in n.items:
                _walk(v)
            return
        if isinstance(n, SubscriptExpr):
            _walk(n.value)
            _walk(n.key)
            return
        if isinstance(n, SliceExpr):
            _walk(n.value)
            if n.start is not None:
                _walk(n.start)
            if n.stop is not None:
                _walk(n.stop)
            return

    _walk(node)
    return out


def find_unsupported_functions(node: ExprNode, allowed: FrozenSet[str] = DEFAULT_ALLOWED_FUNCTIONS) -> Set[str]:
    bad: Set[str] = set()

    def _walk(n: ExprNode) -> None:
        if isinstance(n, FunctionCall):
            if n.name not in allowed:
                bad.add(n.name)
            for arg in n.args:
                _walk(arg)
        elif isinstance(n, UnaryOp):
            _walk(n.operand)
        elif isinstance(n, BinaryOp):
            _walk(n.left)
            _walk(n.right)
        elif isinstance(n, IsNullOp):
            _walk(n.value)
        elif isinstance(n, CaseWhen):
            _walk(n.condition)
            _walk(n.when_true)
            _walk(n.when_false)
        elif isinstance(n, QuantifierExpr):
            _walk(n.source)
            _walk(n.predicate)
        elif isinstance(n, ListComprehension):
            _walk(n.source)
            if n.predicate is not None:
                _walk(n.predicate)
            if n.projection is not None:
                _walk(n.projection)
        elif isinstance(n, ListLiteral):
            for i in n.items:
                _walk(i)
        elif isinstance(n, MapLiteral):
            for _, v in n.items:
                _walk(v)
        elif isinstance(n, SubscriptExpr):
            _walk(n.value)
            _walk(n.key)
        elif isinstance(n, SliceExpr):
            _walk(n.value)
            if n.start is not None:
                _walk(n.start)
            if n.stop is not None:
                _walk(n.stop)

    _walk(node)
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

    def _walk(n: ExprNode) -> None:
        if isinstance(n, Identifier):
            return
        if isinstance(n, Literal):
            return
        if isinstance(n, UnaryOp):
            if n.op.lower() not in allowed_unary_ops:
                errors.append(f"unsupported unary op: {n.op}")
            _walk(n.operand)
            return
        if isinstance(n, BinaryOp):
            if n.op.lower() not in allowed_binary_ops:
                errors.append(f"unsupported binary op: {n.op}")
            _walk(n.left)
            _walk(n.right)
            return
        if isinstance(n, IsNullOp):
            _walk(n.value)
            return
        if isinstance(n, FunctionCall):
            if n.name not in allowed_functions:
                errors.append(f"unsupported function: {n.name}")
            for arg in n.args:
                _walk(arg)
            return
        if isinstance(n, CaseWhen):
            _walk(n.condition)
            _walk(n.when_true)
            _walk(n.when_false)
            return
        if isinstance(n, QuantifierExpr):
            if n.fn not in allowed_quantifiers:
                errors.append(f"unsupported quantifier: {n.fn}")
            _walk(n.source)
            _walk(n.predicate)
            return
        if isinstance(n, ListComprehension):
            _walk(n.source)
            if n.predicate is not None:
                _walk(n.predicate)
            if n.projection is not None:
                _walk(n.projection)
            return
        if isinstance(n, ListLiteral):
            for i in n.items:
                _walk(i)
            return
        if isinstance(n, MapLiteral):
            for _, v in n.items:
                _walk(v)
            return
        if isinstance(n, SubscriptExpr):
            _walk(n.value)
            _walk(n.key)
            return
        if isinstance(n, SliceExpr):
            _walk(n.value)
            if n.start is not None:
                _walk(n.start)
            if n.stop is not None:
                _walk(n.stop)
            return

        errors.append(f"unsupported node type: {type(n).__name__}")

    _walk(node)
    return errors
