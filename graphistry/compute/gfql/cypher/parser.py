from __future__ import annotations

import ast as pyast
from dataclasses import dataclass, replace
from functools import lru_cache
import re
from typing import Any, Callable, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union, cast

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
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


# ---------------------------------------------------------------------------
# GRAMMAR — the declarative source of truth for Cypher syntax.
#
# EDITING THIS GRAMMAR (the safe-by-construction extension flow):
#   1. Add / change a rule here.
#   2. Add at least one query exercising the new syntax to
#      DIFFERENTIAL_CORPUS in tests/.../test_grammar_invariants.py (or, if the
#      construct is grammatical-but-intentionally-unsupported, to
#      GRAMMAR_ONLY_COVERAGE). test_every_grammar_rule_is_exercised_by_the_corpus
#      FAILS with the new rule's name until you do — you cannot land a rule
#      with no coverage.
#   3. Run test_grammar_invariants.py. The machine checks that guard you:
#        - the grammar has ZERO LALR conflicts and builds under strict=True —
#          provably unambiguous; a new ambiguity introduces a conflict and
#          fails the build. Fix it IN THE GRAMMAR (as WITH..WHERE bundling,
#          the dotted-name split, and no-top-level-IN list elements did) —
#          never by resolving a conflict in Python;
#        - semantic ambiguity is ZERO (every Earley derivation -> same AST);
#        - LALR == Earley over the corpus + full-repo scrape (AST-identical).
#   4. If the edit deliberately changes accept/reject, pin it in
#      DELIBERATE_LANGUAGE_FIXES. Otherwise a language change fails the
#      differential.
# The grammar carries the correctness argument; the tests make its properties
# machine-checked. Do not resolve ambiguity in Python — fix it in the grammar.
# ---------------------------------------------------------------------------
_GRAMMAR = r"""
?start: graph_query

graph_query: graph_binding* union_query          -> graph_query_with_bindings
           | graph_binding* graph_constructor     -> graph_query_standalone

graph_binding: "GRAPH"i NAME "=" graph_constructor

graph_constructor: "GRAPH"i "{" graph_constructor_body "}"

graph_constructor_body: graph_constructor_item+

graph_constructor_item: match_clause
                      | call_clause
                      | use_clause

use_clause: "USE"i NAME

union_query: query_body (union_op query_body)* SEMI?
query_body: query_item+
union_op: "UNION"i "ALL"i       -> union_all
        | "UNION"i              -> union_distinct

query_item: match_clause
          | call_clause
          | unwind_clause
          | use_clause
          | stage

stage: with_stage
     | return_stage

with_stage: with_clause with_where_clause? order_by_clause? skip_clause? limit_clause?
return_stage: return_clause order_by_clause? skip_clause? limit_clause?

// WHERE binds to its preceding clause IN THE GRAMMAR (openCypher scoping):
// after MATCH it is part of match_clause; after WITH it is with_where_clause.
// A standalone where_clause item would make `MATCH .. WHERE ..` derivable two
// ways (bundled vs standalone) -- a genuine ambiguity -- so it does not exist.
match_clause: "MATCH"i match_item ("," match_item)* where_clause?              -> match_clause
            | "OPTIONAL"i "MATCH"i match_item ("," match_item)* where_clause?  -> optional_match_clause
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

// Unified: every WHERE parses as a generic boolean ``expr`` (so LALR(1) accepts
// OR/XOR/NOT/parenthesized clauses, no Earley). ``generic_where_clause`` lifts the
// structured (filter_dict) predicates back out of the parsed tree. ``where_predicate``
// is retained as the building block for the ``where_predicate_chain`` lift parser.
where_clause: "WHERE"i expr                 -> generic_where_clause
where_predicate: property_ref COMP_OP where_rhs -> cmp_where
               | property_ref "IS"i "NULL"i -> is_null_where
               | property_ref "IS"i "NOT"i "NULL"i -> is_not_null_where
               | property_ref "CONTAINS"i where_rhs -> contains_where
               | property_ref "STARTS"i "WITH"i where_rhs -> starts_with_where
               | property_ref "ENDS"i "WITH"i where_rhs -> ends_with_where
               | property_ref "=~" where_rhs -> regex_where
               | variable labels -> has_labels_where
// Flat AND-chain of bare predicates; the start symbol for the lift parser. Parens
// / OR / XOR / NOT are not ``where_predicate``s, so such clauses fail this parse and
// stay on the row-filter path (which treats a missing property as null).
where_predicate_chain: where_predicate ("AND"i where_predicate)*
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
// A parenthesized label predicate ``(n:Admin)`` parses via ``expr`` ->
// ``grouped_expr`` over ``bare_label_predicate_expr`` (identical AST), so a
// dedicated ``label_predicate_expr`` rule here would be a pure derivation
// redundancy -- exactly the RPAR shift/reduce conflict. Omitted on purpose.
return_expr: "*"                              -> projection_star
           | expr
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
           | additive "=~" additive          -> regex_op

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

// Dotted chains rooted at a bare NAME derive ONLY via qualified_name (a.b.c);
// property_access applies ONLY to composite roots (calls, groups, subscripts:
// f(x).y, (e).y, xs[0].y). Splitting the two keeps the grammar unambiguous --
// a single derivation for every dotted expression -- and LALR(1) conflict-free
// at DOT (no reduce-qualified_name vs shift-extend race).
?postfix: qualified_name
        | postfix_composite
?postfix_composite: primary_composite
        | postfix "[" subscript_key "]"     -> subscript
        | postfix_composite "." NAME        -> property_access

?primary_composite: parameter
        | literal
        | function_call
        | case_expr
        | quantifier_expr
        | list_comprehension
        | list_literal
        | map_literal
        | WHERE_PATTERN                     -> pattern_atom
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

// A list literal's elements are expressions with NO top-level ``IN`` operator.
// Rationale: ``[x IN xs ...`` is list-comprehension syntax; if a list element
// could also be a bare ``in_op`` (``[x IN xs, y]``), the two overlap and the
// grammar is ambiguous at ``[ NAME . IN`` (this was the last shift/reduce
// conflict). Excluding top-level ``IN`` from list elements makes the grammar
// provably unambiguous (builds under ``strict=True``) AND rejects the invalid
// "list of IN-booleans" uniformly (``[1 IN a, 2]``, ``[n.x IN a, y]`` too, not
// just the bare-name case) — a construct GFQL cannot execute anyway. A genuine
// list of membership booleans is still expressible with parens: ``[(x IN xs)]``.
// ``list_element`` mirrors the ``expr`` hierarchy exactly, minus the ``in_op``
// alternative at the comparable level; ``additive`` and below are shared.
list_literal: "[" [list_element_list] "]"
list_element_list: list_element ("," list_element)*
?list_element: or_expr_no_in
?or_expr_no_in: xor_expr_no_in | or_expr_no_in "OR"i xor_expr_no_in     -> or_op
?xor_expr_no_in: and_expr_no_in | xor_expr_no_in "XOR"i and_expr_no_in  -> xor_op
?and_expr_no_in: not_expr_no_in | and_expr_no_in "AND"i not_expr_no_in  -> and_op
?not_expr_no_in: "NOT"i not_expr_no_in                                  -> not_op
              | predicate_no_in
?predicate_no_in: comparable_no_in
               | comparable_no_in COMP_OP comparable_no_in COMP_OP comparable_no_in -> chained_cmp
               | comparable_no_in COMP_OP comparable_no_in              -> cmp_op
               | bare_label_predicate_expr
?comparable_no_in: additive
                | additive "IS"i "NULL"i                                -> expr_is_null
                | additive "IS"i "NOT"i "NULL"i                         -> expr_is_not_null
                | additive "CONTAINS"i additive                         -> contains_op
                | additive "STARTS"i "WITH"i additive                   -> starts_with_op
                | additive "ENDS"i "WITH"i additive                     -> ends_with_op

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


def _match_bare_label_atom(text: Optional[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Return ``(alias, labels)`` if *text* fullmatches the bare-label-predicate
    shape; else ``None``.  ``fullmatch`` is load-bearing as the false-positive
    guard from #1125 — atom fragments that merely *look* label-shaped (e.g.
    parts of string literals) must not lift into structured predicates.
    """
    if text is None:
        return None
    m = _BARE_LABEL_PREDICATE_RE.fullmatch(text.strip())
    if m is None:
        return None
    labels = tuple(label for label in m.group(2).split(":") if label)
    return (m.group("alias"), labels)


def _lift_label_only_and_spine(
    node: BooleanExpr,
) -> Optional[Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    """Return lifted ``(alias, labels)`` tuples if *node* is an AND-spine whose
    every leaf atom is a bare-label predicate; else ``None``.

    Walks Lark's parsed boolean tree (single source of truth) instead of
    re-splitting the WHERE body text on top-level ``AND``.
    """
    out: List[Tuple[str, Tuple[str, ...]]] = []
    stack: List[BooleanExpr] = [node]
    while stack:
        cur = stack.pop()
        if cur.op == "and":
            if cur.left is None or cur.right is None:
                return None
            stack.append(cur.right)
            stack.append(cur.left)
        elif cur.op == "atom":
            atom = _match_bare_label_atom(cur.atom_text)
            if atom is None:
                return None
            out.append(atom)
        else:
            return None
    return tuple(out)


def _split_top_level_and_pattern_leaves(
    expr: BooleanExpr,
) -> Tuple[List[BooleanExpr], List[BooleanExpr], List[BooleanExpr], bool]:
    """Split *expr* at top-level AND boundaries.

    Returns ``(positive_patterns, negated_patterns, others, has_nested_pattern)``.

    - ``positive_patterns``: leaf ``BooleanExpr(op="pattern")`` nodes at top-level
      AND positions.  Lifted to ``WherePatternPredicate(negated=False)``.
    - ``negated_patterns``: leaf ``BooleanExpr(op="pattern")`` nodes wrapped in a
      single top-level ``not`` (i.e. ``NOT (n)-[:R]->()``).  Lifted to
      ``WherePatternPredicate(negated=True)`` for #1031 slice 2 anti-semi-join
      lowering.  Returns the inner ``pattern`` leaf (the ``not`` is consumed).
    - ``others``: non-pattern conjuncts that should remain in ``expr_tree``.
    - ``has_nested_pattern``: True when a pattern atom appears in a deeper
      non-AND/non-direct-NOT context (e.g. ``OR`` with a pattern leaf, or
      ``NOT (and-tree-of-patterns)``).  Lowering consumes this by keeping
      such leaves in ``expr_tree`` instead of lifting them to predicates.
    """
    if expr.op == "and":
        if expr.left is None or expr.right is None:
            return [], [], [expr], False
        l_pos, l_neg, l_oth, l_bad = _split_top_level_and_pattern_leaves(expr.left)
        r_pos, r_neg, r_oth, r_bad = _split_top_level_and_pattern_leaves(expr.right)
        return l_pos + r_pos, l_neg + r_neg, l_oth + r_oth, l_bad or r_bad
    if expr.op == "pattern":
        return [expr], [], [], False
    if expr.op == "not" and expr.left is not None and expr.left.op == "pattern":
        # `WHERE NOT (pattern)` — slice 2 anti-semi-join target.  Strip the NOT
        # and emit the inner pattern leaf as a negated pattern.  No nested
        # pattern; rest of the tree continues structural traversal as usual.
        return [], [expr.left], [], False
    return [], [], [expr], _has_pattern_descendant(expr)


def _has_pattern_descendant(expr: BooleanExpr) -> bool:
    if expr.op == "pattern":
        return True
    if expr.left is not None and _has_pattern_descendant(expr.left):
        return True
    if expr.right is not None and _has_pattern_descendant(expr.right):
        return True
    return False


def _rebuild_and_tree(conjuncts: List[BooleanExpr]) -> Optional[BooleanExpr]:
    if not conjuncts:
        return None
    if len(conjuncts) == 1:
        return conjuncts[0]
    tree = conjuncts[0]
    for conjunct in conjuncts[1:]:
        new_span = SourceSpan(
            line=tree.span.line,
            column=tree.span.column,
            end_line=conjunct.span.end_line,
            end_column=conjunct.span.end_column,
            start_pos=tree.span.start_pos,
            end_pos=conjunct.span.end_pos,
        )
        tree = BooleanExpr(op="and", span=new_span, left=tree, right=conjunct)
    return tree


# Substring "WHERE pattern predicates" is load-bearing for legacy
# test_lowering.py + test_parser.py error-message contracts.
def _build_where_with_pattern_lift(
    *,
    pattern_leaves: List[BooleanExpr],
    negated_pattern_leaves: List[BooleanExpr],
    other_conjuncts: List[BooleanExpr],
    nested_pattern: bool,
    expr_text: str,
    span: SourceSpan,
) -> WhereClause:
    # Slice 3 (#1031): N positive patterns each become a WherePatternPredicate
    # (negated=False).  Slice 2 (#1031): N NOT-patterns each become a
    # WherePatternPredicate (negated=True) for downstream anti-semi-join
    # lowering.  Both groups travel together in WhereClause.predicates;
    # ast_normalizer dispatches by the negated flag.
    pattern_preds: List[WherePatternPredicate] = []
    for leaf in pattern_leaves:
        assert leaf.pattern is not None, "pattern_atom invariant: pattern payload always set"
        pattern_preds.append(WherePatternPredicate(pattern=leaf.pattern, span=leaf.span, negated=False))
    for leaf in negated_pattern_leaves:
        assert leaf.pattern is not None, "pattern_atom invariant: pattern payload always set"
        pattern_preds.append(WherePatternPredicate(pattern=leaf.pattern, span=leaf.span, negated=True))
    new_expr_tree = _rebuild_and_tree(other_conjuncts)
    if new_expr_tree is None:
        return WhereClause(predicates=tuple(pattern_preds), expr_tree=None, span=span)
    # Nested pattern leaves (OR/XOR/complex NOT contexts) stay in expr_tree;
    # lowering rewrites them to correlated semi-apply marker columns.
    _ = nested_pattern
    return WhereClause(
        predicates=tuple(pattern_preds),
        expr_tree=new_expr_tree,
        span=span,
    )


_RESERVED_IDENTIFIER_GRAPH = "graph"
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


@dataclass(frozen=True)
class _PrimitiveLiteral:
    value: Union[None, bool, int, float, str]
    text: str
    span: SourceSpan


def _unwrap_primitive_literal(value: object) -> object:
    if isinstance(value, _PrimitiveLiteral):
        return value.value
    return value


@dataclass(frozen=True)
class _BoundPattern:
    alias: str
    pattern: Tuple[PatternElement, ...]
    kind: Literal["pattern", "shortestPath", "allShortestPaths"] = "pattern"


@lru_cache(maxsize=1)
def _parser_lalr() -> _ParserLike:
    Lark, _, _, _ = _lark_imports()
    # Sole whole-query parser. The unified WHERE grammar is LALR(1)-parseable, so
    # this accepts every supported query (~80x faster than the former Earley
    # parser); generic_where_clause lifts the structured predicates back out.
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
    # Parses a single pattern fragment (e.g. ``(n)-[:R]->()``) for WHERE pattern
    # predicates. LALR(1) accepts the pattern grammar, so no Earley is needed here.
    parser = Lark(
        _GRAMMAR,
        start="pattern",
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )
    return cast(_ParserLike, parser)


@lru_cache(maxsize=1)
def _where_predicate_chain_parser() -> _ParserLike:
    Lark, _, _, _ = _lark_imports()
    # Parses a flat ``where_predicate ("AND" where_predicate)*`` chain to lift a WHERE
    # body into structured (filter_dict) form. Parens / OR / XOR / NOT / arithmetic make
    # the body NOT a flat chain, so this parse fails and the clause stays on the
    # ``where_rows`` row-filter path (see _lift_and_spine_predicates for why).
    parser = Lark(
        _GRAMMAR,
        start="where_predicate_chain",
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )
    return cast(_ParserLike, parser)


def _retarget_span(s: SourceSpan, base: SourceSpan) -> SourceSpan:
    """Shift an atom-relative span (from re-parsing an isolated atom substring) into
    absolute query coordinates, given the atom's absolute ``base`` span. Exact for
    single-line atoms (the normal case); best-effort across embedded newlines."""
    return SourceSpan(
        line=base.line + (s.line - 1),
        column=(base.column + s.column - 1) if s.line == 1 else s.column,
        end_line=base.line + (s.end_line - 1),
        end_column=(base.column + s.end_column - 1) if s.end_line == 1 else s.end_column,
        start_pos=base.start_pos + s.start_pos,
        end_pos=base.start_pos + s.end_pos,
    )


def _retarget_predicate_spans(pred: "WherePredicate", base: SourceSpan) -> "WherePredicate":
    """Rebuild a lifted predicate with every span shifted from atom-relative to
    absolute (see ``_retarget_span``), so downstream errors (e.g. E108 in lowering)
    point at the predicate's real position in the query instead of column 1."""
    left = replace(pred.left, span=_retarget_span(pred.left.span, base))
    right = pred.right
    if isinstance(right, (PropertyRef, ParameterRef)):
        right = replace(right, span=_retarget_span(right.span, base))
    return replace(pred, left=left, right=right, span=_retarget_span(pred.span, base))


def _lift_and_spine_predicates(
    expr_text: str, base_span: Optional[SourceSpan] = None
) -> Optional[List["WherePredicate"]]:
    """Lift the WHERE body to structured ``filter_dict`` predicates iff it parses as a
    flat ``where_predicate ("AND" where_predicate)*`` chain (cmp / IS NULL / CONTAINS /
    STARTS|ENDS WITH / label); else ``None``, leaving it on ``expr_tree`` -> ``where_rows``.

    Only flat chains lift; parentheses, OR / XOR / NOT and arithmetic stay on
    ``where_rows``. This is a correctness boundary, not just routing: a parenthesized
    predicate over a *missing* property (e.g. ``a IS NULL AND (b = 1)`` where ``b`` is
    absent) must stay on ``where_rows``, which treats an absent property as null
    (Cypher semantics); ``filter_dict`` requires the column to exist and would raise.

    ``base_span`` is the WHERE body's absolute position; the body is parsed in isolation
    (spans relative to it), so predicate spans are shifted back to absolute."""
    _, _, LarkError, _ = _lark_imports()
    try:
        tree = _where_predicate_chain_parser().parse(expr_text)
        preds = _build_transformer(expr_text).transform(tree)
    except (LarkError, GFQLSyntaxError, GFQLValidationError):
        return None
    if not (isinstance(preds, tuple) and preds and all(isinstance(p, WherePredicate) for p in preds)):
        return None
    if base_span is None:
        return list(preds)
    return [_retarget_predicate_spans(p, base_span) for p in preds]


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

    def _relationship_rule(direction: str, *, simple: bool = False) -> Callable[[Any, Any, Sequence[Any]], RelationshipPattern]:
        def _rule(self: Any, meta: Any, items: Sequence[Any]) -> RelationshipPattern:
            return self._relationship(meta, () if simple else items, direction=direction)
        return _rule

    @v_args(meta=True)  # type: ignore[misc]
    class _CypherAstBuilder(Transformer):  # type: ignore[valid-type,misc]
        def _slice(self, span: SourceSpan) -> str:
            return source[span.start_pos:span.end_pos]

        def _expression_slice(self, meta: Any, _items: Sequence[Any]) -> _ExpressionSlice:
            span = _span_from_meta(meta)
            return _ExpressionSlice(text=self._slice(span).strip(), span=span)

        def _string_tuple(self, _meta: Any, items: Sequence[Any]) -> Tuple[str, ...]:
            return tuple(str(item) for item in items)

        def variable(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid variable reference", line=meta.line, column=meta.column)
            return str(items[0])

        def label(self, meta: Any, items: Sequence[Any]) -> str:
            if len(items) != 1:
                raise _to_syntax_error("Invalid node label", line=meta.line, column=meta.column)
            return str(items[0])

        labels = rel_types = _string_tuple

        def parameter(self, meta: Any, items: Sequence[Any]) -> ParameterRef:
            if len(items) != 1:
                raise _to_syntax_error("Invalid parameter reference", line=meta.line, column=meta.column)
            return ParameterRef(name=str(items[0]), span=_span_from_meta(meta))

        def null_lit(self, meta: Any, _items: Sequence[Any]) -> _PrimitiveLiteral:
            span = _span_from_meta(meta)
            return _PrimitiveLiteral(value=None, text=self._slice(span), span=span)

        def true_lit(self, meta: Any, _items: Sequence[Any]) -> _PrimitiveLiteral:
            span = _span_from_meta(meta)
            return _PrimitiveLiteral(value=True, text=self._slice(span), span=span)

        def false_lit(self, meta: Any, _items: Sequence[Any]) -> _PrimitiveLiteral:
            span = _span_from_meta(meta)
            return _PrimitiveLiteral(value=False, text=self._slice(span), span=span)

        def number_lit(self, meta: Any, items: Sequence[Any]) -> _PrimitiveLiteral:
            if len(items) != 1:
                raise _to_syntax_error("Invalid numeric literal", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return _PrimitiveLiteral(value=_parse_number_token(str(items[0])), text=self._slice(span), span=span)

        def string_lit(self, meta: Any, items: Sequence[Any]) -> _PrimitiveLiteral:
            if len(items) != 1:
                raise _to_syntax_error("Invalid string literal", line=meta.line, column=meta.column)
            try:
                value = _parse_string_token(str(items[0]))
            except ValueError as exc:
                raise _to_syntax_error(str(exc), line=meta.line, column=meta.column) from exc
            span = _span_from_meta(meta)
            return _PrimitiveLiteral(value=value, text=self._slice(span), span=span)

        def property_entry(self, meta: Any, items: Sequence[Any]) -> PropertyEntry:
            if len(items) != 2:
                raise _to_syntax_error("Invalid property entry", line=meta.line, column=meta.column)
            key = str(items[0])
            raw_value = _unwrap_primitive_literal(items[1])
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
            value = self._rel_hops(meta, items[0])
            return {"min_hops": value, "max_hops": None, "to_fixed_point": True}

        def rel_range_fixed(self, meta: Any, _items: Sequence[Any]) -> dict[str, Any]:
            return {"min_hops": None, "max_hops": None, "to_fixed_point": True}

        rel_forward = _relationship_rule("forward")
        rel_reverse = _relationship_rule("reverse")
        rel_undirected = _relationship_rule("undirected")
        rel_forward_simple = _relationship_rule("forward", simple=True)
        rel_reverse_simple = _relationship_rule("reverse", simple=True)
        rel_undirected_simple = _relationship_rule("undirected", simple=True)
        rel_bidirectional_simple = _relationship_rule("undirected", simple=True)

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
            return self._match_clause(meta, items, optional=False)

        def optional_match_clause(self, meta: Any, items: Sequence[Any]) -> MatchClause:
            return self._match_clause(meta, items, optional=True)

        def _match_clause(self, meta: Any, items: Sequence[Any], *, optional: bool) -> MatchClause:
            # The grammar bundles a trailing WHERE into the match_clause (WHERE
            # binds to its preceding clause declaratively). Split it back off;
            # query_body's assembly consumes it as if it were a standalone item.
            where: Optional[WhereClause] = None
            if items and isinstance(items[-1], WhereClause):
                where = items[-1]
                items = items[:-1]
            if len(items) < 1:
                clause_name = "OPTIONAL MATCH" if optional else "MATCH"
                raise _to_syntax_error(f"Cypher {clause_name} clause cannot be empty", line=meta.line, column=meta.column)
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
            # Span covers the MATCH..patterns text only (not the bundled WHERE),
            # preserving the clause span the AST has always carried: end at the
            # last non-whitespace character before the WHERE keyword.
            span = _span_from_meta(meta)
            if where is not None:
                end_pos = span.start_pos + len(source[span.start_pos:where.span.start_pos].rstrip())
                line_start = source.rfind("\n", 0, end_pos) + 1
                span = replace(
                    span,
                    end_pos=end_pos,
                    end_line=source.count("\n", 0, end_pos) + 1,
                    end_column=end_pos - line_start + 1,
                )
            return MatchClause(
                patterns=tuple(patterns),
                span=span,
                optional=optional,
                pattern_aliases=tuple(pattern_aliases),
                pattern_alias_kinds=tuple(pattern_alias_kinds),
                where=where,
            )

        def distinct(self, _meta: Any, _items: Sequence[Any]) -> bool:
            return True

        qualified_name = return_expr = order_expr = unwind_expr = _expression_slice

        def property_ref(self, meta: Any, items: Sequence[Any]) -> PropertyRef:
            if len(items) != 2:
                raise _to_syntax_error("Invalid property reference", line=meta.line, column=meta.column)
            return PropertyRef(alias=str(items[0]), property=str(items[1]), span=_span_from_meta(meta))

        def where_rhs(self, _meta: Any, items: Sequence[Any]) -> object:
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE right-hand side")
            return _unwrap_primitive_literal(items[0])

        def where_predicate_chain(self, _meta: Any, items: Sequence[Any]) -> Tuple[WherePredicate, ...]:
            # Flat AND-chain of bare predicates (the lift parser's start symbol).
            return tuple(cast(WherePredicate, p) for p in items)

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

        def _where_predicate(
            self,
            meta: Any,
            items: Sequence[Any],
            *,
            op: str,
            message: str,
            rhs: bool = False,
        ) -> WherePredicate:
            expected = 2 if rhs else 1
            if len(items) != expected:
                raise _to_syntax_error(message, line=meta.line, column=meta.column)
            return WherePredicate(
                left=cast(PropertyRef, items[0]),
                op=cast(Any, op),
                right=cast(Any, items[1]) if rhs else None,
                span=_span_from_meta(meta),
            )

        def is_null_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            return self._where_predicate(meta, items, op="is_null", message="Invalid WHERE IS NULL predicate")

        def is_not_null_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            return self._where_predicate(meta, items, op="is_not_null", message="Invalid WHERE IS NOT NULL predicate")

        def contains_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            return self._where_predicate(meta, items, op="contains", message="Invalid WHERE CONTAINS predicate", rhs=True)

        def starts_with_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            return self._where_predicate(meta, items, op="starts_with", message="Invalid WHERE STARTS WITH predicate", rhs=True)

        def ends_with_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            return self._where_predicate(meta, items, op="ends_with", message="Invalid WHERE ENDS WITH predicate", rhs=True)

        def regex_where(self, meta: Any, items: Sequence[Any]) -> WherePredicate:
            # openCypher/neo4j `=~` — Java-regex, full/anchored match (lowered to fullmatch).
            return self._where_predicate(meta, items, op="regex", message="Invalid WHERE =~ regex predicate", rhs=True)

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

        def _parse_single_where_pattern_predicate_text(self, pattern_item_text: str, span: SourceSpan) -> WherePatternPredicate:
            """Parse one bare-pattern-item text (no AND-chain) into a WherePatternPredicate.

            Caller is responsible for splitting the WHERE_PATTERN token text into individual
            pattern-item segments via ``_WHERE_PATTERN_ITEM_RE``.  This helper handles the
            single-item parse + Lark-error wrapping.
            """
            try:
                pattern_tree = _pattern_parser().parse(pattern_item_text)
                pattern_node = _build_transformer(pattern_item_text).transform(pattern_tree)
            except Exception as exc:
                _, _, LarkError, _ = _lark_imports()
                if isinstance(exc, (GFQLSyntaxError, GFQLValidationError)):
                    raise
                if isinstance(exc, LarkError):
                    raise GFQLValidationError(
                        ErrorCode.E108,
                        "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
                        field="where",
                        value=pattern_item_text,
                        suggestion="Use a positive fixed-length relationship existence pattern in WHERE.",
                        line=span.line,
                        column=span.column,
                        language="cypher",
                    ) from exc
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
                    field="where",
                    value=pattern_item_text,
                    suggestion="Use a positive fixed-length relationship existence pattern in WHERE.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                ) from exc
            if not isinstance(pattern_node, tuple):
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher WHERE pattern predicate is outside the currently supported GFQL Cypher subset",
                    field="where",
                    value=pattern_item_text,
                    suggestion="Use a positive fixed-length relationship existence pattern in WHERE.",
                    line=span.line,
                    column=span.column,
                    language="cypher",
                )
            return WherePatternPredicate(pattern=cast(Tuple[PatternElement, ...], pattern_node), span=span)

        def pattern_atom(self, meta: Any, items: Sequence[Any]) -> BooleanExpr:
            # The ``WHERE_PATTERN`` lexer token is greedy and gobbles
            # ``pattern AND pattern AND ...`` chains as a single match.  Split it
            # back into individual pattern-item texts here and emit one
            # ``BooleanExpr(op="pattern")`` per item; multi-item input becomes an
            # AND-tree that ``_split_top_level_and_pattern_leaves`` upstream
            # reassembles as N pattern leaves (#1031 slice 3).
            if len(items) != 1:
                raise _to_syntax_error("Invalid WHERE pattern predicate", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            pattern_text = str(items[0]).strip()
            pattern_item_texts = [match.group(0).strip() for match in _WHERE_PATTERN_ITEM_RE.finditer(pattern_text)]
            if not pattern_item_texts:
                raise _to_syntax_error("Invalid WHERE pattern predicate", line=meta.line, column=meta.column)
            atoms: List[BooleanExpr] = []
            for item_text in pattern_item_texts:
                pattern_pred = self._parse_single_where_pattern_predicate_text(item_text, span)
                atoms.append(BooleanExpr(
                    op="pattern",
                    span=span,
                    atom_text=item_text,
                    atom_span=span,
                    pattern=pattern_pred.pattern,
                ))
            tree = _rebuild_and_tree(atoms)
            assert tree is not None  # gated above by `if not pattern_item_texts`
            return tree

        def _wrap_as_boolean_atom(self, operand: Any, enclosing_meta: Any) -> BooleanExpr:
            """Coerce a parsed expression operand into a ``BooleanExpr`` atom.

            Recursion bottom-out for ``and_op`` / ``or_op`` / ``xor_op`` /
            ``not_op``.  ``BooleanExpr`` passes through.  Lark ``Tree`` and
            ``_ExpressionSlice`` operands carry their own span, so we use
            it to extract the source slice precisely.

            Primitive literal transformers preserve raw semantic values
            inside ``_PrimitiveLiteral`` so boolean operators can keep the
            operand's exact source text/span while structured predicates and
            properties still receive ordinary Python literal values.
            """
            if isinstance(operand, BooleanExpr):
                return operand
            if isinstance(operand, _ExpressionSlice):
                span = operand.span
                text = operand.text
            elif isinstance(operand, _PrimitiveLiteral):
                span = operand.span
                text = operand.text
            else:
                operand_meta = getattr(operand, "meta", None)
                if operand_meta is None:
                    span = _span_from_meta(enclosing_meta)
                    text = self._slice(span)
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

        def generic_where_clause(self, meta: Any, items: Sequence[Any]) -> WhereClause:
            span = _span_from_meta(meta)
            expr_text = self._slice(span)[len("WHERE"):].strip()
            # Absolute span of the WHERE body (drops "WHERE" + whitespace): aligns the
            # synth single-atom node and is the lift's span base (the lift parses
            # ``expr_text`` in isolation, so its spans need shifting to absolute).
            _body = self._slice(span)[len("WHERE"):]
            _start_off = len("WHERE") + (len(_body) - len(_body.lstrip()))
            body_span = replace(
                span,
                column=span.column + _start_off,
                start_pos=span.start_pos + _start_off,
                end_column=span.column + _start_off + len(expr_text),
                end_pos=span.start_pos + _start_off + len(expr_text),
            )
            # Capture Lark's structural expression tree.  Only ``and_op`` /
            # ``or_op`` / ``xor_op`` / ``not_op`` produce ``BooleanExpr``;
            # atomic WHERE expressions (single predicate) route here with a
            # non-BooleanExpr operand and are wrapped as a single-atom tree
            # so the invariant ``(expr is None) == (expr_tree is None)``
            # holds (#1213 sub-PR A / #1214).
            if items and isinstance(items[0], BooleanExpr):
                expr_tree: BooleanExpr = cast(BooleanExpr, items[0])
            else:
                expr_tree = BooleanExpr(op="atom", span=body_span, atom_text=expr_text, atom_span=body_span)
            # The unified grammar parses every WHERE as a generic `expr`, so bare
            # label predicates and AND chains of them arrive here as a ``BooleanExpr``.
            # Walk the parsed tree (single source of truth) to lift them to structured
            # predicates instead of re-splitting the WHERE body text on top-level
            # ``AND``.  Any non-AND boolean op, non-atom node, or non-bare-label atom
            # causes a conservative fallback to raw expr (preserved on ``expr_tree``).
            lifted = _lift_label_only_and_spine(expr_tree)
            if lifted:
                predicates = tuple(
                    WherePredicate(
                        left=LabelRef(alias=alias, labels=labels, span=span),
                        op="has_labels",
                        right=None,
                        span=span,
                    )
                    for alias, labels in lifted
                )
                return WhereClause(predicates=predicates, span=span)
            # Pattern-leaf lift (slice 1 of #1031).  Pattern leaves
            # produced by the new ``pattern_atom`` grammar rule sit at
            # top-level AND positions; extract them as
            # ``WherePatternPredicate`` entries so lowering can evaluate them
            # as existence checks. Top-level ``NOT (pattern)`` leaves are also
            # lifted, marked ``negated=True`` for anti-semi-join lowering.
            # Patterns nested deeper (under OR/XOR or double-NOT) trip the
            # legacy E108 reject.
            pos_leaves, neg_leaves, other_conjuncts, nested_pattern = _split_top_level_and_pattern_leaves(expr_tree)
            if pos_leaves or neg_leaves or nested_pattern:
                return _build_where_with_pattern_lift(
                    pattern_leaves=pos_leaves,
                    negated_pattern_leaves=neg_leaves,
                    other_conjuncts=other_conjuncts,
                    nested_pattern=nested_pattern,
                    expr_text=expr_text,
                    span=span,
                )
            # Lift a flat AND chain of bare predicates to structured ``filter_dict``
            # predicates; parens / OR / XOR / NOT keep the clause on ``expr_tree`` ->
            # ``where_rows`` (see _lift_and_spine_predicates for why parens matter).
            lifted_preds = _lift_and_spine_predicates(expr_text, body_span)
            if lifted_preds is not None:
                return WhereClause(predicates=tuple(lifted_preds), span=span)
            return WhereClause(
                predicates=(),
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

        call_arg = _expression_slice

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

        def _page_clause(self, meta: Any, items: Sequence[Any], *, keyword: str, clause_type: Type[Any]) -> Any:
            if len(items) != 1:
                raise _to_syntax_error(f"Invalid {keyword} clause", line=meta.line, column=meta.column)
            span = _span_from_meta(meta)
            return clause_type(value=ExpressionText(text=self._slice(span)[len(keyword):].strip(), span=span), span=span)

        def skip_clause(self, meta: Any, items: Sequence[Any]) -> SkipClause:
            return cast(SkipClause, self._page_clause(meta, items, keyword="SKIP", clause_type=SkipClause))

        def limit_clause(self, meta: Any, items: Sequence[Any]) -> LimitClause:
            return cast(LimitClause, self._page_clause(meta, items, keyword="LIMIT", clause_type=LimitClause))

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
            # The grammar bundles WHERE into its MATCH clause. Re-emit it as a
            # follow-on item so the clause-sequence state machine below (which
            # predates the bundling and encodes all ordering/support rules)
            # processes the exact sequence the source text spells.
            flat: List[Any] = []
            for item in items:
                if isinstance(item, MatchClause) and item.where is not None:
                    flat.append(replace(item, where=None))
                    flat.append(item.where)
                else:
                    flat.append(item)
            items = flat
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
                too_many_suffix_withs = (
                    len(reentry_match_clauses) > 1
                    and len(with_stages) > len(reentry_match_clauses) + 1
                )
                if (
                    stages[0].clause.kind != "with"
                    or stages[-1].clause.kind != "return"
                    or any(stage.clause.kind != "with" for stage in stages[:-1])
                    or len(with_stages) < len(reentry_match_clauses)
                    or too_many_suffix_withs
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

        def union_all(self, _meta: Any, _items: Sequence[Any]) -> str:
            return "all"

        def union_distinct(self, _meta: Any, _items: Sequence[Any]) -> str:
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
                    return replace(branch, trailing_semicolon=True)
                return branch
            if len(union_kinds) != len(branches) - 1:
                raise _to_syntax_error("Invalid UNION query", line=meta.line, column=meta.column)
            if len(set(union_kinds)) != 1:
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
            # Split grammar-bundled MATCH..WHERE back into the item sequence the
            # constructor-body rules below were written against (see query_body).
            flat: List[Any] = []
            for item in body_items:
                if isinstance(item, MatchClause) and item.where is not None:
                    flat.append(replace(item, where=None))
                    flat.append(item.where)
                else:
                    flat.append(item)
            body_items = flat
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
            if not bindings:
                return query
            if isinstance(query, CypherUnionQuery):
                raise _to_unsupported(
                    "GRAPH bindings with UNION queries are not yet supported",
                    line=meta.line, column=meta.column,
                )
            return replace(query, graph_bindings=tuple(bindings))

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
    not\s*\(\s*\(\s*[^)\n]*\)\s*          # not((<node>)
    (?:<--|-->|--|<-\[[^\]\n]*\]-|-\[[^\]\n]*\]->|-\[[^\]\n]*\]-)
    |
    not\s+exists\s*\{                        # not exists {
    |
    exists\s*\{                              # exists {
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _mask_quoted_backticked_and_commented_for_scan(expr: str) -> str:
    """Replace quoted/backticked/commented segments with spaces before regex scans."""
    chars = list(expr)
    quote: str | None = None
    source = expr
    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]
        if quote is None:
            if i + 1 < n and source[i] == "/" and source[i + 1] == "/":
                chars[i] = " "
                chars[i + 1] = " "
                i += 2
                while i < n and source[i] != "\n":
                    chars[i] = " "
                    i += 1
                continue
            if i + 1 < n and source[i] == "/" and source[i + 1] == "*":
                chars[i] = " "
                chars[i + 1] = " "
                i += 2
                while i < n:
                    if i + 1 < n and source[i] == "*" and source[i + 1] == "/":
                        chars[i] = " "
                        chars[i + 1] = " "
                        i += 2
                        break
                    chars[i] = " "
                    i += 1
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                chars[i] = " "
            i += 1
            continue

        chars[i] = " "
        if ch == "\\" and quote in {"'", '"'}:
            if i + 1 < n:
                chars[i + 1] = " "
                i += 2
                continue
        if ch == quote:
            quote = None
        i += 1
    return "".join(chars)


def _check_unsupported_syntax_patterns(query: str) -> None:
    """Detect known-but-unsupported Cypher syntax and raise a clear error."""
    if _PATTERN_EXISTENCE_RE.search(_mask_quoted_backticked_and_commented_for_scan(query)):
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
    return _parse_cypher_cached(query)


@lru_cache(maxsize=512)
def _parse_cypher_cached(query: str) -> Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]:
    """Cached parse body: pure function of the query text -> immutable frozen AST.

    Memoizing skips the ~15ms lark parse+transform on repeated identical queries
    (the dominant per-call cost of small cypher queries). Safe to share the cached
    result: every cypher AST node is ``@dataclass(frozen=True)`` and the downstream
    ``compile_cypher_query`` does not mutate the parsed tree (verified). Validation
    errors raise (and are not cached by ``lru_cache``), preserving error behavior.
    """
    # Pre-parse detection of known-but-unsupported Cypher forms
    _check_unsupported_syntax_patterns(query)

    transformer = _build_transformer(query)
    try:
        tree = _parser_lalr().parse(query)
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

    if not isinstance(node, (CypherQuery, CypherUnionQuery, CypherGraphQuery)):
        raise _to_syntax_error("Cypher parser did not produce a query")
    _validate_reserved_identifiers(node)
    if isinstance(node, (CypherQuery, CypherGraphQuery)):
        _validate_graph_bindings(node)
    return node


def _validate_graph_bindings(node: Union[CypherQuery, CypherGraphQuery]) -> None:
    """Validate graph binding names: no duplicates, no forward/circular refs."""
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
