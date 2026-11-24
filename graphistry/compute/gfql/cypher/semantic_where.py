"""Lift ``WHERE ... AND sem_decide(question, {evidence}[, options])`` into ``CALL graphistry.sem_decide``.

``MATCH .. WHERE <structural> AND sem_decide(q, {k: expr}, opts) RETURN ..`` becomes

    MATCH .. WHERE <structural> WITH expr AS k, ..., <return exprs> AS __r0, ...
    CALL graphistry.sem_decide({question: q, evidence: ['k', ..], ..opts}) YIELD verdict AS __sem_verdict
    WHERE __sem_verdict
    RETURN __r0 AS <return name>, ...

so the structural predicates prune rows before one batched model call, and the model call sees only the survivors.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, List, NoReturn, Optional, Sequence, Tuple, Union

from graphistry.compute.gfql.cypher.ast import (
    BooleanExpr,
    CallClause,
    CypherYieldItem,
    ExpressionText,
    MatchClause,
    OrderByClause,
    OrderItem,
    ProjectionStage,
    ReturnClause,
    ReturnItem,
    SourceSpan,
    UnwindClause,
    WhereClause,
    WherePatternPredicate,
    WherePredicate,
)

QueryItem = Union[MatchClause, UnwindClause, CallClause, ProjectionStage, str]
LiftPredicates = Callable[[str, SourceSpan], Optional[List[WherePredicate]]]
Fail = Callable[[str, SourceSpan], NoReturn]

# The lifted form asks one predicate question; these shape other kinds of answer or the output names.
_NOT_IN_LIFTED_OPTIONS = frozenset({
    "question", "questions", "evidence", "template", "kind", "options", "levels", "out_col", "score_col",
    "status_col",
})
_CALL_NAME = re.compile(r"\bsem_decide\s*\(", re.I)
_WHOLE_CALL = re.compile(r"^sem_decide\s*\((.*)\)$", re.S | re.I)
_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_VERDICT = "__sem_verdict"
_STATUS = "__sem_status"
_FORM = "Use CALL graphistry.sem_decide({question: ..., evidence: [...]}) YIELD verdict, score, status for other shapes."


def _names_sem_decide(text: str) -> bool:
    """Whether ``text`` calls sem_decide(...); a string literal that spells it does not count."""
    return _CALL_NAME.search(_QUOTED.sub("''", text)) is not None


def _split_top_level(text: str, sep: str) -> List[str]:
    """Split ``text`` at ``sep`` outside quotes and brackets."""
    parts: List[str] = []
    depth, start, quote, escaped = 0, 0, "", False
    for pos, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "'\"`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:pos].strip())
            start = pos + 1
    parts.append(text[start:].strip())
    return parts


def _map_items(text: str) -> Optional[List[Tuple[str, str]]]:
    body = text.strip()
    if not (body.startswith("{") and body.endswith("}")):
        return None
    items: List[Tuple[str, str]] = []
    for part in _split_top_level(body[1:-1], ","):
        if not part:
            continue
        pieces = _split_top_level(part, ":")
        if len(pieces) < 2:
            return None
        items.append((pieces[0].strip("`'\" "), ":".join(pieces[1:]).strip()))
    return items


def _spine(tree: BooleanExpr, out: List[BooleanExpr]) -> None:
    """Collect the terms of the top-level AND spine; an OR, XOR or NOT subtree is one term."""
    if tree.op == "and" and tree.left is not None and tree.right is not None:
        _spine(tree.left, out)
        _spine(tree.right, out)
    else:
        out.append(tree)


def _mentions(tree: BooleanExpr) -> bool:
    if tree.atom_text is not None and _names_sem_decide(tree.atom_text):
        return True
    return any(_mentions(child) for child in (tree.left, tree.right) if child is not None)


def _stage_mentions(item: QueryItem) -> bool:
    if isinstance(item, ProjectionStage):
        texts = [ret.expression.text for ret in item.clause.items]
        if item.where is not None:
            texts.append(item.where.text)
        return any(_names_sem_decide(text) for text in texts)
    return False


def _rebuild_where(
    atoms: Sequence[BooleanExpr], original: WhereClause, lift: LiftPredicates,
) -> Optional[WhereClause]:
    if not atoms:
        return None
    text = " AND ".join(atom.atom_text or "" for atom in atoms)
    if all(atom.op == "atom" for atom in atoms):
        lifted = lift(text, original.span)
        if lifted is not None:
            return WhereClause(predicates=tuple(lifted), span=original.span)
    tree = atoms[0]
    for atom in atoms[1:]:
        tree = BooleanExpr(op="and", span=original.span, left=tree, right=atom)
    return WhereClause(predicates=(), expr_tree=tree, span=original.span)


def _return_name(item: ReturnItem) -> str:
    return item.alias or item.expression.text


def lift_semantic_where(
    items: Sequence[QueryItem], *, lift_predicates: LiftPredicates, fail: Fail,
) -> Optional[List[QueryItem]]:
    """Rewrite ``items`` (see module docstring), or return None when no WHERE mentions ``sem_decide``."""
    matches = [item for item in items if isinstance(item, MatchClause)]
    mentioning = [m for m in matches if m.where is not None and m.where.expr_tree is not None and _mentions(m.where.expr_tree)]
    if not mentioning:
        for item in items:
            if _stage_mentions(item) and isinstance(item, ProjectionStage):
                fail(f"sem_decide(...) is a term of the WHERE of a MATCH ... RETURN query only. {_FORM}", item.span)
        return None
    match = mentioning[0]
    assert match.where is not None and match.where.expr_tree is not None
    where, tree = match.where, match.where.expr_tree
    if any(isinstance(term, WherePatternPredicate) for term in where.predicates):
        fail(f"WHERE sem_decide(...) cannot combine with a pattern or EXISTS term in the same WHERE. {_FORM}", where.span)
    atoms: List[BooleanExpr] = []
    _spine(tree, atoms)
    if any(term.op != "atom" and _mentions(term) for term in atoms):
        fail(f"sem_decide(...) must be a top-level AND term of WHERE (not under OR, XOR or NOT). {_FORM}", where.span)
    semantic = [term for term in atoms if term.op == "atom" and _names_sem_decide(term.atom_text or "")]
    structural = [term for term in atoms if term not in semantic]
    if len(semantic) != 1:
        fail(f"WHERE accepts one sem_decide(...) term. {_FORM}", where.span)
    atom = semantic[0]
    atom_span = atom.atom_span or where.span
    whole = _WHOLE_CALL.match((atom.atom_text or "").strip())
    if whole is None:
        fail(f"sem_decide(...) must stand alone as a WHERE term, not inside a larger expression. {_FORM}", atom_span)
    args = _split_top_level(whole.group(1), ",")
    if len(args) not in (2, 3) or any(not arg for arg in args):
        fail("sem_decide(question, {evidence: expression, ...}[, {options}]) takes two or three arguments", atom_span)
    evidence = _map_items(args[1])
    if not evidence:
        fail("The second sem_decide argument must be a map such as {summary: a.summary}", atom_span)
    options = _map_items(args[2]) if len(args) == 3 else []
    if options is None:
        fail("The third sem_decide argument must be a map such as {max_usd: $max_usd}", atom_span)
    misplaced = sorted({key for key, _ in options} & _NOT_IN_LIFTED_OPTIONS)
    if misplaced:
        fail(f"sem_decide(...) sets option {misplaced[0]!r} through its own arguments. {_FORM}", atom_span)
    names = [key for key, _ in evidence]
    if len(set(names)) != len(names) or any(not name.isidentifier() for name in names):
        fail("Evidence keys in sem_decide(...) must be unique plain names", atom_span)

    others = [item for item in items if item is not match]
    stages = [item for item in others if isinstance(item, ProjectionStage)]
    trailing = [item for item in others if isinstance(item, str)]
    if len(stages) != 1 or stages[0].clause.kind != "return" or len(matches) != 1 or len(others) != len(stages) + len(trailing):
        fail(f"WHERE sem_decide(...) supports one MATCH ... WHERE ... RETURN query. {_FORM}", atom_span)
    final = stages[0]
    return_items = final.clause.items
    if any(item.expression.text.isidentifier() for item in return_items):
        fail(f"WHERE sem_decide(...) cannot return a whole node or relationship; return its properties. {_FORM}", final.span)
    if any(item.expression.text == "*" for item in return_items):
        fail(f"WHERE sem_decide(...) needs RETURN items that name their columns, not *. {_FORM}", final.span)
    if any(re.search(r"\b(count|sum|avg|min|max|collect|stdev|percentile\w*)\s*\(", item.expression.text, re.I)
           for item in return_items):
        fail(f"WHERE sem_decide(...) cannot feed an aggregating RETURN. {_FORM}", final.span)

    carried = [f"__r{idx}" for idx in range(len(return_items))]
    with_items = tuple(
        ReturnItem(expression=ExpressionText(text=expr, span=atom_span), alias=name, span=atom_span)
        for name, expr in evidence
    ) + tuple(
        ReturnItem(expression=item.expression, alias=column, span=item.span)
        for column, item in zip(carried, return_items)
    )
    with_stage = ProjectionStage(
        clause=ReturnClause(items=with_items, distinct=False, kind="with", span=atom_span),
        where=None, order_by=None, skip=None, limit=None, span=atom_span,
    )
    parts = [f"question: {args[0]}", "evidence: [" + ", ".join(f"'{name}'" for name in names) + "]"]
    parts += [f"{key}: {value}" for key, value in options]
    if "fail_on_incomplete" not in {key for key, _ in options}:
        parts.append("fail_on_incomplete: true")
    call = CallClause(
        procedure="graphistry.sem_decide",
        args=(ExpressionText(text="{" + ", ".join(parts) + "}", span=atom_span),),
        yield_items=(
            CypherYieldItem(name="verdict", alias=_VERDICT, span=atom_span),
            CypherYieldItem(name="status", alias=_STATUS, span=atom_span),
        ),
        span=atom_span,
        where=ExpressionText(text=f"{_VERDICT} AND {_STATUS} = 'ok'", span=atom_span),
    )

    out_names = [
        item.alias or item.expression.text for item in return_items
    ]
    by_text = {item.expression.text: name for name, item in zip(out_names, return_items)}
    by_name = {_return_name(item): name for name, item in zip(out_names, return_items)}
    order: Optional[OrderByClause] = None
    if final.order_by is not None:
        ordered: List[OrderItem] = []
        for entry in final.order_by.items:
            text = entry.expression.text
            if text in by_name:
                name = by_name[text]
            elif text in by_text:
                name = by_text[text]
            else:
                fail(f"ORDER BY after WHERE sem_decide(...) must name a RETURN column. {_FORM}", entry.span)
            if not name.isidentifier():
                fail(f"ORDER BY after WHERE sem_decide(...) needs RETURN columns named with AS. {_FORM}", entry.span)
            ordered.append(replace(entry, expression=ExpressionText(text=name, span=entry.expression.span)))
        order = replace(final.order_by, items=tuple(ordered))
    final_return = replace(
        final,
        clause=replace(
            final.clause,
            items=tuple(
                ReturnItem(expression=ExpressionText(text=column, span=item.expression.span), alias=name, span=item.span)
                for column, name, item in zip(carried, out_names, return_items)
            ),
        ),
        order_by=order,
    )
    lifted_match = replace(match, where=_rebuild_where(structural, where, lift_predicates))
    lifted: List[QueryItem] = [lifted_match, with_stage, call, final_return]
    lifted.extend(trailing)
    return lifted
