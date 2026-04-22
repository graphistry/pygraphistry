"""Predicate pushdown logical pass.

Rewrites ``Filter(input=PatternMatch(...))`` by moving safe predicate
conjuncts into ``PatternMatch.predicates`` while preserving optional-arm
null-extension semantics.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, FrozenSet, List, Tuple, cast

from graphistry.compute.gfql.ir.logical_plan import Filter, LogicalPlan, PatternMatch
from graphistry.compute.gfql.ir.pushdown_safety import is_null_rejecting
from graphistry.compute.gfql.ir.types import BoundPredicate
from graphistry.compute.gfql.passes.manager import LogicalPass, PassResult


class PredicatePushdownPass(LogicalPass):
    """Push safe filter predicates into PatternMatch operators."""

    name = "predicate_pushdown"

    def run(self, plan: LogicalPlan, ctx) -> PassResult:  # noqa: ANN001
        _ = ctx
        rewritten, pushed, residual = _rewrite_tree(plan)
        return PassResult(
            plan=rewritten,
            metadata={
                "pushed_predicates": pushed,
                "residual_predicates": residual,
            },
        )


def _rewrite_tree(plan: LogicalPlan) -> Tuple[LogicalPlan, int, int]:
    pushed = 0
    residual = 0
    children_updates = {}
    for slot in ("input", "left", "right", "subquery"):
        child = getattr(plan, slot, None)
        if isinstance(child, LogicalPlan):
            rewritten_child, child_pushed, child_residual = _rewrite_tree(child)
            pushed += child_pushed
            residual += child_residual
            if rewritten_child is not child:
                children_updates[slot] = rewritten_child

    current = (
        cast(LogicalPlan, replace(cast(Any, plan), **children_updates))
        if children_updates
        else plan
    )
    if isinstance(current, Filter) and isinstance(current.input, PatternMatch):
        rewritten_filter, local_pushed, local_residual = _push_filter_into_pattern(current)
        return rewritten_filter, pushed + local_pushed, residual + local_residual

    return current, pushed, residual


def _push_filter_into_pattern(filter_op: Filter) -> Tuple[LogicalPlan, int, int]:
    assert isinstance(filter_op.input, PatternMatch)
    pattern = cast(PatternMatch, filter_op.input)
    conjuncts = _split_conjuncts(filter_op.predicate)
    if not conjuncts:
        return filter_op, 0, 1

    null_extended_aliases = _optional_arm_aliases(pattern)
    pushable: List[BoundPredicate] = []
    kept: List[BoundPredicate] = []

    for conjunct in conjuncts:
        analysis_predicate = BoundPredicate(
            expression=conjunct.expression,
            references=_predicate_refs_for_analysis(conjunct),
        )
        if pattern.optional and is_null_rejecting(analysis_predicate, null_extended_aliases):
            kept.append(conjunct)
            continue
        pushable.append(conjunct)

    if not pushable:
        return filter_op, 0, len(kept)

    pushed_pattern = cast(
        PatternMatch,
        replace(pattern, predicates=[*pattern.predicates, *pushable]),
    )
    if not kept:
        return pushed_pattern, len(pushable), 0

    residual_predicate = _combine_conjuncts(kept)
    return replace(filter_op, input=pushed_pattern, predicate=residual_predicate), len(pushable), len(kept)


def _optional_arm_aliases(pattern: PatternMatch) -> FrozenSet[str]:
    """Approximate aliases introduced by this optional arm."""
    pattern_aliases = frozenset(pattern.output_schema.columns.keys())
    input_aliases = (
        frozenset(pattern.input.output_schema.columns.keys())
        if pattern.input is not None
        else frozenset()
    )
    return pattern_aliases - input_aliases


def _split_conjuncts(predicate: BoundPredicate) -> List[BoundPredicate]:
    """Split ``A AND B`` into top-level conjunct predicates."""
    expression = predicate.expression.strip()
    if not expression:
        return []
    parts = _split_top_level_and(expression)
    if len(parts) <= 1:
        return [predicate]
    return [
        BoundPredicate(expression=part, references=_refs_for_segment(part, predicate.references))
        for part in parts
    ]


def _combine_conjuncts(predicates: List[BoundPredicate]) -> BoundPredicate:
    expression = " AND ".join(pred.expression.strip() for pred in predicates if pred.expression.strip())
    refs = frozenset().union(*(pred.references for pred in predicates))
    return BoundPredicate(expression=expression, references=refs)


def _split_top_level_and(expression: str) -> List[str]:
    pieces: List[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    i = 0

    while i < len(expression):
        ch = expression[i]
        if quote is not None:
            if ch == quote and (i == 0 or expression[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")" and depth > 0:
            depth -= 1
            i += 1
            continue

        if depth == 0 and expression[i:i + 3].lower() == "and":
            prev_ch = expression[i - 1] if i > 0 else " "
            next_ch = expression[i + 3] if i + 3 < len(expression) else " "
            if prev_ch.isspace() and next_ch.isspace():
                part = expression[start:i].strip()
                if part:
                    pieces.append(part)
                start = i + 3
                i += 3
                continue
        i += 1

    tail = expression[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _refs_for_segment(segment: str, original_refs: FrozenSet[str]) -> FrozenSet[str]:
    detected = {
        alias
        for alias in original_refs
        if re.search(rf"\\b{re.escape(alias)}\\b", segment)
    }
    if detected:
        return frozenset(detected)
    # Conservative fallback for uncommon expressions where alias extraction fails.
    return original_refs


def _predicate_refs_for_analysis(predicate: BoundPredicate) -> FrozenSet[str]:
    return predicate.references if predicate.references else _infer_refs(predicate.expression)


def _infer_refs(expression: str) -> FrozenSet[str]:
    dotted_refs = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", expression))
    ast_alias_refs = set(re.findall(r"alias='([A-Za-z_][A-Za-z0-9_]*)'", expression))
    return frozenset(dotted_refs | ast_alias_refs)
