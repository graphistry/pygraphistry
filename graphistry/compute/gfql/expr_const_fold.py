"""Plan-time constant folding for GFQL row expressions.

Collapses the literal-only part of a predicate so downstream matchers see ONE
canonical shape: ``toLower(i.x) = toLower('Fine Dining')`` becomes
``toLower(i.x) = 'fine dining'``.

Which calls fold, and every decline, is specified by ``test_expr_const_fold.py``
(49 tests, grouped by criterion: TestAsciiGate, TestArgumentClosure,
TestNullAndBooleanPolicy, TestTotality).

The one thing those tests cannot tell you is WHY the ASCII gate exists, and it is
not a formality (#1802): pandas>=3 defaults to an Arrow-backed ``str`` whose
``utf8_lower``/``utf8_upper`` are SIMPLE per-codepoint case mappings, while polars'
and Python's are FULL mappings. Folding a non-ASCII literal in Python would bake
in the FULL mapping and silently disagree with the engine that evaluates the
column side. Hence: fold only where every engine provably agrees. Widening the
gate requires proving that agreement, not sampling it.
"""
from __future__ import annotations

from typing import Callable, FrozenSet, List, Mapping, Optional, Tuple, Union

from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    FunctionCall,
    Literal,
    _rebuild_expr_node,
)

__all__ = [
    "LiteralArgs",
    "FOLDABLE_FUNCTIONS",
    "DENIED_AGGREGATE",
    "DENIED_NOT_ARGUMENT_CLOSED",
    "DENIED_RESULT_TYPE",
    "DENIED_BY_POLICY",
    "DENIED_UNVERIFIED",
    "DECLINED_FUNCTIONS",
    "FoldedValue",
    "ConstantFolder",
    "fold_constants",
]


#: The only value types this pass will substitute for a call.  Deliberately excludes
#: ``None`` (see NULL POLICY), ``float`` (engine rounding/formatting kernels are
#: pinned per engine, see :data:`DENIED_RESULT_TYPE`) and ``bool``.
FoldedValue = Union[str, int]

#: Literal argument values, in call order.  ``object`` rather than a union so each
#: folder must narrow explicitly before using a value.
LiteralArgs = Tuple[object, ...]

#: A folder returns the folded value, or ``None`` to decline.  It may also raise;
#: the driver treats a raise exactly like a decline.
ConstantFolder = Callable[[LiteralArgs], Optional[FoldedValue]]


def _plain_int(value: object) -> Optional[int]:
    """Narrow to ``int`` but not ``bool`` (``isinstance(True, int)`` is ``True``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ascii_str(value: object) -> Optional[str]:
    """Narrow to a ``str`` inside the provably engine-invariant ASCII region, else ``None``.

    Criterion (E).  ``str.isascii`` is exact — no sampling, no table lookup.
    """
    if not isinstance(value, str):
        return None
    if not value.isascii():
        return None
    return value


def _fold_to_lower(args: LiteralArgs) -> Optional[FoldedValue]:
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    return None if text is None else text.lower()


def _fold_to_upper(args: LiteralArgs) -> Optional[FoldedValue]:
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    return None if text is None else text.upper()


def _fold_size(args: LiteralArgs) -> Optional[FoldedValue]:
    """``size(<string>)`` = character count.

    Gated on ASCII like the case folds: for ASCII, ``len`` (Python), ``str.len_chars``
    (polars) and the pandas evaluator's ``len`` are the same count under every
    encoding, so no claim is made about astral planes or combining sequences that
    this pass does not need.
    """
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    return None if text is None else len(text)


def _fold_substring(args: LiteralArgs) -> Optional[FoldedValue]:
    """``substring(<string>, start[, length])``, 0-based, non-negative, IN RANGE only.

    ASCII (criterion E) makes character-indexed and byte-indexed slicing identical,
    which is what removes the polars ``str.slice`` offset-unit question.  Out-of-range
    slices DECLINE (criterion T): Python yields ``''``, Neo4j raises, polars clamps —
    there is no engine-invariant answer to fold to.
    """
    if len(args) not in (2, 3):
        return None
    text = _ascii_str(args[0])
    if text is None:
        return None
    start_idx = _plain_int(args[1])
    if start_idx is None or start_idx < 0:
        return None
    length: Optional[int] = None
    if len(args) == 3:
        length = _plain_int(args[2])
        if length is None or length < 0:
            return None
    stop_idx = None if length is None else start_idx + length
    if start_idx > len(text):
        return None
    if stop_idx is not None and stop_idx > len(text):
        return None
    return text[start_idx:stop_idx]


def _fold_head(args: LiteralArgs) -> Optional[FoldedValue]:
    """``head(<string>)`` = first character.  STRING OVERLOAD ONLY.

    On GFQL's surface ``head`` is a string op: ``eval_sequence_fn_scalar`` answers
    ``value[0] if len(value) > 0 else None`` and the Series op is ``.str.get(0)``
    (``row/dispatch.py``).  ASCII (criterion E) makes "first character" the same for a
    codepoint-indexed and a byte-indexed implementation.  ``head('')`` is ``null`` at
    runtime, and NULL POLICY forbids synthesizing one, so the empty string DECLINES.
    A LIST argument never arrives: it parses to ``ListLiteral``, which the driver's
    per-argument guard rejects before any folder runs.
    """
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    if text is None or text == "":
        return None
    return text[0]


def _fold_tail(args: LiteralArgs) -> Optional[FoldedValue]:
    """``tail(<string>)`` = everything after the first character.  STRING OVERLOAD ONLY.

    ``eval_sequence_fn_scalar`` answers ``value[1:]``; the Series op is
    ``.str.slice(start=1)``.  Under ASCII those agree with Python's slice for every
    input including ``''`` (which answers ``''``, not null), so there is no empty-string
    carve-out here.
    """
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    return None if text is None else text[1:]


def _fold_reverse(args: LiteralArgs) -> Optional[FoldedValue]:
    """``reverse(<string>)``.  STRING OVERLOAD ONLY.

    ``eval_sequence_fn_scalar`` checks ``isinstance(value, str)`` FIRST and answers
    ``value[::-1]``; the Series op is ``.str[::-1]``.  Both reverse CODEPOINTS, which is
    only the same as reversing user-perceived characters when there are no combining
    sequences or surrogate pairs — hence the ASCII gate, exactly as for the case folds.
    ``reverse([1, 2, 3])`` has a ``ListLiteral`` argument and never reaches here.
    """
    if len(args) != 1:
        return None
    text = _ascii_str(args[0])
    return None if text is None else text[::-1]


#: FOLDABLE.  Keys are lowercase function names as the parser normalizes them.
FOLDABLE_FUNCTIONS: Mapping[str, ConstantFolder] = {
    "tolower": _fold_to_lower,
    "lower": _fold_to_lower,
    "toupper": _fold_to_upper,
    "upper": _fold_to_upper,
    "size": _fold_size,
    "substring": _fold_substring,
    "head": _fold_head,
    "tail": _fold_tail,
    "reverse": _fold_reverse,
}


# ================================================================================
# THE DECLINE TAXONOMY.  Every name below is filed under the MECHANISM that stops it,
# and every mechanism carries an executable witness.  Together with
# ``FOLDABLE_FUNCTIONS`` these must partition GFQL's entire Cypher function surface
# (``GFQL_ALLOWED_FUNCTIONS | GFQL_AGGREGATION_FUNCTIONS``); a test enforces that, so a
# newly added function cannot silently default to any side.
# ================================================================================

#: (P) THE ONE LOAD-BEARING DENY-SET.  An aggregate's value is a function of the ROW
#: SET, not of its arguments, so ``count(1)`` — argument-closed, ``int``-valued — passes
#: every structural guard the driver has and would fold to a wrong literal.  Nothing but
#: this list's absence from ``FOLDABLE_FUNCTIONS`` stops it.
#:
#: WITNESS (executed in ``test_const_fold_engine_parity.py``): the same
#: ``RETURN count(1)`` answers ``1`` over a one-row match and ``5`` over a five-row one.
DENIED_AGGREGATE: FrozenSet[str] = frozenset({
    "count",
    "count_distinct",
    "sum",
    "min",
    "max",
    "avg",
    "mean",
    "collect",
    "collect_distinct",
})


#: (A) NOT ARGUMENT-CLOSED IN THE SHAPE THE SYSTEM EMITS.  Maps each name to that
#: shape, which is the witness: the parsed node is either not a ``FunctionCall`` at all
#: (the quantifiers parse to ``QuantifierExpr``, so they can never reach the name
#: lookup) or carries a non-``Literal`` argument, which the driver's per-argument guard
#: declines.
#:
#: SCOPE OF THE CLAIM, stated because the parser is looser than the language: the
#: GRAMMAR will also accept ``keys('x')``, which IS argument-closed.  That spelling is
#: not something the lowering produces, and the runtime rejects it for
#: ``keys``/``labels``/``type``/``properties`` — but what stops it HERE is the name
#: lookup, not criterion (A).  A test pins both halves of that so the distinction cannot
#: quietly rot back into an unqualified claim.
DENIED_NOT_ARGUMENT_CLOSED: Mapping[str, str] = {
    "keys": "keys(n)",
    "labels": "labels(n)",
    "type": "type(e)",
    "properties": "properties(n)",
    "nodes": "nodes(p)",
    "relationships": "relationships(p)",
    # internal lowering markers over a match alias (lowering.py emits `__node_entity__(a)`)
    "__node_keys__": "__node_keys__(n)",
    "__edge_keys__": "__edge_keys__(e)",
    "__node_entity__": "__node_entity__(n)",
    "__edge_entity__": "__edge_entity__(e)",
    # quantifiers bind a variable over a source; they are not FunctionCall nodes
    "any": "any(x IN n.c WHERE x > 1)",
    "all": "all(x IN n.c WHERE x > 1)",
    "none": "none(x IN n.c WHERE x > 1)",
    "single": "single(x IN n.c WHERE x > 1)",
}


#: RESULT TYPE THE DRIVER'S CONTRACT GUARD REJECTS.  Maps each name to a LITERAL-ONLY
#: call — argument-closed, so the argument guard does NOT stop it — whose value the
#: engine computes as a ``float``, ``bool`` or ``list``.  ``FoldedValue`` is
#: ``str | int``, so even a perfect folder could not fold these: the witness is the
#: engine's own answer, asserted in ``test_const_fold_engine_parity.py``.
#:
#: This is where the numeric kernels land, and it is worth being blunt about what that
#: means: the elaborate neo4j-tie / JDK-6430675 reasoning behind ``round`` is real and
#: necessary, but it does no work HERE — ``round(1.5)`` is ``2.0``, a ``float``, and the
#: guard rejects it before any tie rule matters.  That reasoning lives in the row
#: KERNEL's docstring (``row/pipeline.py``), where it is load-bearing.
DENIED_RESULT_TYPE: Mapping[str, str] = {
    "sqrt": "sqrt(4)",
    "floor": "floor(1.5)",
    "ceil": "ceil(1.5)",
    "ceiling": "ceiling(1.5)",
    "round": "round(1.5)",
    "tofloat": "tofloat(1)",
    "toboolean": "toboolean('true')",
    "range": "range(1, 5)",
    "__cypher_case_eq__": "__cypher_case_eq__(1, 1)",
}


#: POLICY, NOT CORRECTNESS.  NO WITNESS EXISTS for these: the literal-only call is
#: argument-closed, the engine's answer is a guard-passing ``str``/``int``, and it is
#: identical to a plain Python fold — so a fold here could not change an answer.  Only
#: the name lookup declines them.  Moving one into ``FOLDABLE_FUNCTIONS`` would be a
#: PERF decision, and the perf case is nil: nothing downstream matches on arithmetic
#: text, and this pass exists to canonicalize predicate spellings, not to evaluate
#: constants for their own sake.  A test pins the absence of a witness, so if an engine
#: ever stops agreeing with Python here the claim fails loudly instead of aging.
DENIED_BY_POLICY: Mapping[str, str] = {
    "abs": "abs(-3)",
    "sign": "sign(-3)",
    # coalesce(1, 2) folds correctly to 1; the null case is already covered by the
    # driver's `result is None` decline and by NULL POLICY, so nothing here is unsafe.
    "coalesce": "coalesce(1, 2)",
}


#: UNVERIFIED, NOT ASSERTED.  A cross-engine divergence is CLAIMED for these — cuDF and
#: pandas are said to format ``float``->``string`` differently, which would make
#: ``toString`` engine-dependent, and ``toInteger``'s truncation/NaN masking is said to
#: be engine-specific.  Both return guard-passing values, so they are the only declines
#: where such a witness COULD exist.  It has not been exhibited: every literal-only
#: spelling reachable from CI (pandas and polars) agrees with the plain Python answer,
#: and the claim is specifically about a GPU engine no CI lane here runs.  They stay
#: declined and the reason stays labelled UNVERIFIED rather than promoted to a fact.
#: ``test_const_fold_engine_parity.py`` asserts agreement across every engine it can
#: reach, so the day a GPU lane covers it, a real divergence surfaces as a failure —
#: which is the witness — and its absence exposes these as ``DENIED_BY_POLICY``.
DENIED_UNVERIFIED: Mapping[str, str] = {
    "tostring": "tostring(1.5)",
    "tointeger": "tointeger(1.9)",
}


#: Every declined name, for the partition test.
DECLINED_FUNCTIONS: FrozenSet[str] = frozenset(
    DENIED_AGGREGATE
    | set(DENIED_NOT_ARGUMENT_CLOSED)
    | set(DENIED_RESULT_TYPE)
    | set(DENIED_BY_POLICY)
    | set(DENIED_UNVERIFIED)
)


def fold_constants(
    node: ExprNode,
    *,
    registry: Optional[Mapping[str, ConstantFolder]] = None,
) -> ExprNode:
    """Fold literal-only calls to literals, bottom-up.  Never raises on a fold.

    ``registry`` defaults to :data:`FOLDABLE_FUNCTIONS`; it is a parameter so tests
    can prove the driver's decline paths (a folder that raises, a folder that returns
    an out-of-contract value) without monkeypatching module state.
    """
    table: Mapping[str, ConstantFolder] = (
        FOLDABLE_FUNCTIONS if registry is None else registry
    )
    folded = _rebuild_expr_node(
        node,
        rewrite=lambda child: fold_constants(child, registry=table),
        error_context="constant folding",
    )
    if not isinstance(folded, FunctionCall) or folded.distinct:
        return folded
    folder = table.get(folded.name.lower())
    if folder is None:
        return folded
    values: List[object] = []
    for arg in folded.args:
        if not isinstance(arg, Literal):
            return folded  # (A) not argument-closed
        values.append(arg.value)
    try:
        result = folder(tuple(values))
    except Exception:
        # (T) a folder that raises is a decline, never a plan-time crash: the node is
        # returned untouched and the runtime produces whatever it produced before.
        return folded
    if result is None:
        return folded
    if not isinstance(result, (str, int)) or isinstance(result, bool):
        # Contract guard: a folder that returns something outside FoldedValue (a float,
        # a null, a bool) is a bug in the folder, not a licence to rewrite the plan.
        return folded
    return Literal(result)
