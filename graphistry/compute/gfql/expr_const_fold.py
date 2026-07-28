"""Plan-time constant folding for GFQL row expressions.

WHAT THIS IS FOR
================
The Cypher lowering serializes every row predicate it cannot push into
``filter_dict`` back to canonical predicate *text* (``_row_expr_arg``), and both the
row evaluators and the connected-join fast-path residual translator consume that
text.  Without folding, ``toLower(i.interest) = toLower('Fine Dining')`` and
``toLower(i.interest) = 'fine dining'`` are two different spellings of the same
predicate, and every consumer has to learn both.  Folding the literal-only
sub-expression at plan time collapses the first into the second, so downstream
matchers need ONE canonical shape and every other foldable function is handled
without its own case.

THE FOLDABILITY CRITERION
=========================
A ``FunctionCall`` node folds to a ``Literal`` if and only if all four hold:

(P) PURE AND DETERMINISTIC.  Its value is a total function of its arguments: no
    clock, no RNG, no locale, no timezone, no environment, no filesystem or
    network, no graph context, no dependence on the row set.  ``rand()``,
    ``randomUUID()``, ``timestamp()``, ``now()``, ``date()`` and friends fail here
    permanently.  (None of those are on GFQL's Cypher surface today — see
    ``test_expr_const_fold.py::TestClassification`` — but the criterion is stated
    so that adding one does not accidentally make it foldable.)

(A) ARGUMENT-CLOSED.  Every argument is already a ``Literal`` node.  Folding runs
    bottom-up, so a nested call that folds first satisfies this for its parent:
    ``toLower(substring('ABCDEF', 0, 3))`` -> ``toLower('ABC')`` -> ``'abc'``.
    A ``ListLiteral`` / ``MapLiteral`` is deliberately NOT a ``Literal``: this pass
    synthesizes scalar literals only.

(E) ENGINE-INVARIANT ON THESE ARGUMENT VALUES.  The Python-computed result must be
    provably identical to what EVERY supported engine would compute for the same
    literal expression — provable from the argument values themselves, not from a
    spot check on a sample.

    THIS IS NOT A FORMALITY.  Issue #1802: pandas>=3 defaults to an Arrow-backed
    ``str`` dtype whose ``utf8_lower``/``utf8_upper`` are SIMPLE per-codepoint case
    mappings, while polars' (and Python's) are FULL mappings, so
    ``toUpper(n.name) = 'STRASSE'`` already answers ``[9]`` on pandas and ``[8, 9]``
    on polars against the same data.  Folding a case function with the wrong
    semantics silently changes answers on exactly the shapes this pass targets.
    The region where every implementation provably agrees — Python's ``str``, Rust's
    ``str`` (polars), Arrow's ``utf8_*`` kernels (pandas>=3), libcudf, and Java's
    ``String.toLowerCase`` (the Cypher reference) — is the ASCII block, where case
    mapping is a fixed 26-character bijection defined by the Unicode standard's
    invariant range and is independent of every implementation's Unicode table
    version.  So: **string folds require ASCII arguments, and decline otherwise.**
    Declining costs a slower query; folding outside the provable region costs a
    wrong answer.

(T) TOTAL ON THESE ARGUMENTS.  Evaluation must neither raise nor land in a region
    where implementations are known to disagree.  Out-of-range ``substring`` is the
    worked example: ``substring('abc', 99)`` is ``''`` in Python, an error in Neo4j,
    and clamped in polars, so it DECLINES rather than picking one.  Any exception
    raised by a folder is caught by the driver and turned into a decline, so a fold
    can never convert a runtime error into a plan-time crash — the node is left
    exactly as it was and the runtime produces exactly what it produced before.

Anything failing any of the four is left untouched.  A decline is always safe: the
expression text is unchanged, so every downstream consumer behaves as it did.

NULL POLICY
===========
This pass never synthesizes a ``null`` literal and never folds a call whose
arguments include ``null``.  Substituting ``null`` for a call changes which branch
of the evaluators' three-valued logic runs downstream, and null handling has been
this area's most frequent source of silent wrong answers.  ``toLower(null)`` is
therefore left for the runtime, which already answers it.

Booleans are excluded from the integer folds (``isinstance(True, int)`` is ``True``
in Python, and ``substring('abc', True)`` is not ``substring('abc', 1)`` in Cypher).
"""
from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Tuple, Union

from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    FunctionCall,
    Literal,
    _rebuild_expr_node,
)

__all__ = [
    "LiteralArgs",
    "FOLDABLE_FUNCTIONS",
    "NON_FOLDABLE_REASONS",
    "FoldedValue",
    "ConstantFolder",
    "fold_constants",
]


#: The only value types this pass will substitute for a call.  Deliberately excludes
#: ``None`` (see NULL POLICY), ``float`` (engine rounding/formatting kernels are
#: pinned per engine, see ``NON_FOLDABLE_REASONS``) and ``bool``.
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


#: FOLDABLE.  Keys are lowercase function names as the parser normalizes them.
FOLDABLE_FUNCTIONS: Mapping[str, ConstantFolder] = {
    "tolower": _fold_to_lower,
    "lower": _fold_to_lower,
    "toupper": _fold_to_upper,
    "upper": _fold_to_upper,
    "size": _fold_size,
    "substring": _fold_substring,
}


#: NOT FOLDABLE, with the criterion each one fails.  Together with
#: ``FOLDABLE_FUNCTIONS`` this must partition GFQL's entire Cypher function surface
#: (``GFQL_ALLOWED_FUNCTIONS | GFQL_AGGREGATION_FUNCTIONS``); a test enforces that,
#: so a newly added function cannot silently default to either side.
NON_FOLDABLE_REASONS: Mapping[str, str] = {
    # --- numeric kernels: (E).  Each engine implements a DELIBERATELY non-default
    # kernel to match neo4j (ties toward +inf at precision 0, HALF_UP above it,
    # -0.0 normalization, Float64/Int64 result casts, the p>308 identity window,
    # explicit NaN masking).  A plan-time fold would have to re-derive every one of
    # them exactly, and the argument values alone do not establish agreement.  The
    # perf value of folding literal-only arithmetic is nil.
    "abs": "(E) engine-pinned numeric kernel/result dtype; no gain over a literal",
    "sqrt": "(E) engine-pinned numeric kernel/result dtype; no gain over a literal",
    "sign": "(E) engine-pinned Int64 result cast; no gain over a literal",
    "floor": "(E) engine-pinned Float64 cast and neo4j tie rules",
    "ceil": "(E) engine-pinned Float64 cast and neo4j tie rules",
    "ceiling": "(E) engine-pinned Float64 cast and neo4j tie rules",
    "round": "(E) neo4j tie rules differ per precision; polars/pandas kernels are hand-written",
    "tointeger": "(E) NaN/null masking and truncation rules are engine-specific",
    "tofloat": "(E) NaN preservation differs from toInteger and is engine-specific",
    "toboolean": "(E) the evaluators DECLINE on unrecognized tokens; a fold cannot reproduce a decline",
    "tostring": "(E) float->string formatting diverges between cuDF and pandas (host round-trip workaround)",
    # --- null-valued results: (T) + NULL POLICY.
    "coalesce": "(T) can yield null; this pass never substitutes a null literal",
    # --- graph context / non-scalar results: (A).
    "keys": "(A) operates on an entity, never on a literal",
    "labels": "(A) operates on an entity, never on a literal",
    "type": "(A) operates on an edge entity, never on a literal",
    "properties": "(A) operates on an entity, never on a literal",
    "nodes": "(A) path-valued, graph context",
    "relationships": "(A) path-valued, graph context",
    "range": "(A) list-valued; this pass synthesizes scalar literals only",
    "head": "(A) sequence op; list literals parse to ListLiteral, not Literal",
    "tail": "(A) sequence op; list literals parse to ListLiteral, not Literal",
    "reverse": "(A) sequence op; list literals parse to ListLiteral, not Literal",
    # --- internal lowering markers: (A)/(P).
    "__node_keys__": "(A) internal marker over a match alias",
    "__edge_keys__": "(A) internal marker over a match alias",
    "__node_entity__": "(A) internal marker over a match alias",
    "__edge_entity__": "(A) internal marker over a match alias",
    "__cypher_case_eq__": "(P) internal simple-CASE marker with engine-branching null-matching semantics",
    # --- quantifiers: (A).  Parsed as QuantifierExpr, not FunctionCall; they bind a
    # variable over a source, so they are never argument-closed.
    "any": "(A) quantifier binds a variable over a source",
    "all": "(A) quantifier binds a variable over a source",
    "none": "(A) quantifier binds a variable over a source",
    "single": "(A) quantifier binds a variable over a source",
    # --- aggregates: (P).  Value depends on the ROW SET, not on the arguments.
    "count": "(P) aggregate over the row set, not a function of its arguments",
    "count_distinct": "(P) aggregate over the row set, not a function of its arguments",
    "sum": "(P) aggregate over the row set, not a function of its arguments",
    "min": "(P) aggregate over the row set, not a function of its arguments",
    "max": "(P) aggregate over the row set, not a function of its arguments",
    "avg": "(P) aggregate over the row set, not a function of its arguments",
    "mean": "(P) aggregate over the row set, not a function of its arguments",
    "collect": "(P) aggregate over the row set, not a function of its arguments",
    "collect_distinct": "(P) aggregate over the row set, not a function of its arguments",
}


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
