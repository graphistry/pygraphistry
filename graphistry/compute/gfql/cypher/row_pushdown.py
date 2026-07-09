"""L4 single-alias predicate pushdown for the Cypher row pipeline.

A disjunctive / ``searchAny`` WHERE collapses the whole MATCH onto the row
pipeline: ``rows(binding_ops=[a,e,b])`` materializes the full ``(a)-[e]->(b)``
binding table (O(E) joins) and only THEN filters it with ``where_rows`` /
``search_any``.  But a conjunct that references a single alias depends only on
that alias's columns, which the (inner) binding join copies verbatim into the
row table.  So we can peel such conjuncts off the post-join filter and hand them
to the ``rows`` op as per-alias ``alias_prefilters``, which the binding builder
applies to each alias frame BEFORE the join — shrinking every downstream hop.

Parity is exact (see ``_gfql_apply_alias_prefilter``): the SAME evaluator and the
SAME ``search_any`` kernel run, just earlier and on a subset.  The pass is
deliberately conservative — anything it does not recognize is left untouched, so
the rewrite can only ever move work earlier, never change results.
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

from graphistry.compute.ast import ASTCall, ASTObject
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.row.prefilter import (
    AliasPrefilterSpec, MutableAliasPrefilters, is_alias_prefilters,
)
from graphistry.utils.json import JSONVal
from typing_extensions import TypeGuard

# Ops that consume/close the post-``rows`` filter region.  Reaching any of these
# ends the contiguous filter block we are allowed to peel from.
_FILTER_REGION_OPS = frozenset({"where_rows", "search_any", "semi_apply_mark", "anti_semi_apply"})


def _is_binding_ops(value: object) -> TypeGuard[List[Dict[str, JSONVal]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _strip_redundant_parens(expr: str) -> str:
    """Drop one or more fully-enclosing ``( ... )`` wrappers.

    The Cypher lowering emits fully-parenthesized left-associated ANDs
    (``((((C1) AND (C2)) AND (C3)) AND (C4))``); the quote/bracket-aware AND
    splitter only finds top-level ANDs, so the outer wrapper must come off first.
    Only strips when the leading ``(`` matches the trailing ``)`` (i.e. the pair
    genuinely encloses the whole string).
    """
    expr = expr.strip()
    while len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
        depth = 0
        encloses = True
        quote: Optional[str] = None
        escaped = False
        for idx, ch in enumerate(expr):
            if quote is not None:
                # skip the char after a backslash (escaped quote) — mirror the
                # sibling split_top_level_and_conjuncts scanner exactly.
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote:
                    quote = None
                continue
            if ch in {"'", '"', "`"}:
                quote = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    encloses = False
                    break
        if not encloses or depth != 0:
            break
        expr = expr[1:-1].strip()
    return expr


def _flatten_and_conjuncts(expr: str) -> List[str]:
    """Recursively split *expr* into top-level AND conjuncts.

    Returns the fully-flattened conjunct list (each with redundant parens
    stripped).  A leaf that is not an AND at its level comes back as a single
    entry, so ``NOT (x AND y)`` and ``(p OR q)`` stay intact.
    """
    from graphistry.compute.gfql.passes.predicate_pushdown import split_top_level_and_conjuncts

    inner = _strip_redundant_parens(expr)
    parts = split_top_level_and_conjuncts(inner)
    if len(parts) <= 1:
        return [inner]
    out: List[str] = []
    for part in parts:
        out.extend(_flatten_and_conjuncts(part))
    return out


def _binding_alias_targets(
    binding_ops: Sequence[Dict[str, JSONVal]],
) -> Optional[Tuple[Dict[str, ASTObject], Set[str]]]:
    """Deserialize ``binding_ops`` into ``{alias: ASTObject}`` plus the set of
    aliases eligible for pushdown.

    Node aliases are always eligible (inner-joined in the connected/cartesian
    binding path).  Edge aliases are eligible only when the relationship is a
    plain single hop — a variable-length / fixed-point edge does not map onto a
    single pre-join edge-property filter.  Returns ``None`` if any op is
    un-parseable (bail out of the whole pushdown for safety).
    """
    from graphistry.compute.ast import from_json as ast_from_json, ASTEdge, ASTNode

    alias_targets: Dict[str, ASTObject] = {}
    eligible: Set[str] = set()
    for op_json in binding_ops:
        name = op_json.get("name") if isinstance(op_json, dict) else None
        # shortestPath scalar bindings route to a separate builder that does not
        # honor alias_prefilters — pushing there would silently drop the filter.
        # Bail out of the entire pushdown for any such pattern.
        lnh = op_json.get("label_node_hops") if isinstance(op_json, dict) else None
        if isinstance(lnh, str) and lnh.startswith("__cypher_shortest_path_hops__"):
            return None
        try:
            obj = ast_from_json(op_json, validate=False)
        except (AssertionError, KeyError, TypeError, ValueError, GFQLValidationError):
            return None
        if not isinstance(name, str):
            continue
        alias_targets[name] = obj
        if isinstance(obj, ASTNode):
            eligible.add(name)
        elif isinstance(obj, ASTEdge):
            hops = op_json.get("hops", 1)
            if (
                op_json.get("to_fixed_point") is not True
                and hops == 1
                and op_json.get("min_hops") in (None, 1)
                and op_json.get("max_hops") in (None, 1)
            ):
                eligible.add(name)
    return alias_targets, eligible


def _positive_search_any_markers(steps: Sequence[ASTObject], start_idx: int) -> Set[str]:
    """Return searchAny marker columns used as positive top-level AND conjuncts.

    ``search_any`` row ops are positive kernels. They are safe as alias prefilters
    only when the retained ``where_rows`` filter also requires that marker to be
    true. ``NOT marker`` and ``marker OR ...`` must stay post-join only.
    """
    markers: Set[str] = set()
    idx = start_idx
    while idx < len(steps):
        op = steps[idx]
        if not isinstance(op, ASTCall) or op.function not in _FILTER_REGION_OPS:
            break
        if op.function == "where_rows":
            expr = op.params.get("expr")
            if isinstance(expr, str):
                for conjunct in _flatten_and_conjuncts(expr):
                    leaf = _strip_redundant_parens(conjunct)
                    if leaf.startswith("__gfql_search_any_") and leaf.endswith("__"):
                        markers.add(leaf)
        idx += 1
    return markers


def _conjunct_single_alias(
    conjunct: str,
    alias_targets: Dict[str, ASTObject],
) -> Optional[str]:
    """Return the sole binding alias a conjunct references, else ``None``.

    ``None`` when the conjunct touches zero binding aliases (constants, internal
    marker columns), more than one alias, or any aggregate — none of which are
    safe to evaluate before the join.
    """
    from graphistry.compute.gfql.cypher.lowering import _expr_match_alias_usage

    # Reject any conjunct that names an INTERNAL machinery column. These reserved
    # ``__gfql_`` / ``__cypher_`` names are produced by downstream ops (EXISTS /
    # searchAny markers ``__gfql_where_pattern_*`` / ``__gfql_search_any_*``) or
    # carry a value from another alias (reentry ``__cypher_reentry_*``); the alias
    # walker hides their roots, so they falsely read as single-alias and are not
    # resolvable on the pre-join alias frame. User columns never use these prefixes.
    if "__gfql_" in conjunct or "__cypher_" in conjunct:
        return None

    try:
        non_aggregate, aggregate = _expr_match_alias_usage(
            conjunct,
            alias_targets=alias_targets,
            params=None,
            field="where",
            line=0,
            column=0,
        )
    except GFQLValidationError:
        return None
    if aggregate:
        return None
    if len(non_aggregate) != 1:
        return None
    return next(iter(non_aggregate))


def apply_row_prefilter_pushdown(chain: Chain) -> Chain:
    """Attach single-alias predicates to the ``rows(binding_ops=...)`` op as an
    advisory ``alias_prefilters`` hint — WITHOUT removing the post-join filters.

    Design note (redundant-filter safety): the peeled predicates are added as a
    pre-join HINT only; the original post-join ``where_rows`` / ``search_any`` ops
    are left in place untouched. Engines whose row builder honors the hint
    (pandas / cuDF via RowPipelineMixin) pre-filter each alias frame before the
    binding join and win big; engines that ignore it (e.g. polars' native
    ``rows_binding_ops_polars``, the shortestPath scalar builder) simply run the
    unchanged post-join filter and stay correct. Because a pushed conjunct is a
    literal sub-conjunct of the retained AND, the pre-filter can only remove rows
    the post-filter would also remove — never a result change, on any engine.

    Safe no-op for any chain without a ``rows(binding_ops)`` op, or whose filters
    are all multi-alias / marker-only.
    """
    steps: List[ASTObject] = list(chain.chain)
    if not steps:
        return chain

    rows_match: Optional[Tuple[int, ASTCall]] = None
    for idx, op in enumerate(steps):
        if isinstance(op, ASTCall) and op.function == "rows" and op.params.get("binding_ops"):
            rows_match = (idx, op)
            break
    if rows_match is None:
        return chain
    rows_idx, rows_op = rows_match
    raw_binding_ops = rows_op.params.get("binding_ops")
    if not _is_binding_ops(raw_binding_ops):
        return chain
    binding_ops = raw_binding_ops
    targets = _binding_alias_targets(binding_ops)
    if targets is None:
        return chain
    alias_targets, eligible = targets
    if not eligible:
        return chain

    prefilters: DefaultDict[str, List[AliasPrefilterSpec]] = defaultdict(list)
    positive_search_markers = _positive_search_any_markers(steps, rows_idx + 1)

    # Scan the contiguous filter region immediately following the rows op and
    # COLLECT (never remove) pushable single-alias predicates.
    idx = rows_idx + 1
    while idx < len(steps):
        op = steps[idx]
        if not isinstance(op, ASTCall) or op.function not in _FILTER_REGION_OPS:
            break
        if op.function == "search_any":
            alias = op.params.get("alias")
            out_col = op.params.get("out_col")
            term = op.params.get("term")
            if isinstance(alias, str) and isinstance(term, str) and out_col in positive_search_markers and alias in eligible:
                spec: AliasPrefilterSpec = {"kind": "search_any", "term": term}
                if op.params.get("case_sensitive"):
                    spec["case_sensitive"] = True
                if op.params.get("regex"):
                    spec["regex"] = True
                columns = op.params.get("columns")
                if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
                    spec["columns"] = columns
                prefilters[alias].append(spec)
        elif op.function == "where_rows":
            expr = op.params.get("expr")
            if isinstance(expr, str):
                for conjunct in _flatten_and_conjuncts(expr):
                    alias = _conjunct_single_alias(conjunct, alias_targets)
                    if alias is not None and alias in eligible:
                        prefilters[alias].append({"kind": "expr", "text": conjunct})
        idx += 1

    if not prefilters:
        return chain

    # Attach the hint to the rows op; every other op is left byte-for-byte intact.
    new_params = dict(rows_op.params)
    merged: MutableAliasPrefilters = {}
    existing_prefilters = new_params.get("alias_prefilters")
    if is_alias_prefilters(existing_prefilters):
        for alias, specs in existing_prefilters.items():
            merged[alias] = list(specs)
    for alias, specs in prefilters.items():
        merged.setdefault(alias, []).extend(specs)
    new_params["alias_prefilters"] = merged

    new_steps = list(steps)
    new_steps[rows_idx] = ASTCall("rows", new_params)
    return Chain(new_steps, where=chain.where)
