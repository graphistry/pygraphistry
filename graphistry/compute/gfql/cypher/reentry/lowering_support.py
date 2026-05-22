"""Reentry lowering support shared by compile-time rewrite stages."""
from __future__ import annotations

from dataclasses import fields as _dataclass_fields, replace
import re
from typing import AbstractSet, Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast

from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    ExpressionText,
    MatchClause,
    NodePattern,
    OrderByClause,
    OrderItem,
    PatternElement,
    ProjectionStage,
    PropertyRef,
    RelationshipPattern,
    ReturnItem,
    WhereClause,
)
from graphistry.compute.gfql.cypher.reentry.naming import _secondary_reentry_hidden_column_name
from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    GFQLExprParseError,
    Identifier,
    ListComprehension,
    PropertyAccessExpr,
    QuantifierExpr,
    collect_identifiers,
    parse_expr,
    walk_expr_nodes,
)


def _post_processing_with(
    *,
    result_projection: Optional[Any],
    empty_result_row: Optional[Dict[str, Any]],
    optional_null_fill: Optional[Any],
    optional_projection_row_guard: Optional[Any],
) -> Optional[Any]:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    return _lowering._normalize_post_processing(
        _lowering.CompiledCypherPostProcessing(
            result_projection=result_projection,
            empty_result_row=empty_result_row,
            optional_null_fill=optional_null_fill,
            optional_projection_row_guard=optional_projection_row_guard,
        )
    )


def _rewrite_order_by_expressions(
    order_by: Optional[OrderByClause],
    rewrite_expr: Callable[[ExpressionText, str], Optional[ExpressionText]],
) -> Optional[OrderByClause]:
    if order_by is None:
        return None
    rewritten_items: List[OrderItem] = []
    for item in order_by.items:
        rewritten_expr = rewrite_expr(item.expression, "order_by")
        if rewritten_expr is None:
            return None
        rewritten_items.append(replace(item, expression=rewritten_expr))
    return replace(order_by, items=tuple(rewritten_items))


def _drop_bare_alias_items_from_stage(
    stage: ProjectionStage,
    aliases: AbstractSet[str],
    *,
    identifier_re: "re.Pattern[str]",
) -> ProjectionStage:
    """Drop bare-identifier projection items whose name is in ``aliases``."""
    new_items = tuple(
        item
        for item in stage.clause.items
        if not (
            item.alias is None
            and identifier_re.fullmatch(item.expression.text.strip())
            and item.expression.text.strip() in aliases
        )
    )
    if len(new_items) == len(stage.clause.items):
        return stage
    return replace(stage, clause=replace(stage.clause, items=new_items))


def _iter_property_refs(node: object) -> Iterable[PropertyRef]:
    """Yield ``PropertyRef`` leaves reachable from a structured ``WhereClause`` predicate."""
    if isinstance(node, PropertyRef):
        yield node
        return
    dataclass_node = cast(Any, node)
    if hasattr(dataclass_node, "__dataclass_fields__"):
        for f in _dataclass_fields(dataclass_node):
            yield from _iter_property_refs(getattr(node, f.name))
        return
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _iter_property_refs(item)


def _collect_non_source_alias_property_refs(
    *,
    query: CypherQuery,
    non_source_aliases: Sequence[str],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Scan trailing clauses for `<non_source>.<prop>` references and bare `<non_source>`."""
    if not non_source_aliases:
        return ({}, set())
    targets = set(non_source_aliases)

    props_by_alias: Dict[str, Set[str]] = {alias: set() for alias in targets}
    bare_referenced: Set[str] = set()

    def _scan_text(text: Optional[str]) -> None:
        if text is None or not text:
            return
        try:
            node = parse_expr(text)
        except (GFQLExprParseError, ImportError):
            return
        property_refs_seen: Set[Tuple[str, str]] = set()

        def _enter(child: ExprNode) -> None:
            if (
                isinstance(child, PropertyAccessExpr)
                and isinstance(child.value, Identifier)
                and child.value.name in targets
            ):
                property_refs_seen.add((child.value.name, child.property))

        walk_expr_nodes(node, enter=_enter)
        for alias, prop in property_refs_seen:
            props_by_alias[alias].add(prop)
        property_alias_names = {alias for alias, _prop in property_refs_seen}
        for ident in collect_identifiers(node):
            if ident in targets and ident not in property_alias_names:
                bare_referenced.add(ident)

    def _scan_where(where_clause: Optional[WhereClause]) -> None:
        if where_clause is None:
            return
        if where_clause.expr_tree is not None:
            _scan_text(boolean_expr_to_text(where_clause.expr_tree))
        for term in where_clause.predicates:
            for ref in _iter_property_refs(term):
                if ref.alias in targets:
                    props_by_alias[ref.alias].add(ref.property)

    for match_clause in query.reentry_matches:
        _scan_where(match_clause.where)
    for where_clause in query.reentry_wheres:
        _scan_where(where_clause)
    for stage in query.with_stages[1:]:
        for projection_item in stage.clause.items:
            _scan_text(projection_item.expression.text)
        if stage.where is not None:
            _scan_text(stage.where.text)
    for unwind_clause in query.reentry_unwinds:
        _scan_text(unwind_clause.expression.text)
    for return_item in query.return_.items:
        _scan_text(return_item.expression.text)
    if query.order_by is not None:
        for order_item in query.order_by.items:
            _scan_text(order_item.expression.text)

    return props_by_alias, bare_referenced


def _first_pattern_node_alias(clause: MatchClause) -> Optional[str]:
    if clause.patterns:
        first_pattern = clause.patterns[0]
        if first_pattern and isinstance(first_pattern[0], NodePattern):
            return first_pattern[0].variable
    from graphistry.compute.gfql.cypher import lowering as _lowering

    pattern = _lowering._match_pattern_elements(clause)
    if not pattern or not isinstance(pattern[0], NodePattern):
        return None
    return pattern[0].variable


_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_whole_row_with_item(item: ReturnItem, *, match_node_aliases: Set[str]) -> bool:
    """A WITH item is a whole-row carry when it is a bare prior node alias."""
    text = item.expression.text
    if not _BARE_IDENT_RE.match(text):
        return False
    if item.alias is not None and item.alias != text:
        return False
    return text in match_node_aliases


def _all_match_alias_kinds(query: CypherQuery) -> Dict[str, str]:
    """Map aliases bound by ``query.matches`` to ``node``, ``rel``, or ``path``."""
    from graphistry.compute.gfql.cypher import lowering as _lowering

    node_aliases: Set[str] = set()
    rel_aliases: Set[str] = set()
    path_aliases: Set[str] = set()
    for clause in query.matches:
        for pattern in clause.patterns:
            for element in pattern:
                if isinstance(element, NodePattern) and element.variable is not None:
                    node_aliases.add(element.variable)
                elif isinstance(element, RelationshipPattern) and element.variable is not None:
                    rel_aliases.add(element.variable)
        for element in _lowering._match_pattern_elements(clause):
            if isinstance(element, NodePattern) and element.variable is not None:
                node_aliases.add(element.variable)
            elif isinstance(element, RelationshipPattern) and element.variable is not None:
                rel_aliases.add(element.variable)
        for path_alias in clause.pattern_aliases:
            if path_alias is not None:
                path_aliases.add(path_alias)
    out: Dict[str, str] = {name: "node" for name in node_aliases}
    for name in rel_aliases:
        out[name] = "rel"
    for name in path_aliases:
        out[name] = "path"
    return out


def _is_bare_carry_with_item(item: ReturnItem) -> Optional[str]:
    """Return the bare carried identifier for unrenamed/self-renamed WITH items."""
    text = item.expression.text
    if not _BARE_IDENT_RE.match(text):
        return None
    if item.alias is not None and item.alias != text:
        return None
    return text


def _collect_secondary_property_refs(
    expr: ExpressionText,
    *,
    secondary_aliases: Set[str],
    field: str,
) -> Tuple[ExpressionText, Set[Tuple[str, str]], Set[str]]:
    """Rewrite secondary ``S.X`` references to hidden scalar carry identifiers."""
    if not secondary_aliases:
        return expr, set(), set()
    if not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", expr.text) for name in secondary_aliases):
        return expr, set(), set()
    from graphistry.compute.gfql.cypher import lowering as _lowering

    try:
        node = parse_expr(expr.text)
    except (GFQLExprParseError, ImportError) as exc:
        raise _lowering._unsupported(
            "Cypher MATCH after WITH multi-alias carry rewrite requires a locally supported scalar expression",
            field=field,
            value=expr.text,
            line=expr.span.line,
            column=expr.span.column,
        ) from exc
    refs: Set[Tuple[str, str]] = set()
    bare: Set[str] = set()
    rewritten = _rewrite_secondary_alias_property_refs(
        node,
        secondary_aliases=secondary_aliases,
        refs=refs,
        bare=bare,
    )
    if not refs and not bare:
        return expr, set(), set()
    new_text = _lowering._render_expr_node(rewritten)
    if new_text == expr.text:
        return expr, refs, bare
    return ExpressionText(text=new_text, span=expr.span), refs, bare


def _rewrite_secondary_alias_property_refs(
    node: ExprNode,
    *,
    secondary_aliases: Set[str],
    refs: Set[Tuple[str, str]],
    bare: Set[str],
    shadowed: Optional[FrozenSet[str]] = None,
) -> ExprNode:
    active_shadow = shadowed or frozenset()
    if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
        alias_name = node.value.name
        if alias_name in secondary_aliases and alias_name not in active_shadow:
            refs.add((alias_name, node.property))
            return Identifier(_secondary_reentry_hidden_column_name(alias_name, node.property))
    if isinstance(node, Identifier):
        if node.name in secondary_aliases and node.name not in active_shadow:
            bare.add(node.name)
        return node
    if isinstance(node, QuantifierExpr):
        next_shadow = active_shadow | {node.var}
        return QuantifierExpr(
            node.fn,
            node.var,
            _rewrite_secondary_alias_property_refs(
                node.source, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=next_shadow,
            ),
            _rewrite_secondary_alias_property_refs(
                node.predicate, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=next_shadow,
            ),
        )
    if isinstance(node, ListComprehension):
        next_shadow = active_shadow | {node.var}
        return ListComprehension(
            node.var,
            _rewrite_secondary_alias_property_refs(
                node.source, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=next_shadow,
            ),
            predicate=None if node.predicate is None else _rewrite_secondary_alias_property_refs(
                node.predicate, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=next_shadow,
            ),
            projection=None if node.projection is None else _rewrite_secondary_alias_property_refs(
                node.projection, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=next_shadow,
            ),
        )
    from graphistry.compute.gfql.cypher import lowering as _lowering

    return _lowering._rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_secondary_alias_property_refs(
            child, secondary_aliases=secondary_aliases, refs=refs, bare=bare, shadowed=active_shadow,
        ),
        error_context="secondary alias rewrite",
    )


def _all_match_node_aliases(query: CypherQuery) -> Set[str]:
    from graphistry.compute.gfql.cypher import lowering as _lowering

    out: Set[str] = set()
    for clause in query.matches:
        for pattern in clause.patterns:
            out.update(_lowering._pattern_node_aliases(pattern))
        for element in _lowering._match_pattern_elements(clause):
            if isinstance(element, NodePattern) and element.variable is not None:
                out.add(element.variable)
    return out


def _demote_secondary_whole_row_aliases(
    query: CypherQuery,
    *,
    prefix_stage: ProjectionStage,
    primary_alias: Optional[str],
) -> Tuple[CypherQuery, ProjectionStage, Tuple[str, ...], Mapping[str, Tuple[str, ...]]]:
    """Demote secondary whole-row aliases in a reentry prefix into scalar carries."""
    if not query.reentry_matches:
        return query, prefix_stage, (), {}
    from graphistry.compute.gfql.cypher import lowering as _lowering
    from graphistry.compute.gfql.cypher.reentry.rewrite import (
        _rewrite_reentry_match_clause,
        _rewrite_reentry_projection_clause,
        _rewrite_reentry_projection_stage,
    )

    match_node_aliases = _all_match_node_aliases(query)
    whole_row_items: List[Tuple[int, ReturnItem]] = [
        (idx, item)
        for idx, item in enumerate(prefix_stage.clause.items)
        if _is_whole_row_with_item(item, match_node_aliases=match_node_aliases)
    ]
    if len(whole_row_items) <= 1:
        return query, prefix_stage, (), {}
    if primary_alias is None:
        return query, prefix_stage, (), {}

    primary_indices = {idx for idx, item in whole_row_items if item.expression.text == primary_alias}
    if not primary_indices:
        return query, prefix_stage, (), {}
    secondary_items = [(idx, item) for idx, item in whole_row_items if idx not in primary_indices]
    secondary_aliases: Set[str] = {item.expression.text for _idx, item in secondary_items}

    for trailing in (*query.reentry_matches,):
        trailing_aliases: Set[str] = set()
        for pattern in trailing.patterns:
            trailing_aliases.update(_lowering._pattern_node_aliases(pattern))
        rebound = sorted(trailing_aliases & secondary_aliases)
        if rebound:
            raise _lowering._unsupported_at_span(
                "Cypher MATCH after WITH does not yet support re-binding a carried secondary alias as a node variable in the trailing MATCH",
                field="match",
                value=rebound,
                span=trailing.span,
            )

    refs_collected: Set[Tuple[str, str]] = set()
    bare_collected: Set[str] = set()

    def rewrite_text(expr: ExpressionText, field: str) -> ExpressionText:
        rewritten, refs, bare = _collect_secondary_property_refs(
            expr,
            secondary_aliases=secondary_aliases,
            field=field,
        )
        refs_collected.update(refs)
        bare_collected.update(bare)
        return rewritten

    rewritten_reentry_matches = tuple(
        _rewrite_reentry_match_clause(clause, rewrite_expr=rewrite_text)
        for clause in query.reentry_matches
    )
    rewritten_reentry_wheres = tuple(
        where_clause if where_clause is None else _lowering._rewrite_where_clause_and_resync(where_clause, rewrite_text, "where")
        for where_clause in query.reentry_wheres
    )
    secondary_forwarding_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    cleaned_with_stages_tail = tuple(
        _drop_bare_alias_items_from_stage(
            stage, secondary_aliases, identifier_re=secondary_forwarding_re
        )
        for stage in query.with_stages[1:]
    )
    rewritten_with_stages_tail = tuple(
        _rewrite_reentry_projection_stage(stage, rewrite_expr=rewrite_text)
        for stage in cleaned_with_stages_tail
    )
    rewritten_unwinds = tuple(
        replace(unwind, expression=rewrite_text(unwind.expression, "unwind"))
        for unwind in query.reentry_unwinds
    )
    rewritten_return = _rewrite_reentry_projection_clause(query.return_, rewrite_expr=rewrite_text)
    rewritten_order_by = _rewrite_order_by_expressions(query.order_by, rewrite_text)

    if bare_collected:
        raise _lowering._unsupported_at_span(
            "Cypher MATCH after WITH does not yet support carrying secondary whole-row aliases as whole-row outputs; reference them by property only",
            field="return",
            value=sorted(bare_collected),
            span=query.return_.span,
        )

    secondary_drop_indices = {idx for idx, _item in secondary_items}
    new_items: List[ReturnItem] = [
        item for idx, item in enumerate(prefix_stage.clause.items)
        if idx not in secondary_drop_indices
    ]
    template_span = prefix_stage.span
    for alias_name, prop in sorted(refs_collected):
        new_items.append(
            ReturnItem(
                expression=ExpressionText(text=f"{alias_name}.{prop}", span=template_span),
                alias=_secondary_reentry_hidden_column_name(alias_name, prop),
                span=template_span,
            )
        )
    rewritten_prefix_stage = replace(
        prefix_stage,
        clause=replace(prefix_stage.clause, items=tuple(new_items)),
    )

    if refs_collected and rewritten_with_stages_tail:
        forwarded_tuple = tuple(
            ReturnItem(
                expression=ExpressionText(
                    text=_secondary_reentry_hidden_column_name(alias_name, prop),
                    span=template_span,
                ),
                alias=None,
                span=template_span,
            )
            for alias_name, prop in sorted(refs_collected)
        )
        rewritten_with_stages_tail = tuple(
            replace(
                stage,
                clause=replace(
                    stage.clause,
                    items=stage.clause.items + forwarded_tuple,
                ),
            )
            for stage in rewritten_with_stages_tail
        )

    rewritten_query = replace(
        query,
        with_stages=(rewritten_prefix_stage,) + rewritten_with_stages_tail,
        reentry_matches=rewritten_reentry_matches,
        reentry_wheres=rewritten_reentry_wheres,
        reentry_unwinds=rewritten_unwinds,
        return_=rewritten_return,
        order_by=rewritten_order_by,
    )
    secondary_props: Dict[str, Set[str]] = {alias: set() for alias in secondary_aliases}
    for alias_name, prop in refs_collected:
        secondary_props.setdefault(alias_name, set()).add(prop)
    secondary_props_sorted = {
        alias_name: tuple(sorted(props))
        for alias_name, props in secondary_props.items()
        if props
    }
    return (
        rewritten_query,
        rewritten_prefix_stage,
        tuple(sorted(secondary_aliases)),
        secondary_props_sorted,
    )
