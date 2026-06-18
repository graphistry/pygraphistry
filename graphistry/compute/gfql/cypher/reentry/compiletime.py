"""Bounded reentry compile-time orchestration."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, NoReturn, Optional, Tuple, cast

from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    ExpressionText,
    MatchClause,
    OrderByClause,
    ProjectionStage,
    ReturnClause,
    ReturnItem,
    UnwindClause,
)
from graphistry.compute.gfql.cypher.lowering import (
    CompiledCypherPostProcessing,
    CompiledCypherExecutionExtras,
    CompiledCypherQuery,
    _connected_component_from_pattern,
    _match_pattern_elements,
    _normalize_post_processing,
    _pattern_node_aliases,
    _render_expr_node,
    _rewrite_expr_identifiers,
    _rewrite_where_clause_and_resync,
    _unsupported_at_span,
    _verify_selected_logical_plan,
    compile_cypher_query,
)
from graphistry.compute.gfql.cypher.reentry.carry import (
    _bounded_reentry_carry_columns,
    _bounded_reentry_prefix_order_is_safe,
    _bounded_reentry_scalar_prefix_columns,
)
from graphistry.compute.gfql.cypher.reentry.lowering_support import (
    _all_match_alias_kinds,
    _all_match_node_aliases,
    _collect_non_source_alias_property_refs,
    _demote_secondary_whole_row_aliases,
    _drop_bare_alias_items_from_stage,
    _first_pattern_node_alias,
    _is_bare_carry_with_item,
    _is_whole_row_with_item,
    _rewrite_order_by_expressions,
)
from graphistry.compute.gfql.cypher.reentry.naming import (
    _reentry_hidden_column_name,
    _reentry_property_carry_name,
)
from graphistry.compute.gfql.cypher.reentry.rewrite import (
    _rewrite_collect_unwind_reentry_query,
    _rewrite_reentry_expr_to_hidden_properties,
    _rewrite_reentry_match_clause,
    _rewrite_reentry_projection_clause,
    _rewrite_reentry_projection_stage,
)
from graphistry.compute.gfql.cypher.reentry_plan import CarriedAlias, ReentryPlan
from graphistry.compute.gfql.expr_parser import (
    GFQLExprParseError,
    Identifier,
    ListLiteral,
    parse_expr,
)
from graphistry.compute.gfql.ir.logical_plan import (
    LogicalPlan,
    PatternMatch,
    Project as LogicalProject,
    RowSchema as LogicalRowSchema,
)


def _raise_requires_named_reentry_alias(reentry_match: MatchClause, first_alias: Optional[str]) -> NoReturn:
    raise _unsupported_at_span(
        "Cypher MATCH after WITH currently requires the trailing MATCH to start from a named node alias",
        field="match",
        value=first_alias,
        span=reentry_match.span,
    )


def _map_terminal_reentry_query(
    compiled_query: CompiledCypherQuery,
    *,
    transform: Callable[[CompiledCypherQuery], CompiledCypherQuery],
) -> CompiledCypherQuery:
    if compiled_query.start_nodes_query is None:
        return transform(compiled_query)
    mapped_start_nodes = _map_terminal_reentry_query(
        compiled_query.start_nodes_query,
        transform=transform,
    )
    return replace(
        compiled_query,
        execution_extras=replace(
            compiled_query.execution_extras or CompiledCypherExecutionExtras(),
            start_nodes_query=mapped_start_nodes,
            scope_stack=(),
        ),
    )


def _without_outer_query_context(query: CypherQuery, **updates: Any) -> CypherQuery:
    return replace(query, call=None, row_sequence=(), graph_bindings=(), use=None, **updates)


def _rewrite_terminal_singleton_reentry_unwind(
    *,
    reentry_unwinds: Tuple[UnwindClause, ...],
    reentry_return: ReturnClause,
    reentry_order_by: Optional[OrderByClause],
) -> Optional[Tuple[Tuple[UnwindClause, ...], ReturnClause, Optional[OrderByClause]]]:
    """Rewrite terminal `UNWIND [x] AS y` into a downstream identifier rename."""
    if len(reentry_unwinds) != 1:
        return None
    unwind_clause = reentry_unwinds[0]
    try:
        unwind_node = parse_expr(unwind_clause.expression.text.strip())
    except (GFQLExprParseError, ImportError):
        return None
    if not isinstance(unwind_node, ListLiteral) or len(unwind_node.items) != 1:
        return None
    source_node = unwind_node.items[0]
    if not isinstance(source_node, Identifier):
        return None
    source_name = source_node.name
    unwind_alias = unwind_clause.alias
    if source_name == unwind_alias:
        return (), reentry_return, reentry_order_by
    replacements = {unwind_alias: source_name}

    def _rewrite_expr(expr: ExpressionText, _field: str) -> Optional[ExpressionText]:
        try:
            node = parse_expr(expr.text)
        except (GFQLExprParseError, ImportError):
            return None
        rewritten = _rewrite_expr_identifiers(node, replacements)
        return ExpressionText(text=_render_expr_node(rewritten), span=expr.span)

    rewritten_return_items: List[ReturnItem] = []
    for item in reentry_return.items:
        rewritten_expr = _rewrite_expr(item.expression, "return")
        if rewritten_expr is None:
            return None
        rewritten_return_items.append(replace(item, expression=rewritten_expr))
    rewritten_return = replace(reentry_return, items=tuple(rewritten_return_items))
    rewritten_order_by = _rewrite_order_by_expressions(reentry_order_by, _rewrite_expr)
    if reentry_order_by is not None and rewritten_order_by is None:
        return None
    return (), rewritten_return, rewritten_order_by


def _rewrite_multi_whole_row_prefix(
    prefix_stage: ProjectionStage,
    *,
    query: CypherQuery,
    reentry_first_alias: Optional[str],
) -> Tuple[ProjectionStage, Tuple[ProjectionStage, ...], Dict[str, Tuple[str, ...]]]:
    """Decompose non-source whole-row aliases in the prefix WITH into scalar carries."""

    original_tail = tuple(query.with_stages[1:])
    if reentry_first_alias is None:
        return prefix_stage, original_tail, {}

    bare_item_indices: Dict[str, int] = {}
    for idx, item in enumerate(prefix_stage.clause.items):
        carry_name = _is_bare_carry_with_item(item)
        if carry_name is not None:
            bare_item_indices[carry_name] = idx

    non_source_aliases = tuple(
        name for name in bare_item_indices if name != reentry_first_alias
    )
    if not non_source_aliases:
        return prefix_stage, original_tail, {}

    candidate_set = set(non_source_aliases)
    cleaned_tail = tuple(
        _drop_bare_alias_items_from_stage(stage, candidate_set)
        for stage in original_tail
    )
    cleaned_query = replace(query, with_stages=(prefix_stage,) + cleaned_tail)

    props_by_alias, bare_referenced = _collect_non_source_alias_property_refs(
        query=cleaned_query,
        non_source_aliases=non_source_aliases,
    )
    if bare_referenced:
        return prefix_stage, original_tail, {}

    referenced = tuple(name for name in non_source_aliases if props_by_alias.get(name))
    if not referenced:
        return prefix_stage, cleaned_tail, {}

    drop_indices = {bare_item_indices[name] for name in referenced}
    new_items: List[ReturnItem] = [item for idx, item in enumerate(prefix_stage.clause.items) if idx not in drop_indices]
    span = prefix_stage.clause.span
    carried_props: Dict[str, Tuple[str, ...]] = {}
    for alias in referenced:
        props = tuple(sorted(props_by_alias[alias]))
        carried_props[alias] = props
        for prop in props:
            new_items.append(
                ReturnItem(
                    expression=ExpressionText(text=f"{alias}.{prop}", span=span),
                    alias=_reentry_property_carry_name(alias, prop),
                    span=span,
                )
            )

    new_clause = replace(prefix_stage.clause, items=tuple(new_items))
    return replace(prefix_stage, clause=new_clause), cleaned_tail, carried_props


def _optional_reentry_logical_route_marker(reentry_match: MatchClause) -> LogicalPlan:
    component = _connected_component_from_pattern(
        _match_pattern_elements(reentry_match),
        entry_points=(),
    )
    logical_plan = LogicalProject(
        op_id=2,
        input=PatternMatch(
            op_id=1,
            pattern={
                "node_aliases": tuple(component.node_aliases),
                "edge_aliases": tuple(component.edge_aliases),
                "entry_points": tuple(component.entry_points),
            },
            optional=True,
            arm_id="optional_reentry_arm",
            output_schema=LogicalRowSchema(columns={}),
        ),
        expressions=["optional_reentry"],
        output_schema=LogicalRowSchema(columns={}),
    )
    _verify_selected_logical_plan(logical_plan)
    return logical_plan


def _compile_bounded_reentry_query(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherQuery:
    if query.unwinds:
        rewritten_query = _rewrite_collect_unwind_reentry_query(query)
        if rewritten_query is None:
            first_unwind = query.unwinds[0]
            raise _unsupported_at_span(
                "Cypher UNWIND after WITH/RETURN currently supports only a single WITH collect([distinct] alias) AS list UNWIND list AS alias MATCH ... RETURN shape",
                field="unwind",
                value=first_unwind.expression.text,
                span=first_unwind.span,
            )
        query = rewritten_query
    too_few_withs = len(query.with_stages) < len(query.reentry_matches)
    too_many_suffix_withs = (
        len(query.reentry_matches) > 1
        and len(query.with_stages) > len(query.reentry_matches) + 1
    )
    if not query.reentry_matches or too_few_withs or too_many_suffix_withs:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH is only supported for alternating MATCH ... WITH ... MATCH ... [WITH ... MATCH ...] ... [WITH] RETURN read shapes in the local compiler",
            field="match",
            value=len(query.reentry_matches),
            span=query.return_.span,
        )
    prefix_stage = query.with_stages[0]
    if prefix_stage.where is not None:
        raise _unsupported_at_span(
            "Cypher MATCH after WITH does not yet support WITH ... WHERE in the prefix stage",
            field="with.where",
            value=prefix_stage.where.text,
            span=prefix_stage.span,
        )
    match_alias_kinds = _all_match_alias_kinds(query)
    for item in prefix_stage.clause.items:
        carry_name = _is_bare_carry_with_item(item)
        if carry_name is None:
            continue
        kind = match_alias_kinds.get(carry_name)
        if kind == "rel":
            raise _unsupported_at_span(
                "Cypher MATCH after WITH does not yet support carrying a relationship "
                "variable across re-entry; project its properties (`<rel>.<prop>`) instead",
                field="with",
                value=carry_name,
                span=item.span,
            )
        if kind == "path":
            raise _unsupported_at_span(
                "Cypher MATCH after WITH does not yet support carrying a named path alias "
                "across re-entry; project derived scalars (e.g. `length(<path>)`) instead",
                field="with",
                value=carry_name,
                span=item.span,
            )
    primary_alias_hint = _first_pattern_node_alias(query.reentry_matches[0])
    multi_alias_carries: Dict[str, Tuple[str, ...]] = {}
    if primary_alias_hint is not None:
        match_node_aliases = _all_match_node_aliases(query)
        whole_row_aliases = {
            item.expression.text.strip()
            for item in prefix_stage.clause.items
            if _is_whole_row_with_item(item, match_node_aliases=match_node_aliases)
        }
        if primary_alias_hint not in whole_row_aliases:
            rewritten_prefix, rewritten_tail, multi_alias_carries = _rewrite_multi_whole_row_prefix(
                prefix_stage,
                query=query,
                reentry_first_alias=primary_alias_hint,
            )
            if multi_alias_carries:
                query = replace(query, with_stages=(rewritten_prefix,) + rewritten_tail)
                prefix_stage = rewritten_prefix
    query, prefix_stage, demoted_secondary_aliases, demoted_secondary_props = _demote_secondary_whole_row_aliases(
        query,
        prefix_stage=prefix_stage,
        primary_alias=primary_alias_hint,
    )
    non_source_carried_props_map: Dict[str, Tuple[str, ...]] = dict(demoted_secondary_props)
    projection_items = [item.expression.text for item in prefix_stage.clause.items]
    prefix_query = _without_outer_query_context(
        query,
        reentry_matches=(),
        reentry_wheres=(),
        with_stages=(),
        return_=replace(prefix_stage.clause, kind="return"),
        order_by=prefix_stage.order_by,
        skip=prefix_stage.skip,
        limit=prefix_stage.limit,
        trailing_semicolon=False,
        reentry_unwinds=(),
    )
    prefix_compiled = compile_cypher_query(prefix_query, params=params)
    if not isinstance(prefix_compiled, CompiledCypherQuery):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH prefix compilation produced an unexpected UNION program",
            field="with",
            value="union",
            span=prefix_stage.span,
        )
    reentry_match = query.reentry_matches[0]
    remaining_with_stages = query.with_stages[1:]
    remaining_reentry_matches = query.reentry_matches[1:]
    first_alias = _first_pattern_node_alias(reentry_match)
    prefix_projection = prefix_compiled.result_projection
    scalar_only_prefix = prefix_projection is None
    if scalar_only_prefix:
        reused_scalar_aliases = sorted(
            {item.alias for item in prefix_stage.clause.items if item.alias is not None}
            & set().union(*(_pattern_node_aliases(pattern) for pattern in reentry_match.patterns))
        )
        if reused_scalar_aliases:
            # Binder cross-kind guards catch this for standard compile paths;
            # keep as a defensive backstop for any path that bypasses binder.
            raise _unsupported_at_span(
                "Cypher MATCH after WITH scalar-only prefix aliases cannot be reused as node variables in the trailing MATCH",
                field="match",
                value=reused_scalar_aliases,
                span=reentry_match.span,
            )
        if first_alias is None:
            _raise_requires_named_reentry_alias(reentry_match, first_alias)
        reentry_alias = first_alias
        carry_columns = _bounded_reentry_scalar_prefix_columns(
            prefix_stage,
            projection_items=projection_items,
        )
        free_form = False
    else:
        assert prefix_projection is not None
        reentry_alias, carry_columns, non_source_alias_names = _bounded_reentry_carry_columns(
            prefix_projection,
            projection_items=projection_items,
            query=query,
            prefix_stage=prefix_stage,
            reentry_alias_hint=first_alias,
        )
        whole_row_carried = tuple(
            column.output_name for column in prefix_projection.columns if column.kind == "whole_row"
        )
        free_form = first_alias is not None and first_alias not in whole_row_carried
        if free_form:
            reentry_alias = cast(str, first_alias)
            non_source_alias_names = whole_row_carried
        if non_source_alias_names:
            props_by_alias, bare_referenced = _collect_non_source_alias_property_refs(
                query=query,
                non_source_aliases=non_source_alias_names,
            )
            if bare_referenced:
                raise _unsupported_at_span(
                    "Cypher MATCH after WITH currently carries non-source whole-row aliases only via property "
                    "access (`<alias>.<prop>`); bare references like `WHERE x = ...` or `RETURN x` require "
                    "the full row-carrier rewrite tracked under #989",
                    field="with",
                    value=sorted(bare_referenced),
                    span=prefix_stage.span,
                )
            for alias_name, props in props_by_alias.items():
                if not props:
                    continue
                non_source_carried_props_map[alias_name] = tuple(
                    sorted({*non_source_carried_props_map.get(alias_name, ()), *props})
                )
        for alias_name, carried_props in multi_alias_carries.items():
            non_source_carried_props_map[alias_name] = tuple(
                sorted({*non_source_carried_props_map.get(alias_name, ()), *carried_props})
            )
    if not _bounded_reentry_prefix_order_is_safe(prefix_stage=prefix_stage, query=query, params=params):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH requires bounded literal LIMIT (and no SKIP) to preserve prefix WITH row ordering across MATCH re-entry when the trailing query has no ORDER BY",
            field="with.order_by",
            value=(
                [item.expression.text for item in prefix_stage.order_by.items]
                if prefix_stage.order_by is not None
                else None
            ),
            span=prefix_stage.order_by.span if prefix_stage.order_by is not None else prefix_stage.span,
        )
    if prefix_projection is not None and prefix_projection.table != "nodes":
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently supports node re-entry only",
            field="with",
            value=prefix_projection.table,
            span=prefix_stage.span,
        )
    if len(query.return_.items) == 1 and query.return_.items[0].expression.text == "*":
        raise _unsupported_at_span(
            "Cypher MATCH after WITH does not yet support RETURN * from the trailing MATCH re-entry stage",
            field=query.return_.kind,
            value="*",
            span=query.return_.span,
        )
    if first_alias is None:
        _raise_requires_named_reentry_alias(reentry_match, first_alias)

    hidden_columns = tuple(_reentry_hidden_column_name(output_name) for output_name in carry_columns)

    if scalar_only_prefix:
        current_reentry_plan: ReentryPlan = ReentryPlan(
            reentry_alias_name=reentry_alias,
            aliases=(),
            scalar_columns=tuple(carry_columns),
            scalar_only=True,
        )
    else:
        assert prefix_projection is not None
        carried_names = tuple(
            name
            for name in dict.fromkeys(non_source_alias_names + demoted_secondary_aliases + tuple(multi_alias_carries))
            if free_form or name != reentry_alias
        )
        current_aliases = (
            (() if free_form else (
                CarriedAlias(
                    output_name=reentry_alias,
                    table=prefix_projection.table,
                    is_reentry_alias=True,
                ),
            ))
            + tuple(
                CarriedAlias(
                    output_name=name,
                    table=prefix_projection.table,
                    is_reentry_alias=False,
                    carried_properties=non_source_carried_props_map.get(name, ()),
                )
                for name in carried_names
            )
        )
        current_reentry_plan = ReentryPlan(
            reentry_alias_name=reentry_alias,
            aliases=current_aliases,
            scalar_columns=tuple(carry_columns),
            scalar_only=False,
            free_form=free_form,
        )

    def rewrite_expr(expr: ExpressionText, field: str) -> ExpressionText:
        return _rewrite_reentry_expr_to_hidden_properties(
            expr,
            carried_alias=reentry_alias,
            carried_columns=carry_columns,
            field=field,
            non_source_carried_props=non_source_carried_props_map or None,
        )

    reentry_where = query.reentry_where
    reentry_return = query.return_
    reentry_order_by = query.order_by
    rewritten_with_stages = remaining_with_stages
    rewritten_reentry_unwinds = query.reentry_unwinds
    remaining_reentry_wheres = query.reentry_wheres[1:]
    rewritten_remaining_reentry_wheres = remaining_reentry_wheres
    rewritten_reentry_match = reentry_match
    rewritten_remaining_reentry_matches = remaining_reentry_matches
    if hidden_columns:
        rewritten_reentry_match = _rewrite_reentry_match_clause(reentry_match, rewrite_expr=rewrite_expr)
        rewritten_remaining_reentry_matches = tuple(
            _rewrite_reentry_match_clause(match_clause, rewrite_expr=rewrite_expr)
            for match_clause in remaining_reentry_matches
        )
        rewritten_remaining_reentry_wheres = tuple(
            None
            if where_clause is None
            else _rewrite_where_clause_and_resync(where_clause, rewrite_expr, "where")
            for where_clause in remaining_reentry_wheres
        )
        rewritten_with_stages = tuple(
            _rewrite_reentry_projection_stage(stage, rewrite_expr=rewrite_expr)
            for stage in remaining_with_stages
        )
        rewritten_reentry_unwinds = tuple(
            replace(
                unwind_clause,
                expression=rewrite_expr(unwind_clause.expression, "unwind"),
            )
            for unwind_clause in query.reentry_unwinds
        )
        if query.reentry_where is not None:
            reentry_where = _rewrite_where_clause_and_resync(query.reentry_where, rewrite_expr, "where")
        if not remaining_reentry_matches:
            reentry_return = _rewrite_reentry_projection_clause(query.return_, rewrite_expr=rewrite_expr)
            if reentry_order_by is not None:
                rewritten_order_by = _rewrite_order_by_expressions(reentry_order_by, rewrite_expr)
                assert rewritten_order_by is not None
                reentry_order_by = rewritten_order_by
    if rewritten_reentry_unwinds and rewritten_with_stages and not rewritten_remaining_reentry_matches:
        singleton_rewrite = _rewrite_terminal_singleton_reentry_unwind(
            reentry_unwinds=rewritten_reentry_unwinds,
            reentry_return=reentry_return,
            reentry_order_by=reentry_order_by,
        )
        if singleton_rewrite is not None:
            rewritten_reentry_unwinds, reentry_return, reentry_order_by = singleton_rewrite
        else:
            first_unwind = rewritten_reentry_unwinds[0]
            raise _unsupported_at_span(
                "Cypher UNWIND after WITH/RETURN is not yet supported once MATCH has introduced graph aliases",
                field="unwind",
                value=first_unwind.expression.text,
                span=first_unwind.span,
            )
    suffix_query = _without_outer_query_context(
        query,
        matches=(rewritten_reentry_match,),
        where=reentry_where,
        unwinds=rewritten_reentry_unwinds,
        with_stages=rewritten_with_stages,
        return_=reentry_return,
        order_by=reentry_order_by,
        reentry_matches=rewritten_remaining_reentry_matches,
        reentry_wheres=rewritten_remaining_reentry_wheres,
        reentry_unwinds=(),
    )
    suffix_compiled = compile_cypher_query(suffix_query, params=params)
    if not isinstance(suffix_compiled, CompiledCypherQuery):
        raise _unsupported_at_span(
            "Cypher MATCH after WITH suffix compilation produced an unexpected UNION program",
            field="match",
            value="union",
            span=reentry_match.span,
        )
    def attach_current_reentry(target: CompiledCypherQuery) -> CompiledCypherQuery:
        target_projection = target.result_projection
        if target_projection is not None and target_projection.alias == reentry_alias and hidden_columns:
            target_projection = replace(
                target_projection,
                exclude_columns=tuple(
                    dict.fromkeys(target_projection.exclude_columns + hidden_columns)
                ),
            )
        is_optional = reentry_match.optional or target.optional_reentry
        return replace(
            target,
            post_processing=_normalize_post_processing(
                CompiledCypherPostProcessing(
                    result_projection=target_projection,
                    # Clear empty_result_row when optional_reentry is set — the
                    # reentry null-fill handles missing rows instead.
                    empty_result_row=None if is_optional else target.empty_result_row,
                    optional_null_fill=target.optional_null_fill,
                    optional_projection_row_guard=target.optional_projection_row_guard,
                )
            ),
            execution_extras=replace(
                target.execution_extras or CompiledCypherExecutionExtras(),
                start_nodes_query=prefix_compiled,
                optional_reentry=is_optional,
                reentry_plan=current_reentry_plan,
                scope_stack=(),
                logical_plan=(
                    target.logical_plan
                    if target.logical_plan is not None or not is_optional
                    else _optional_reentry_logical_route_marker(reentry_match)
                ),
                logical_plan_defer_reason=(
                    None
                    if is_optional and target.logical_plan is None
                    else target.logical_plan_defer_reason
                ),
                logical_plan_defer_code=(
                    None
                    if is_optional and target.logical_plan is None
                    else target.logical_plan_defer_code
                ),
            ),
        )

    return _map_terminal_reentry_query(
        suffix_compiled,
        transform=attach_current_reentry,
    )
