"""Bounded reentry compile-time orchestration extracted from ``cypher.lowering``.

S3 under #1260 moves reentry orchestration helpers into a reentry-focused
module while preserving behavior and compatibility via lowering-level shims.
"""
# mypy: ignore-errors
# ruff: noqa: F821
from __future__ import annotations

# Rebind the lowering symbol table (including private helpers) so extracted
# functions stay behavior-identical without re-plumbing deep dependencies.
from graphistry.compute.gfql.cypher import lowering as _lowering

globals().update(vars(_lowering))

def _map_terminal_reentry_query(
    compiled_query: CompiledCypherQuery,
    *,
    transform: Callable[[CompiledCypherQuery], CompiledCypherQuery],
) -> CompiledCypherQuery:
    compiled_extras = compiled_query.execution_extras or CompiledCypherExecutionExtras()
    if compiled_query.start_nodes_query is None:
        return transform(compiled_query)
    mapped_start_nodes = _map_terminal_reentry_query(
        compiled_query.start_nodes_query,
        transform=transform,
    )
    return replace(
        compiled_query,
        execution_extras=_execution_extras_with(
            compiled_query,
            connected_optional_match=compiled_extras.connected_optional_match,
            connected_match_join=compiled_extras.connected_match_join,
            query_graph=compiled_extras.query_graph,
            start_nodes_query=mapped_start_nodes,
            optional_reentry=compiled_query.optional_reentry,
            reentry_plan=compiled_query.reentry_plan,
            logical_plan=compiled_query.logical_plan,
            logical_plan_defer_reason=compiled_query.logical_plan_defer_reason,
        ),
    )


def _drop_bare_alias_items_from_stage(
    stage: ProjectionStage,
    aliases: AbstractSet[str],
    *,
    identifier_re: "re.Pattern[str]",
) -> ProjectionStage:
    """Drop bare-identifier projection items whose name is in ``aliases``.

    Used by the multi-stage carry chain (slice 4.3c): downstream
    ``WITH a, x, y, collect(...)`` re-projects carried whole-row aliases as
    forwarding items. Once their properties are carried as hidden columns on
    the reentry-alias's row table, the bare forwarding items are noise — the
    carry continues through the chain regardless. Drop them so the bare-ref
    scanner doesn't fail-fast on what is in fact a forwarding pattern.
    """
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


def _rewrite_terminal_singleton_reentry_unwind(
    *,
    reentry_unwinds: Tuple[UnwindClause, ...],
    reentry_return: ReturnClause,
    reentry_order_by: Optional[OrderByClause],
) -> Optional[Tuple[Tuple[UnwindClause, ...], ReturnClause, Optional[OrderByClause]]]:
    """Rewrite `UNWIND [x] AS y` (terminal reentry tail) into an identifier rename.

    For the bounded-reentry suffix shape `... WITH ... UNWIND [x] AS y RETURN ...`,
    the current query model cannot encode WITH-before-UNWIND ordering. When the
    unwind is a singleton list literal over an identifier, the operation is an
    identity row mapping (`y := x`) and can be removed by rewriting downstream
    references from `y` to `x`.
    """
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

    def _rewrite_expr(expr: ExpressionText) -> Optional[ExpressionText]:
        try:
            node = parse_expr(expr.text)
        except (GFQLExprParseError, ImportError):
            return None
        rewritten = _rewrite_expr_identifiers(node, replacements)
        return ExpressionText(text=_render_expr_node(rewritten), span=expr.span)

    rewritten_return_items: List[ReturnItem] = []
    for item in reentry_return.items:
        rewritten_expr = _rewrite_expr(item.expression)
        if rewritten_expr is None:
            return None
        rewritten_return_items.append(replace(item, expression=rewritten_expr))
    rewritten_return = replace(reentry_return, items=tuple(rewritten_return_items))
    rewritten_order_by = None
    if reentry_order_by is not None:
        rewritten_order_items: List[OrderByItem] = []
        for item in reentry_order_by.items:
            rewritten_expr = _rewrite_expr(item.expression)
            if rewritten_expr is None:
                return None
            rewritten_order_items.append(replace(item, expression=rewritten_expr))
        rewritten_order_by = replace(
            reentry_order_by,
            items=tuple(rewritten_order_items),
        )
    return (), rewritten_return, rewritten_order_by


def _rewrite_multi_whole_row_prefix(
    prefix_stage: ProjectionStage,
    *,
    query: CypherQuery,
    reentry_first_alias: Optional[str],
) -> Tuple[ProjectionStage, Tuple[ProjectionStage, ...], Dict[str, Tuple[str, ...]]]:
    """Decompose non-source whole-row aliases in the prefix WITH into scalar carries.

    Returns ``(rewritten_prefix, rewritten_tail, multi_alias_carries)``:
    * ``rewritten_prefix`` — prefix with non-source bare items replaced by scalar
      carries (``x.id AS __carry_x__id__``), which the existing scalar-carry
      plumbing forwards onto the reentry-alias's row table as hidden columns.
    * ``rewritten_tail`` — ``query.with_stages[1:]`` with bare non-source-alias
      forwarding items dropped (slice 4.3c). Carried properties survive the
      chain via the hidden columns; bare items in downstream WITH stages were
      pure forwarding noise.
    * ``multi_alias_carries`` — ``{alias: (props...)}`` driving downstream
      AST rewrites of ``<alias>.<prop>`` references.

    Bare-identifier non-source references in actual USE positions (WHERE
    expressions, RETURN items, ORDER BY) cause this pre-flight to return
    untouched query state so the main flow's failfast surfaces the issue.
    """

    original_tail = tuple(query.with_stages[1:])
    if reentry_first_alias is None:
        return prefix_stage, original_tail, {}

    identifier_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    bare_item_indices: Dict[str, int] = {}
    for idx, item in enumerate(prefix_stage.clause.items):
        if item.alias is not None:
            continue
        text = item.expression.text.strip()
        if identifier_re.fullmatch(text):
            bare_item_indices[text] = idx

    non_source_aliases = tuple(
        name for name in bare_item_indices if name != reentry_first_alias
    )
    if not non_source_aliases:
        return prefix_stage, original_tail, {}

    # Slice 4.3c: drop bare forwarding items in downstream WITH stages BEFORE
    # the bare-ref scan so we don't false-positive on `WITH a, x, y, ...` chain
    # forwards.
    candidate_set = set(non_source_aliases)
    cleaned_tail = tuple(
        _drop_bare_alias_items_from_stage(stage, candidate_set, identifier_re=identifier_re)
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

    # Rewrite items: drop bare entries for `referenced` aliases, then append a
    # scalar projection per (alias, property) pair using the hidden column name.
    drop_set = set(referenced)
    new_items: List[ReturnItem] = [
        item
        for idx, item in enumerate(prefix_stage.clause.items)
        if not any(idx == bare_item_indices[name] for name in drop_set)
    ]
    carried_props: Dict[str, Tuple[str, ...]] = {}
    for alias in referenced:
        # Sort properties for deterministic prefix-result column ordering;
        # downstream AST rewrites match by name, so order is purely cosmetic.
        props = tuple(sorted(props_by_alias[alias]))
        carried_props[alias] = props
        for prop in props:
            hidden_name = _reentry_property_carry_name(alias, prop)
            new_items.append(
                ReturnItem(
                    expression=ExpressionText(
                        text=f"{alias}.{prop}",
                        span=prefix_stage.clause.span,
                    ),
                    alias=hidden_name,
                    span=prefix_stage.clause.span,
                )
            )

    new_clause = replace(prefix_stage.clause, items=tuple(new_items))
    return replace(prefix_stage, clause=new_clause), cleaned_tail, carried_props


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
    # #1358: classify carried aliases by kind (node / rel / path). The
    # downstream reentry pipeline only handles whole-row node carries; bare
    # carries of relationship variables or named-path aliases must surface as
    # a clean scope error instead of falling into untested code paths in the
    # multi-whole-row prefix rewriter.
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
        # #1275: for free-form trailing MATCH (first alias not carried by prefix
        # whole-row items), pre-rewrite non-source whole-row carries into scalar
        # hidden columns so trailing `<carried>.<prop>` references can rewrite
        # cleanly without double-wrapping hidden names.
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
    prefix_query = replace(
        query,
        call=None,
        row_sequence=(),
        reentry_matches=(),
        reentry_wheres=(),
        graph_bindings=(),
        use=None,
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
    prefix_projection_table: Optional[Literal["nodes", "edges"]] = None
    if scalar_only_prefix:
        scalar_prefix_aliases = {
            item.alias
            for item in prefix_stage.clause.items
            if item.alias is not None
        }
        reused_scalar_aliases = sorted(
            scalar_prefix_aliases
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
            raise _unsupported_at_span(
                "Cypher MATCH after WITH currently requires the trailing MATCH to start from a named node alias",
                field="match",
                value=first_alias,
                span=reentry_match.span,
            )
        reentry_alias = first_alias
        carry_columns = _bounded_reentry_scalar_prefix_columns(
            prefix_stage,
            projection_items=projection_items,
        )
        free_form = False
    else:
        assert prefix_projection is not None
        prefix_projection_table = prefix_projection.table
        reentry_alias, carry_columns, non_source_alias_names = _bounded_reentry_carry_columns(
            prefix_projection,
            projection_items=projection_items,
            query=query,
            prefix_stage=prefix_stage,
            reentry_alias_hint=first_alias,
        )
        # #1263 (LDBC SNB IC3 endpoint): detect free-form intermediate MATCH —
        # the trailing MATCH's first alias is NOT in the prefix's carried
        # whole-row set. Treat every carried whole-row alias as non-source so
        # the existing property-carry rewriter materializes them as hidden
        # columns; use ``first_alias`` as the carrier label so downstream
        # ``<carried>.<prop>`` rewrites resolve against the trailing-MATCH row
        # table at runtime (the runtime broadcasts the hidden columns onto
        # every base node, so any alias the trailing MATCH binds carries them).
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
                merged = set(non_source_carried_props_map.get(alias_name, ()))
                merged.update(props)
                non_source_carried_props_map[alias_name] = tuple(sorted(merged))
            # #1275: free-form + carried-property bridge refs are admitted via
            # `multi_alias_carries` pre-rewrite and downstream hidden-column
            # expression rewrites.
            # Non-source aliases that survived the prefix rewrite (i.e. had no
            # property refs and no bare refs) are simply unused; safe to admit.
            # `multi_alias_carries` (computed before the prefix compile) holds
            # the {alias: props} for the rewritten ones — used below to
            # populate ReentryPlan.aliases and rewrite trailing PropertyAccessExpr.
        for alias_name, props in multi_alias_carries.items():
            merged = set(non_source_carried_props_map.get(alias_name, ()))
            merged.update(props)
            non_source_carried_props_map[alias_name] = tuple(sorted(merged))
    if not _bounded_reentry_prefix_order_is_safe(prefix_stage=prefix_stage, query=query, params=params):
        raise _unsupported(
            "Cypher MATCH after WITH requires bounded literal LIMIT (and no SKIP) to preserve prefix WITH row ordering across MATCH re-entry when the trailing query has no ORDER BY",
            field="with.order_by",
            value=(
                [item.expression.text for item in prefix_stage.order_by.items]
                if prefix_stage.order_by is not None
                else None
            ),
            line=prefix_stage.order_by.span.line if prefix_stage.order_by is not None else prefix_stage.span.line,
            column=prefix_stage.order_by.span.column if prefix_stage.order_by is not None else prefix_stage.span.column,
        )
    if prefix_projection_table is not None and prefix_projection_table != "nodes":
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently supports node re-entry only",
            field="with",
            value=prefix_projection_table,
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
        raise _unsupported_at_span(
            "Cypher MATCH after WITH currently requires the trailing MATCH to start from a named node alias",
            field="match",
            value=first_alias,
            span=reentry_match.span,
        )

    hidden_columns = tuple(_reentry_hidden_column_name(output_name) for output_name in carry_columns)

    # Build the explicit ReentryPlan contract. Scalar-only prefixes carry
    # scalar columns only, while whole-row prefixes (including free-form)
    # record carried aliases explicitly.
    if scalar_only_prefix:
        current_reentry_plan: ReentryPlan = ReentryPlan(
            reentry_alias_name=reentry_alias,
            aliases=(),
            scalar_columns=tuple(carry_columns),
            scalar_only=True,
        )
    else:
        assert prefix_projection is not None
        # #1263 free-form: no carried alias is the trailing-MATCH source, so
        # every entry is recorded with is_reentry_alias=False. The carried-alias
        # path keeps exactly one entry with is_reentry_alias=True.
        current_aliases: List[CarriedAlias] = []
        if not free_form:
            current_aliases.append(
                CarriedAlias(
                    output_name=reentry_alias,
                    table=prefix_projection.table,
                    is_reentry_alias=True,
                )
            )
        for name in non_source_alias_names:
            current_aliases.append(
                CarriedAlias(
                    output_name=name,
                    table=prefix_projection.table,
                    is_reentry_alias=False,
                    carried_properties=non_source_carried_props_map.get(name, ()),
                )
            )
        # Keep secondary aliases recorded on the plan even when demoted to
        # scalar carries by the #1071 rewrite path.
        for alias in demoted_secondary_aliases:
            if alias not in {entry.output_name for entry in current_aliases}:
                current_aliases.append(
                    CarriedAlias(
                        output_name=alias,
                        table=prefix_projection.table,
                        is_reentry_alias=False,
                        carried_properties=non_source_carried_props_map.get(alias, ()),
                    )
                )
        for alias in multi_alias_carries:
            if alias not in {entry.output_name for entry in current_aliases}:
                current_aliases.append(
                    CarriedAlias(
                        output_name=alias,
                        table=prefix_projection.table,
                        is_reentry_alias=False,
                        carried_properties=non_source_carried_props_map.get(alias, ()),
                    )
                )
        current_reentry_plan = ReentryPlan(
            reentry_alias_name=reentry_alias,
            aliases=tuple(current_aliases),
            scalar_columns=tuple(carry_columns),
            scalar_only=False,
            free_form=free_form,
        )

    non_source_carried_props: Optional[Mapping[str, Tuple[str, ...]]] = (
        non_source_carried_props_map if non_source_carried_props_map else None
    )

    def rewrite_expr(expr: ExpressionText, field: str) -> ExpressionText:
        return _rewrite_reentry_expr_to_hidden_properties(
            expr,
            carried_alias=reentry_alias,
            carried_columns=carry_columns,
            field=field,
            non_source_carried_props=non_source_carried_props,
        )

    reentry_where = query.reentry_where
    reentry_return = query.return_
    reentry_order_by = query.order_by
    rewritten_with_stages = remaining_with_stages
    rewritten_reentry_unwinds = query.reentry_unwinds
    remaining_reentry_wheres = query.reentry_wheres[1:] if query.reentry_wheres else ()
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
                reentry_order_by = replace(
                    reentry_order_by,
                    items=tuple(
                        replace(
                            item,
                            expression=rewrite_expr(item.expression, "order_by"),
                        )
                        for item in reentry_order_by.items
                    ),
                )
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
    suffix_query = replace(
        query,
        call=None,
        row_sequence=(),
        graph_bindings=(),
        use=None,
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
        target_extras = target.execution_extras or CompiledCypherExecutionExtras()
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
            post_processing=_post_processing_with(
                result_projection=target_projection,
                # Clear empty_result_row when optional_reentry is set — the
                # reentry null-fill handles missing rows instead.
                empty_result_row=None if is_optional else target.empty_result_row,
                optional_null_fill=target.optional_null_fill,
                optional_projection_row_guard=target.optional_projection_row_guard,
            ),
            execution_extras=_execution_extras_with(
                target,
                connected_optional_match=target_extras.connected_optional_match,
                connected_match_join=target_extras.connected_match_join,
                query_graph=target_extras.query_graph,
                start_nodes_query=prefix_compiled,
                optional_reentry=is_optional,
                reentry_plan=current_reentry_plan,
                logical_plan=target.logical_plan,
                logical_plan_defer_reason=target.logical_plan_defer_reason,
            ),
        )

    return _map_terminal_reentry_query(
        suffix_compiled,
        transform=attach_current_reentry,
    )
