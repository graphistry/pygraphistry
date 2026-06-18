"""Projection-planning helpers extracted from ``cypher.lowering``."""
# mypy: ignore-errors
# ruff: noqa: F821
from __future__ import annotations

from graphistry.compute.gfql.cypher import lowering as _lowering

globals().update(vars(_lowering))

def _split_qualified_name(expr: str, *, line: int, column: int) -> Tuple[str, Optional[str]]:
    parts = expr.split(".")
    if len(parts) <= 2:
        return parts[0], parts[1] if len(parts) == 2 else None
    raise _unsupported(
        "Only simple aliases and alias.property expressions are supported in Cypher RETURN/ORDER BY",
        field="expression",
        value=expr,
        line=line,
        column=column,
    )


def _qualified_ref_from_node(
    node: ExprNode,
    *,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    field: str,
    value: str,
    line: int,
    column: int,
) -> Optional[Tuple[str, Optional[str]]]:
    if isinstance(node, Identifier):
        try:
            return _split_qualified_name(node.name, line=line, column=column)
        except GFQLValidationError:
            return None
    if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
        try:
            alias_name, prop = _split_qualified_name(node.value.name, line=line, column=column)
        except GFQLValidationError:
            return None
        if prop is not None:
            return None
        return alias_name, node.property
    if isinstance(node, FunctionCall) and node.name == "type":
        if len(node.args) != 1 or not isinstance(node.args[0], Identifier):
            return None
        alias_name, prop = _split_qualified_name(node.args[0].name, line=line, column=column)
        if prop is not None:
            return None
        target = (alias_targets or {}).get(alias_name)
        if target is None:
            return None
        if isinstance(target, ASTEdge):
            return alias_name, "type"
        raise _unsupported(
            "type(...) is only supported for relationship aliases in this phase",
            field=field,
            value=value,
            line=line,
            column=column,
        )
    return None


def _projection_ref_from_expr(
    expr: str,
    *,
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]] = None,
    field: str,
    line: int,
    column: int,
) -> Tuple[str, Optional[str]]:
    node = _parse_row_expr(expr, params=params, field=field, line=line, column=column)
    ref = _qualified_ref_from_node(
        node,
        alias_targets=alias_targets,
        field=field,
        value=expr,
        line=line,
        column=column,
    )
    if ref is not None:
        return ref
    raise _unsupported(
        "Only simple aliases, alias.property, and type(rel_alias) expressions are supported in Cypher RETURN/ORDER BY",
        field=field,
        value=expr,
        line=line,
        column=column,
    )


def _reject_duplicate_alias_row_refs(
    query: CypherQuery,
    *,
    alias_targets: Mapping[str, ASTObject],
    duplicated_aliases: Set[str],
    params: Optional[Mapping[str, Any]],
) -> None:
    if not duplicated_aliases:
        return

    def _check(expr_text: str, *, field: str, line: int, column: int) -> None:
        refs = _expr_match_aliases(
            expr_text,
            alias_targets=alias_targets,
            params=params,
            field=field,
            line=line,
            column=column,
        )
        if refs & duplicated_aliases:
            raise _unsupported(
                "Cypher row projection from repeated MATCH aliases is not yet supported in the local compiler",
                field=field,
                value=expr_text,
                line=line,
                column=column,
            )

    for unwind_clause in query.unwinds:
        _check(
            unwind_clause.expression.text,
            field="unwind",
            line=unwind_clause.span.line,
            column=unwind_clause.span.column,
        )
    for item in query.return_.items:
        _check(
            item.expression.text,
            field=query.return_.kind,
            line=item.span.line,
            column=item.span.column,
        )
    if query.order_by is not None:
        for order_item in query.order_by.items:
            _check(
                order_item.expression.text,
                field="order_by",
                line=order_item.span.line,
                column=order_item.span.column,
            )


def _build_projection_plan(
    clause: ReturnClause,
    *,
    alias_targets: Dict[str, ASTObject],
    active_alias: Optional[str] = None,
    projected_columns: Optional[Mapping[str, _StageColumnBinding]] = None,
    params: Optional[Mapping[str, Any]] = None,
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> _ProjectionPlan:
    source_alias: Optional[str] = None
    all_source_aliases: Optional[Set[str]] = None
    whole_row_output_names: List[str] = []
    whole_row_sources: Dict[str, str] = {}
    projection_items: List[Tuple[str, Any]] = []
    projection_columns: List[ResultProjectionColumn] = []
    available_columns: Set[str] = set()
    projected_property_outputs: Dict[str, str] = {}
    output_to_source_property: Dict[str, str] = {}
    output_to_expr_source: Dict[str, str] = {}

    for item in clause.items:
        binding: Optional[_StageColumnBinding] = None
        projected_expr_binding = False
        simple_ref = True
        if item.expression.text == "*":
            if len(alias_targets) > 1:
                raise _unsupported(
                    "Cypher RETURN * currently requires a single MATCH alias in the local compiler",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
            if active_alias is not None:
                alias_name = active_alias
            elif len(alias_targets) == 1:
                alias_name = next(iter(alias_targets.keys()))
            else:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher RETURN/WITH * currently requires a single active alias in the local compiler",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    suggestion="Use a single MATCH alias or project explicit columns.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            prop = None
        else:
            node = _parse_row_expr(
                item.expression.text,
                params=params,
                field=f"{clause.kind}.items",
                line=item.span.line,
                column=item.span.column,
            )
            ref = _qualified_ref_from_node(
                node,
                alias_targets=alias_targets,
                field=f"{clause.kind}.items",
                value=item.expression.text,
                line=item.span.line,
                column=item.span.column,
            )
            if ref is None:
                simple_ref = False
                unsupported_ref = _unsupported(
                    "Only simple aliases, alias.property, and type(rel_alias) expressions are supported in Cypher RETURN/ORDER BY",
                    field=f"{clause.kind}.items",
                    value=item.expression.text,
                    line=item.span.line,
                    column=item.span.column,
                )
                aliases = sorted(
                    _expr_match_aliases(
                        item.expression.text,
                        alias_targets=alias_targets,
                        params=params,
                        field=f"{clause.kind}.items",
                        line=item.span.line,
                        column=item.span.column,
                    )
                )
                if len(aliases) == 1:
                    alias_name = aliases[0]
                    prop = None
                elif len(aliases) == 0:
                    if active_alias is not None:
                        alias_name = active_alias
                        prop = None
                    elif len(alias_targets) == 1:
                        alias_name = next(iter(alias_targets.keys()))
                        prop = None
                    else:
                        raise unsupported_ref
                else:
                    raise unsupported_ref
            else:
                alias_name, prop = ref
        if alias_name not in alias_targets and projected_columns is not None:
            binding = projected_columns.get(alias_name)
            if binding is not None:
                if prop is None and binding.kind == "property":
                    if active_alias is None:
                        raise _unsupported(
                            "Projected Cypher column references require an active MATCH alias in this phase",
                            field=f"{clause.kind}.items",
                            value=item.expression.text,
                            line=item.span.line,
                            column=item.span.column,
                        )
                    alias_name = active_alias
                    simple_ref = True
                    prop = binding.source_name
                else:
                    simple_ref = False
                    projected_expr_binding = True
        if alias_name not in alias_targets:
            if projected_expr_binding:
                alias_name = active_alias or alias_name
            else:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    f"Unknown Cypher alias '{alias_name}' in {clause.kind.upper()} clause",
                    field=f"{clause.kind}.alias",
                    value=alias_name,
                    suggestion="Reference an alias declared in the MATCH pattern.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
        if source_alias is None:
            source_alias = alias_name
        elif source_alias != alias_name:
            if all_source_aliases is None:
                all_source_aliases = {source_alias}
            all_source_aliases.add(alias_name)
        if item.expression.text == "*" or (simple_ref and prop is None):
            output_name = item.alias or alias_name
            if output_name in available_columns or output_name in whole_row_output_names:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Duplicate Cypher projection names are not yet supported in local lowering",
                    field=f"{clause.kind}.items",
                    value=output_name,
                    suggestion="Use distinct output names in RETURN/WITH.",
                    line=item.span.line,
                    column=item.span.column,
                    language="cypher",
                )
            whole_row_output_names.append(output_name)
            whole_row_sources[output_name] = alias_name
            continue
        output_name = item.alias or item.expression.text
        if output_name in available_columns or output_name in whole_row_output_names:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Duplicate Cypher projection names are not yet supported in local lowering",
                field=f"{clause.kind}.items",
                value=output_name,
                suggestion="Use distinct output names in RETURN/WITH.",
                line=item.span.line,
                column=item.span.column,
                language="cypher",
            )
        row_expr = _rewrite_expr_to_projected_sources(
            item.expression,
            projected_columns=projected_columns,
            params=params,
            alias_targets=alias_targets,
            field=f"{clause.kind}.items",
        )
        runtime_expr = (
            f"{alias_name}.{prop}"
            if simple_ref and prop is not None
            else (
                binding.source_name
                if binding is not None and binding.kind == "expr" and prop is None
                else _row_expr_arg(
                    row_expr,
                    params=params,
                    alias_targets=alias_targets,
                    field=f"{clause.kind}.items",
                )
            )
        )
        projection_items.append((output_name, runtime_expr))
        available_columns.add(output_name)
        if simple_ref and prop is not None:
            projected_property_outputs.setdefault(prop, output_name)
            source_property_name = prop if alias_name == source_alias else f"{alias_name}.{prop}"
            output_to_source_property[output_name] = source_property_name
            projection_columns.append(
                ResultProjectionColumn(
                    output_name=output_name,
                    kind="property",
                    source_name=source_property_name,
                )
            )
        else:
            if isinstance(runtime_expr, str):
                output_to_expr_source[output_name] = runtime_expr
            projection_columns.append(
                ResultProjectionColumn(
                    output_name=output_name,
                    kind="expr",
                    source_name=runtime_expr if isinstance(runtime_expr, str) else item.expression.text,
                )
            )

    if source_alias is None:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Cypher {clause.kind.upper()} clause must project at least one supported expression",
            field=clause.kind,
            value=None,
            suggestion="Project a match alias or alias.property expression.",
            line=clause.span.line,
            column=clause.span.column,
            language="cypher",
        )
    if whole_row_output_names:
        available_columns = set()
    table = _alias_table(
        alias_targets[source_alias],
        alias=source_alias,
        line=clause.span.line,
        column=clause.span.column,
        semantic_entity_kinds=semantic_entity_kinds,
    )
    return _ProjectionPlan(
        source_alias=source_alias,
        table=table,
        whole_row_output_names=whole_row_output_names,
        whole_row_sources=whole_row_sources,
        clause_kind=clause.kind,
        projection_items=projection_items,
        projection_columns=projection_columns,
        available_columns=available_columns,
        projected_property_outputs=projected_property_outputs,
        output_to_source_property=output_to_source_property,
        output_to_expr_source=output_to_expr_source,
        all_source_aliases=all_source_aliases,
    )


def _can_lower_multi_alias_projection_bindings(
    plan: _ProjectionPlan,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> bool:
    all_refs = (plan.all_source_aliases or set()) | {plan.source_alias}
    all_are_edges = all(isinstance(alias_targets.get(alias_name), ASTEdge) for alias_name in all_refs)
    has_non_scalar = bool(plan.whole_row_output_names) or bool(plan.output_to_expr_source)
    if not has_non_scalar:
        return not all_are_edges
    if not plan.whole_row_output_names:
        return False
    source_target = alias_targets.get(plan.source_alias)
    if source_target is None:
        return False
    source_is_edge = isinstance(source_target, ASTEdge)
    whole_row_sources = {
        plan.whole_row_sources.get(output_name, plan.source_alias)
        for output_name in plan.whole_row_output_names
    }
    if any(alias_name not in alias_targets for alias_name in whole_row_sources):
        return False
    if any(isinstance(alias_targets.get(alias_name), ASTEdge) != source_is_edge for alias_name in whole_row_sources):
        return False
    simple_qualified_ref = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
    for source_name in plan.output_to_expr_source.values():
        match = simple_qualified_ref.fullmatch(source_name)
        if match is None:
            return False
        alias_name = source_name.split(".", 1)[0]
        expr_target = alias_targets.get(alias_name)
        if expr_target is None:
            return False
        if isinstance(expr_target, ASTEdge) != source_is_edge:
            return False
    return True


def _result_projection_plan(
    plan: _ProjectionPlan,
    *,
    alias_targets: Mapping[str, ASTObject],
) -> Optional[ResultProjectionPlan]:
    if not plan.whole_row_output_names:
        return None
    columns = tuple(
        ResultProjectionColumn(
            output_name=output_name,
            kind="whole_row",
            source_name=plan.whole_row_sources.get(output_name, plan.source_alias),
        )
        for output_name in plan.whole_row_output_names
    ) + tuple(plan.projection_columns)
    return ResultProjectionPlan(
        alias=plan.source_alias,
        table=cast(Literal["nodes", "edges"], plan.table),
        columns=columns,
        exclude_columns=tuple(sorted(alias_targets.keys())),
    )


def _optional_null_fill_projection_value(
    *,
    expression_text: str,
    optional_aliases: AbstractSet[str],
    alias_targets: Mapping[str, ASTObject],
    params: Optional[Mapping[str, Any]],
    field: str,
    line: int,
    column: int,
) -> Optional[Any]:
    node = _parse_row_expr(
        expression_text,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    if not isinstance(node, IsNullOp):
        return None

    operand_text = _render_expr_node(node.value)
    operand_refs = _expr_match_aliases(
        operand_text,
        alias_targets=alias_targets,
        params=params,
        field=field,
        line=line,
        column=column,
    )
    if operand_refs and operand_refs <= optional_aliases:
        return False if node.negated else True
    if not operand_refs and isinstance(node.value, Literal):
        try:
            is_null = bool(pd.isna(node.value.value))
        except Exception:
            is_null = node.value.value is None
        return (not is_null) if node.negated else is_null
    return None


def _empty_optional_projection_row(
    plan: _ProjectionPlan,
    *,
    query: Optional[CypherQuery] = None,
    optional_aliases: Optional[AbstractSet[str]] = None,
    alias_targets: Optional[Mapping[str, ASTObject]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {name: None for name in plan.whole_row_output_names}
    if query is None or optional_aliases is None or alias_targets is None:
        for column in plan.projection_columns:
            out[column.output_name] = None
        return out
    item_expr_by_alias: Dict[str, str] = {
        (item.alias or item.expression.text): item.expression.text
        for item in query.return_.items
    }
    for column in plan.projection_columns:
        default_value: Any = None
        expr_text = item_expr_by_alias.get(column.output_name)
        if expr_text is not None:
            inferred = _optional_null_fill_projection_value(
                expression_text=expr_text,
                optional_aliases=optional_aliases,
                alias_targets=alias_targets,
                params=params,
                field=query.return_.kind,
                line=query.return_.span.line,
                column=query.return_.span.column,
            )
            if inferred is not None:
                default_value = inferred
        out[column.output_name] = default_value
    return out


def _eligible_optional_projection_query(query: CypherQuery) -> bool:
    return (
        len(query.matches) == 2
        and not query.matches[0].optional
        and query.matches[1].optional
        and query.where is None
        and not query.with_stages
        and not query.unwinds
        and query.order_by is None
        and query.skip is None
        and query.limit is None
    )


def _optional_null_fill_plan(
    query: CypherQuery,
    *,
    lowered: LoweredCypherMatch,
    alias_targets: Mapping[str, ASTObject],
    plan: _ProjectionPlan,
    params: Optional[Mapping[str, Any]],
    bound_visible_aliases: AbstractSet[str] = frozenset(),
    semantic_entity_kinds: Optional[Mapping[str, Literal["node", "edge", "scalar"]]] = None,
) -> Optional[OptionalNullFillPlan]:
    if not _eligible_optional_projection_query(query):
        return None

    seed_alias = _single_node_seed_alias(query.matches[0])
    if seed_alias is None:
        return None

    optional_aliases = _match_clause_aliases(query.matches[1]) - {seed_alias}
    if not optional_aliases:
        return None

    referenced: Set[str] = set()
    for item in query.return_.items:
        referenced.update(
            _expr_match_aliases(
                item.expression.text,
                alias_targets=alias_targets,
                params=params,
                field=query.return_.kind,
                line=item.span.line,
                column=item.span.column,
            )
        )
    if not referenced or not referenced <= optional_aliases:
        return None

    alignment_output_name = "__cypher_optional_seed__"
    alignment_table = cast(
        Literal["nodes", "edges"],
        _alias_table(
            alias_targets[seed_alias],
            alias=seed_alias,
            line=query.return_.span.line,
            column=query.return_.span.column,
            semantic_entity_kinds=semantic_entity_kinds,
        ),
    )
    alignment_plan = _ProjectionPlan(
        source_alias=seed_alias,
        table=alignment_table,
        whole_row_output_names=[alignment_output_name],
        whole_row_sources={alignment_output_name: seed_alias},
        clause_kind="return",
        projection_items=[],
        projection_columns=[],
        available_columns=set(),
        projected_property_outputs={},
        output_to_source_property={},
        output_to_expr_source={},
    )
    alignment_projection = _result_projection_plan(alignment_plan, alias_targets=alias_targets)
    if alignment_projection is None:
        raise _unsupported(
            "Cypher OPTIONAL MATCH null-row alignment could not construct a seed-row projection",
            field="return",
            value=[item.expression.text for item in query.return_.items],
            line=query.return_.span.line,
            column=query.return_.span.column,
        )

    return OptionalNullFillPlan(
        base_chain=Chain(lower_match_clause(query.matches[0], params=params)),
        null_row=_empty_optional_projection_row(
            plan,
            query=query,
            optional_aliases=optional_aliases,
            alias_targets=alias_targets,
            params=params,
        ),
        alignment_chain=Chain(
            _lower_projection_chain(
                query,
                lowered,
                params=params,
                plan=alignment_plan,
                bound_visible_aliases=bound_visible_aliases,
                semantic_entity_kinds=semantic_entity_kinds,
            )
        ),
        alignment_projection=alignment_projection,
        alignment_output_name=alignment_output_name,
    )


def _optional_projection_row_guard_plan(
    query: CypherQuery,
    *,
    params: Optional[Mapping[str, Any]],
) -> Optional[OptionalProjectionRowGuardPlan]:
    if not _eligible_optional_projection_query(query):
        return None
    base_clause = query.matches[0]
    if len(base_clause.patterns) == 1:
        return OptionalProjectionRowGuardPlan(
            base_chains=(Chain(lower_match_clause(base_clause, params=params)),)
        )
    base_chains: List[Chain] = []
    for idx, pattern in enumerate(base_clause.patterns):
        if len(pattern) != 1 or not isinstance(pattern[0], NodePattern):
            return None
        pattern_clause = MatchClause(
            patterns=(pattern,),
            span=base_clause.span,
            optional=False,
            pattern_aliases=((base_clause.pattern_aliases[idx] if idx < len(base_clause.pattern_aliases) else None),),
            pattern_alias_kinds=(_match_pattern_alias_kinds(base_clause)[idx],),
        )
        base_chains.append(Chain(lower_match_clause(pattern_clause, params=params)))
    return OptionalProjectionRowGuardPlan(base_chains=tuple(base_chains))


def _plan_with_visible_projected_columns(
    plan: _ProjectionPlan,
    projected_columns: Mapping[str, _StageColumnBinding],
) -> _ProjectionPlan:
    if not projected_columns:
        return plan
    output_to_source_property = dict(plan.output_to_source_property)
    output_to_expr_source = dict(plan.output_to_expr_source)
    available_columns = set(plan.available_columns)
    for name, binding in projected_columns.items():
        available_columns.add(name)
        if name in output_to_source_property or name in output_to_expr_source:
            continue
        if binding.kind == "property":
            output_to_source_property[name] = binding.source_name
        else:
            output_to_expr_source[name] = binding.source_name
    return replace(
        plan,
        available_columns=available_columns,
        output_to_source_property=output_to_source_property,
        output_to_expr_source=output_to_expr_source,
    )


def _projection_output_names(plan: _ProjectionPlan) -> Set[str]:
    return {name for name, _ in plan.projection_items} | set(plan.whole_row_output_names)
