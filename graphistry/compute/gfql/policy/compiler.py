"""Compiler policy payload helpers."""
from __future__ import annotations

from hashlib import sha256
import re
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

from graphistry.compute.gfql.cypher.ast import (
    CypherGraphQuery,
    CypherQuery,
    CypherUnionQuery,
    ProjectionStage,
    ReturnClause,
    ReturnItem,
)
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.types import EdgeRef, NodeRef, ScalarType

from .types import (
    CompilerAggregateSummary,
    CompilerAliasSummary,
    CompilerPolicySummary,
    CompilerProjectionSummary,
)

CypherAST = Union[CypherQuery, CypherUnionQuery, CypherGraphQuery]

_AGGREGATE_RE = re.compile(
    r"(?is)^(?P<fn>count|sum|min|max|avg|collect)\s*\(\s*(?P<distinct>distinct\s+)?(?P<input>.*?)\s*\)$"
)
_PROPERTY_RE = re.compile(r"^(?P<source>[A-Za-z_][A-Za-z0-9_]*)\.(?P<property>[A-Za-z_][A-Za-z0-9_]*)$")


def build_cypher_postcompile_summary(
    *,
    query_text: str,
    parsed_query: CypherAST,
    params: Optional[Mapping[str, object]] = None,
) -> CompilerPolicySummary:
    """Build the stable summary emitted to the opt-in ``postcompile`` hook."""
    bound = FrontendBinder().bind(parsed_query, PlanContext(), strict_name_resolution=True)
    entity_kinds = {
        name: variable.entity_kind
        for name, variable in bound.semantic_table.variables.items()
    }
    nullable_aliases = {
        name
        for name, variable in bound.semantic_table.variables.items()
        if variable.nullable or bool(variable.null_extended_from)
    }
    for scope in bound.scope_stack:
        for name in sorted(scope.visible_vars):
            logical_type = scope.schema.columns.get(name)
            if logical_type is not None:
                entity_kinds.setdefault(name, _entity_kind_from_logical_type(logical_type))
            nullable_aliases.update(
                name
                for name, logical_type in scope.schema.columns.items()
                if isinstance(logical_type, ScalarType) and logical_type.nullable
            )
    aliases: List[CompilerAliasSummary] = [
        {
            "name": name,
            "kind": kind,
            "nullable": name in nullable_aliases,
        }
        for name, kind in sorted(entity_kinds.items())
    ]

    projections: List[CompilerProjectionSummary] = []
    aggregates: List[CompilerAggregateSummary] = []
    group_keys: List[str] = []
    for clause in _projection_clauses(parsed_query):
        clause_projections, clause_aggregates, clause_group_keys = _summarize_clause(
            clause,
            entity_kinds=entity_kinds,
        )
        projections.extend(clause_projections)
        aggregates.extend(clause_aggregates)
        group_keys.extend(clause_group_keys)

    summary: CompilerPolicySummary = {
        "phase": "postcompile",
        "language": "cypher",
        "query_type": "chain",
        "query_hash": sha256(query_text.encode("utf-8")).hexdigest(),
        "aliases": aliases,
        "projections": projections,
        "group_keys": group_keys,
        "aggregates": aggregates,
    }
    if params:
        # Keep values out of the contract; callers can correlate by name only.
        summary["param_keys"] = sorted(str(key) for key in params.keys())
    return summary


def _entity_kind_from_logical_type(logical_type: object) -> Literal["node", "edge", "scalar"]:
    if isinstance(logical_type, NodeRef):
        return "node"
    if isinstance(logical_type, EdgeRef):
        return "edge"
    return "scalar"


def _projection_clauses(parsed_query: CypherAST) -> Sequence[ReturnClause]:
    if isinstance(parsed_query, CypherUnionQuery):
        clauses: List[ReturnClause] = []
        for branch in parsed_query.branches:
            clauses.extend(_projection_clauses(branch))
        return tuple(clauses)
    if isinstance(parsed_query, CypherGraphQuery):
        return ()
    clauses = [stage.clause for stage in _projection_stages(parsed_query)]
    clauses.append(parsed_query.return_)
    return tuple(clauses)


def _projection_stages(query: CypherQuery) -> Sequence[ProjectionStage]:
    stages: List[ProjectionStage] = list(query.with_stages)
    for item in query.row_sequence:
        if isinstance(item, ProjectionStage):
            stages.append(item)
    return tuple(stages)


def _summarize_clause(
    clause: ReturnClause,
    *,
    entity_kinds: Mapping[str, str],
) -> Tuple[List[CompilerProjectionSummary], List[CompilerAggregateSummary], List[str]]:
    projections: List[CompilerProjectionSummary] = []
    aggregates: List[CompilerAggregateSummary] = []
    aggregate_outputs = set()

    for item in clause.items:
        aggregate = _aggregate_summary(item, clause_kind=clause.kind)
        if aggregate is not None:
            aggregates.append(aggregate)
            aggregate_outputs.add(aggregate["output"])
        projections.append(_projection_summary(item, clause_kind=clause.kind, entity_kinds=entity_kinds))

    group_keys = [
        item.alias or item.expression.text
        for item in clause.items
        if aggregates and (item.alias or item.expression.text) not in aggregate_outputs
    ]
    return projections, aggregates, group_keys


def _aggregate_summary(
    item: ReturnItem,
    *,
    clause_kind: str,
) -> Optional[CompilerAggregateSummary]:
    match = _AGGREGATE_RE.match(item.expression.text)
    if match is None:
        return None
    input_text = match.group("input").strip()
    if input_text == "":
        input_text = "*"
    summary: CompilerAggregateSummary = {
        "clause": cast(Literal["with", "return"], clause_kind),
        "output": item.alias or item.expression.text,
        "fn": match.group("fn").lower(),
        "input": input_text,
        "distinct": bool(match.group("distinct")),
    }
    return summary


def _projection_summary(
    item: ReturnItem,
    *,
    clause_kind: str,
    entity_kinds: Mapping[str, str],
) -> CompilerProjectionSummary:
    expr = item.expression.text
    output = item.alias or expr
    aggregate = _aggregate_summary(item, clause_kind=clause_kind)
    if aggregate is not None:
        return {
            "clause": clause_kind,  # type: ignore[typeddict-item]
            "output": output,
            "expr": expr,
            "expr_kind": "aggregate",
            "source": aggregate["input"],
            "entity_kind": None,
        }
    if expr == "*":
        return {
            "clause": clause_kind,  # type: ignore[typeddict-item]
            "output": output,
            "expr": expr,
            "expr_kind": "wildcard",
            "source": None,
            "entity_kind": None,
        }
    if expr in entity_kinds:
        return {
            "clause": clause_kind,  # type: ignore[typeddict-item]
            "output": output,
            "expr": expr,
            "expr_kind": "alias",
            "source": expr,
            "entity_kind": entity_kinds[expr],  # type: ignore[typeddict-item]
        }
    property_match = _PROPERTY_RE.match(expr)
    if property_match is not None:
        source = property_match.group("source")
        return {
            "clause": clause_kind,  # type: ignore[typeddict-item]
            "output": output,
            "expr": expr,
            "expr_kind": "property",
            "source": source,
            "property": property_match.group("property"),
            "entity_kind": entity_kinds.get(source),  # type: ignore[typeddict-item]
        }
    return {
        "clause": clause_kind,  # type: ignore[typeddict-item]
        "output": output,
        "expr": expr,
        "expr_kind": "expr",
        "source": None,
        "entity_kind": None,
    }
