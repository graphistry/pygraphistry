"""GFQL ``call()`` entries for semantic operators: parameter validators and schema effects.

Validators and schema-effect callables never raise; exactly-one and template-shape problems surface from
the operator itself, so only missing columns become schema errors.
"""
from typing import Callable, Dict, List, Mapping, Optional

from graphistry.compute.semantic.rules import (
    SEM_DECIDE_RULES, SEM_MAP_RULES, Rule, decision_columns, evidence_columns, map_columns)
from graphistry.models.compute.features import GraphEntityKind


SEMANTIC_CALLS = frozenset({"sem_decide", "sem_map"})
SEM_DECIDE_PARAMS = frozenset(SEM_DECIDE_RULES)
SEM_DECIDE_VALIDATORS: Dict[str, Rule] = dict(SEM_DECIDE_RULES)
SEM_MAP_PARAMS = frozenset(SEM_MAP_RULES)
SEM_MAP_VALIDATORS: Dict[str, Rule] = dict(SEM_MAP_RULES)


def typed_output_columns(function: str, params: Mapping[str, object]) -> Optional[Dict[str, str]]:
    """Output column -> scalar type name (``bool``, ``string``, ``float64``) for a semantic call."""
    if function == "sem_map":
        out_col = params.get("out_col")
        return map_columns(out_col) if isinstance(out_col, str) else None
    questions = params.get("questions")
    if function != "sem_decide" or not isinstance(questions, list) or not all(
            isinstance(spec, dict) and isinstance(spec.get("id"), str) for spec in questions):
        return None
    return decision_columns([(spec["id"], spec.get("kind")) for spec in questions])


def _evidence_reads(params: Mapping[str, object]) -> List[str]:
    try:
        return evidence_columns(params.get("evidence_cols"), params.get("template"))
    except ValueError:
        return []


def _required(params: Mapping[str, object]) -> List[str]:
    id_col = params.get("id_col")
    cols = _evidence_reads(params)
    return cols + [id_col] if isinstance(id_col, str) and id_col not in cols else cols


def _schema_effects(function: str) -> Dict[str, Callable[[Dict[str, object]], List[str]]]:
    def added(params: Mapping[str, object]) -> List[str]:
        return list(typed_output_columns(function, params) or {})

    return {"adds_node_cols": _for_kind("nodes", added), "adds_edge_cols": _for_kind("edges", added),
            "requires_node_cols": _for_kind("nodes", _required), "requires_edge_cols": _for_kind("edges", _required)}


def _for_kind(kind: GraphEntityKind, compute: Callable[[Mapping[str, object]], List[str]]
              ) -> Callable[[Dict[str, object]], List[str]]:
    def effect(params: Dict[str, object]) -> List[str]:
        return compute(params) if params.get("kind", "nodes") == kind else []
    return effect


SEM_DECIDE_SCHEMA_EFFECTS = _schema_effects("sem_decide")
SEM_MAP_SCHEMA_EFFECTS = _schema_effects("sem_map")
