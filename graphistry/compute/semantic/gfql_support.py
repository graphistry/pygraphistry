"""GFQL ``call()`` entries for semantic operators: parameter validators and schema effects.

Validators and schema-effect callables never raise; exactly-one and template-shape problems surface from
the operator itself, so only missing columns become schema errors.
"""
from typing import Callable, Dict, List, Mapping, Optional

from graphistry.compute.semantic.rules import RULES, Rule, decision_columns, evidence_columns
from graphistry.models.compute.features import GraphEntityKind


SEMANTIC_CALLS = frozenset({"sem_decide"})
SEM_DECIDE_PARAMS = frozenset(RULES)
SEM_DECIDE_VALIDATORS: Dict[str, Rule] = dict(RULES)


def typed_output_columns(params: Mapping[str, object]) -> Optional[Dict[str, str]]:
    """Output column -> scalar type name (``bool``, ``string``, ``float64``) of a ``sem_decide`` call."""
    questions = params.get("questions")
    if not isinstance(questions, list) or not all(
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


def _added(params: Mapping[str, object]) -> List[str]:
    return list(typed_output_columns(params) or {})


def _for_kind(kind: GraphEntityKind, compute: Callable[[Mapping[str, object]], List[str]]
              ) -> Callable[[Dict[str, object]], List[str]]:
    def effect(params: Dict[str, object]) -> List[str]:
        return compute(params) if params.get("kind", "nodes") == kind else []
    return effect


SEM_DECIDE_SCHEMA_EFFECTS = {
    "adds_node_cols": _for_kind("nodes", _added),
    "adds_edge_cols": _for_kind("edges", _added),
    "requires_node_cols": _for_kind("nodes", _required),
    "requires_edge_cols": _for_kind("edges", _required),
}
