"""GFQL ``call()`` support for ``sem_decide`` and ``sem_map``: parameter validators and schema effects.

Validators return ``True`` or a message string (the safelist turns a message into E201); they never
raise. Schema-effect callables never raise either, so cross-parameter and template-shape problems
surface from the operator itself.
"""
import string
from typing import Callable, Dict, List, Mapping, Optional, Union

from graphistry.compute.gfql.identifiers import validate_column_name
from graphistry.compute.semantic.contract import (
    DECISION_SUFFIXES,
    GENERATION_SUFFIXES,
    DecisionQuestion,
    finite_float,
)
from graphistry.models.compute.features import GraphEntityKind


Validator = Callable[[object], Union[bool, str]]

SEM_COMMON_PARAMS = frozenset({
    "provider", "task_id", "kind", "evidence_cols", "template", "id_col", "task_version", "batch_size",
    "max_concurrency", "deadline_s", "budget", "max_rows", "column_max_chars", "overwrite",
})
SEM_DECIDE_PARAMS = SEM_COMMON_PARAMS | {"questions", "min_p"}
SEM_MAP_PARAMS = SEM_COMMON_PARAMS | {"instruction", "out_col", "max_output_tokens"}


def non_empty_str(value: object) -> Union[bool, str]:
    return True if isinstance(value, str) and value != "" else "must be a non-empty string"


def positive_int(value: object) -> Union[bool, str]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return True
    return "must be an int >= 1"


def positive_number(value: object) -> Union[bool, str]:
    number = finite_float(value)
    return True if number is not None and number > 0 else "must be a finite number > 0"


def probability(value: object) -> Union[bool, str]:
    number = finite_float(value)
    return True if number is not None and 0 <= number <= 1 else "must be a number in [0, 1]"


def boolean(value: object) -> Union[bool, str]:
    return True if isinstance(value, bool) else "must be a bool"


def table_kind(value: object) -> Union[bool, str]:
    return True if value in ("nodes", "edges") else "must be 'nodes' or 'edges'"


def evidence_column_list(value: object) -> Union[bool, str]:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list of column names"
    if not all(isinstance(col, str) and col != "" for col in value):
        return "must be a non-empty list of column names"
    if len(set(value)) != len(value):
        return "column names must be unique"
    return True


def budget_spec(value: object) -> Union[bool, str]:
    if not isinstance(value, dict):
        return "must be a dict with max_cost_usd and/or max_requests"
    unknown = set(value) - {"max_cost_usd", "max_requests"}
    if unknown:
        return f"unknown keys {sorted(map(str, unknown))}"
    cost = value.get("max_cost_usd")
    cost_number = finite_float(cost)
    if cost is not None and (cost_number is None or cost_number < 0):
        return "max_cost_usd must be a finite number >= 0"
    requests = value.get("max_requests")
    if requests is not None and positive_int(requests) is not True:
        return "max_requests must be an int >= 1"
    return True


def question_list(value: object) -> Union[bool, str]:
    """Questions as JSON: a non-empty list of dicts that pass the contract's construction checks."""
    if not isinstance(value, list) or not value:
        return "must be a non-empty list of question dicts"
    ids: List[str] = []
    for spec in value:
        if not isinstance(spec, dict):
            return "each question must be a dict with id, kind, text and optional options or levels"
        try:
            question = DecisionQuestion.from_mapping(spec)
            validate_column_name(question.id, "Question id")
        except ValueError as error:
            return str(error)
        ids.append(question.id)
    if len(set(ids)) != len(ids):
        return "question ids must be unique"
    return True


def output_column(value: object) -> Union[bool, str]:
    if not isinstance(value, str) or value == "":
        return "must be a non-empty string"
    try:
        validate_column_name(value, "out_col")
    except ValueError as error:
        return str(error)
    return True


COMMON_VALIDATORS: Dict[str, Validator] = {
    "provider": non_empty_str,
    "task_id": non_empty_str,
    "task_version": non_empty_str,
    "kind": table_kind,
    "evidence_cols": evidence_column_list,
    "template": non_empty_str,
    "id_col": non_empty_str,
    "batch_size": positive_int,
    "max_concurrency": positive_int,
    "deadline_s": positive_number,
    "budget": budget_spec,
    "max_rows": positive_int,
    "column_max_chars": positive_int,
    "overwrite": boolean,
}
SEM_DECIDE_VALIDATORS: Dict[str, Validator] = dict(COMMON_VALIDATORS, questions=question_list, min_p=probability)
SEM_MAP_VALIDATORS: Dict[str, Validator] = dict(
    COMMON_VALIDATORS, instruction=non_empty_str, out_col=output_column, max_output_tokens=positive_int
)


def _kind(params: Mapping[str, object]) -> str:
    kind = params.get("kind", "nodes")
    return kind if isinstance(kind, str) else "nodes"


def _bare_template_fields(template: str) -> List[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return []
    fields: List[str] = []
    for _literal, name, spec, conversion in parsed:
        if name is None:
            continue
        if name == "" or name.isdigit() or any(ch in name for ch in ".[]") or spec or conversion is not None:
            return []
        if name not in fields:
            fields.append(name)
    return fields


def _evidence_reads(params: Mapping[str, object]) -> List[str]:
    cols = params.get("evidence_cols")
    template = params.get("template")
    if (cols is None) == (template is None):
        return []
    if isinstance(cols, list):
        return [col for col in cols if isinstance(col, str)]
    if isinstance(template, str):
        return _bare_template_fields(template)
    return []


def _question_ids(params: Mapping[str, object]) -> List[str]:
    questions = params.get("questions")
    if not isinstance(questions, list):
        return []
    return [spec["id"] for spec in questions if isinstance(spec, dict) and isinstance(spec.get("id"), str)]


def decide_output_columns(params: Mapping[str, object]) -> List[str]:
    return [name for qid in _question_ids(params) for name in [qid] + [qid + s for s in DECISION_SUFFIXES]]


def map_output_columns(params: Mapping[str, object]) -> List[str]:
    out_col = params.get("out_col")
    if not isinstance(out_col, str):
        return []
    return [out_col] + [out_col + suffix for suffix in GENERATION_SUFFIXES]


def _required(params: Mapping[str, object]) -> List[str]:
    id_col = params.get("id_col")
    cols = _evidence_reads(params)
    if isinstance(id_col, str) and id_col not in cols:
        cols = cols + [id_col]
    return cols


def _for_kind(kind: GraphEntityKind, compute: Callable[[Mapping[str, object]], List[str]]
              ) -> Callable[[Dict[str, object]], List[str]]:
    def effect(params: Dict[str, object]) -> List[str]:
        return compute(params) if _kind(params) == kind else []
    return effect


SEM_DECIDE_SCHEMA_EFFECTS = {
    "adds_node_cols": _for_kind("nodes", decide_output_columns),
    "adds_edge_cols": _for_kind("edges", decide_output_columns),
    "requires_node_cols": _for_kind("nodes", _required),
    "requires_edge_cols": _for_kind("edges", _required),
}
SEM_MAP_SCHEMA_EFFECTS = {
    "adds_node_cols": _for_kind("nodes", map_output_columns),
    "adds_edge_cols": _for_kind("edges", map_output_columns),
    "requires_node_cols": _for_kind("nodes", _required),
    "requires_edge_cols": _for_kind("edges", _required),
}


def typed_output_columns(function: str, params: Mapping[str, object]) -> Optional[Dict[str, str]]:
    """Output column -> scalar type name (``bool``, ``string``, ``float64``) for a semantic call."""
    if function == "sem_decide":
        questions = params.get("questions")
        if not isinstance(questions, list):
            return None
        out: Dict[str, str] = {}
        for spec in questions:
            if not isinstance(spec, dict) or not isinstance(spec.get("id"), str):
                return None
            qid = spec["id"]
            out[qid] = "bool" if spec.get("kind") == "predicate" else "string"
            for suffix in DECISION_SUFFIXES:
                out[qid + suffix] = "float64" if suffix == "__p" else "string"
        return out
    if function == "sem_map":
        names = map_output_columns(params)
        return {name: "string" for name in names} if names else None
    return None

