"""Parameter rules, evidence parsing and output columns of ``sem_decide``, shared by the kernel and GFQL.

A rule returns ``True`` or a problem message; the kernel raises ``ValueError`` on a message and the GFQL
safelist reports it as E201.
"""
import string
from collections import Counter
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from graphistry.compute.gfql.identifiers import validate_column_name
from graphistry.compute.semantic.contract import Budget, DecisionQuestion, canonicalize, finite_float


Rule = Callable[[object], Union[bool, str]]
QuestionLike = Union[DecisionQuestion, Mapping[str, object]]
BudgetSpec = Union[Budget, Mapping[str, object], None]

DECISION_COLUMN_TYPES: Tuple[Tuple[str, str], ...] = (
    ("", "answer"), ("__status", "string"), ("__p", "float64"), ("__dist", "string"),
    ("__probability_kind", "string"), ("__error_code", "string"), ("__meta", "string"),
)


def non_empty_str(value: object) -> Union[bool, str]:
    return True if isinstance(value, str) and value != "" else "must be a non-empty string"


def positive_int(value: object) -> Union[bool, str]:
    number = canonicalize(value)
    return True if isinstance(number, int) and not isinstance(number, bool) and number >= 1 else "must be an int >= 1"


def positive_number(value: object) -> Union[bool, str]:
    number = finite_float(canonicalize(value))
    return True if number is not None and number > 0 else "must be a finite number > 0"


def probability(value: object) -> Union[bool, str]:
    number = finite_float(canonicalize(value))
    return True if number is not None and 0 <= number <= 1 else "must be a number in [0, 1]"


def boolean(value: object) -> Union[bool, str]:
    return True if isinstance(value, bool) else "must be a bool"


def table_kind(value: object) -> Union[bool, str]:
    return True if isinstance(value, str) and value in ("nodes", "edges") else "must be 'nodes' or 'edges'"


def column_list(value: object) -> Union[bool, str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value \
            or not all(isinstance(col, str) and col != "" for col in value):
        return "must be a non-empty list of column names"
    return True if len(set(value)) == len(value) else "column names must be unique"


def _message_of(build: Callable[[object], object]) -> Rule:
    def rule(value: object) -> Union[bool, str]:
        try:
            build(value)
        except ValueError as error:
            return str(error)
        return True
    return rule


def coerce_questions(questions: object) -> Tuple[DecisionQuestion, ...]:
    """Questions as ``DecisionQuestion`` objects; dicts use the keys id, kind, text, options, levels."""
    if isinstance(questions, (str, bytes, Mapping)) or not isinstance(questions, (list, tuple)) or not questions:
        raise ValueError("questions must be a non-empty list of questions")
    coerced: List[DecisionQuestion] = []
    for spec in questions:
        question = DecisionQuestion.from_mapping(spec) if isinstance(spec, Mapping) else spec
        if not isinstance(question, DecisionQuestion):
            raise ValueError(f"each question must be a DecisionQuestion or a dict, got {type(spec).__name__}")
        coerced.append(question)
    ids = [question.id for question in coerced]
    if len(set(ids)) != len(ids):
        raise ValueError(f"question ids must be unique, got {ids}")
    for question_id in ids:
        validate_column_name(question_id, "Question id")
    counts = Counter(question_id + suffix for question_id in ids for suffix, _ in DECISION_COLUMN_TYPES)
    collisions = sorted(name for name, count in counts.items() if count > 1)
    if collisions:
        raise ValueError(f"derived output column names collide: {collisions}")
    return tuple(coerced)


def coerce_budget(budget: BudgetSpec) -> Optional[Budget]:
    if budget is None or isinstance(budget, Budget):
        return budget
    if not isinstance(budget, Mapping):
        raise ValueError(f"budget must be a Budget or a dict, got {type(budget).__name__}")
    return Budget.from_mapping(budget)


def _json_questions(value: object) -> object:
    if not isinstance(value, list) or not all(isinstance(spec, dict) for spec in value):
        raise ValueError("must be a non-empty list of question dicts with id, kind, text, options or levels")
    return coerce_questions(value)


def _json_budget(value: object) -> object:
    if not isinstance(value, dict):
        raise ValueError("must be a dict with max_cost_usd and/or max_requests")
    return Budget.from_mapping(value)


RULES: Dict[str, Rule] = {
    "questions": _message_of(_json_questions),
    "provider": non_empty_str,
    "task_id": non_empty_str,
    "task_version": non_empty_str,
    "kind": table_kind,
    "evidence_cols": column_list,
    "template": non_empty_str,
    "id_col": non_empty_str,
    "batch_size": positive_int,
    "max_concurrency": positive_int,
    "deadline_s": positive_number,
    "budget": _message_of(_json_budget),
    "max_rows": positive_int,
    "min_p": probability,
    "column_max_chars": positive_int,
    "overwrite": boolean,
}


def check_params(required: Mapping[str, object], optional: Mapping[str, object]) -> None:
    """Raise ``ValueError`` for the first parameter that breaks its rule; None leaves an optional one unset."""
    for name, value in list(required.items()) + [(name, value) for name, value in optional.items() if value is not None]:
        verdict = RULES[name](value)
        if verdict is not True:
            raise ValueError(f"{name} {verdict}, got {value!r}")


def template_fields(template: object) -> List[str]:
    """Placeholder names of a template; only bare ``{column}`` fields are allowed."""
    if not isinstance(template, str) or template == "":
        raise ValueError("template must be a non-empty string")
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"template is not a valid format string: {error}") from error
    fields: List[str] = []
    for _literal, name, format_spec, conversion in parsed:
        if name is None:
            continue
        if name == "" or name.isdigit() or any(ch in name for ch in ".[]") or format_spec or conversion is not None:
            raise ValueError(f"template field {{{name}}} must be a bare column name, without a conversion or format spec")
        if name not in fields:
            fields.append(name)
    if not fields:
        raise ValueError("template must reference at least one column")
    return fields


def evidence_columns(evidence_cols: object, template: object) -> List[str]:
    """Columns the evidence reads, after the exactly-one-of check."""
    if (evidence_cols is None) == (template is None):
        raise ValueError("pass exactly one of evidence_cols or template")
    if template is not None:
        return template_fields(template)
    check_params(dict(evidence_cols=evidence_cols), {})
    return list(evidence_cols) if isinstance(evidence_cols, (list, tuple)) else []


def decision_columns(questions: Sequence[Tuple[str, object]]) -> Dict[str, str]:
    """Output column -> logical type (``bool``, ``string``, ``float64``) for ``(question id, kind)`` pairs."""
    return {question_id + suffix: ("bool" if kind == "predicate" else "string") if logical == "answer" else logical
            for question_id, kind in questions for suffix, logical in DECISION_COLUMN_TYPES}
