"""Provider-neutral contract for semantic table operators.

Requests, results, capabilities and the provider protocol shared by ``sem_decide`` and by any host that
plugs in its own model provider. Standard library only: the file loads by path without pandas or graphistry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union, get_args, runtime_checkable


QuestionKind = Literal["predicate", "choice", "score"]
ProbabilityKind = Literal["native_decision", "token_logprob", "verbalized", "none"]
ResultStatus = Literal["ok", "uncertain", "error", "skipped"]
UncertainReason = Literal["refusal", "null_answer", "tie", "min_p"]
SkippedReason = Literal["max_rows", "deadline", "budget", "null_evidence"]
StatusReason = Union[UncertainReason, SkippedReason]
ExtraValue = Union[str, int, float, bool, None]
Evidence = Union[str, Mapping[str, object]]

QUESTION_KINDS: Tuple[QuestionKind, ...] = get_args(QuestionKind)
PROBABILITY_KINDS: Tuple[str, ...] = get_args(ProbabilityKind)
RESULT_STATUSES: Tuple[str, ...] = get_args(ResultStatus)
UNCERTAIN_REASONS: Tuple[str, ...] = get_args(UncertainReason)
SKIPPED_REASONS: Tuple[str, ...] = get_args(SkippedReason)
_REASONS_BY_STATUS: Dict[str, Tuple[str, ...]] = {"uncertain": UNCERTAIN_REASONS, "skipped": SKIPPED_REASONS}
_EMPTY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "error": ("value", "distribution", "confidence"), "skipped": ("value", "distribution", "confidence"),
    "refusal": ("value", "distribution"), "null_answer": ("value", "distribution"), "tie": ("value",)}
DISTRIBUTION_SUM_TOLERANCE = 0.01
QUESTION_KEYS = frozenset({"id", "kind", "text", "options", "levels"})
BUDGET_KEYS = frozenset({"max_requests"})


def _is_member(value: object, allowed: Sequence[str]) -> bool:
    return isinstance(value, str) and value in allowed


def finite_float(value: object) -> Optional[float]:
    """The value as a float if it is a real, finite int or float (bools excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        as_float = float(value)
    except OverflowError:
        return None
    return as_float if math.isfinite(as_float) else None


def _min_int(value: object, field_name: str, minimum: int, optional: bool = True) -> Optional[int]:
    if value is None and optional:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    raise ValueError(f"{field_name} must be an int >= {minimum}, got {value!r}")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _known_keys(spec: Mapping[str, object], allowed: FrozenSet[str], what: str) -> Mapping[str, object]:
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown {what} keys {sorted(map(str, unknown))}; allowed {sorted(allowed)}")
    return spec


def _question_kind(value: object, question_id: object) -> QuestionKind:
    for kind in QUESTION_KINDS:
        if isinstance(value, str) and value == kind:
            return kind
    raise ValueError(f"question {question_id!r} kind must be one of {QUESTION_KINDS}, got {value!r}")


def _labels_tuple(value: object, field_name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple of strings, got {value!r}")
    labels = tuple(value)
    if not all(isinstance(label, str) and label != "" for label in labels):
        raise ValueError(f"{field_name} entries must be non-empty strings, got {labels!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field_name} entries must be unique, got {labels!r}")
    return labels


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do; optional limits are None when the provider sets no limit.

    ``concurrency_limit`` is how many calls one operator run sends at once (1 if not thread-safe; calls a run
    abandoned at its deadline may still be running when the next run starts) and
    ``max_rows_per_request`` how many rows one transport request answers. ``supported_kinds`` None means
    all question kinds. Operators check questions against these limits before any call.
    """

    supports_native_probabilities: bool
    max_questions_per_request: int
    concurrency_limit: int
    max_rows_per_request: Optional[int] = None
    supported_kinds: Optional[Tuple[QuestionKind, ...]] = None
    max_choice_options: Optional[int] = None
    max_levels: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.supports_native_probabilities, bool):
            raise ValueError("supports_native_probabilities must be a bool")
        for field_name, minimum, optional in (("max_questions_per_request", 1, False), ("concurrency_limit", 1, False),
                                              ("max_rows_per_request", 1, True), ("max_choice_options", 2, True),
                                              ("max_levels", 2, True)):
            _min_int(getattr(self, field_name), field_name, minimum, optional)
        if self.supported_kinds is not None:
            kinds = _labels_tuple(self.supported_kinds, "supported_kinds")
            if not kinds or not all(kind in QUESTION_KINDS for kind in kinds):
                raise ValueError(f"supported_kinds must name at least one of {QUESTION_KINDS}, got {kinds!r}")
            object.__setattr__(self, "supported_kinds", kinds)


@dataclass(frozen=True)
class DecisionQuestion:
    """One question asked of every row: ``predicate`` (yes/no), ``choice`` (one of ``options``) or
    ``score`` (one of ``levels``, ordered lowest to highest)."""

    id: str
    kind: QuestionKind
    text: str
    options: Optional[Tuple[str, ...]] = None
    levels: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        _text(self.id, "question id")
        _text(self.text, f"question {self.id!r} text")
        kind = _question_kind(self.kind, self.id)
        object.__setattr__(self, "kind", kind)
        own_labels = self._labels_field()
        for field_name in ("options", "levels"):
            labels = _labels_tuple(getattr(self, field_name), f"question {self.id!r} {field_name}")
            object.__setattr__(self, field_name, labels)
            if field_name == own_labels and (labels is None or len(labels) < 2):
                raise ValueError(f"{kind} question {self.id!r} needs at least 2 {field_name}")
            if field_name != own_labels and labels is not None:
                raise ValueError(f"{kind} question {self.id!r} must not set {field_name}")

    @classmethod
    def from_mapping(cls, spec: Mapping[str, object]) -> "DecisionQuestion":
        """A question from a dict with the keys id, kind, text and optional options or levels."""
        _known_keys(spec, QUESTION_KEYS, "question")
        question_id = _text(spec.get("id"), "question id")
        text = _text(spec.get("text"), f"question {question_id!r} text")
        return cls(id=question_id, kind=_question_kind(spec.get("kind"), question_id), text=text,
                   options=_labels_tuple(spec.get("options"), f"question {question_id!r} options"),
                   levels=_labels_tuple(spec.get("levels"), f"question {question_id!r} levels"))

    def _labels_field(self) -> Optional[str]:
        return {"choice": "options", "score": "levels"}.get(self.kind)

    def labels(self) -> Optional[Tuple[str, ...]]:
        """Allowed answer labels: options for choice, levels for score, None for predicate."""
        field_name = self._labels_field()
        return None if field_name is None else getattr(self, field_name)

    def distribution_keys(self) -> Tuple[str, ...]:
        """Distribution keys in label order (``true``/``false`` for a predicate)."""
        labels = self.labels()
        return labels if labels is not None else ("true", "false")


@dataclass(frozen=True)
class DecisionRequest:
    """One row's evidence and the questions to answer about it; ``request_id`` identifies the row."""

    request_id: str
    evidence: Evidence
    questions: Tuple[DecisionQuestion, ...]
    task_id: str
    task_version: str
    deadline_s: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "questions", tuple(self.questions))


@dataclass(frozen=True)
class Budget:
    """Run limit on the number of provider calls."""

    max_requests: Optional[int] = None

    def __post_init__(self) -> None:
        _min_int(self.max_requests, "max_requests", 1)

    @classmethod
    def from_mapping(cls, spec: Mapping[str, object]) -> "Budget":
        """A budget from a dict with the optional key max_requests; a missing marker such as NaN raises."""
        limit = _known_keys(spec, BUDGET_KEYS, "budget").get("max_requests")
        canonical = canonicalize(limit)
        return cls(max_requests=_min_int(limit if canonical is None else canonical, "max_requests", 1))


@dataclass(frozen=True)
class DecisionResult:
    """Exactly one result per (request, question).

    ``status``: ``ok`` (answered), ``uncertain`` (abstained, tied, or below ``min_p``), ``error``
    (``error_code`` set) or ``skipped`` (not attempted; ``status_reason`` says why).
    ``distribution`` maps ``"true"``/``"false"`` (both, for a predicate) or labels (all or some, for choice and
    score) to probabilities and is None exactly when ``probability_kind`` is ``"none"``; distributions are never
    fabricated. ``confidence`` is the
    provider's own number, verbatim, else None. ``retry_count`` None means unknown. ``extra`` holds scalar
    metadata such as ``probability_mechanism``, ``score_index``, ``evidence_truncated`` and
    ``cached_input_tokens``; it never carries evidence, credentials or tenant identifiers.
    """

    request_id: str
    question_id: str
    value: Union[bool, str, None]
    distribution: Optional[Mapping[str, float]]
    confidence: Optional[float]
    probability_kind: ProbabilityKind
    status: ResultStatus
    error_code: Optional[str]
    model: str
    provider: str
    task_version: str
    input_hash: str
    latency_ms: Optional[float]
    cost_usd: Optional[float]
    cache_hit: Optional[bool]
    retry_count: Optional[int]
    status_reason: Optional[StatusReason] = None
    extra: Optional[Mapping[str, ExtraValue]] = None


@runtime_checkable
class DecisionProvider(Protocol):
    """A model backend for ``sem_decide``.

    ``decide_batch`` is synchronous and may be called from up to ``concurrency_limit`` threads at once;
    ``deadline_s`` and ``budget`` are what remains of the run.
    """

    name: str
    model: str

    def capabilities(self) -> ProviderCapabilities:
        ...

    def decide_batch(self, requests: Sequence[DecisionRequest], *,
                     deadline_s: Optional[float], budget: Optional[Budget]) -> List[DecisionResult]:
        ...


def canonicalize(value: object) -> object:
    """Plain Python for hashing and evidence: mappings become dicts with string keys, sequences and arrays
    become lists, numpy scalars become Python scalars, and float NaN and pandas NA/NaT become None."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if type(value).__name__ in ("NAType", "NaTType") and type(value).__module__.startswith("pandas"):
        return None
    if isinstance(value, Mapping):
        return {str(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    to_python = getattr(value, "tolist", None)
    if callable(to_python) and not isinstance(value, type):
        plain = to_python()
        if type(plain) is not type(value):
            return canonicalize(plain)
    return value


def compute_decision_hash(evidence: Evidence, question: DecisionQuestion, task_version: str) -> str:
    """Stable hash of what a decision depends on; excludes question id, model and provider."""
    payload = {
        "evidence": evidence,
        "question": {"kind": question.kind, "text": question.text, "options": question.options,
                     "levels": question.levels},
        "task_version": task_version,
    }
    text = json.dumps(canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def align_results(requests: Sequence[DecisionRequest], results: object
                  ) -> Optional[Dict[str, Optional[Dict[str, DecisionResult]]]]:
    """Results of one provider call by request id, then question id.

    None when the call as a whole cannot be trusted: not a list, a non-result entry, an unknown request id or
    a duplicate (request, question) key. A row maps to None when it misses a question or carries an unknown one.
    """
    if not isinstance(results, (list, tuple)):
        return None
    expected = {request.request_id: {question.id for question in request.questions} for request in requests}
    rows: Dict[str, Dict[str, DecisionResult]] = {request_id: {} for request_id in expected}
    broken_rows = set()
    seen = set()
    for result in results:
        if not isinstance(result, DecisionResult) or not (
                isinstance(result.request_id, str) and result.request_id in expected):
            return None
        if not isinstance(result.question_id, str):
            broken_rows.add(result.request_id)
            continue
        key = (result.request_id, result.question_id)
        if key in seen:
            return None
        seen.add(key)
        if result.question_id in expected[result.request_id]:
            rows[result.request_id][result.question_id] = result
        else:
            broken_rows.add(result.request_id)
    return {request_id: None if request_id in broken_rows or len(found) < len(expected[request_id]) else found
            for request_id, found in rows.items()}


def _distribution_problems(question: DecisionQuestion, distribution: object) -> List[str]:
    if not isinstance(distribution, Mapping):
        return ["distribution must be None or a mapping"]
    if not distribution:
        return ["an empty distribution must be None"]
    allowed = question.distribution_keys()
    problems = [f"distribution key {key!r} is not one of {allowed}" for key in distribution if not _is_member(key, allowed)]
    numbers = [finite_float(probability) for probability in distribution.values()]
    if not all(number is not None and 0.0 <= number <= 1.0 for number in numbers):
        return problems + [f"each distribution value must be a number in [0, 1], got {dict(distribution)!r}"]
    total = sum(number for number in numbers if number is not None)
    slack = DISTRIBUTION_SUM_TOLERANCE + 1e-9
    if question.kind == "predicate" and not problems and set(distribution) != set(allowed):
        problems.append("a predicate distribution needs both true and false")
    elif set(distribution) == set(allowed) and abs(total - 1.0) > slack:
        problems.append(f"a full distribution must sum to 1 +/- {DISTRIBUTION_SUM_TOLERANCE}, got {total}")
    elif total > 1.0 + slack:
        problems.append(f"a partial distribution must not sum above 1, got {total}")
    return problems


def check_decision_result(request: DecisionRequest, question: DecisionQuestion, result: DecisionResult,
                          capabilities: Optional[ProviderCapabilities] = None) -> List[str]:
    """Content violations of one decision result against its request and question (empty when valid)."""
    status, reason = result.status, result.status_reason
    value, distribution = canonicalize(result.value), canonicalize(result.distribution)
    if not _is_member(status, RESULT_STATUSES):
        return [f"status must be a string in {RESULT_STATUSES}, got {status!r}"]
    if not _is_member(result.probability_kind, PROBABILITY_KINDS):
        return [f"probability_kind must be a string in {PROBABILITY_KINDS}, got {result.probability_kind!r}"]
    problems = [f"{name} does not match {expected!r}" for name, got, expected in (
        ("request_id", result.request_id, request.request_id), ("question_id", result.question_id, question.id))
        if not _is_member(got, [expected])]
    if status == "skipped" and not _is_member(reason, SKIPPED_REASONS):
        problems.append(f"a skipped result needs a status_reason in {SKIPPED_REASONS}")
    elif reason is not None and not _is_member(reason, _REASONS_BY_STATUS.get(status, ())):
        problems.append(f"status_reason {reason!r} is not allowed with status {status!r}")
    if status == "error" and not (isinstance(result.error_code, str) and result.error_code != ""):
        problems.append("an error result needs a non-empty error_code")
    if status != "error" and result.error_code is not None:
        problems.append(f"error_code must be None unless status is error, got {result.error_code!r}")
    label = reason if isinstance(reason, str) and reason in _EMPTY_FIELDS else status
    problems.extend(f"{label} results must have {name} None" for name in _EMPTY_FIELDS.get(label, ())
                    if getattr(result, name) is not None)
    if status == "ok" and value is None:
        problems.append("an ok result must carry a value")
    if (result.probability_kind == "none") != (distribution is None):
        problems.append("probability_kind must be 'none' exactly when distribution is None")
    labels = question.labels()
    if value is not None and labels is None and not isinstance(value, bool):
        problems.append(f"a predicate value must be a bool or None, got {result.value!r}")
    if value is not None and labels is not None and not _is_member(value, labels):
        problems.append(f"value {result.value!r} is not one of {labels}")
    if distribution is not None:
        problems.extend(_distribution_problems(question, distribution))
    if capabilities is not None and result.probability_kind == "native_decision" \
            and not capabilities.supports_native_probabilities:
        problems.append("native_decision reported by a provider without native probabilities")
    return problems


def unanswered_result(request: DecisionRequest, question: DecisionQuestion, *, provider: str, model: str,
                      error_code: Optional[str] = None, status_reason: Optional[SkippedReason] = None,
                      latency_ms: Optional[float] = None, retry_count: Optional[int] = None,
                      input_hash: Optional[str] = None) -> DecisionResult:
    """An ``error`` result when ``error_code`` is set, else a ``skipped`` result with ``status_reason``."""
    return DecisionResult(
        request_id=request.request_id, question_id=question.id, value=None, distribution=None, confidence=None,
        probability_kind="none", status="error" if error_code is not None else "skipped", error_code=error_code,
        model=model, provider=provider, task_version=request.task_version,
        input_hash=input_hash or compute_decision_hash(request.evidence, question, request.task_version),
        latency_ms=latency_ms, cost_usd=None, cache_hit=None, retry_count=retry_count, status_reason=status_reason,
    )
