"""Provider-neutral contract for semantic table operators.

Requests, results, capabilities and provider protocols shared by ``sem_decide`` and
``sem_map`` and by any host that plugs in its own model provider. The module uses only the
standard library: the file can be loaded by path without pandas or the rest of graphistry.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import (
    Dict, List, Literal, Mapping, Optional, Protocol, Sequence, SupportsComplex, SupportsFloat, Tuple, Union, get_args,
    runtime_checkable,
)


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
DISTRIBUTION_SUM_TOLERANCE = 0.01
QUESTION_KEYS = frozenset({"id", "kind", "text", "options", "levels"})
BUDGET_KEYS = frozenset({"max_cost_usd", "max_requests"})
DECISION_SUFFIXES: Tuple[str, ...] = ("__status", "__p", "__dist", "__probability_kind", "__error_code", "__meta")
GENERATION_SUFFIXES: Tuple[str, ...] = ("__status", "__error_code", "__meta")
EXTRA_KEY_TYPES: Dict[str, Tuple[type, ...]] = {
    "probability_mechanism": (str,),
    "score_index": (int, float),
    "evidence_truncated": (bool,),
    "cached_input_tokens": (int,),
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_float(value: object) -> Optional[float]:
    """The value as a float if it is a real, finite int or float (bools excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        as_float = float(value)
    except OverflowError:
        return None
    return as_float if math.isfinite(as_float) else None


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _require_min_int(value: object, field_name: str, minimum: int, optional: bool) -> Optional[int]:
    if value is None and optional:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return value
    raise ValueError(f"{field_name} must be an int >= {minimum}, got {value!r}")


def _labels_tuple(value: object, field_name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple of strings, got {value!r}")
    labels = tuple(value)
    for label in labels:
        if not isinstance(label, str) or label == "":
            raise ValueError(f"{field_name} entries must be non-empty strings, got {label!r}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field_name} entries must be unique, got {labels!r}")
    return labels


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do. Optional limits are None when the provider sets no limit.

    ``concurrency_limit`` is how many provider calls may run at once (1 if not thread-safe);
    ``max_rows_per_request`` is how many rows one transport request can answer;
    ``supported_kinds`` None means all question kinds; ``max_choice_options`` and ``max_levels``
    bound choice options and score levels. Operators check questions against these before any call.
    """

    supports_native_probabilities: bool
    max_questions_per_request: int
    max_evidence_tokens: Optional[int]
    supports_shared_prefix_cache: bool
    concurrency_limit: int
    max_rows_per_request: Optional[int] = None
    supported_kinds: Optional[Tuple[QuestionKind, ...]] = None
    max_choice_options: Optional[int] = None
    max_levels: Optional[int] = None

    def __post_init__(self) -> None:
        for flag in ("supports_native_probabilities", "supports_shared_prefix_cache"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be a bool")
        _require_min_int(self.max_questions_per_request, "max_questions_per_request", 1, optional=False)
        _require_min_int(self.concurrency_limit, "concurrency_limit", 1, optional=False)
        _require_min_int(self.max_evidence_tokens, "max_evidence_tokens", 1, optional=True)
        _require_min_int(self.max_rows_per_request, "max_rows_per_request", 1, optional=True)
        _require_min_int(self.max_choice_options, "max_choice_options", 2, optional=True)
        _require_min_int(self.max_levels, "max_levels", 2, optional=True)
        if self.supported_kinds is not None:
            kinds = _labels_tuple(self.supported_kinds, "supported_kinds")
            if not kinds:
                raise ValueError("supported_kinds must be None or name at least one kind")
            for kind in kinds:
                if kind not in QUESTION_KINDS:
                    raise ValueError(f"supported_kinds entry {kind!r} is not one of {QUESTION_KINDS}")
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
        _require_non_empty_str(self.id, "question id")
        _require_non_empty_str(self.text, f"question {self.id!r} text")
        object.__setattr__(self, "kind", _question_kind(self.kind, self.id))
        options = _labels_tuple(self.options, f"question {self.id!r} options")
        levels = _labels_tuple(self.levels, f"question {self.id!r} levels")
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "levels", levels)
        if self.kind == "choice":
            if options is None or len(options) < 2:
                raise ValueError(f"choice question {self.id!r} needs at least 2 options")
            if levels is not None:
                raise ValueError(f"choice question {self.id!r} must not set levels")
        elif self.kind == "score":
            if levels is None or len(levels) < 2:
                raise ValueError(f"score question {self.id!r} needs at least 2 levels")
            if options is not None:
                raise ValueError(f"score question {self.id!r} must not set options")
        elif options is not None or levels is not None:
            raise ValueError(f"predicate question {self.id!r} must not set options or levels")

    @classmethod
    def from_mapping(cls, spec: Mapping[str, object]) -> "DecisionQuestion":
        """A question from a dict with the keys id, kind, text and optional options or levels."""
        unknown = set(spec) - QUESTION_KEYS
        if unknown:
            raise ValueError(f"unknown question keys {sorted(map(str, unknown))}; allowed {sorted(QUESTION_KEYS)}")
        question_id = _require_non_empty_str(spec.get("id"), "question id")
        text = _require_non_empty_str(spec.get("text"), f"question {question_id!r} text")
        kind = _question_kind(spec.get("kind"), question_id)
        return cls(id=question_id, kind=kind, text=text,
                   options=_labels_tuple(spec.get("options"), f"question {question_id!r} options"),
                   levels=_labels_tuple(spec.get("levels"), f"question {question_id!r} levels"))

    def labels(self) -> Optional[Tuple[str, ...]]:
        """Allowed answer labels: options for choice, levels for score, None for predicate."""
        if self.kind == "choice":
            return self.options
        if self.kind == "score":
            return self.levels
        return None

    def distribution_keys(self) -> Tuple[str, ...]:
        """Distribution keys in label order (``true``/``false`` for a predicate)."""
        labels = self.labels()
        return labels if labels is not None else ("true", "false")


def _questions_tuple(value: object) -> Tuple[DecisionQuestion, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"questions must be a list or tuple, got {value!r}")
    questions = tuple(value)
    if not questions:
        raise ValueError("a request needs at least one question")
    for question in questions:
        if not isinstance(question, DecisionQuestion):
            raise ValueError(f"questions entries must be DecisionQuestion, got {question!r}")
    ids = [question.id for question in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"question ids must be unique, got {ids!r}")
    return questions


def _require_cost(value: object, field_name: str) -> Optional[float]:
    if value is None:
        return None
    number = finite_float(value)
    if number is None or number < 0 or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be None or a finite number >= 0, got {value!r}")
    return value


def _question_kind(value: object, question_id: object) -> QuestionKind:
    for kind in QUESTION_KINDS:
        if value == kind:
            return kind
    raise ValueError(f"question {question_id!r} kind must be one of {QUESTION_KINDS}, got {value!r}")


def _check_request_common(request_id: object, evidence: object, task_id: object, task_version: object,
                          deadline_s: object, max_cost_usd: object) -> None:
    _require_non_empty_str(request_id, "request_id")
    _require_non_empty_str(task_id, "task_id")
    _require_non_empty_str(task_version, "task_version")
    if not isinstance(evidence, (str, Mapping)):
        raise ValueError(f"evidence must be a string or a mapping, got {type(evidence).__name__}")
    deadline = finite_float(deadline_s)
    if deadline_s is not None and (deadline is None or deadline <= 0):
        raise ValueError(f"deadline_s must be None or a finite number > 0, got {deadline_s!r}")
    _require_cost(max_cost_usd, "max_cost_usd")


@dataclass(frozen=True)
class DecisionRequest:
    """One row's evidence and the questions to answer about it; ``request_id`` identifies the row."""

    request_id: str
    evidence: Evidence
    questions: Tuple[DecisionQuestion, ...]
    task_id: str
    task_version: str
    deadline_s: Optional[float] = None
    max_cost_usd: Optional[float] = None

    def __post_init__(self) -> None:
        _check_request_common(self.request_id, self.evidence, self.task_id, self.task_version, self.deadline_s,
                              self.max_cost_usd)
        object.__setattr__(self, "questions", _questions_tuple(self.questions))


@dataclass(frozen=True)
class GenerationRequest:
    """One row's evidence and the instruction to generate a value for it."""

    request_id: str
    evidence: Evidence
    instruction: str
    task_id: str
    task_version: str
    max_output_tokens: Optional[int] = None
    deadline_s: Optional[float] = None
    max_cost_usd: Optional[float] = None

    def __post_init__(self) -> None:
        _check_request_common(self.request_id, self.evidence, self.task_id, self.task_version, self.deadline_s,
                              self.max_cost_usd)
        _require_non_empty_str(self.instruction, "instruction")
        _require_min_int(self.max_output_tokens, "max_output_tokens", 1, optional=True)


@dataclass(frozen=True)
class Budget:
    """Run limits: estimated cost in USD and number of provider calls."""

    max_cost_usd: Optional[float] = None
    max_requests: Optional[int] = None

    def __post_init__(self) -> None:
        _require_cost(self.max_cost_usd, "max_cost_usd")
        _require_min_int(self.max_requests, "max_requests", 1, optional=True)

    @classmethod
    def from_mapping(cls, spec: Mapping[str, object]) -> "Budget":
        """A budget from a dict with the optional keys max_cost_usd and max_requests."""
        unknown = set(spec) - BUDGET_KEYS
        if unknown:
            raise ValueError(f"unknown budget keys {sorted(map(str, unknown))}; allowed {sorted(BUDGET_KEYS)}")
        return cls(max_cost_usd=_require_cost(_limit_value(spec.get("max_cost_usd")), "max_cost_usd"),
                   max_requests=_require_min_int(_limit_value(spec.get("max_requests")), "max_requests", 1,
                                                 optional=True))


def _limit_value(value: object) -> object:
    """Canonical limit; a missing marker such as NaN stays visible so a cap can never turn into no cap."""
    canonical = canonicalize(value)
    return value if canonical is None else canonical


@dataclass(frozen=True)
class DecisionResult:
    """Exactly one result per (request, question).

    ``status``: ``ok`` (answered), ``uncertain`` (abstained, tied, or below ``min_p``), ``error``
    (``error_code`` set) or ``skipped`` (not attempted; ``status_reason`` says why).
    ``distribution`` maps ``"true"``/``"false"`` or labels to probabilities and is None exactly when
    ``probability_kind`` is ``"none"``; distributions are never fabricated. ``confidence`` is the
    provider's own number, verbatim, else None; it is never derived from the distribution.
    ``retry_count`` None means unknown. ``extra`` holds scalar metadata such as
    ``probability_mechanism``, ``score_index``, ``evidence_truncated`` and ``cached_input_tokens``; it
    never carries evidence, credentials or tenant identifiers.
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


@dataclass(frozen=True)
class GenerationResult:
    """Exactly one result per request. Status and ``extra`` as in ``DecisionResult``; ``uncertain``
    results carry ``refusal`` or ``null_answer`` (never ``tie`` or ``min_p``)."""

    request_id: str
    value: Optional[str]
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

    Methods are synchronous. ``decide_batch`` may be called from up to ``concurrency_limit`` threads at
    once (declare 1 if the provider is not thread-safe); ``deadline_s`` and ``budget`` are what remains
    of the run. ``estimate_cost_usd`` returns None when it cannot estimate, else a finite value >= 0;
    operators treat any other value as "cannot estimate".
    """

    name: str
    model: str

    def capabilities(self) -> ProviderCapabilities:
        ...

    def estimate_cost_usd(self, requests: Sequence[DecisionRequest]) -> Optional[float]:
        ...

    def decide_batch(self, requests: Sequence[DecisionRequest], *,
                     deadline_s: Optional[float], budget: Optional[Budget]) -> List[DecisionResult]:
        ...


@runtime_checkable
class GenerationProvider(Protocol):
    """A model backend for ``sem_map``; same calling rules as ``DecisionProvider``."""

    name: str
    model: str

    def capabilities(self) -> ProviderCapabilities:
        ...

    def estimate_cost_usd(self, requests: Sequence[GenerationRequest]) -> Optional[float]:
        ...

    def generate_batch(self, requests: Sequence[GenerationRequest], *,
                       deadline_s: Optional[float], budget: Optional[Budget]) -> List[GenerationResult]:
        ...


def _is_pandas_missing(value: object) -> bool:
    kind = type(value)
    if kind.__name__ in ("NAType", "NaTType") and kind.__module__.startswith("pandas"):
        return True
    return kind.__name__ == "MaskedConstant" and kind.__module__.startswith("numpy")


@runtime_checkable
class _ArrayLike(Protocol):
    ndim: int

    def tolist(self) -> object:
        ...


@runtime_checkable
class _ScalarLike(Protocol):
    def item(self) -> object:
        ...


def _sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def canonicalize(value: object) -> object:
    """Normalize a value for hashing and for sending as evidence.

    Mappings become dicts with string keys; lists, tuples and arrays become lists; sets become lists
    sorted by their JSON text; numpy scalars become Python scalars, with finite long doubles rounded to
    float (complex values with an imaginary part and long doubles that overflow float stay as they are);
    float NaN, pandas NA/NaT and masked values become None. Other objects are returned unchanged (hashing
    then stringifies them).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if _is_pandas_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(item) for item in value), key=_sort_key)
    if isinstance(value, _ArrayLike) and value.ndim > 0:
        return canonicalize(value.tolist())
    if isinstance(value, _ScalarLike):
        try:
            scalar = value.item()
        except (TypeError, ValueError):
            return value
        if type(scalar) is not type(value):
            return canonicalize(scalar)
        if not isinstance(scalar, (SupportsComplex, SupportsFloat)):
            return value
        try:
            as_complex = complex(scalar)
        except (TypeError, ValueError, OverflowError):
            return value
        if as_complex.imag != 0 or (math.isinf(as_complex.real) and scalar != as_complex.real):
            return value
        return None if math.isnan(as_complex.real) else as_complex.real
    return value


def _sha256_json(payload: Mapping[str, object]) -> str:
    text = json.dumps(canonicalize(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_decision_hash(evidence: Evidence, question: DecisionQuestion, task_version: str) -> str:
    """Stable hash of what a decision depends on; excludes question id, model and provider."""
    return _sha256_json({
        "evidence": evidence,
        "question": {
            "kind": question.kind,
            "text": question.text,
            "options": question.options,
            "levels": question.levels,
        },
        "task_version": task_version,
    })


def compute_generation_hash(evidence: Evidence, instruction: str, task_version: str) -> str:
    """Stable hash of what a generation depends on; excludes model and provider."""
    return _sha256_json({"evidence": evidence, "instruction": instruction, "task_version": task_version})


def valid_cost_estimate(value: object) -> Optional[float]:
    """A provider cost estimate if it is a finite number >= 0, else None (treated as "cannot estimate")."""
    number = finite_float(canonicalize(value))
    return number if number is not None and number >= 0 else None


def check_requests(requests: Sequence[Union[DecisionRequest, GenerationRequest]]) -> None:
    """Raise ``ValueError`` when one provider call would carry a duplicate ``request_id``."""
    seen = set()
    for request in requests:
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id {request.request_id!r} within one provider call")
        seen.add(request.request_id)


def _check_optional_number(value: object, field_name: str, problems: List[str], upper: Optional[float] = None) -> None:
    if value is None:
        return
    number = finite_float(value)
    if number is None or number < 0:
        problems.append(f"{field_name} must be None or a finite number >= 0, got {value!r}")
        return
    if upper is not None and number > upper:
        problems.append(f"{field_name} must be <= {upper}, got {value!r}")


def _check_common_result(result: Union[DecisionResult, GenerationResult], problems: List[str]) -> None:
    if not isinstance(result.request_id, str):
        problems.append(f"request_id must be a string, got {type(result.request_id).__name__}")
    if not isinstance(result.status, str) or result.status not in RESULT_STATUSES:
        problems.append(f"status must be one of {RESULT_STATUSES}, got {result.status!r}")
    if result.status == "skipped" and result.status_reason not in SKIPPED_REASONS:
        problems.append(f"a skipped result needs a status_reason in {SKIPPED_REASONS}")
    elif result.status_reason is not None:
        allowed: Tuple[str, ...] = ()
        if result.status == "uncertain":
            allowed = UNCERTAIN_REASONS
        elif result.status == "skipped":
            allowed = SKIPPED_REASONS
        if not isinstance(result.status_reason, str) or result.status_reason not in allowed:
            problems.append(f"status_reason {result.status_reason!r} is not allowed with status {result.status!r}")
    if result.status == "error":
        if not isinstance(result.error_code, str) or result.error_code == "":
            problems.append("an error result needs a non-empty error_code")
    elif result.error_code is not None:
        problems.append(f"error_code must be None unless status is error, got {result.error_code!r}")
    if result.status_reason in ("refusal", "null_answer", "tie") and result.value is not None:
        problems.append(f"a {result.status_reason} result must have value None")
    if result.status in ("error", "skipped") and result.value is not None:
        problems.append(f"a {result.status} result must have value None")
    if result.status == "ok" and result.value is None:
        problems.append("an ok result must carry a value")
    for field_name in ("model", "provider", "task_version", "input_hash"):
        if not isinstance(getattr(result, field_name), str):
            problems.append(f"{field_name} must be a string")
    _check_optional_number(result.latency_ms, "latency_ms", problems)
    _check_optional_number(result.cost_usd, "cost_usd", problems)
    if result.cache_hit is not None and not isinstance(result.cache_hit, bool):
        problems.append(f"cache_hit must be None or a bool, got {result.cache_hit!r}")
    if result.retry_count is not None and (not _is_int(result.retry_count) or result.retry_count < 0):
        problems.append(f"retry_count must be None or an int >= 0, got {result.retry_count!r}")
    if result.extra is not None:
        if not isinstance(result.extra, Mapping):
            problems.append("extra must be None or a mapping")
        else:
            for key, item in result.extra.items():
                if not isinstance(key, str) or not (item is None or isinstance(item, (str, int, float, bool))):
                    problems.append(f"extra entry {key!r} must map a string to a scalar")
                elif isinstance(item, float) and not math.isfinite(item):
                    problems.append(f"extra entry {key!r} must be finite, got {item!r}")
                elif key in EXTRA_KEY_TYPES and item is not None and (
                    not isinstance(item, EXTRA_KEY_TYPES[key]) or (isinstance(item, bool) and bool not in EXTRA_KEY_TYPES[key])
                ):
                    problems.append(f"extra entry {key!r} has the wrong type: {type(item).__name__}")


def _check_distribution(question: DecisionQuestion, distribution: Mapping[str, float], problems: List[str]) -> None:
    if not isinstance(distribution, Mapping):
        problems.append("distribution must be None or a mapping")
        return
    if len(distribution) == 0:
        problems.append("an empty distribution must be None")
        return
    allowed = question.distribution_keys()
    total = 0.0
    all_valid = True
    for key, probability in distribution.items():
        if not isinstance(key, str) or key not in allowed:
            problems.append(f"distribution key {key!r} is not one of {allowed}")
        number = finite_float(probability)
        if number is None or not 0.0 <= number <= 1.0:
            problems.append(f"distribution value for {key!r} must be a number in [0, 1], got "
                            f"{type(probability).__name__} {probability!r}")
            all_valid = False
        else:
            total += number
    if not all_valid:
        return
    if set(distribution.keys()) == set(allowed) and abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE + 1e-9:
        problems.append(f"a full distribution must sum to 1 +/- {DISTRIBUTION_SUM_TOLERANCE}, got {total}")
    elif total > 1.0 + DISTRIBUTION_SUM_TOLERANCE + 1e-9:
        problems.append(f"a partial distribution must not sum above 1, got {total}")


def _malformed_scalar_fields(result: Union[DecisionResult, GenerationResult], fields: Sequence[str]) -> List[str]:
    """Identity and vocabulary fields that are not strings; later checks compare them, so check these first."""
    problems: List[str] = []
    for name in fields:
        value = getattr(result, name)
        if not (isinstance(value, str) or (name in ("status_reason", "error_code") and value is None)):
            problems.append(f"{name} must be a string, got {type(value).__name__}")
    return problems


def check_decision_result(request: DecisionRequest, question: DecisionQuestion, result: DecisionResult,
                          capabilities: Optional[ProviderCapabilities] = None) -> List[str]:
    """Per-result violations of one decision result against its request and question."""
    problems = _malformed_scalar_fields(
        result, ("request_id", "question_id", "status", "status_reason", "probability_kind", "error_code"))
    if problems:
        return problems
    if result.request_id != request.request_id:
        problems.append(f"request_id {result.request_id!r} does not match {request.request_id!r}")
    if not isinstance(result.question_id, str) or result.question_id != question.id:
        problems.append(f"question_id {result.question_id!r} does not match {question.id!r}")
    _check_common_result(result, problems)
    if not isinstance(result.probability_kind, str) or result.probability_kind not in PROBABILITY_KINDS:
        problems.append(f"probability_kind must be one of {PROBABILITY_KINDS}, got {result.probability_kind!r}")
    if (result.probability_kind == "none") != (result.distribution is None):
        problems.append("probability_kind must be 'none' exactly when distribution is None")
    if result.status in ("error", "skipped") and result.distribution is not None:
        problems.append(f"a {result.status} result must have distribution None")
    if result.status in ("error", "skipped") and result.confidence is not None:
        problems.append(f"a {result.status} result must have confidence None")
    if result.status_reason in ("refusal", "null_answer") and result.distribution is not None:
        problems.append(f"a {result.status_reason} result must have distribution None")
    if result.value is not None:
        if question.kind == "predicate":
            if not isinstance(result.value, bool):
                problems.append(f"a predicate value must be a bool or None, got {result.value!r}")
        else:
            labels = question.labels() or ()
            if not isinstance(result.value, str) or result.value not in labels:
                problems.append(f"value {result.value!r} is not one of {labels}")
    if result.distribution is not None:
        _check_distribution(question, result.distribution, problems)
    _check_optional_number(result.confidence, "confidence", problems, upper=1.0)
    if capabilities is not None and result.probability_kind == "native_decision" \
            and not capabilities.supports_native_probabilities:
        problems.append("native_decision reported by a provider without native probabilities")
    return problems


def check_generation_result(request: GenerationRequest, result: GenerationResult) -> List[str]:
    """Per-result violations of one generation result against its request."""
    problems = _malformed_scalar_fields(result, ("request_id", "status", "status_reason", "error_code"))
    if problems:
        return problems
    if result.request_id != request.request_id:
        problems.append(f"request_id {result.request_id!r} does not match {request.request_id!r}")
    _check_common_result(result, problems)
    if result.value is not None and not isinstance(result.value, str):
        problems.append(f"a generation value must be a string or None, got {result.value!r}")
    if result.status_reason in ("tie", "min_p"):
        problems.append(f"status_reason {result.status_reason!r} does not apply to generation")
    return problems


def check_decision_results(requests: Sequence[DecisionRequest], results: Sequence[DecisionResult],
                           capabilities: Optional[ProviderCapabilities] = None) -> List[str]:
    """Every violation of one provider return: alignment (one result per key) plus per-result checks."""
    problems: List[str] = []
    expected: Dict[Tuple[str, str], Tuple[DecisionRequest, DecisionQuestion]] = {}
    for request in requests:
        for question in request.questions:
            expected[(request.request_id, question.id)] = (request, question)
    if not isinstance(results, (list, tuple)):
        return problems + [f"results must be a list, got {type(results).__name__}"]
    seen = set()
    for result in results:
        if not isinstance(result, DecisionResult):
            problems.append(f"expected DecisionResult, got {type(result).__name__}")
            continue
        if not isinstance(result.request_id, str) or not isinstance(result.question_id, str):
            problems.append("result request_id and question_id must be strings")
            continue
        key = (result.request_id, result.question_id)
        if key not in expected:
            problems.append(f"unexpected result for request {key[0]!r}, question {key[1]!r}")
            continue
        if key in seen:
            problems.append(f"duplicate result for request {key[0]!r}, question {key[1]!r}")
            continue
        seen.add(key)
        request, question = expected[key]
        problems.extend(f"{key[0]}/{key[1]}: {problem}"
                        for problem in check_decision_result(request, question, result, capabilities))
    for key in expected:
        if key not in seen:
            problems.append(f"missing result for request {key[0]!r}, question {key[1]!r}")
    return problems


def check_generation_results(requests: Sequence[GenerationRequest], results: Sequence[GenerationResult]) -> List[str]:
    """Every violation of one provider return: alignment (one result per request) plus per-result checks."""
    problems: List[str] = []
    expected = {request.request_id: request for request in requests}
    if not isinstance(results, (list, tuple)):
        return problems + [f"results must be a list, got {type(results).__name__}"]
    seen = set()
    for result in results:
        if not isinstance(result, GenerationResult):
            problems.append(f"expected GenerationResult, got {type(result).__name__}")
            continue
        if not isinstance(result.request_id, str):
            problems.append("result request_id must be a string")
            continue
        if result.request_id not in expected:
            problems.append(f"unexpected result for request {result.request_id!r}")
            continue
        if result.request_id in seen:
            problems.append(f"duplicate result for request {result.request_id!r}")
            continue
        seen.add(result.request_id)
        problems.extend(f"{result.request_id}: {problem}"
                        for problem in check_generation_result(expected[result.request_id], result))
    for request_id in expected:
        if request_id not in seen:
            problems.append(f"missing result for request {request_id!r}")
    return problems


def assert_decision_results_valid(requests: Sequence[DecisionRequest], results: Sequence[DecisionResult],
                                  capabilities: Optional[ProviderCapabilities] = None) -> None:
    problems = check_decision_results(requests, results, capabilities)
    if problems:
        raise ValueError("invalid decision results: " + "; ".join(problems))


def assert_generation_results_valid(requests: Sequence[GenerationRequest],
                                    results: Sequence[GenerationResult]) -> None:
    problems = check_generation_results(requests, results)
    if problems:
        raise ValueError("invalid generation results: " + "; ".join(problems))


def error_result(request: DecisionRequest, question: DecisionQuestion, *, error_code: str, provider: str,
                 model: str, latency_ms: Optional[float] = None, retry_count: Optional[int] = 0) -> DecisionResult:
    return DecisionResult(
        request_id=request.request_id, question_id=question.id, value=None, distribution=None, confidence=None,
        probability_kind="none", status="error", error_code=error_code, model=model, provider=provider,
        task_version=request.task_version, input_hash=compute_decision_hash(request.evidence, question,
                                                                           request.task_version),
        latency_ms=latency_ms, cost_usd=None, cache_hit=None, retry_count=retry_count,
    )


def skipped_result(request: DecisionRequest, question: DecisionQuestion, *, provider: str, model: str,
                   status_reason: SkippedReason) -> DecisionResult:
    return DecisionResult(
        request_id=request.request_id, question_id=question.id, value=None, distribution=None, confidence=None,
        probability_kind="none", status="skipped", error_code=None, model=model, provider=provider,
        task_version=request.task_version, input_hash=compute_decision_hash(request.evidence, question,
                                                                           request.task_version),
        latency_ms=None, cost_usd=None, cache_hit=None, retry_count=None, status_reason=status_reason,
    )


def generation_error_result(request: GenerationRequest, *, error_code: str, provider: str, model: str,
                            latency_ms: Optional[float] = None, retry_count: Optional[int] = 0) -> GenerationResult:
    return GenerationResult(
        request_id=request.request_id, value=None, status="error", error_code=error_code, model=model,
        provider=provider, task_version=request.task_version,
        input_hash=compute_generation_hash(request.evidence, request.instruction, request.task_version),
        latency_ms=latency_ms, cost_usd=None, cache_hit=None, retry_count=retry_count,
    )


def generation_skipped_result(request: GenerationRequest, *, provider: str, model: str,
                              status_reason: SkippedReason) -> GenerationResult:
    return GenerationResult(
        request_id=request.request_id, value=None, status="skipped", error_code=None, model=model,
        provider=provider, task_version=request.task_version,
        input_hash=compute_generation_hash(request.evidence, request.instruction, request.task_version),
        latency_ms=None, cost_usd=None, cache_hit=None, retry_count=None, status_reason=status_reason,
    )
