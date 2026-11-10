"""Semantic table operators: ``sem_decide`` and ``sem_map``.

Each operator keeps every input row, in order and with its index, and appends typed result
columns. Provider faults become per-row ``error`` results; validation failures raise
``ValueError`` before any provider call.
"""
import dataclasses
import json
from collections import Counter
import math
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from graphistry.compute.gfql.effects import record_effect
from graphistry.compute.gfql.identifiers import validate_column_name
from graphistry.compute.semantic.binding import resolve_decision_provider, resolve_generation_provider
from graphistry.compute.semantic.contract import (
    DECISION_SUFFIXES,
    GENERATION_SUFFIXES,
    Budget,
    DecisionProvider,
    DecisionQuestion,
    DecisionRequest,
    DecisionResult,
    Evidence,
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    ProviderCapabilities,
    SkippedReason,
    canonicalize,
    check_decision_result,
    check_generation_result,
    compute_decision_hash,
    compute_generation_hash,
    error_result,
    finite_float,
    valid_cost_estimate,
    generation_error_result,
    generation_skipped_result,
    skipped_result,
)
from graphistry.compute.typing import DataFrameT


QuestionLike = Union[DecisionQuestion, Mapping[str, object]]
BudgetSpec = Union[Budget, Mapping[str, object], None]

DEFAULT_DECIDE_BATCH_SIZE = 32
DEFAULT_MAP_BATCH_SIZE = 16

COST_SLACK_USD = 1e-12
COST_SLACK_RELATIVE = 1e-9
RequestT = TypeVar("RequestT", DecisionRequest, GenerationRequest)
ResultT = TypeVar("ResultT", DecisionResult, GenerationResult)


@dataclasses.dataclass(frozen=True)
class _CallOutcome(Generic[ResultT]):
    returned: Optional[Sequence[ResultT]]
    raised: bool
    skipped_reason: Optional[SkippedReason]


@dataclasses.dataclass(frozen=True)
class _Prepared:
    frame: pd.DataFrame
    request_ids: List[str]
    evidence: List[Evidence]
    attempt: List[bool]
    skip_reason: List[Optional[SkippedReason]]


def coerce_questions(questions: Sequence[QuestionLike]) -> Tuple[DecisionQuestion, ...]:
    """Questions as ``DecisionQuestion`` objects; dicts use the keys id, kind, text, options, levels."""
    if isinstance(questions, (str, bytes, Mapping)) or not isinstance(questions, (list, tuple)):
        raise ValueError(f"questions must be a list of questions, got {type(questions).__name__}")
    if len(questions) == 0:
        raise ValueError("questions must contain at least one question")
    coerced: List[DecisionQuestion] = []
    for spec in questions:
        if isinstance(spec, DecisionQuestion):
            coerced.append(spec)
        elif isinstance(spec, Mapping):
            coerced.append(DecisionQuestion.from_mapping(spec))
        else:
            raise ValueError(f"each question must be a DecisionQuestion or a dict, got {type(spec).__name__}")
    ids = [question.id for question in coerced]
    if len(set(ids)) != len(ids):
        raise ValueError(f"question ids must be unique, got {ids}")
    for question_id in ids:
        validate_column_name(question_id, "Question id")
    return tuple(coerced)


def coerce_budget(budget: BudgetSpec) -> Optional[Budget]:
    if budget is None or isinstance(budget, Budget):
        return budget
    if not isinstance(budget, Mapping):
        raise ValueError(f"budget must be a Budget or a dict, got {type(budget).__name__}")
    return Budget.from_mapping(budget)


def template_fields(template: str) -> List[str]:
    """Placeholder names of a template; only bare ``{column}`` fields are allowed."""
    if not isinstance(template, str) or template == "":
        raise ValueError("template must be a non-empty string")
    fields: List[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"template is not a valid format string: {error}") from error
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(f"template fields must name a column, got {{{field_name}}}")
        if any(ch in field_name for ch in ".[]"):
            raise ValueError(f"template field {{{field_name}}} must be a bare column name")
        if conversion is not None or format_spec:
            raise ValueError(f"template field {{{field_name}}} must not use a conversion or format spec")
        if field_name not in fields:
            fields.append(field_name)
    if not fields:
        raise ValueError("template must reference at least one column")
    return fields


def evidence_columns(evidence_cols: Optional[Sequence[str]], template: Optional[str]) -> List[str]:
    """Columns the evidence reads, after the exactly-one-of check."""
    if (evidence_cols is None) == (template is None):
        raise ValueError("pass exactly one of evidence_cols or template")
    if template is not None:
        return template_fields(template)
    if isinstance(evidence_cols, (str, bytes)) or not isinstance(evidence_cols, (list, tuple)):
        raise ValueError("evidence_cols must be a list of column names")
    cols = list(evidence_cols)
    if not cols or not all(isinstance(col, str) and col != "" for col in cols):
        raise ValueError("evidence_cols must be a non-empty list of column names")
    if len(set(cols)) != len(cols):
        raise ValueError(f"evidence_cols must be unique, got {cols}")
    return cols


def _check_int(value: object, name: str, minimum: int = 1) -> Optional[int]:
    if value is None:
        return None
    number = canonicalize(value)
    if isinstance(number, int) and not isinstance(number, bool) and number >= minimum:
        return number
    raise ValueError(f"{name} must be an int >= {minimum}, got {number!r}")


def _check_number(value: object, name: str, low: float, high: float, low_open: bool = False) -> None:
    if value is None:
        return
    number = finite_float(canonicalize(value))
    if number is None:
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if number > high or number < low or (low_open and number == low):
        raise ValueError(f"{name} out of range, got {value!r}")


def _check_text(value: object, name: str) -> None:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _is_cudf(df: object) -> bool:
    return type(df).__module__.startswith("cudf")


def _to_host(frame: DataFrameT) -> pd.DataFrame:
    if _is_cudf(frame):
        return frame.to_pandas()
    return frame


def _truncate_strings(frame: pd.DataFrame, max_chars: Optional[int]) -> pd.DataFrame:
    if max_chars is None:
        return frame
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        elif series.dtype != object and not pd.api.types.is_string_dtype(series.dtype):
            continue
        is_text = series.map(lambda value: isinstance(value, str)).to_numpy(dtype=bool)
        if not is_text.any():
            continue
        values = series.to_numpy(dtype=object).copy()
        positions = np.flatnonzero(is_text)
        values[positions] = series.iloc[positions].str.slice(0, max_chars).to_numpy(dtype=object)
        out[col] = values
    return out


def _json_ready(value: object) -> object:
    value = canonicalize(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _row_request_ids(df: DataFrameT, host: pd.DataFrame, id_col: Optional[str]) -> List[str]:
    if id_col is not None:
        ids = host[id_col]
        if ids.isna().any():
            raise ValueError(f"id_col {id_col!r} contains nulls")
        as_text = [str(value) for value in ids.tolist()]
        if len(set(as_text)) != len(as_text):
            raise ValueError(f"id_col {id_col!r} values are not unique after str()")
        if "" in as_text:
            raise ValueError(f"id_col {id_col!r} contains an empty id")
        return as_text
    labels = df.index.to_pandas().tolist() if _is_cudf(df) else df.index.tolist()
    from_index = [str(label) for label in labels]
    if df.index.is_unique and "" not in from_index and len(set(from_index)) == len(from_index):
        return from_index
    return [str(position) for position in range(len(df))]


def _prepare_rows(df: DataFrameT, cols: List[str], template: Optional[str], id_col: Optional[str],
                  column_max_chars: Optional[int], max_rows: Optional[int]) -> _Prepared:
    missing = [col for col in cols + ([id_col] if id_col is not None else []) if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    duplicated = sorted({col for col in cols + ([id_col] if id_col is not None else [])
                         if list(df.columns).count(col) > 1})
    if duplicated:
        raise ValueError(f"columns appear more than once in the table: {duplicated}")
    needed = list(dict.fromkeys(cols + ([id_col] if id_col is not None else [])))
    host = _to_host(df[needed])
    request_ids = _row_request_ids(df, host, id_col)
    frame = _truncate_strings(host[cols], column_max_chars)
    all_null = frame.isna().all(axis=1).tolist()
    evidence: List[Evidence] = []
    # one provider item per row is inherent; cuDF evidence moved to host once above
    for record in frame.to_dict("records"):
        if template is None:
            evidence.append({str(key): _json_ready(value) for key, value in record.items()})
        else:
            evidence.append(template.format_map({
                key: ("" if canonicalize(value) is None else value) for key, value in record.items()
            }))
    skip_reason: List[Optional[SkippedReason]] = []
    for position, is_null in enumerate(all_null):
        if max_rows is not None and position >= max_rows:
            skip_reason.append("max_rows")
        elif is_null:
            skip_reason.append("null_evidence")
        else:
            skip_reason.append(None)
    attempt = [reason is None for reason in skip_reason]
    return _Prepared(frame=frame, request_ids=request_ids, evidence=evidence, attempt=attempt,
                     skip_reason=skip_reason)


def _output_names(names: Sequence[str], df: DataFrameT, overwrite: bool) -> None:
    if len(set(names)) != len(names):
        dupes = sorted(name for name, count in Counter(names).items() if count > 1)
        raise ValueError(f"derived output column names collide: {dupes}")
    existing = [name for name in names if name in df.columns]
    if existing and not overwrite:
        raise ValueError(f"output columns already exist: {existing}; pass overwrite=True to replace them")


def _provider_capabilities(provider: Union[DecisionProvider, GenerationProvider]) -> ProviderCapabilities:
    try:
        capabilities = provider.capabilities()
    except Exception as error:
        raise ValueError(f"provider {provider.name!r} capabilities() failed: {type(error).__name__}") from error
    if not isinstance(capabilities, ProviderCapabilities):
        raise ValueError(f"provider {provider.name!r} capabilities() must return ProviderCapabilities")
    return capabilities


def _first_estimate(estimate: Callable[[], Optional[float]]) -> Optional[float]:
    try:
        return valid_cost_estimate(estimate())
    except Exception:
        return None


def _thread_count(max_concurrency: Optional[int], capabilities: ProviderCapabilities, calls: int) -> int:
    limit = capabilities.concurrency_limit
    if max_concurrency is not None:
        limit = min(limit, max_concurrency)
    return max(1, min(limit, calls))


def _run_provider_calls(
    batches: Sequence[Sequence[RequestT]],
    estimate_raw: Callable[[Sequence[RequestT]], Optional[float]],
    invoke: Callable[[Sequence[RequestT], Optional[float], Optional[Budget]], Sequence[ResultT]],
    budget: Optional[Budget],
    deadline_at: Optional[float],
    threads: int,
    effect_name: str,
) -> List[_CallOutcome[ResultT]]:
    """Submit provider calls in order under the request, cost and deadline limits.

    Provider calls are the isolation boundary: any exception from a provider becomes that call's
    ``provider_exception`` results, never an exception out of the run.
    """

    def estimate(batch: Sequence[RequestT]) -> Optional[float]:
        try:
            return valid_cost_estimate(estimate_raw(batch))
        except Exception:
            return None

    outcomes: List[Optional[_CallOutcome[ResultT]]] = [None] * len(batches)
    plan: List[Tuple[int, Optional[Budget]]] = []
    reserved = 0.0
    limit_hit = False
    for position, batch in enumerate(batches):
        cost: Optional[float] = None
        if not limit_hit and budget is not None:
            if budget.max_requests is not None and len(plan) >= budget.max_requests:
                limit_hit = True
            elif budget.max_cost_usd is not None:
                cost = estimate(batch)
                if cost is None or reserved + cost > budget.max_cost_usd * (1 + COST_SLACK_RELATIVE) + COST_SLACK_USD:
                    limit_hit = True
        if limit_hit:
            outcomes[position] = _CallOutcome(None, False, "budget")
            continue
        remaining_budget: Optional[Budget] = None
        if budget is not None:
            remaining_cost: Optional[float] = None
            if budget.max_cost_usd is not None:
                remaining_cost = max(0.0, budget.max_cost_usd - reserved)
                reserved += cost or 0.0
            remaining_requests = None if budget.max_requests is None else budget.max_requests - len(plan)
            remaining_budget = Budget(max_cost_usd=remaining_cost, max_requests=remaining_requests)
        plan.append((position, remaining_budget))

    start_lock = threading.Lock()
    started_calls: List[int] = []
    closed: List[bool] = []

    def run_one(position: int, call_budget: Optional[Budget]) -> _CallOutcome[ResultT]:
        with start_lock:
            remaining_s = None if deadline_at is None else deadline_at - time.monotonic()
            if closed or (remaining_s is not None and remaining_s <= 0):
                return _CallOutcome(None, False, "deadline")
            started_calls.append(position)
        try:
            return _CallOutcome(invoke(batches[position], remaining_s, call_budget), False, None)
        except Exception:
            return _CallOutcome(None, True, None)

    if plan:
        pool = ThreadPoolExecutor(max_workers=threads)
        try:
            futures = {pool.submit(run_one, position, call_budget): position for position, call_budget in plan}
            timeout = None if deadline_at is None else min(threading.TIMEOUT_MAX, max(0.0, deadline_at - time.monotonic()))
            done, not_done = wait(list(futures), timeout=timeout)
            for future in done:
                outcomes[futures[future]] = future.result()
            for future in not_done:
                future.cancel()
                outcomes[futures[future]] = _CallOutcome(None, False, "deadline")
        finally:
            with start_lock:
                closed.append(True)
                any_started = bool(started_calls)
            pool.shutdown(wait=False)
            if any_started:
                record_effect(effect_name)
    return [outcome if outcome is not None else _CallOutcome(None, False, "deadline") for outcome in outcomes]


def _valid_latency(value: object) -> Optional[float]:
    """A provider-reported latency kept only when it is a finite number >= 0."""
    return valid_cost_estimate(value)


def _valid_retries(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _chunks(items: Sequence[int], size: int) -> List[List[int]]:
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def _probability_of_value(question: DecisionQuestion, result: DecisionResult) -> float:
    if result.distribution is None or result.value is None:
        return math.nan
    key = ("true" if result.value else "false") if question.kind == "predicate" else str(result.value)
    probability = result.distribution.get(key)
    return float(probability) if probability is not None else math.nan


def _meta_json(result: Union[DecisionResult, GenerationResult], with_confidence: bool) -> str:
    meta: Dict[str, object] = {
        "model": result.model,
        "provider": result.provider,
        "task_version": result.task_version,
        "input_hash": result.input_hash,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "cache_hit": result.cache_hit,
        "retry_count": result.retry_count,
        "status_reason": result.status_reason,
        "extra": None if result.extra is None else dict(result.extra),
    }
    if with_confidence and isinstance(result, DecisionResult):
        meta["confidence"] = result.confidence
    return json.dumps(_json_ready(meta), sort_keys=True, separators=(",", ":"))


def _dist_json(question: DecisionQuestion, result: DecisionResult) -> Optional[str]:
    if result.distribution is None:
        return None
    ordered = {key: float(result.distribution[key]) for key in question.distribution_keys()
               if key in result.distribution}
    return json.dumps(ordered, separators=(",", ":"))


def _assign_columns(df: DataFrameT, columns: Dict[str, object]) -> DataFrameT:
    out = df.copy(deep=False)
    if _is_cudf(df):
        import cudf  # type: ignore[import-not-found]
        import pyarrow as pa
        for name, values in columns.items():
            # explicit arrow strings: object dtype on every cudf version and mode, even when all null
            series = (cudf.Series(pa.array(values, type=pa.string())) if pd.api.types.is_object_dtype(values)
                      else cudf.from_pandas(pd.Series(values)))
            series.index = out.index
            out[name] = series
        return out
    for name, values in columns.items():
        out[name] = values
    return out


def _decision_columns(questions: Sequence[DecisionQuestion], grid: Dict[Tuple[int, str], DecisionResult],
                      rows: int) -> Dict[str, object]:
    columns: Dict[str, object] = {}
    for question in questions:
        results = [grid[(position, question.id)] for position in range(rows)]
        if question.kind == "predicate":
            columns[question.id] = pd.array([result.value for result in results], dtype="boolean")
        else:
            columns[question.id] = np.array([result.value for result in results], dtype=object)
        columns[f"{question.id}__status"] = np.array([result.status for result in results], dtype=object)
        columns[f"{question.id}__p"] = np.array([_probability_of_value(question, result) for result in results],
                                                dtype="float64")
        columns[f"{question.id}__dist"] = np.array([_dist_json(question, result) for result in results],
                                                   dtype=object)
        columns[f"{question.id}__probability_kind"] = np.array(
            [result.probability_kind for result in results], dtype=object)
        columns[f"{question.id}__error_code"] = np.array([result.error_code for result in results], dtype=object)
        columns[f"{question.id}__meta"] = np.array([_meta_json(result, True) for result in results], dtype=object)
    return columns


def _apply_min_p(question: DecisionQuestion, result: DecisionResult, min_p: Optional[float]) -> DecisionResult:
    if min_p is None or result.status != "ok":
        return result
    probability = _probability_of_value(question, result)
    if math.isnan(probability) or probability < min_p:
        return dataclasses.replace(result, status="uncertain", status_reason="min_p")
    return result


def _decision_call_results(
    requests: Sequence[DecisionRequest], positions: Sequence[int], outcome: _CallOutcome[DecisionResult],
    provider: DecisionProvider, capabilities: ProviderCapabilities, hashes: Dict[Tuple[int, str], str],
) -> Dict[Tuple[int, str], DecisionResult]:
    """Apply the one checking rule to a provider return, keyed by (row position, question id)."""
    by_request = {request.request_id: (request, position) for request, position in zip(requests, positions)}
    expected = {(request.request_id, question.id): (request, position, question)
                for request, position in zip(requests, positions) for question in request.questions}

    def everything(make: Callable[[DecisionRequest, DecisionQuestion], DecisionResult]
                   ) -> Dict[Tuple[int, str], DecisionResult]:
        return {(position, question.id): make(request, question)
                for (_rid, _qid), (request, position, question) in expected.items()}

    def as_error(code: str) -> Callable[[DecisionRequest, DecisionQuestion], DecisionResult]:
        return lambda request, question: error_result(request, question, error_code=code, provider=provider.name,
                                                      model=provider.model, retry_count=None)

    if outcome.skipped_reason is not None:
        reason = outcome.skipped_reason
        return everything(lambda request, question: skipped_result(
            request, question, provider=provider.name, model=provider.model, status_reason=reason))
    if outcome.raised:
        return everything(as_error("provider_exception"))
    returned = outcome.returned
    if not isinstance(returned, (list, tuple)):
        return everything(as_error("invalid_provider_result"))
    found: Dict[Tuple[str, str], DecisionResult] = {}
    seen_keys = set()
    bad_rows = set()
    for result in returned:
        if not isinstance(result, DecisionResult) or not isinstance(result.request_id, str) \
                or result.request_id not in by_request:
            return everything(as_error("invalid_provider_result"))
        if not isinstance(result.question_id, str):
            bad_rows.add(result.request_id)
            continue
        key = (result.request_id, result.question_id)
        if key in seen_keys:
            return everything(as_error("invalid_provider_result"))
        seen_keys.add(key)
        if key not in expected:
            bad_rows.add(result.request_id)
            continue
        found[key] = result
    out: Dict[Tuple[int, str], DecisionResult] = {}
    for request_id, (request, position) in by_request.items():
        row_complete = request_id not in bad_rows and all(
            (request_id, question.id) in found for question in request.questions)
        for question in request.questions:
            result = found.get((request_id, question.id))
            if not row_complete or result is None:
                out[(position, question.id)] = as_error("invalid_provider_result")(request, question)
                continue
            if check_decision_result(request, question, result, capabilities):
                out[(position, question.id)] = error_result(
                    request, question, error_code="invalid_provider_result", provider=provider.name,
                    model=provider.model, latency_ms=_valid_latency(result.latency_ms),
                    retry_count=_valid_retries(result.retry_count))
                continue
            out[(position, question.id)] = dataclasses.replace(result, input_hash=hashes[(position, question.id)])
    return out


def sem_decide(
    df: DataFrameT,
    *,
    questions: Sequence[QuestionLike],
    provider: Union[str, DecisionProvider],
    task_id: str,
    evidence_cols: Optional[Sequence[str]] = None,
    template: Optional[str] = None,
    id_col: Optional[str] = None,
    task_version: str = "1",
    batch_size: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    deadline_s: Optional[float] = None,
    budget: BudgetSpec = None,
    max_rows: Optional[int] = None,
    min_p: Optional[float] = None,
    column_max_chars: Optional[int] = None,
    overwrite: bool = False,
) -> DataFrameT:
    """Ask a model typed questions about every row and append one set of result columns per question.

    For each question ``q`` the output adds ``q`` (the answer), ``q__status`` (``ok``, ``uncertain``,
    ``error`` or ``skipped``), ``q__p`` (the probability the answer distribution gives the answer),
    ``q__dist`` (the distribution as JSON), ``q__probability_kind``, ``q__error_code`` and ``q__meta``
    (JSON with model, provider, input hash, cost, latency, retries, provider confidence, status reason).

    :param df: pandas or cuDF table; rows, order and index are preserved
    :param questions: ``DecisionQuestion`` objects or dicts ``{id, kind, text, options?, levels?}`` where
        ``kind`` is ``predicate``, ``choice`` (needs ``options``) or ``score`` (needs ordered ``levels``)
    :param provider: a ``DecisionProvider`` or an alias bound with ``bind_providers``
    :param task_id: caller label for this task
    :param evidence_cols: columns sent as a ``{column: value}`` mapping (exactly one of this or ``template``)
    :param template: text with bare ``{column}`` placeholders, substituted verbatim
    :param id_col: column whose ``str`` values identify rows; default the index when its labels are unique (also
        after ``str()``, and non-empty), else row positions
    :param task_version: version label included in the input hash
    :param batch_size: rows per provider call; default the provider's ``max_rows_per_request``, else 32
    :param max_concurrency: upper bound on concurrent provider calls
    :param deadline_s: seconds for the whole run; unfinished rows become ``skipped`` (calls already sent are
        abandoned, not cancelled, so they may still finish in the background)
    :param budget: ``Budget`` or dict with ``max_cost_usd`` and/or ``max_requests``
    :param max_rows: rows at positions >= ``max_rows`` are ``skipped``, never dropped
    :param min_p: ``ok`` answers whose ``q__p`` is below this (or undefined) become ``uncertain``
    :param column_max_chars: truncate each string evidence value to this many characters
    :param overwrite: replace existing output columns instead of raising
    :returns: a copy of ``df`` with the result columns appended

    **Example**
        ::

            out = sem_decide(df, questions=[{"id": "is_vendor", "kind": "predicate",
                                             "text": "Is this company a software vendor?"}],
                             provider=my_provider, evidence_cols=["company", "description"],
                             task_id="vendors")
    """
    started = time.monotonic()
    coerced = coerce_questions(questions)
    _check_text(task_id, "task_id")
    _check_text(task_version, "task_version")
    cols = evidence_columns(evidence_cols, template)
    _check_int(batch_size, "batch_size")
    _check_int(max_concurrency, "max_concurrency")
    _check_int(max_rows, "max_rows")
    _check_int(column_max_chars, "column_max_chars")
    _check_number(deadline_s, "deadline_s", 0.0, math.inf, low_open=True)
    _check_number(min_p, "min_p", 0.0, 1.0)
    if not isinstance(overwrite, bool):
        raise ValueError(f"overwrite must be a bool, got {overwrite!r}")
    run_budget = coerce_budget(budget)
    names = [name for question in coerced for name in [question.id] + [question.id + s for s in DECISION_SUFFIXES]]
    _output_names(names, df, overwrite)
    resolved = resolve_decision_provider(provider)
    capabilities = _provider_capabilities(resolved)
    _check_question_capabilities(coerced, capabilities)
    prepared = _prepare_rows(df, cols, template, id_col, column_max_chars, max_rows)

    rows = len(prepared.request_ids)
    hashes = {(position, question.id): compute_decision_hash(prepared.evidence[position], question, task_version)
              for position in range(rows) for question in coerced}
    question_chunks = [list(coerced[start:start + capabilities.max_questions_per_request])
                       for start in range(0, len(coerced), capabilities.max_questions_per_request)]
    size = batch_size or capabilities.max_rows_per_request or DEFAULT_DECIDE_BATCH_SIZE
    attempted = [position for position in range(rows) if prepared.attempt[position]]
    calls: List[Tuple[List[int], List[DecisionRequest]]] = []
    for row_batch in _chunks(attempted, size):
        for chunk in question_chunks:
            calls.append((row_batch, [
                DecisionRequest(request_id=prepared.request_ids[position], evidence=prepared.evidence[position],
                                questions=tuple(chunk), task_id=task_id, task_version=task_version)
                for position in row_batch
            ]))
    if run_budget is not None and run_budget.max_cost_usd is not None and calls \
            and _first_estimate(lambda: resolved.estimate_cost_usd(calls[0][1])) is None:
        raise ValueError("budget.max_cost_usd is set but the provider cannot estimate cost; drop the cost cap")

    deadline_at = None if deadline_s is None else started + deadline_s
    outcomes = _run_provider_calls(
        [requests for _positions, requests in calls],
        resolved.estimate_cost_usd,
        lambda batch, remaining, call_budget: resolved.decide_batch(batch, deadline_s=remaining, budget=call_budget),
        run_budget, deadline_at, _thread_count(max_concurrency, capabilities, len(calls)), "sem_decide",
    )
    grid: Dict[Tuple[int, str], DecisionResult] = {}
    for (positions, requests), outcome in zip(calls, outcomes):
        grid.update(_decision_call_results(requests, positions, outcome, resolved, capabilities, hashes))
    for position in range(rows):
        reason = prepared.skip_reason[position]
        if reason is None:
            continue
        request = DecisionRequest(request_id=prepared.request_ids[position], evidence=prepared.evidence[position],
                                  questions=coerced, task_id=task_id, task_version=task_version)
        for question in coerced:
            grid[(position, question.id)] = skipped_result(request, question, provider=resolved.name,
                                                           model=resolved.model, status_reason=reason)
    by_id = {question.id: question for question in coerced}
    for key, result in list(grid.items()):
        grid[key] = _apply_min_p(by_id[key[1]], result, min_p)
    return _assign_columns(df, _decision_columns(coerced, grid, rows))


def _check_question_capabilities(questions: Sequence[DecisionQuestion], capabilities: ProviderCapabilities) -> None:
    for question in questions:
        if capabilities.supported_kinds is not None and question.kind not in capabilities.supported_kinds:
            raise ValueError(f"provider does not support {question.kind} questions (question {question.id!r})")
        if question.kind == "choice" and capabilities.max_choice_options is not None \
                and len(question.options or ()) > capabilities.max_choice_options:
            raise ValueError(f"question {question.id!r} has more than {capabilities.max_choice_options} options")
        if question.kind == "score" and capabilities.max_levels is not None \
                and len(question.levels or ()) > capabilities.max_levels:
            raise ValueError(f"question {question.id!r} has more than {capabilities.max_levels} levels")


def _generation_call_results(
    requests: Sequence[GenerationRequest], positions: Sequence[int], outcome: _CallOutcome[GenerationResult],
    provider: GenerationProvider, hashes: Dict[int, str],
) -> Dict[int, GenerationResult]:
    """Apply the one checking rule to a generation provider return, keyed by row position."""
    by_request = {request.request_id: (request, position) for request, position in zip(requests, positions)}

    def everything(code: str) -> Dict[int, GenerationResult]:
        return {position: generation_error_result(request, error_code=code, provider=provider.name,
                                                  model=provider.model, retry_count=None)
                for request, position in by_request.values()}

    if outcome.skipped_reason is not None:
        reason = outcome.skipped_reason
        return {position: generation_skipped_result(request, provider=provider.name, model=provider.model,
                                                    status_reason=reason)
                for request, position in by_request.values()}
    if outcome.raised:
        return everything("provider_exception")
    returned = outcome.returned
    if not isinstance(returned, (list, tuple)):
        return everything("invalid_provider_result")
    found: Dict[str, GenerationResult] = {}
    for result in returned:
        if not isinstance(result, GenerationResult) or not isinstance(result.request_id, str) \
                or result.request_id not in by_request or result.request_id in found:
            return everything("invalid_provider_result")
        found[result.request_id] = result
    out: Dict[int, GenerationResult] = {}
    for request_id, (request, position) in by_request.items():
        result = found.get(request_id)
        if result is None or check_generation_result(request, result):
            out[position] = generation_error_result(
                request, error_code="invalid_provider_result", provider=provider.name, model=provider.model,
                latency_ms=None if result is None else _valid_latency(result.latency_ms),
                retry_count=None if result is None else _valid_retries(result.retry_count))
            continue
        out[position] = dataclasses.replace(result, input_hash=hashes[position])
    return out


def sem_map(
    df: DataFrameT,
    *,
    instruction: str,
    out_col: str,
    provider: Union[str, GenerationProvider],
    task_id: str,
    evidence_cols: Optional[Sequence[str]] = None,
    template: Optional[str] = None,
    id_col: Optional[str] = None,
    task_version: str = "1",
    max_output_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    deadline_s: Optional[float] = None,
    budget: BudgetSpec = None,
    max_rows: Optional[int] = None,
    column_max_chars: Optional[int] = None,
    overwrite: bool = False,
) -> DataFrameT:
    """Generate one text value per row and append ``out_col`` plus its status, error code and meta columns.

    Parameters, row identity, limits and failure handling match :func:`sem_decide`, except that ``batch_size``
    defaults to the provider's ``max_rows_per_request``, else 16; ``instruction`` is the generation instruction
    and ``max_output_tokens`` bounds each generated value.

    **Example**
        ::

            out = sem_map(df, instruction="Summarize the description in one sentence.",
                          out_col="summary", provider=my_generator, evidence_cols=["description"],
                          task_id="summaries")
    """
    started = time.monotonic()
    _check_text(instruction, "instruction")
    _check_text(out_col, "out_col")
    validate_column_name(out_col, "out_col")
    _check_text(task_id, "task_id")
    _check_text(task_version, "task_version")
    cols = evidence_columns(evidence_cols, template)
    max_output_tokens = _check_int(max_output_tokens, "max_output_tokens")
    _check_int(batch_size, "batch_size")
    _check_int(max_concurrency, "max_concurrency")
    _check_int(max_rows, "max_rows")
    _check_int(column_max_chars, "column_max_chars")
    _check_number(deadline_s, "deadline_s", 0.0, math.inf, low_open=True)
    if not isinstance(overwrite, bool):
        raise ValueError(f"overwrite must be a bool, got {overwrite!r}")
    run_budget = coerce_budget(budget)
    _output_names([out_col] + [out_col + suffix for suffix in GENERATION_SUFFIXES], df, overwrite)
    resolved = resolve_generation_provider(provider)
    capabilities = _provider_capabilities(resolved)
    prepared = _prepare_rows(df, cols, template, id_col, column_max_chars, max_rows)

    rows = len(prepared.request_ids)
    hashes = {position: compute_generation_hash(prepared.evidence[position], instruction, task_version)
              for position in range(rows)}
    size = batch_size or capabilities.max_rows_per_request or DEFAULT_MAP_BATCH_SIZE
    attempted = [position for position in range(rows) if prepared.attempt[position]]
    calls: List[Tuple[List[int], List[GenerationRequest]]] = [
        (row_batch, [
            GenerationRequest(request_id=prepared.request_ids[position], evidence=prepared.evidence[position],
                              instruction=instruction, task_id=task_id, task_version=task_version,
                              max_output_tokens=max_output_tokens)
            for position in row_batch
        ])
        for row_batch in _chunks(attempted, size)
    ]
    if run_budget is not None and run_budget.max_cost_usd is not None and calls \
            and _first_estimate(lambda: resolved.estimate_cost_usd(calls[0][1])) is None:
        raise ValueError("budget.max_cost_usd is set but the provider cannot estimate cost; drop the cost cap")

    deadline_at = None if deadline_s is None else started + deadline_s
    outcomes = _run_provider_calls(
        [requests for _positions, requests in calls],
        resolved.estimate_cost_usd,
        lambda batch, remaining, call_budget: resolved.generate_batch(batch, deadline_s=remaining,
                                                                      budget=call_budget),
        run_budget, deadline_at, _thread_count(max_concurrency, capabilities, len(calls)), "sem_map",
    )
    results: Dict[int, GenerationResult] = {}
    for (positions, requests), outcome in zip(calls, outcomes):
        results.update(_generation_call_results(requests, positions, outcome, resolved, hashes))
    for position in range(rows):
        reason = prepared.skip_reason[position]
        if reason is None:
            continue
        request = GenerationRequest(request_id=prepared.request_ids[position], evidence=prepared.evidence[position],
                                    instruction=instruction, task_id=task_id, task_version=task_version)
        results[position] = generation_skipped_result(request, provider=resolved.name, model=resolved.model,
                                                      status_reason=reason)
    ordered = [results[position] for position in range(rows)]
    return _assign_columns(df, {
        out_col: np.array([result.value for result in ordered], dtype=object),
        f"{out_col}__status": np.array([result.status for result in ordered], dtype=object),
        f"{out_col}__error_code": np.array([result.error_code for result in ordered], dtype=object),
        f"{out_col}__meta": np.array([_meta_json(result, False) for result in ordered], dtype=object),
    })
