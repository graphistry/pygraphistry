"""``sem_decide``: ask a model provider typed questions about every row of a table.

Every input row is kept, in order and with its index, and gets typed result columns. Provider faults become
per-row ``error`` results; invalid parameters raise ``ValueError`` before any provider call.
"""
import dataclasses
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from graphistry.compute.gfql.effects import record_effect
from graphistry.compute.semantic.binding import resolve_provider
from graphistry.compute.semantic.contract import (
    Budget, DecisionProvider, DecisionQuestion, DecisionRequest, DecisionResult, Evidence, ProviderCapabilities,
    SkippedReason, align_results, canonicalize, check_decision_result, compute_decision_hash, finite_float,
    unanswered_result)
from graphistry.compute.semantic.rules import (
    DECISION_COLUMN_TYPES, BudgetSpec, QuestionLike, check_params, coerce_budget, coerce_questions, decision_columns,
    evidence_columns)
from graphistry.compute.typing import DataFrameT


DEFAULT_BATCH_SIZE = 32
Cell = Tuple[int, str]
Invoke = Callable[[Sequence[DecisionRequest], Optional[float], Optional[Budget]], object]


@dataclasses.dataclass(frozen=True)
class _Rows:
    request_ids: List[str]
    evidence: List[Evidence]
    skip_reason: List[Optional[SkippedReason]]


@dataclasses.dataclass(frozen=True)
class _CallOutcome:
    returned: object = None
    raised: bool = False
    skipped: Optional[SkippedReason] = None


def _is_cudf(df: object) -> bool:
    return type(df).__module__.startswith("cudf")


def _json_ready(value: object) -> object:
    value = canonicalize(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value if value is None or isinstance(value, (str, bool, int, float)) else str(value)


def _distinct_ids(ids: List[str]) -> bool:
    return "" not in ids and len(set(ids)) == len(ids)


def _row_request_ids(host: pd.DataFrame, id_col: Optional[str]) -> List[str]:
    if id_col is not None:
        ids = [str(value) for value in host[id_col].tolist()]
        if host[id_col].isna().any() or not _distinct_ids(ids):
            raise ValueError(f"id_col {id_col!r} must hold non-null, unique ids with no empty id after str()")
        return ids
    labels = [str(label) for label in host.index.tolist()]
    return labels if host.index.is_unique and _distinct_ids(labels) else [str(position) for position in range(len(host))]


def _row_evidence(record: Mapping[object, object], template: Optional[str], max_chars: Optional[int]) -> Evidence:
    values = {str(key): value[:max_chars] if max_chars is not None and isinstance(value, str) else value
              for key, value in record.items()}
    if template is None:
        return {key: _json_ready(value) for key, value in values.items()}
    return template.format_map({key: "" if canonicalize(value) is None else value for key, value in values.items()})


def _skip_reason(position: int, all_null: bool, max_rows: Optional[int]) -> Optional[SkippedReason]:
    if max_rows is not None and position >= max_rows:
        return "max_rows"
    return "null_evidence" if all_null else None


def _prepare_rows(df: DataFrameT, cols: List[str], template: Optional[str], id_col: Optional[str],
                  max_chars: Optional[int], max_rows: Optional[int]) -> _Rows:
    needed = list(dict.fromkeys(cols + ([] if id_col is None else [id_col])))
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    duplicated = [col for col in needed if list(df.columns).count(col) > 1]
    if duplicated:
        raise ValueError(f"columns appear more than once in the table: {duplicated}")
    host = df[needed].to_pandas() if _is_cudf(df) else df[needed]
    return _Rows(
        request_ids=_row_request_ids(host, id_col),
        evidence=[_row_evidence(record, template, max_chars) for record in host[cols].to_dict("records")],
        skip_reason=[_skip_reason(position, all_null, max_rows)
                     for position, all_null in enumerate(host[cols].isna().all(axis=1).tolist())],
    )


def _check_existing_columns(names: Sequence[str], df: DataFrameT, overwrite: bool) -> None:
    existing = [name for name in names if name in df.columns]
    if existing and not overwrite:
        raise ValueError(f"output columns already exist: {existing}; pass overwrite=True to replace them")


def _provider_capabilities(provider: DecisionProvider) -> ProviderCapabilities:
    """Provider code is untrusted: a failing ``capabilities()`` is a ``ValueError`` before any call."""
    try:
        capabilities = provider.capabilities()
    except Exception as error:
        raise ValueError(f"provider {provider.name!r} capabilities() failed: {type(error).__name__}") from error
    if not isinstance(capabilities, ProviderCapabilities):
        raise ValueError(f"provider {provider.name!r} capabilities() must return ProviderCapabilities")
    return capabilities


def _check_question_capabilities(questions: Sequence[DecisionQuestion], capabilities: ProviderCapabilities) -> None:
    limits = {"choice": (capabilities.max_choice_options, "options"), "score": (capabilities.max_levels, "levels")}
    for question in questions:
        if capabilities.supported_kinds is not None and question.kind not in capabilities.supported_kinds:
            raise ValueError(f"provider does not support {question.kind} questions (question {question.id!r})")
        limit, labels = limits.get(question.kind, (None, ""))
        if limit is not None and len(question.labels() or ()) > limit:
            raise ValueError(f"question {question.id!r} has more than {limit} {labels}")


def _estimate_cost(provider: DecisionProvider, batch: Sequence[DecisionRequest]) -> Optional[float]:
    """Provider code is untrusted: a failing or invalid estimate means "cannot estimate"."""
    try:
        return _nonnegative(canonicalize(provider.estimate_cost_usd(batch)))
    except Exception:
        return None


def _plan_calls(batches: Sequence[Sequence[DecisionRequest]], budget: Optional[Budget],
                estimate: Callable[[Sequence[DecisionRequest]], Optional[float]]) -> List[Optional[Budget]]:
    """The remaining budget handed to each call the budget allows, in order; later calls are skipped (``budget``).

    Each call reserves its estimated cost before submission, so which calls run never depends on completion order.
    """
    if budget is None:
        return [None] * len(batches)
    plans: List[Optional[Budget]] = []
    reserved = 0.0
    for index, batch in enumerate(batches):
        if budget.max_requests is not None and index >= budget.max_requests:
            break
        cost = None if budget.max_cost_usd is None else estimate(batch)
        if budget.max_cost_usd is not None and (cost is None or reserved + cost > budget.max_cost_usd * (1 + 1e-9) + 1e-12):
            break
        plans.append(Budget(max_cost_usd=None if budget.max_cost_usd is None else max(0.0, budget.max_cost_usd - reserved),
                            max_requests=None if budget.max_requests is None else budget.max_requests - index))
        reserved += cost or 0.0
    return plans


def _call_provider(invoke: Invoke, batch: Sequence[DecisionRequest], remaining_s: Optional[float],
                   budget: Optional[Budget]) -> _CallOutcome:
    """Provider calls are the isolation boundary: any exception is that call's ``provider_exception``."""
    try:
        return _CallOutcome(returned=invoke(batch, remaining_s, budget))
    except Exception:
        return _CallOutcome(raised=True)


def _execute(batches: Sequence[Sequence[DecisionRequest]], budgets: Sequence[Optional[Budget]], invoke: Invoke,
             deadline_at: Optional[float], threads: int, effect_name: str) -> List[_CallOutcome]:
    """Run the planned calls in order on a thread pool; at the deadline, unstarted and running calls are skipped."""
    outcomes = [_CallOutcome(skipped="budget")] * len(batches)
    lock, started, closed = threading.Lock(), threading.Event(), threading.Event()

    def run_one(position: int) -> _CallOutcome:
        with lock:
            remaining_s = None if deadline_at is None else deadline_at - time.monotonic()
            if closed.is_set() or (remaining_s is not None and remaining_s <= 0):
                return _CallOutcome(skipped="deadline")
            started.set()
        return _call_provider(invoke, batches[position], remaining_s, budgets[position])

    pool = ThreadPoolExecutor(max_workers=threads)
    try:
        futures = {pool.submit(run_one, position): position for position in range(len(budgets))}
        timeout = None if deadline_at is None else min(threading.TIMEOUT_MAX, max(0.0, deadline_at - time.monotonic()))
        done, not_done = wait(list(futures), timeout=timeout)
        for future in done:
            outcomes[futures[future]] = future.result()
        for future in not_done:
            future.cancel()
            outcomes[futures[future]] = _CallOutcome(skipped="deadline")
    finally:
        with lock:
            closed.set()
            any_started = started.is_set()
        pool.shutdown(wait=False)
        if any_started:
            record_effect(effect_name)
    return outcomes


def _probability_of_value(question: DecisionQuestion, result: DecisionResult) -> float:
    if result.distribution is None or result.value is None:
        return math.nan
    key = ("true" if result.value else "false") if question.kind == "predicate" else str(result.value)
    probability = result.distribution.get(key)
    return float(probability) if probability is not None else math.nan


def _nonnegative(value: object) -> Optional[float]:
    number = finite_float(value)
    return number if number is not None and number >= 0 else None


def _cell_result(request: DecisionRequest, question: DecisionQuestion, outcome: _CallOutcome,
                 row: Optional[Dict[str, DecisionResult]], identity: Tuple[str, str],
                 capabilities: ProviderCapabilities, min_p: Optional[float], input_hash: str) -> DecisionResult:
    """The kernel's verdict on one cell. Provider data is untrusted: if reading it fails, or its metadata cannot
    become JSON, the cell is an error built from kernel fields. ``input_hash`` (taken before any call) and
    ``task_version`` come from the kernel, so a provider cannot change them."""
    provider, model = identity
    if outcome.skipped is not None:
        return unanswered_result(request, question, provider=provider, model=model, input_hash=input_hash,
                                 status_reason=outcome.skipped)
    latency_ms: Optional[float] = None
    retry_count: Optional[int] = None
    try:
        result = None if row is None else _plain_answer(row[question.id])
        if result is not None and not check_decision_result(request, question, result, capabilities):
            result = dataclasses.replace(result, input_hash=input_hash, task_version=request.task_version)
            if min_p is not None and result.status == "ok":
                probability = _probability_of_value(question, result)
                if math.isnan(probability) or probability < min_p:
                    result = dataclasses.replace(result, status="uncertain", status_reason="min_p")
            if _meta_json_or_none(result) is not None:
                return result
        elif result is not None:
            retries = result.retry_count
            latency_ms = _nonnegative(result.latency_ms)
            retry_count = retries if isinstance(retries, int) and not isinstance(retries, bool) \
                and 0 <= retries < 2 ** 31 else None
    except Exception:
        latency_ms, retry_count = None, None
    return unanswered_result(
        request, question, provider=provider, model=model, input_hash=input_hash,
        error_code="provider_exception" if outcome.raised else "invalid_provider_result",
        latency_ms=latency_ms, retry_count=retry_count)


def _plain_answer(result: DecisionResult) -> Optional[DecisionResult]:
    """The result as an exact ``DecisionResult`` whose answer, distribution and status labels are plain Python
    values, so what is checked is what is written; None when they are not plain answer values. Reading provider
    values may raise; the caller's boundary turns that into an error cell."""
    value, distribution = canonicalize(result.value), canonicalize(result.distribution)
    raw_size = len(result.distribution) if isinstance(result.distribution, Mapping) else None
    if not isinstance(value, (bool, str, type(None))) or not isinstance(distribution, (dict, type(None))) \
            or (distribution is not None and raw_size != len(distribution)):
        return None
    fields = {field.name: getattr(result, field.name) for field in dataclasses.fields(DecisionResult)}
    labels = {name: str.__str__(fields[name]) for name in ("status", "status_reason", "probability_kind", "error_code")
              if isinstance(fields[name], str)}
    return DecisionResult(**dict(fields, value=value, distribution=distribution, **labels))


def _aligned_or_none(requests: Sequence[DecisionRequest], returned: object
                     ) -> Optional[Dict[str, Optional[Dict[str, DecisionResult]]]]:
    """``align_results`` on untrusted provider output; a failure means the whole call is untrusted."""
    try:
        return align_results(requests, returned)
    except Exception:
        return None


def _provider_identity(provider: DecisionProvider) -> Tuple[str, str]:
    """The provider name and model the kernel stamps on results it builds itself, always strings."""
    fallback = type(provider).__name__
    return (provider.name if isinstance(provider.name, str) else fallback,
            provider.model if isinstance(provider.model, str) else fallback)


def _meta_json(result: DecisionResult) -> str:
    meta = {name: getattr(result, name) for name in (
        "model", "provider", "task_version", "input_hash", "latency_ms", "cost_usd", "cache_hit", "retry_count",
        "status_reason", "confidence")}
    meta["extra"] = None if result.extra is None else dict(result.extra)
    return json.dumps(_json_ready(meta), sort_keys=True, separators=(",", ":"))


def _meta_json_or_none(result: DecisionResult) -> Optional[str]:
    """Provider metadata is untrusted: any that cannot become JSON (a non-mapping ``extra``, a cycle) is invalid."""
    try:
        return _meta_json(result)
    except Exception:
        return None


def _dist_json(question: DecisionQuestion, result: DecisionResult) -> Optional[str]:
    if result.distribution is None:
        return None
    return json.dumps({key: float(result.distribution[key]) for key in question.distribution_keys()
                       if key in result.distribution}, separators=(",", ":"))


CELL_VALUES: Dict[str, Callable[[DecisionQuestion, DecisionResult], object]] = {
    "": lambda question, result: result.value,
    "__status": lambda question, result: result.status,
    "__p": _probability_of_value,
    "__dist": _dist_json,
    "__probability_kind": lambda question, result: result.probability_kind,
    "__error_code": lambda question, result: result.error_code,
    "__meta": lambda question, result: _meta_json(result),
}
NUMPY_DTYPES: Dict[str, Union[type, str]] = {"string": object, "float64": "float64"}


def _result_columns(questions: Sequence[DecisionQuestion], grid: Dict[Cell, DecisionResult],
                    rows: int) -> Dict[str, object]:
    types = decision_columns([(question.id, question.kind) for question in questions])
    columns: Dict[str, object] = {}
    for question in questions:
        results = [grid[(position, question.id)] for position in range(rows)]
        for suffix, _logical in DECISION_COLUMN_TYPES:
            name = question.id + suffix
            values = [CELL_VALUES[suffix](question, result) for result in results]
            columns[name] = (pd.array(values, dtype="boolean") if types[name] == "bool"
                             else np.array(values, dtype=NUMPY_DTYPES[types[name]]))
    return columns


def _cudf_columns(df: DataFrameT, columns: Dict[str, object]) -> Dict[str, object]:
    import cudf
    import pyarrow as pa
    converted: Dict[str, object] = {}
    for name, values in columns.items():
        # explicit arrow strings: object dtype on every cudf version and mode, even when all null
        series = (cudf.Series(pa.array(values, type=pa.string())) if pd.api.types.is_object_dtype(values)
                  else cudf.from_pandas(pd.Series(values)))
        series.index = df.index
        converted[name] = series
    return converted


def _with_columns(df: DataFrameT, columns: Dict[str, object]) -> DataFrameT:
    """A copy of ``df`` with the columns set one by one (``assign`` cannot take a column named ``self``)."""
    out = df.assign()
    for name, values in (_cudf_columns(df, columns) if _is_cudf(df) else columns).items():
        out[name] = values
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
    :param batch_size: rows per provider call, at most the provider's ``max_rows_per_request``; default that
        limit, else 32
    :param max_concurrency: upper bound on concurrent provider calls within this run
    :param deadline_s: seconds for the whole run; unfinished rows become ``skipped`` (calls already sent are
        abandoned, not cancelled: they may still finish, and keep the process alive, in the background)
    :param budget: ``Budget`` or dict with ``max_cost_usd`` (summed provider estimates; a provider that cannot
        estimate raises) and/or ``max_requests`` (provider calls)
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
    check_params(dict(task_id=task_id, task_version=task_version, overwrite=overwrite),
                 dict(batch_size=batch_size, max_concurrency=max_concurrency, deadline_s=deadline_s,
                      max_rows=max_rows, min_p=min_p, column_max_chars=column_max_chars))
    cols = evidence_columns(evidence_cols, template)
    run_budget = coerce_budget(budget)
    deadline = None if deadline_s is None else float(deadline_s)
    threshold = None if min_p is None else float(min_p)
    _check_existing_columns([question.id + suffix for question in coerced for suffix, _ in DECISION_COLUMN_TYPES], df,
                            overwrite)
    resolved = resolve_provider(provider)
    if not isinstance(resolved, DecisionProvider):
        raise ValueError(f"semantic provider {provider if isinstance(provider, str) else type(provider).__name__!r} "
                         "is not a DecisionProvider")
    capabilities = _provider_capabilities(resolved)
    identity = _provider_identity(resolved)
    _check_question_capabilities(coerced, capabilities)
    rows = _prepare_rows(df, cols, template, id_col, column_max_chars, max_rows)

    def request(position: int, chunk: Sequence[DecisionQuestion]) -> DecisionRequest:
        return DecisionRequest(request_id=rows.request_ids[position], evidence=rows.evidence[position],
                               questions=tuple(chunk), task_id=task_id, task_version=task_version)

    per_call = capabilities.max_questions_per_request
    size = batch_size or capabilities.max_rows_per_request or DEFAULT_BATCH_SIZE
    if capabilities.max_rows_per_request is not None:
        size = min(size, capabilities.max_rows_per_request)
    attempted = [position for position, reason in enumerate(rows.skip_reason) if reason is None]
    calls: List[List[DecisionRequest]] = []
    for first in range(0, len(attempted), size):
        for start in range(0, len(coerced), per_call):
            calls.append([request(position, coerced[start:start + per_call]) for position in attempted[first:first + size]])
    hashes = {(position, question.id): compute_decision_hash(evidence, question, task_version)
              for position, evidence in enumerate(rows.evidence) for question in coerced}
    threads = max(1, min(capabilities.concurrency_limit, max_concurrency or capabilities.concurrency_limit, len(calls)))
    if run_budget is not None and run_budget.max_cost_usd is not None and calls \
            and _estimate_cost(resolved, calls[0]) is None:
        raise ValueError("budget.max_cost_usd is set but the provider cannot estimate cost; drop the cost cap")
    outcomes = _execute(calls, _plan_calls(calls, run_budget, lambda batch: _estimate_cost(resolved, batch)),
                        lambda batch, remaining, call_budget: resolved.decide_batch(
                            batch, deadline_s=remaining, budget=call_budget),
                        None if deadline is None else started + deadline, threads, "sem_decide")
    never_sent = [([request(position, coerced)], _CallOutcome(skipped=reason))
                  for position, reason in enumerate(rows.skip_reason) if reason is not None]
    position_of = {request_id: position for position, request_id in enumerate(rows.request_ids)}
    grid: Dict[Cell, DecisionResult] = {}
    for requests, outcome in list(zip(calls, outcomes)) + never_sent:
        aligned = None if outcome.skipped or outcome.raised else _aligned_or_none(requests, outcome.returned)
        for request_ in requests:
            row = None if aligned is None else aligned[request_.request_id]
            position = position_of[request_.request_id]
            for question in request_.questions:
                grid[(position, question.id)] = _cell_result(request_, question, outcome, row, identity, capabilities,
                                                             threshold, hashes[(position, question.id)])
    return _with_columns(df, _result_columns(coerced, grid, len(rows.request_ids)))
