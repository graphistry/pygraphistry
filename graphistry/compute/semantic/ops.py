"""Semantic row operators: ``sem_decide`` (typed questions) and ``sem_map`` (generated text) for every row.

Every input row is kept, in order and with its index, and gets typed result columns. Provider faults become
per-row ``error`` results; invalid parameters raise ``ValueError`` before any provider call.
"""
import collections
import dataclasses
import functools
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Dict, Generic, Hashable, List, Mapping, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd

from graphistry.compute.gfql.effects import record_effect
from graphistry.compute.semantic.binding import bound_cache, resolve_provider
from graphistry.compute.semantic.cache import SemanticCache, cache_key, provider_fingerprint
from graphistry.compute.semantic.observe import SemanticRunStats, record_run, trace_active
from graphistry.compute.semantic.contract import (
    Budget, DecisionProvider, DecisionQuestion, DecisionRequest, DecisionResult, Evidence, GenerationProvider,
    GenerationRequest, GenerationResult, ProviderCapabilities, ResultT, SkippedReason, align_results, canonicalize,
    check_decision_result, check_generation_result, compute_decision_hash, compute_generation_hash, finite_float,
    unanswered_generation_result, unanswered_result)
from graphistry.compute.semantic.probability import enforce_probability, known_probability_gap
from graphistry.compute.semantic.rules import (
    DECISION_COLUMN_TYPES, MAP_COLUMN_TYPES, BudgetSpec, ProbabilityRequirement, QuestionLike, check_params,
    coerce_budget, coerce_map_budget, coerce_questions, decision_columns, evidence_columns)
from graphistry.compute.typing import DataFrameT


DEFAULT_DECIDE_BATCH_SIZE = 32
DEFAULT_MAP_BATCH_SIZE = 16
COST_SLACK = 1e-9
Cell = Tuple[int, str]
RequestT = TypeVar("RequestT", DecisionRequest, GenerationRequest)
Invoke = Callable[[Sequence[RequestT], Optional[float], Optional[Budget]], object]
Admit = Callable[[object], bool]
UnansweredT = TypeVar("UnansweredT", DecisionResult, GenerationResult, covariant=True)


class _Unanswered(Protocol[UnansweredT]):
    """Builds the kernel's own ``error`` or ``skipped`` result for one cell."""

    def __call__(self, *, input_hash: Optional[str] = None, error_code: Optional[str] = None,
                 status_reason: Optional[SkippedReason] = None, latency_ms: Optional[float] = None,
                 retry_count: Optional[int] = None) -> UnansweredT:
        ...


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


def _is_cudf(df: object) -> bool:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    return type(df).__module__.startswith("cudf")  # hygiene-ok: dynamic-attr -- optional third-party type, no import-time dependency


def _json_ready(value: object) -> object:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
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


def _subrows(rows: _Rows, keep: Sequence[int]) -> _Rows:
    return _Rows(request_ids=[rows.request_ids[position] for position in keep],
                 evidence=[rows.evidence[position] for position in keep],
                 skip_reason=[rows.skip_reason[position] for position in keep])


def _duplicates(rows: _Rows, signatures: Sequence[Hashable]) -> Dict[int, int]:
    """Position of each active row that repeats an earlier active row's signature -> position of that first row.
    Rows that are skipped (``max_rows``, ``null_evidence``) never merge."""
    first: Dict[Hashable, int] = {}
    duplicates: Dict[int, int] = {}
    for position, signature in enumerate(signatures):
        if rows.skip_reason[position] is None:
            representative = first.setdefault(signature, position)
            if representative != position:
                duplicates[position] = representative
    return duplicates


def _copy_of(result: ResultT, request_id: str, representative_id: str) -> ResultT:
    """The answer of a duplicate row: the representative's result, with the cost, latency and retries counted once."""
    return dataclasses.replace(
        result, request_id=request_id, cost_usd=None if result.cost_usd is None else 0.0,
        latency_ms=None if result.latency_ms is None else 0.0,
        retry_count=None if result.retry_count is None else 0,
        extra={**(result.extra or {}), "deduped_from": representative_id})


def _bound_cache(provider: Union[DecisionProvider, GenerationProvider]) -> Tuple[Optional[SemanticCache], str]:
    """The cache bound in this context and the provider's fingerprint; a cache needs a provider that has one."""
    cache = bound_cache()
    if cache is None:
        return None, ""
    fingerprint = provider_fingerprint(provider)
    if fingerprint is None:
        raise ValueError(f"a result cache is bound, but provider {_provider_identity(provider)[0]!r} has no cache "
                         "fingerprint: its cache_fingerprint() is missing or returns None. Callable providers take "
                         "fingerprint=...; a chat provider needs an explicit strategy, not 'auto'. Or bind no cache")
    return cache, fingerprint


def _cache_get(cache: SemanticCache, keys: Sequence[str]) -> Dict[str, str]:
    """Stored payloads for ``keys``. A cache is an optimisation: a failing or malformed cache is a miss."""
    try:
        found = cache.get_many(list(keys))
    except Exception:
        return {}
    wanted = set(keys)
    return {key: value for key, value in found.items() if key in wanted and isinstance(value, str)} \
        if isinstance(found, Mapping) else {}


def _cache_peek(cache: SemanticCache, keys: Sequence[str]) -> Dict[str, str]:
    """Like ``_cache_get`` but without a side effect on the cache, when the cache has ``peek_many``. A plan uses it."""
    peek = getattr(cache, "peek_many", None)
    if not callable(peek):
        return _cache_get(cache, keys)
    try:
        found = peek(list(keys))
    except Exception:
        return {}
    wanted = set(keys)
    return {key: value for key, value in found.items() if key in wanted and isinstance(value, str)} \
        if isinstance(found, Mapping) else {}


def _cache_put(cache: SemanticCache, entries: Dict[str, str]) -> None:
    if entries:
        try:
            cache.put_many(entries)
        except Exception:
            pass


CACHEABLE_UNCERTAIN = ("tie", "refusal")


def _cacheable(result: Union[DecisionResult, GenerationResult]) -> bool:
    """Answers that the same input gives again: ``ok``, and an ``uncertain`` that comes from the answer itself.
    Errors, skips, ``null_answer`` (can be a passing fault) and ``min_p`` (set by this run) are never cached."""
    return result.status == "ok" or (result.status == "uncertain" and result.status_reason in CACHEABLE_UNCERTAIN)


def _result_payload(result: Union[DecisionResult, GenerationResult]) -> str:
    """The result as JSON. It holds no row id: ``request_id`` is blank here and set again on load."""
    fields = {field.name: getattr(result, field.name) for field in dataclasses.fields(result)}  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
    fields["request_id"] = ""
    return json.dumps(_json_ready(fields), sort_keys=True, separators=(",", ":"))


def _load_payload(payload: str, result_type: Type[ResultT], **identity: str) -> Optional[ResultT]:
    """A stored result for the current request, or None when the payload does not fit ``result_type``."""
    try:
        fields = json.loads(payload)
        if not isinstance(fields, dict) or set(fields) != {field.name for field in dataclasses.fields(result_type)}:
            return None
        return result_type(**{**fields, **identity})
    except Exception:
        return None


def _as_cache_hit(result: ResultT) -> ResultT:
    return dataclasses.replace(result, cost_usd=0.0, latency_ms=0.0, retry_count=0, cache_hit=True,
                               extra={**(result.extra or {}), "cache_source": "local_cache"})


def _collector(store: Dict[Cell, ResultT], cell: Cell) -> Callable[[ResultT], None]:
    def keep(result: ResultT) -> None:
        store[cell] = result
    return keep


def _check_existing_columns(names: Sequence[str], df: DataFrameT, overwrite: bool) -> None:
    existing = [name for name in names if name in df.columns]
    if existing and not overwrite:
        raise ValueError(f"output columns already exist: {existing}; pass overwrite=True to replace them")


def _provider_capabilities(provider: Union[DecisionProvider, GenerationProvider]) -> ProviderCapabilities:
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


def _cost_cap(budget: Budget) -> Optional[float]:
    return None if budget.max_cost_usd is None else budget.max_cost_usd * (1 + COST_SLACK) + COST_SLACK


class _Setup:
    """Admits the setup requests a provider sends before a run's calls (host probes) against the run's budget, as
    ``admit(estimated_cost)``; each admitted request counts as one request."""

    def __init__(self, budget: Optional[Budget]) -> None:
        self.budget, self.requests, self.cost_usd = budget, 0, 0.0

    def __call__(self, cost: object) -> bool:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
        number = _nonnegative(canonicalize(cost))
        if self.budget is not None:
            cap = _cost_cap(self.budget)
            if (self.budget.max_requests is not None and self.requests >= self.budget.max_requests) \
                    or (cap is not None and (number is None or self.cost_usd + number > cap)):
                return False
        self.requests += 1
        self.cost_usd += number or 0.0
        return True


def _prepared(provider: DecisionProvider, requests_: Sequence[DecisionRequest], admit: Admit,
              deadline_at: Optional[float]) -> Optional[DecisionProvider]:
    """The provider that answers one run: its optional ``prepare_run`` result (None when the budget stopped the
    setup), else the provider itself."""
    hook = getattr(provider, "prepare_run", None)  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
    if hook is None:
        return provider
    prepared = hook(requests_, admit, deadline_s=None if deadline_at is None else max(0.0, deadline_at - time.monotonic()))
    if prepared is not None and not isinstance(prepared, DecisionProvider):
        raise TypeError("prepare_run must return a DecisionProvider or None")
    return prepared


def _plan_calls(batches: Sequence[Sequence[RequestT]], budget: Optional[Budget],
                estimate: Optional[Callable[[Sequence[RequestT]], Optional[float]]],
                setup: Optional[_Setup] = None) -> List[Optional[Budget]]:
    """The remaining budget handed to each call the budget allows, in order; later calls are skipped (``budget``).

    Each call reserves its estimated cost before submission, so which calls run never depends on completion order.
    Setup requests already admitted are charged first.
    """
    if budget is None:
        return [None] * len(batches)
    max_cost, max_requests = budget.max_cost_usd, budget.max_requests
    cap = _cost_cap(budget)
    plans: List[Optional[Budget]] = []
    spent, reserved = (0, 0.0) if setup is None else (setup.requests, setup.cost_usd)
    for index, batch in enumerate(batches):
        if max_requests is not None and spent + index >= max_requests:
            break
        cost = None if cap is None or estimate is None else estimate(batch)
        if cap is not None and (cost is None or reserved + cost > cap):
            break
        plans.append(Budget(max_cost_usd=None if max_cost is None else max(0.0, max_cost - reserved),
                            max_requests=None if max_requests is None else max_requests - spent - index))
        reserved += cost or 0.0
    return plans


def _call_provider(invoke: Invoke[RequestT], batch: Sequence[RequestT], remaining_s: Optional[float],
                   budget: Optional[Budget]) -> _CallOutcome:
    """Provider calls are the isolation boundary: any exception is that call's ``provider_exception``."""
    try:
        return _CallOutcome(returned=invoke(batch, remaining_s, budget))
    except Exception:
        return _CallOutcome(raised=True)


def _execute(batches: Sequence[Sequence[RequestT]], budgets: Sequence[Optional[Budget]], invoke: Invoke[RequestT],
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


def _nonnegative(value: object) -> Optional[float]:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
    number = finite_float(value)
    return number if number is not None and number >= 0 else None


RequestOutcomes = List[Tuple[int, RequestT, _CallOutcome, Optional[Dict[str, ResultT]]]]


@dataclasses.dataclass(frozen=True)
class _Layout(Generic[RequestT]):
    """How a run groups its attempted rows into provider calls: the rows per call, the calls and the threads."""
    size: int
    calls: List[List[RequestT]]
    threads: int


def _layout(rows: _Rows, requests_for: Callable[[List[int]], List[List[RequestT]]], capabilities: ProviderCapabilities,
            batch_size: Optional[int], default_batch_size: int, max_concurrency: Optional[int]) -> "_Layout[RequestT]":
    """The calls of a run. ``_run_rows`` and the plan reports both use it, so a plan is what the run does."""
    size = batch_size or capabilities.max_rows_per_request or default_batch_size
    if capabilities.max_rows_per_request is not None:
        size = min(size, capabilities.max_rows_per_request)
    attempted = [position for position, reason in enumerate(rows.skip_reason) if reason is None]
    calls = [requests for first in range(0, len(attempted), size)
             for requests in requests_for(attempted[first:first + size])]
    threads = max(1, min(capabilities.concurrency_limit, max_concurrency or capabilities.concurrency_limit, len(calls)))
    return _Layout(size=size, calls=calls, threads=threads)


def _check_estimable(budget: Optional[Budget], calls: Sequence[Sequence[RequestT]],
                     estimate: Optional[Callable[[Sequence[RequestT]], Optional[float]]]) -> None:
    """A cost cap needs a provider that can estimate cost. The run and its plan both raise this error."""
    if budget is not None and budget.max_cost_usd is not None and calls \
            and (estimate is None or estimate(calls[0]) is None):
        raise ValueError("budget.max_cost_usd is set but the provider cannot estimate cost; give the provider "
                         "prices it can estimate with, or drop the cost cap")


def _estimated_total(estimate: Optional[Callable[[Sequence[RequestT]], Optional[float]]],
                     calls: Sequence[Sequence[RequestT]]) -> Optional[float]:
    """The summed estimate of every call, or None when there is no estimator or one call cannot be estimated."""
    if estimate is None:
        return None
    parts = [estimate(call) for call in calls]
    return None if any(part is None for part in parts) else float(sum(part or 0.0 for part in parts))


def _run_rows(rows: _Rows, requests_for: Callable[[List[int]], List[List[RequestT]]], invoke: Invoke[RequestT], *,
              result_type: Type[ResultT], capabilities: ProviderCapabilities, batch_size: Optional[int],
              default_batch_size: int, max_concurrency: Optional[int], budget: Optional[Budget],
              deadline_at: Optional[float], effect_name: str,
              estimate: Optional[Callable[[Sequence[RequestT]], Optional[float]]] = None,
              local_answers: Optional[Callable[[Sequence[RequestT]], object]] = None,
              prepare: Optional[Callable[[Sequence[RequestT], Admit], bool]] = None,
              telemetry: Optional[Dict[str, object]] = None
              ) -> RequestOutcomes[RequestT, ResultT]:
    """The row driver: batch the attempted rows, run the calls under the limits, align each provider return, and
    return ``(position, request, outcome, aligned results of that request)`` for every request, sent or not.
    ``local_answers`` answers every call without the provider, before any budget planning. ``prepare`` runs once
    before planning when there are calls; its admitted setup requests are charged to the budget, False skips every
    call (``budget``) and raising makes every call a ``provider_exception``. ``telemetry``, when given, receives the
    number of planned calls, the setup requests that were admitted and the estimated cost of the calls."""
    layout = _layout(rows, requests_for, capabilities, batch_size, default_batch_size, max_concurrency)
    calls, threads = layout.calls, layout.threads
    if telemetry is not None:       # the estimate is taken only when a cost cap makes the run estimate anyway: a trace adds no provider call
        capped = budget is not None and budget.max_cost_usd is not None
        telemetry.update(planned_calls=len(calls), probe_requests=0,
                         estimated_cost_usd=(0.0 if local_answers is not None else _estimated_total(estimate, calls)) if capped else None)
    if local_answers is not None:
        expired = deadline_at is not None and time.monotonic() >= deadline_at
        outcomes = [_CallOutcome(skipped="deadline") if expired else _CallOutcome(returned=local_answers(batch))
                    for batch in calls]
    else:
        _check_estimable(budget, calls, estimate)
        setup = _Setup(budget)
        try:
            ready: Optional[bool] = prepare is None or not calls or prepare(
                [request for requests in calls for request in requests], setup)
        except Exception:
            ready = None
        if telemetry is not None:
            telemetry["probe_requests"] = setup.requests
        if setup.requests:
            record_effect(effect_name)
        outcomes = [_CallOutcome(raised=True)] * len(calls) if ready is None else _execute(
            calls, _plan_calls(calls, budget, estimate, setup) if ready else [], invoke, deadline_at, threads,
            effect_name)
    never_sent = [(requests, _CallOutcome(skipped=reason)) for position, reason in enumerate(rows.skip_reason)
                  if reason is not None for requests in requests_for([position])]
    position_of = {request_id: position for position, request_id in enumerate(rows.request_ids)}
    cells: RequestOutcomes[RequestT, ResultT] = []
    for requests, outcome in list(zip(calls, outcomes)) + never_sent:
        aligned = None if outcome.skipped or outcome.raised else _aligned_or_none(requests, outcome.returned,
                                                                                   result_type)
        cells.extend((position_of[request.request_id], request, outcome,
                      None if aligned is None else aligned[request.request_id]) for request in requests)
    return cells


def _aligned_or_none(requests: Sequence[RequestT], returned: object, result_type: Type[ResultT]  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
                     ) -> Optional[Dict[str, Optional[Dict[str, ResultT]]]]:
    """``align_results`` on untrusted provider output; a failure means the whole call is untrusted."""
    try:
        return align_results(requests, returned, result_type)
    except Exception:
        return None


def _plain_result(result: ResultT, result_type: Type[ResultT]) -> Optional[ResultT]:
    """The result as an exact ``result_type`` whose answer, distribution and status labels are plain Python values,
    so what is checked is what is written; None when they are not plain answer values. Reading provider values may
    raise; ``_checked`` turns that into an error cell."""
    fields = {field.name: getattr(result, field.name) for field in dataclasses.fields(result_type)}  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
    value, distribution = canonicalize(fields["value"]), canonicalize(fields.get("distribution"))
    raw_size = len(fields["distribution"]) if isinstance(fields.get("distribution"), Mapping) else None
    if not isinstance(value, (bool, str, type(None))) or not isinstance(distribution, (dict, type(None))) \
            or (distribution is not None and raw_size != len(distribution)):
        return None
    fields.update({name: str.__str__(fields[name]) for name in (
        "status", "status_reason", "probability_kind", "error_code") if isinstance(fields.get(name), str)})
    fields["value"] = value
    if "distribution" in fields:
        fields["distribution"] = distribution
    return result_type(**fields)


def _checked(outcome: _CallOutcome, row: Optional[Dict[str, ResultT]], key: str, result_type: Type[ResultT],
             problems: Callable[[ResultT], List[str]], finish: Callable[[ResultT], ResultT],
             unanswered: _Unanswered[ResultT], input_hash: str, task_version: str,
             keep: Optional[Callable[[ResultT], None]] = None) -> ResultT:
    """The kernel's verdict on one cell. Provider data is untrusted: if reading it fails, or its metadata cannot
    become JSON, the cell is an error built from kernel fields. ``input_hash`` (taken before any call) and
    ``task_version`` come from the kernel, so a provider cannot change them. ``keep`` sees each valid result before
    ``finish`` (so before ``min_p``)."""
    if outcome.skipped is not None:
        return unanswered(status_reason=outcome.skipped, input_hash=input_hash)
    latency_ms: Optional[float] = None
    retry_count: Optional[int] = None
    try:
        plain = None if row is None else _plain_result(row[key], result_type)
        if plain is not None and not problems(plain):
            checked = dataclasses.replace(plain, input_hash=input_hash, task_version=task_version)
            if keep is not None:
                keep(checked)
            final = finish(checked)
            if _meta_json_or_none(final) is not None:
                return final
        elif plain is not None:
            retries = plain.retry_count
            latency_ms = _nonnegative(plain.latency_ms)
            retry_count = retries if isinstance(retries, int) and not isinstance(retries, bool) \
                and 0 <= retries < 2 ** 31 else None
    except Exception:
        latency_ms, retry_count = None, None
    return unanswered(error_code="provider_exception" if outcome.raised else "invalid_provider_result",
                      input_hash=input_hash, latency_ms=latency_ms, retry_count=retry_count)


def _with_min_p(question: DecisionQuestion, result: DecisionResult, min_p: Optional[float]) -> DecisionResult:
    if min_p is None or result.status != "ok":
        return result
    probability = _probability_of_value(question, result)
    if math.isnan(probability) or probability < min_p:
        return dataclasses.replace(result, status="uncertain", status_reason="min_p")
    return result


def _decision_cell(request: DecisionRequest, question: DecisionQuestion, outcome: _CallOutcome,
                   row: Optional[Dict[str, DecisionResult]], identity: Tuple[str, str],
                   capabilities: ProviderCapabilities, min_p: Optional[float], input_hash: str,
                   require_probabilities: ProbabilityRequirement,
                   keep: Optional[Callable[[DecisionResult], None]] = None) -> DecisionResult:
    def finish(found: DecisionResult) -> DecisionResult:
        if require_probabilities is not False and _meta_json_or_none(found) is None:
            return found
        return _with_min_p(question, enforce_probability(question, found, require_probabilities), min_p)
    return _checked(outcome, row, question.id, DecisionResult,
                    lambda found: check_decision_result(request, question, found, capabilities), finish,
                    functools.partial(unanswered_result, request, question, provider=identity[0], model=identity[1]),
                    input_hash, request.task_version, keep)


def _unavailable_answers(identity: Tuple[str, str]) -> Callable[[Sequence[DecisionRequest]], object]:
    """Unbilled ``probabilities_unavailable`` answers for a provider known not to report the required kind."""
    return lambda batch: [dataclasses.replace(
        unanswered_result(request_, question, provider=identity[0], model=identity[1],
                          error_code="probabilities_unavailable", latency_ms=0.0, retry_count=0), cost_usd=0.0)
        for request_ in batch for question in request_.questions]


def _generation_cell(request: GenerationRequest, outcome: _CallOutcome, row: Optional[Dict[str, GenerationResult]],
                     identity: Tuple[str, str], input_hash: str,
                     keep: Optional[Callable[[GenerationResult], None]] = None) -> GenerationResult:
    return _checked(outcome, row, "", GenerationResult,
                    lambda found: check_generation_result(request, found), lambda found: found,
                    functools.partial(unanswered_generation_result, request, provider=identity[0], model=identity[1]),
                    input_hash, request.task_version, keep)


def _provider_identity(provider: Union[DecisionProvider, GenerationProvider]) -> Tuple[str, str]:
    """The provider name and model the kernel stamps on results it builds itself, always strings."""
    fallback = type(provider).__name__
    return (provider.name if isinstance(provider.name, str) else fallback,
            provider.model if isinstance(provider.model, str) else fallback)


def _meta_json(result: Union[DecisionResult, GenerationResult]) -> str:
    meta = {name: getattr(result, name) for name in (  # hygiene-ok: dynamic-attr -- duck-typed host object with no common base
        "model", "provider", "task_version", "input_hash", "latency_ms", "cost_usd", "cache_hit", "retry_count",
        "status_reason") + (("confidence",) if isinstance(result, DecisionResult) else ())}
    meta["extra"] = None if result.extra is None else dict(result.extra)
    return json.dumps(_json_ready(meta), sort_keys=True, separators=(",", ":"))


def _meta_json_or_none(result: Union[DecisionResult, GenerationResult]) -> Optional[str]:
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


class _DecideSetup:
    """Everything a ``sem_decide`` run settles before it calls the provider: the checked parameters, the rows, which
    rows are duplicates, which are answered from the cache, and which go to the provider. ``sem_decide`` and
    ``explain_sem_decide`` both build it, so a plan is what the run does. ``peek`` reads the cache without a side effect."""

    def __init__(self, df: DataFrameT, *, questions: Sequence[QuestionLike], provider: Union[str, DecisionProvider],
                 task_id: str, evidence_cols: Optional[Sequence[str]], template: Optional[str], id_col: Optional[str],
                 task_version: str, batch_size: Optional[int], max_concurrency: Optional[int],
                 deadline_s: Optional[float], budget: BudgetSpec, max_rows: Optional[int], min_p: Optional[float],
                 require_probabilities: ProbabilityRequirement, column_max_chars: Optional[int], overwrite: bool,
                 dedupe: bool, peek: bool = False, check_columns: bool = True) -> None:
        self.coerced = coerce_questions(questions)
        check_params(dict(task_id=task_id, task_version=task_version, overwrite=overwrite, dedupe=dedupe,
                          require_probabilities=require_probabilities),
                     dict(batch_size=batch_size, max_concurrency=max_concurrency, deadline_s=deadline_s,
                          max_rows=max_rows, min_p=min_p, column_max_chars=column_max_chars))
        cols = evidence_columns(evidence_cols, template)
        self.run_budget = coerce_budget(budget)
        self.deadline = None if deadline_s is None else float(deadline_s)
        self.threshold = None if min_p is None else float(min_p)
        self.batch_size, self.max_concurrency = batch_size, max_concurrency
        self.task_id, self.task_version, self.dedupe = task_id, task_version, dedupe
        self.require_probabilities = require_probabilities
        if check_columns:
            _check_existing_columns([q.id + suffix for q in self.coerced for suffix, _ in DECISION_COLUMN_TYPES], df, overwrite)
        resolved = resolve_provider(provider)
        if not isinstance(resolved, DecisionProvider):
            raise ValueError(f"semantic provider {provider if isinstance(provider, str) else type(provider).__name__!r} "
                             "is not a DecisionProvider")
        self.resolved: DecisionProvider = resolved
        self.capabilities = _provider_capabilities(resolved)
        self.identity = _provider_identity(resolved)
        _check_question_capabilities(self.coerced, self.capabilities)
        self.rows = _prepare_rows(df, cols, template, id_col, column_max_chars, max_rows)
        self.cache, fingerprint = _bound_cache(resolved)
        self.per_call = self.capabilities.max_questions_per_request
        count = len(self.rows.request_ids)
        self.hashes = {(position, q.id): compute_decision_hash(evidence, q, task_version)
                       for position, evidence in enumerate(self.rows.evidence) for q in self.coerced}
        self.duplicates = _duplicates(self.rows, [tuple(self.hashes[(position, q.id)] for q in self.coerced)
                                                  for position in range(count)]) if dedupe else {}
        self.served: Dict[Cell, DecisionResult] = {}
        self.keys = {(position, q.id): cache_key(fingerprint, self.hashes[(position, q.id)])
                     for position in range(count) for q in self.coerced} if self.cache is not None else {}
        if self.cache is not None:
            asked = [position for position, reason in enumerate(self.rows.skip_reason)
                     if reason is None and position not in self.duplicates]
            fetch = _cache_peek if peek else _cache_get
            stored = fetch(self.cache, [self.keys[(position, q.id)] for position in asked for q in self.coerced])
            for position in asked:
                hits: Dict[Cell, DecisionResult] = {}
                for question in self.coerced:
                    payload = stored.get(self.keys[(position, question.id)])
                    loaded = None if payload is None else _load_payload(
                        payload, DecisionResult, request_id=self.rows.request_ids[position], question_id=question.id)
                    result = None if loaded is None else self.finish_cell(
                        self.rows, position, position, question, _CallOutcome(), {question.id: loaded})
                    if result is None or result.status not in ("ok", "uncertain"):
                        break
                    hits[(position, question.id)] = _as_cache_hit(result)
                else:
                    self.served.update(hits)
        self.served_rows = {position for position, _question in self.served}
        self.send = [position for position in range(count) if position not in self.duplicates and position not in self.served_rows]
        self.work = _subrows(self.rows, self.send)

    def request(self, source: _Rows, position: int, chunk: Sequence[DecisionQuestion]) -> DecisionRequest:
        return DecisionRequest(request_id=source.request_ids[position], evidence=source.evidence[position],
                               questions=tuple(chunk), task_id=self.task_id, task_version=self.task_version)

    def finish_cell(self, source: _Rows, local: int, position: int, question: DecisionQuestion, outcome: _CallOutcome,
                    row: Optional[Dict[str, DecisionResult]], keep: Optional[Callable[[DecisionResult], None]] = None
                    ) -> DecisionResult:
        """The cell of ``question`` for row ``position`` of the table, which is row ``local`` of ``source``."""
        return _decision_cell(self.request(source, local, (question,)), question, outcome, row, self.identity,
                              self.capabilities, self.threshold, self.hashes[(position, question.id)],
                              self.require_probabilities, keep)

    def requests_for(self, positions: List[int]) -> List[List[DecisionRequest]]:
        return [[self.request(self.work, position, self.coerced[start:start + self.per_call]) for position in positions]
                for start in range(0, len(self.coerced), self.per_call)]

    def layout(self) -> "_Layout[DecisionRequest]":
        return _layout(self.work, self.requests_for, self.capabilities, self.batch_size, DEFAULT_DECIDE_BATCH_SIZE,
                       self.max_concurrency)


class _MapSetup:
    """The ``sem_map`` counterpart of :class:`_DecideSetup`."""

    def __init__(self, df: DataFrameT, *, instruction: str, out_col: str, provider: Union[str, GenerationProvider],
                 task_id: str, evidence_cols: Optional[Sequence[str]], template: Optional[str], id_col: Optional[str],
                 task_version: str, max_output_tokens: Optional[int], batch_size: Optional[int],
                 max_concurrency: Optional[int], deadline_s: Optional[float], budget: BudgetSpec,
                 max_rows: Optional[int], column_max_chars: Optional[int], overwrite: bool, dedupe: bool,
                 peek: bool = False, check_columns: bool = True) -> None:
        check_params(dict(instruction=instruction, out_col=out_col, task_id=task_id, task_version=task_version,
                          overwrite=overwrite, dedupe=dedupe),
                     dict(max_output_tokens=max_output_tokens, batch_size=batch_size, max_concurrency=max_concurrency,
                          deadline_s=deadline_s, max_rows=max_rows, column_max_chars=column_max_chars))
        cols = evidence_columns(evidence_cols, template)
        self.run_budget = coerce_map_budget(budget)
        self.deadline = None if deadline_s is None else float(deadline_s)
        self.output_tokens = None if max_output_tokens is None else int(max_output_tokens)
        self.batch_size, self.max_concurrency = batch_size, max_concurrency
        self.instruction, self.task_id, self.task_version, self.dedupe = instruction, task_id, task_version, dedupe
        if check_columns:
            _check_existing_columns([out_col + suffix for suffix, _ in MAP_COLUMN_TYPES], df, overwrite)
        resolved = resolve_provider(provider)
        if not isinstance(resolved, GenerationProvider):
            raise ValueError(f"semantic provider {provider if isinstance(provider, str) else type(provider).__name__!r} "
                             "is not a GenerationProvider")
        self.resolved: GenerationProvider = resolved
        self.capabilities = _provider_capabilities(resolved)
        self.identity = _provider_identity(resolved)
        self.rows = _prepare_rows(df, cols, template, id_col, column_max_chars, max_rows)
        self.cache, fingerprint = _bound_cache(resolved)
        self.hashes = [compute_generation_hash(evidence, instruction, task_version) for evidence in self.rows.evidence]
        self.duplicates = _duplicates(self.rows, self.hashes) if dedupe else {}
        salt = f"max_output_tokens={self.output_tokens}"
        self.keys = [cache_key(fingerprint, input_hash, salt) for input_hash in self.hashes] if self.cache is not None else []
        self.served: Dict[int, GenerationResult] = {}
        if self.cache is not None:
            asked = [position for position, reason in enumerate(self.rows.skip_reason)
                     if reason is None and position not in self.duplicates]
            fetch = _cache_peek if peek else _cache_get
            stored = fetch(self.cache, [self.keys[position] for position in asked])
            for position in asked:
                payload = stored.get(self.keys[position])
                loaded = None if payload is None else _load_payload(payload, GenerationResult,
                                                                    request_id=self.rows.request_ids[position])
                result = None if loaded is None else self.finish_cell(self.rows, position, position, _CallOutcome(), {"": loaded})
                if result is not None and result.status in ("ok", "uncertain"):
                    self.served[position] = _as_cache_hit(result)
        self.send = [position for position in range(len(self.rows.request_ids))
                     if position not in self.duplicates and position not in self.served]
        self.work = _subrows(self.rows, self.send)

    def request(self, source: _Rows, position: int) -> GenerationRequest:
        return GenerationRequest(request_id=source.request_ids[position], evidence=source.evidence[position],
                                 instruction=self.instruction, task_id=self.task_id, task_version=self.task_version,
                                 max_output_tokens=self.output_tokens)

    def finish_cell(self, source: _Rows, local: int, position: int, outcome: _CallOutcome,
                    row: Optional[Dict[str, GenerationResult]],
                    keep: Optional[Callable[[GenerationResult], None]] = None) -> GenerationResult:
        return _generation_cell(self.request(source, local), outcome, row, self.identity, self.hashes[position], keep)

    def requests_for(self, positions: List[int]) -> List[List[GenerationRequest]]:
        return [[self.request(self.work, position) for position in positions]]

    def layout(self) -> "_Layout[GenerationRequest]":
        return _layout(self.work, self.requests_for, self.capabilities, self.batch_size, DEFAULT_MAP_BATCH_SIZE,
                       self.max_concurrency)


class _Flight:
    """Counts the provider calls of one run and the most that were in flight at once. Used only while tracing."""

    def __init__(self) -> None:
        self.lock, self.calls, self.active, self.peak = threading.Lock(), 0, 0, 0

    def enter(self) -> None:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


def _strategy_of(provider: object) -> Optional[str]:
    """The strategy that a provider names, or None. Provider code is untrusted: an attribute that raises is None."""
    try:
        value = getattr(provider, "strategy", None)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _run_stats(op: str, setup: Union[_DecideSetup, _MapSetup], cells: Dict[Cell, Union[DecisionResult, GenerationResult]],
               provider_cells: Dict[Cell, Union[DecisionResult, GenerationResult]], questions: Sequence[str],
               telemetry: Dict[str, object], flight: _Flight, started: float, strategy: Optional[object]) -> SemanticRunStats:
    """The statistics of one finished run. ``cells`` are all the final results; ``provider_cells`` are those that the
    provider answered (a copy of a duplicate or a cache hit costs nothing and is not in it), so cost, retries,
    latency and cached tokens are counted once."""
    rows, send = setup.rows, setup.send
    skipped = collections.Counter(reason for reason in rows.skip_reason if reason is not None)
    active_send = [position for position in send if rows.skip_reason[position] is None]
    served_rows = len({position for position, _question in setup.served}) if isinstance(setup, _DecideSetup) else len(setup.served)
    by_row: Dict[int, List[Union[DecisionResult, GenerationResult]]] = collections.defaultdict(list)
    for (position, _question), result in cells.items():
        by_row[position].append(result)
    budget_skipped = sum(1 for position in active_send if any(r.status == "skipped" and r.status_reason == "budget" for r in by_row[position]))
    deadline_skipped = sum(1 for position in active_send if any(r.status == "skipped" and r.status_reason == "deadline" for r in by_row[position]))
    status_counts: Dict[str, Dict[str, int]] = {question: collections.Counter() for question in questions}  # type: ignore[misc]
    reasons: Dict[str, int] = collections.Counter()
    codes: Dict[str, int] = collections.Counter()
    kinds: Dict[str, int] = collections.Counter()
    for (position, question), result in cells.items():
        status_counts[question][result.status] += 1
        if result.status == "uncertain" and result.status_reason:
            reasons[str(result.status_reason)] += 1
        if result.status == "error" and result.error_code:
            codes[str(result.error_code)] += 1
        kinds[str(getattr(result, "probability_kind", "none"))] += 1
    cached_tokens = 0
    for result in provider_cells.values():
        value = (result.extra or {}).get("cached_input_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cached_tokens += int(value)
    estimated = telemetry.get("estimated_cost_usd")
    probes = telemetry.get("probe_requests")
    return {
        "op": op, "task_id": setup.task_id, "provider": setup.identity[0], "model": setup.identity[1],
        "questions": len(questions) if op == "sem_decide" else 0, "dedupe": setup.dedupe, "cache_bound": setup.cache is not None,
        "total_rows": len(rows.request_ids), "null_evidence_rows": skipped["null_evidence"], "max_rows_skipped": skipped["max_rows"],
        "active_rows": len(rows.request_ids) - sum(skipped.values()), "deduped_rows": len(setup.duplicates), "cache_hits": served_rows,
        "provider_rows": len(active_send), "budget_skipped_rows": budget_skipped, "deadline_skipped_rows": deadline_skipped,
        "status_counts": {question: dict(counter) for question, counter in status_counts.items()},
        "uncertain_reasons": dict(reasons), "error_codes": dict(codes), "probability_kind_counts": dict(kinds),
        "provider_calls": flight.calls, "probe_calls": probes if isinstance(probes, int) else 0,
        "retries": sum(int(r.retry_count or 0) for r in provider_cells.values()),
        "billed_cost_usd": float(sum(float(r.cost_usd or 0.0) for r in provider_cells.values())),
        "estimated_cost_usd": float(estimated) if isinstance(estimated, (int, float)) else None,
        "provider_latency_ms_sum": float(sum(float(r.latency_ms or 0.0) for r in provider_cells.values())),
        "wall_ms": (time.monotonic() - started) * 1000.0, "peak_in_flight": flight.peak, "cached_input_tokens": cached_tokens,
        "strategy_used": strategy if isinstance(strategy, str) else None,
    }


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
    require_probabilities: ProbabilityRequirement = False,
    column_max_chars: Optional[int] = None,
    overwrite: bool = False,
    dedupe: bool = True,
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
        estimate raises) and/or ``max_requests`` (provider calls); host probes a provider sends for the run count
        against both
    :param max_rows: rows at positions >= ``max_rows`` are ``skipped``, never dropped
    :param min_p: ``ok`` answers whose ``q__p`` is below this (or undefined) become ``uncertain``
    :param require_probabilities: False preserves default behavior; True requires native or token-logprob p for
        each successful value; "native" requires native-decision p. Missing required p is probabilities_unavailable.
        Known incompatible shipped chat configurations fail locally without sending evidence or reserving cost;
        this capability check precedes provider-call budget planning. Answerless uncertainty, errors and skips are preserved.
    :param column_max_chars: truncate each string evidence value to this many characters
    :param overwrite: replace existing output columns instead of raising
    :param dedupe: ask once for rows that have the same evidence (after ``max_rows`` and null-evidence skips, before
        ``budget``); a copy gets the answer with ``cost_usd`` and ``latency_ms`` 0 and ``extra["deduped_from"]`` set to
        the id of the row that was asked. ``dedupe=False`` asks for every row, for example to vote over several answers
    :returns: a copy of ``df`` with the result columns appended

    A cache bound with :func:`~graphistry.compute.semantic.bind_providers` answers repeated inputs without a call;
    see :class:`~graphistry.compute.semantic.cache.SemanticCache`. A row is served from it only when every question
    of the row hits.

    **Example**
        ::

            out = sem_decide(df, questions=[{"id": "is_vendor", "kind": "predicate",
                                             "text": "Is this company a software vendor?"}],
                             provider=my_provider, evidence_cols=["company", "description"],
                             task_id="vendors")
    """
    started = time.monotonic()
    setup = _DecideSetup(df, questions=questions, provider=provider, task_id=task_id, evidence_cols=evidence_cols,
                         template=template, id_col=id_col, task_version=task_version, batch_size=batch_size,
                         max_concurrency=max_concurrency, deadline_s=deadline_s, budget=budget, max_rows=max_rows,
                         min_p=min_p, require_probabilities=require_probabilities, column_max_chars=column_max_chars,
                         overwrite=overwrite, dedupe=dedupe)
    resolved, rows, coerced, cache = setup.resolved, setup.rows, setup.coerced, setup.cache
    deadline_at = None if setup.deadline is None else started + setup.deadline
    run = [resolved]
    unavailable = _unavailable_answers(setup.identity)
    settled_gap = [False]
    tracing = trace_active()
    flight = _Flight()
    telemetry: Dict[str, object] = {}

    def prepare(requests_: Sequence[DecisionRequest], admit: Admit) -> bool:
        prepared = _prepared(resolved, requests_, admit, deadline_at)
        run[0] = prepared or resolved
        settled_gap[0] = prepared is not None and known_probability_gap(prepared, require_probabilities)
        return prepared is not None

    def invoke(batch: Sequence[DecisionRequest], remaining: Optional[float], call_budget: Optional[Budget]) -> object:  # hygiene-ok: object-annotation -- untrusted host or user input, narrowed by isinstance
        # A strategy settled by the run's setup (such as a probe) may lack the required probabilities: send no evidence.
        if settled_gap[0]:
            return unavailable(batch)
        if not tracing:
            return run[0].decide_batch(batch, deadline_s=remaining, budget=call_budget)
        flight.enter()
        try:
            return run[0].decide_batch(batch, deadline_s=remaining, budget=call_budget)
        finally:
            flight.leave()

    work, send = setup.work, setup.send
    cells = _run_rows(
        work, setup.requests_for, invoke,
        result_type=DecisionResult, capabilities=setup.capabilities, batch_size=batch_size,
        default_batch_size=DEFAULT_DECIDE_BATCH_SIZE, max_concurrency=max_concurrency, budget=setup.run_budget,
        deadline_at=deadline_at, effect_name="sem_decide",
        estimate=lambda batch: 0.0 if settled_gap[0] else _estimate_cost(run[0], batch),
        local_answers=unavailable if known_probability_gap(resolved, require_probabilities) else None,
        prepare=prepare, telemetry=telemetry if tracing else None)
    fresh: Dict[Cell, DecisionResult] = {}
    grid = {(send[local], question.id): setup.finish_cell(
                work, local, send[local], question, outcome, row,
                _collector(fresh, (send[local], question.id)) if cache is not None else None)
            for local, request_, outcome, row in cells for question in request_.questions}
    if cache is not None:
        _cache_put(cache, {setup.keys[cell]: _result_payload(result) for cell, result in fresh.items() if _cacheable(result)})
    provider_cells = dict(grid)
    grid.update(setup.served)
    for position, representative in setup.duplicates.items():
        for question in coerced:
            grid[(position, question.id)] = _copy_of(grid[(representative, question.id)], rows.request_ids[position],
                                                     rows.request_ids[representative])
    out = _with_columns(df, _result_columns(coerced, grid, len(rows.request_ids)))
    if tracing:
        record_run(_run_stats("sem_decide", setup, grid, provider_cells, [q.id for q in coerced], telemetry, flight,  # type: ignore[arg-type]
                              started, _strategy_of(run[0])))
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
    dedupe: bool = True,
) -> DataFrameT:
    """Generate one text value per row and append ``out_col`` plus its status, error code and meta columns.

    Parameters, row identity, limits and failure handling match :func:`sem_decide`, except that ``batch_size``
    defaults to the provider's ``max_rows_per_request``, else 16; ``instruction`` is the generation instruction
    and ``max_output_tokens`` is passed to the provider as its bound on each generated value. ``budget`` takes
    ``max_requests`` only: ``max_cost_usd`` raises, since generation providers do not estimate cost. ``dedupe`` and a
    bound result cache work as in :func:`sem_decide`; the cache key also holds ``max_output_tokens``.

    **Example**
        ::

            out = sem_map(df, instruction="Summarize the description in one sentence.",
                          out_col="summary", provider=my_generator, evidence_cols=["description"],
                          task_id="summaries")
    """
    started = time.monotonic()
    setup = _MapSetup(df, instruction=instruction, out_col=out_col, provider=provider, task_id=task_id,
                      evidence_cols=evidence_cols, template=template, id_col=id_col, task_version=task_version,
                      max_output_tokens=max_output_tokens, batch_size=batch_size, max_concurrency=max_concurrency,
                      deadline_s=deadline_s, budget=budget, max_rows=max_rows, column_max_chars=column_max_chars,
                      overwrite=overwrite, dedupe=dedupe)
    resolved, rows, cache = setup.resolved, setup.rows, setup.cache
    tracing = trace_active()
    flight = _Flight()
    telemetry: Dict[str, object] = {}

    def invoke(batch: Sequence[GenerationRequest], remaining: Optional[float], call_budget: Optional[Budget]) -> object:
        if not tracing:
            return resolved.generate_batch(batch, deadline_s=remaining, budget=call_budget)
        flight.enter()
        try:
            return resolved.generate_batch(batch, deadline_s=remaining, budget=call_budget)
        finally:
            flight.leave()

    work, send = setup.work, setup.send
    cells = _run_rows(
        work, setup.requests_for, invoke,
        result_type=GenerationResult, capabilities=setup.capabilities, batch_size=batch_size,
        default_batch_size=DEFAULT_MAP_BATCH_SIZE, max_concurrency=max_concurrency, budget=setup.run_budget,
        deadline_at=None if setup.deadline is None else started + setup.deadline, effect_name="sem_map",
        telemetry=telemetry if tracing else None)
    fresh: Dict[Cell, GenerationResult] = {}
    results = {send[local]: setup.finish_cell(work, local, send[local], outcome, row,
                                              _collector(fresh, (send[local], "")) if cache is not None else None)
               for local, _request, outcome, row in cells}
    if cache is not None:
        _cache_put(cache, {setup.keys[position]: _result_payload(result) for (position, _q), result in fresh.items()
                           if _cacheable(result)})
    provider_results = dict(results)
    results.update(setup.served)
    for position, representative in setup.duplicates.items():
        results[position] = _copy_of(results[representative], rows.request_ids[position], rows.request_ids[representative])
    ordered = [results[position] for position in range(len(rows.request_ids))]
    out = _with_columns(df, {
        out_col: np.array([result.value for result in ordered], dtype=object),
        f"{out_col}__status": np.array([result.status for result in ordered], dtype=object),
        f"{out_col}__error_code": np.array([result.error_code for result in ordered], dtype=object),
        f"{out_col}__meta": np.array([_meta_json(result) for result in ordered], dtype=object),
    })
    if tracing:
        record_run(_run_stats("sem_map", setup, {(p, ""): r for p, r in results.items()},  # type: ignore[arg-type]
                              {(p, ""): r for p, r in provider_results.items()}, [""], telemetry, flight, started, None))
    return out
