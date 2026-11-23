"""Observability for semantic operators: run statistics, plan reports and safe verdict helpers.

``semantic_trace()`` collects one :class:`SemanticRunStats` for each ``sem_decide`` or ``sem_map`` run inside its
block, like ``index_trace()`` does for GFQL index decisions. ``explain_sem_decide`` and ``explain_sem_map`` (in
:mod:`graphistry.compute.semantic.explain`) return a :class:`SemanticPlanReport` without any provider call.
``semantic_verdicts`` and ``semantic_summary`` read the status columns of a result without ever counting an error or a
skipped row as an answer. The module keeps no process-wide state: a trace lives in a ``ContextVar``.
"""
import warnings
from contextvars import ContextVar
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, TypedDict, Union

import numpy as np
import pandas as pd


class SemanticPlanReport(TypedDict):
    """What a run would do, from ``explain_sem_decide`` or ``explain_sem_map``. Row counts are rows of the table."""
    op: str
    provider: str
    model: str
    strategy: Optional[str]
    total_rows: int
    null_evidence_rows: int
    max_rows_skipped: int
    active_rows: int
    dedupe: bool
    unique_rows: int
    dedupe_saved_rows: int
    cache_bound: bool
    cache_hits: int
    cache_misses: int
    provider_rows: int
    effective_batch_size: int
    effective_concurrency: int
    planned_calls: int
    calls_within_budget: int
    planned_questions: int
    estimated_input_cost_usd: Optional[float]
    budget_max_cost_usd: Optional[float]
    budget_max_requests: Optional[int]
    would_budget_skip_rows: int
    known_probability_gap: bool


class SemanticRunStats(TypedDict):
    """What one run did, from ``semantic_trace()``. Row counts are rows of the table, call counts are provider calls."""
    op: str
    task_id: str
    provider: str
    model: str
    questions: int
    dedupe: bool
    cache_bound: bool
    total_rows: int
    null_evidence_rows: int
    max_rows_skipped: int
    active_rows: int
    deduped_rows: int
    cache_hits: int
    provider_rows: int
    budget_skipped_rows: int
    deadline_skipped_rows: int
    status_counts: Dict[str, Dict[str, int]]
    uncertain_reasons: Dict[str, int]
    error_codes: Dict[str, int]
    probability_kind_counts: Dict[str, int]
    provider_calls: int
    probe_calls: int
    retries: int
    billed_cost_usd: float
    estimated_cost_usd: Optional[float]
    provider_latency_ms_sum: float
    wall_ms: float
    peak_in_flight: int
    cached_input_tokens: int
    strategy_used: Optional[str]


_RUNS: "ContextVar[Optional[List[SemanticRunStats]]]" = ContextVar("gfql_semantic_trace", default=None)


class semantic_trace:
    """Context manager: collect the statistics of every ``sem_decide`` and ``sem_map`` run inside the block.

    The value of ``with semantic_trace() as runs`` is a list that gets one :class:`SemanticRunStats` for each run that
    ends, in order. This covers a direct call, ``g.sem_decide`` and ``g.gfql([call("sem_decide", ...)])``. A run that
    raises before it finishes adds nothing. Blocks nest. A run goes to the innermost block when it ends, and to each outer
    block when the inner block ends. The trace follows
    ``contextvars`` rules, so it sees the runs of the thread that opens it. A thread that you start yourself does not see
    it unless you start it with ``contextvars.copy_context().run``. Outside a block a run pays nothing for tracing.

    **Example**
        ::

            with semantic_trace() as runs:
                out = g.sem_decide(questions, "rules", task_id="t", evidence_cols=["name"])
            runs[0]["billed_cost_usd"], runs[0]["provider_calls"]
    """

    def __enter__(self) -> List[SemanticRunStats]:
        self.runs: List[SemanticRunStats] = []
        self._previous = _RUNS.get()
        self._token = _RUNS.set(self.runs)
        return self.runs

    def __exit__(self, *exc: object) -> Literal[False]:
        _RUNS.reset(self._token)
        if self._previous is not None:
            self._previous.extend(self.runs)            # an outer block sees the runs of an inner one
        return False


def trace_active() -> bool:
    return _RUNS.get() is not None


def record_run(stats: SemanticRunStats) -> None:
    runs = _RUNS.get()
    if runs is not None:
        runs.append(stats)


STATUSES = ("ok", "uncertain", "error", "skipped")
VERDICTS = ("yes", "no", "uncertain", "error", "skipped")
Labels = Union[str, bool, Iterable[Union[str, bool]]]


def _label(label: object) -> Union[str, bool]:
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    if isinstance(label, str):
        return label
    raise ValueError(f"positive must be a label string, True, or a list of them, not {label!r}")


def _positive_labels(positive: Optional[Labels]) -> Optional[Set[Union[str, bool]]]:
    if positive is None:
        return None
    if isinstance(positive, (str, bool, np.bool_)):
        return {_label(positive)}
    try:
        return {_label(label) for label in positive}
    except TypeError:
        raise ValueError(f"positive must be a label string, True, or a list of them, not {positive!r}") from None


def _host(df: "pd.DataFrame") -> "pd.DataFrame":
    """A pandas table: the helpers read a cuDF result through ``to_pandas``."""
    return df.to_pandas() if type(df).__module__.startswith("cudf") else df


def _checked_status(df: "pd.DataFrame", question_id: str) -> "pd.Series":
    """The status column of a question, as objects. A value that is not a status (a null too) raises, because a row that no
    count shows is the trap that these helpers exist to prevent."""
    status_col = f"{question_id}__status"
    if status_col not in df.columns:
        raise ValueError(f"the table has no status column for question {question_id!r}")
    status = df[status_col].astype(object)
    unknown = sorted({repr(s) for s in set(status) if not isinstance(s, str) or s not in STATUSES})
    if unknown:
        raise ValueError(f"unknown status values in {status_col!r}: {unknown}")
    return status


def _is_predicate_column(value: "pd.Series") -> Optional[bool]:
    """True for a predicate column, False for a choice or score column, None when no row has an answer (then the kind is unknown)."""
    if pd.api.types.is_bool_dtype(value.dtype):
        return True
    answers = value.dropna()
    if len(answers) == 0:
        return None
    return value.dtype == object and all(isinstance(v, (bool, np.bool_)) for v in answers)


def semantic_verdicts(df: "pd.DataFrame", question_id: str, *, positive: Optional[Labels] = None) -> "pd.Series":
    """One verdict for each row of a ``sem_decide`` result: ``yes``, ``no``, ``uncertain``, ``error`` or ``skipped``.

    Only an ``ok`` answer is ``yes`` or ``no``. An ``uncertain``, ``error`` or ``skipped`` row keeps that word, so no
    caller counts a row that the model never answered as ``no`` (as ``~out["q"]`` would, or ``out["q"] == "label"``).
    For a predicate, ``yes`` is a true answer. For a choice or score question, pass ``positive``, a label or a list of
    labels, and ``yes`` means that the answer is one of them. The result has the index of ``df`` and a categorical dtype.
    A row with a status that is not one of the four statuses raises ``ValueError``. So does a ``positive`` of the wrong
    type (a bool for a choice, a string for a predicate), and a ``positive`` label that the answers spell with another case
    gives a ``UserWarning``. A cuDF table is read as pandas, and the result is a pandas Series.

    **Example**
        ::

            verdict = semantic_verdicts(out, "urgency", positive=["patch_now"])
            verdict.value_counts()
    """
    df = _host(df)
    if question_id not in df.columns:
        raise ValueError(f"the table has no result columns for question {question_id!r}")
    status = _checked_status(df, question_id)
    labels = _positive_labels(positive)
    value = df[question_id]
    predicate = _is_predicate_column(value)
    if labels is None:
        if predicate is False:
            raise ValueError(f"question {question_id!r} is not a predicate: pass positive=... with the labels that count as yes")
        answered = pd.Series(["yes" if v else "no" for v in value.fillna(False).astype(bool)], index=df.index, dtype=object)
    else:
        if predicate is True and any(not isinstance(label, bool) for label in labels):
            raise ValueError(f"question {question_id!r} is a predicate: positive must be True, not {sorted(map(repr, labels))}")
        if predicate is False and any(not isinstance(label, str) for label in labels):
            raise ValueError(f"question {question_id!r} is not a predicate: positive must be label strings, not {sorted(map(repr, labels))}")
        seen = {v for v in value[status == "ok"].dropna().astype(object)}
        if predicate is False and seen and not (labels & seen):
            near = {str(label).strip().lower() for label in labels} & {str(v).strip().lower() for v in seen}
            if near:
                warnings.warn(f"no answer of {question_id!r} equals {sorted(labels)}, but the answers {sorted(seen)} match when case and spaces are ignored", UserWarning, stacklevel=2)
        answered = value.astype(object).map(lambda v: "yes" if v in labels else "no")
    out = pd.Series([a if s == "ok" else s for a, s in zip(answered, status)], index=df.index, dtype=object)
    return out.astype(pd.CategoricalDtype(categories=list(VERDICTS)))


def semantic_summary(df: "pd.DataFrame", question_ids: Optional[Sequence[str]] = None) -> "pd.DataFrame":
    """A table of the status of each question of a ``sem_decide`` or ``sem_map`` result.

    One row for each question, with the number of rows that are ``ok``, ``uncertain``, ``error`` and ``skipped``, the
    ``unanswered`` count (the three that are not ``ok``) and ``unanswered_pct``. Read ``unanswered`` before you trust a
    count of ``no``. Without ``question_ids`` it reads every column that ends in ``__status``. A status that is not one of the
    four, a null too, raises ``ValueError``, so every row is counted. A cuDF table is read as pandas.
    """
    df = _host(df)
    suffix = "__status"
    ids = list(question_ids) if question_ids is not None else [c[:-len(suffix)] for c in df.columns if str(c).endswith(suffix)]
    rows: List[Mapping[str, object]] = []
    for question_id in ids:
        counts = _checked_status(df, question_id).value_counts()
        row = {name: int(counts.get(name, 0)) for name in STATUSES}
        unanswered = row["uncertain"] + row["error"] + row["skipped"]
        rows.append({"question": question_id, **row, "unanswered": unanswered,
                     "unanswered_pct": round(100.0 * unanswered / max(len(df), 1), 2)})
    return pd.DataFrame(rows).set_index("question") if rows else pd.DataFrame(
        columns=["ok", "uncertain", "error", "skipped", "unanswered", "unanswered_pct"])
