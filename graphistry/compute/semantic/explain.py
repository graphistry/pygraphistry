"""``explain_sem_decide`` and ``explain_sem_map``: what a run would do, with no provider call.

A plan is built by the code that the run uses (the same row checks, dedupe, cache lookup and call layout), so what a plan says is
what the run does. A plan reads the cache without a side effect when the cache has ``peek_many``, as ``MemorySemanticCache``
does. It never asks the provider for an answer, a probe or a model: it only reads ``capabilities()`` and the cost estimate.
"""
from typing import Callable, Optional, Sequence, Union

from graphistry.Engine import is_polars_df
from graphistry.compute.semantic.contract import DecisionProvider, DecisionRequest, GenerationProvider
from graphistry.compute.semantic.observe import SemanticPlanReport
from graphistry.compute.semantic.ops import (
    RequestT, _DecideSetup, _Layout, _MapSetup, _check_estimable, _estimate_cost, _estimated_total, _plan_calls, _strategy_of)
from graphistry.compute.semantic.probability import known_probability_gap
from graphistry.compute.semantic.rules import BudgetSpec, ProbabilityRequirement, QuestionLike
from graphistry.compute.typing import DataFrameT


def _host_frame(df: DataFrameT) -> DataFrameT:
    """A Polars table is read as pandas, as ``g.sem_decide`` reads the nodes of a Polars graph."""
    if not is_polars_df(df):
        return df
    return (df.collect() if hasattr(df, "collect") else df).to_pandas()  # type: ignore[no-any-return,attr-defined]


def _report(op: str, setup: Union[_DecideSetup, _MapSetup], layout: "_Layout[RequestT]", *, questions: int,
            estimate: Optional[Callable[[Sequence[RequestT]], Optional[float]]], gap: bool) -> SemanticPlanReport:
    rows = setup.rows
    skipped = [reason for reason in rows.skip_reason if reason is not None]
    active = len(rows.request_ids) - len(skipped)
    dedupe_saved = len(setup.duplicates)
    unique = active - dedupe_saved
    hits = len({position for position, _question in setup.served}) if isinstance(setup, _DecideSetup) else len(setup.served)
    budget = setup.run_budget
    would_skip, runnable = 0, len(layout.calls)
    if not gap and budget is not None:
        _check_estimable(budget, layout.calls, estimate)
        taken = len(_plan_calls(layout.calls, budget, estimate, None))
        would_skip = len({request.request_id for call in layout.calls[taken:] for request in call})
        runnable = taken
    total_cost = 0.0 if gap else _estimated_total(estimate, layout.calls)
    strategy = _strategy_of(setup.resolved)
    return {
        "op": op, "provider": setup.identity[0], "model": setup.identity[1],
        "strategy": strategy,
        "total_rows": len(rows.request_ids), "null_evidence_rows": skipped.count("null_evidence"),
        "max_rows_skipped": skipped.count("max_rows"), "active_rows": active, "dedupe": setup.dedupe,
        "unique_rows": unique, "dedupe_saved_rows": dedupe_saved, "cache_bound": setup.cache is not None,
        "cache_hits": hits, "cache_misses": (unique - hits) if setup.cache is not None else 0,
        "provider_rows": unique - hits, "effective_batch_size": layout.size, "effective_concurrency": layout.threads,
        "planned_calls": 0 if gap else len(layout.calls), "calls_within_budget": 0 if gap else runnable, "planned_questions": (unique - hits) * questions,
        "estimated_input_cost_usd": total_cost,
        "budget_max_cost_usd": None if budget is None else budget.max_cost_usd,
        "budget_max_requests": None if budget is None else budget.max_requests,
        "would_budget_skip_rows": would_skip, "known_probability_gap": gap,
    }


def explain_sem_decide(
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
    dedupe: bool = True,
) -> SemanticPlanReport:
    """What :func:`~graphistry.compute.semantic.sem_decide` would do with the same arguments, with no provider call.

    The parameters are those of ``sem_decide`` (``overwrite`` and the output columns do not matter to a plan). The report
    counts the rows at each step (``total_rows``, ``null_evidence_rows``, ``max_rows_skipped``, ``active_rows``,
    ``dedupe_saved_rows``, ``cache_hits``, ``provider_rows``), the calls that follow (``planned_calls``, ``planned_questions``,
    ``effective_batch_size``, ``effective_concurrency``) and the money (``estimated_input_cost_usd``, the budget and
    ``would_budget_skip_rows``). Invalid parameters raise ``ValueError``, as they do in a run, and so does a cost cap on a provider
    that cannot estimate cost. The plan does not count the probe requests of ``strategy="auto"``: it sends none.
    ``estimated_input_cost_usd`` is an estimate of the input cost only.

    **Example**
        ::

            plan = explain_sem_decide(df, questions=questions, provider=provider, task_id="t", evidence_cols=["name"])
            plan["planned_calls"], plan["dedupe_saved_rows"], plan["estimated_input_cost_usd"]
    """
    setup = _DecideSetup(_host_frame(df), questions=questions, provider=provider, task_id=task_id, evidence_cols=evidence_cols,
                         template=template, id_col=id_col, task_version=task_version, batch_size=batch_size,
                         max_concurrency=max_concurrency, deadline_s=deadline_s, budget=budget, max_rows=max_rows,
                         min_p=min_p, require_probabilities=require_probabilities, column_max_chars=column_max_chars,
                         overwrite=True, dedupe=dedupe, peek=True, check_columns=False)
    gap = known_probability_gap(setup.resolved, require_probabilities)
    return _report("sem_decide", setup, setup.layout(), questions=len(setup.coerced), gap=gap,
                   estimate=lambda batch: _estimate_cost(setup.resolved, batch))


def explain_sem_map(
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
    dedupe: bool = True,
) -> SemanticPlanReport:
    """What :func:`~graphistry.compute.semantic.sem_map` would do with the same arguments, with no provider call.

    The report has the fields of :func:`explain_sem_decide`. ``estimated_input_cost_usd`` is None, because generation providers
    do not estimate cost, and ``planned_questions`` counts rows.
    """
    setup = _MapSetup(_host_frame(df), instruction=instruction, out_col=out_col, provider=provider, task_id=task_id,
                      evidence_cols=evidence_cols, template=template, id_col=id_col, task_version=task_version,
                      max_output_tokens=max_output_tokens, batch_size=batch_size, max_concurrency=max_concurrency,
                      deadline_s=deadline_s, budget=budget, max_rows=max_rows, column_max_chars=column_max_chars,
                      overwrite=True, dedupe=dedupe, peek=True, check_columns=False)
    report = _report("sem_map", setup, setup.layout(), questions=1, gap=False, estimate=None)
    report["estimated_input_cost_usd"] = None
    return report
