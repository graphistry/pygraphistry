"""gfql_explain — answer "did/would my query use the index?" honestly.

Seeded queries are cheap once indexed, so explain *executes* the query under a
decision trace and reports the real per-hop index-vs-scan path (plus resident
indexes). This makes "it silently scanned" detectable and assertable rather than
a mystery — the top human-factors need from the design review.
"""
from __future__ import annotations

from typing import Any, List, Optional, TypedDict, cast

from graphistry.Engine import EngineAbstractType, resolve_engine
from .api import index_trace, show_indexes
from .policy import IndexPolicy, validate_index_policy
from .types import IndexTraceStep


class GfqlExplainReport(TypedDict):
    engine: str
    index_policy: IndexPolicy
    resident_indexes: List[str]
    steps: List[IndexTraceStep]
    used_index: bool
    est_seed_cardinality: Optional[int]
    est_result_rows: Optional[int]
    chosen_direction: Optional[str]
    decision_reason: Optional[str]
    error: Optional[str]


def gfql_explain(
    g: Any,
    query: object,
    *,
    index_policy: str = "use",
    engine: EngineAbstractType = "auto",
) -> GfqlExplainReport:
    resolved_policy: IndexPolicy = validate_index_policy(index_policy) or "use"
    eng = resolve_engine(engine, g)
    resident = show_indexes(g)
    with index_trace() as steps:
        try:
            g.gfql(query, engine=engine, index_policy=resolved_policy)
            error = None
        except Exception as ex:  # report, don't raise — explain is diagnostic
            error = f"{type(ex).__name__}: {ex}"
    used_index = any(s.get("path") == "index" for s in steps)
    # Surface the planner's cost signal at the top level (LP1): prefer the step that
    # actually took the index, else the last decision. `est_seed_cardinality` = number
    # of seeds; `est_result_rows` = estimated fanout (Σ seed degree, free from CSR).
    ref = [s for s in steps if s.get("path") == "index"] or list(steps)
    last = ref[-1] if ref else {}
    resident_names = cast(List[str], resident["name"].tolist() if len(resident) else [])
    return {
        "engine": eng.value,
        "index_policy": resolved_policy,
        "resident_indexes": resident_names,
        "steps": list(steps),
        "used_index": used_index,
        "est_seed_cardinality": cast(Optional[int], last.get("frontier_n")),
        "est_result_rows": cast(Optional[int], last.get("est_result_rows")),
        "chosen_direction": cast(Optional[str], last.get("direction")),
        "decision_reason": cast(Optional[str], last.get("decision_reason")),
        "error": error,
    }
