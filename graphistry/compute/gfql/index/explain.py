"""gfql_explain — answer "did/would my query use the index?" honestly.

Seeded queries are cheap once indexed, so explain *executes* the query under a
decision trace and reports the real per-hop index-vs-scan path (plus resident
indexes). This makes "it silently scanned" detectable and assertable rather than
a mystery — the top human-factors need from the design review (P0-1).
"""
from __future__ import annotations

from typing import Any, Dict

from graphistry.Engine import resolve_engine
from .api import index_trace, get_registry, show_indexes


def gfql_explain(g: Any, query: Any, *, index_policy: str = "use", engine: str = "auto") -> Dict[str, Any]:
    eng = resolve_engine(engine, g)
    resident = show_indexes(g)
    with index_trace() as steps:
        try:
            g.gfql(query, engine=engine, index_policy=index_policy)
            error = None
        except Exception as ex:  # report, don't raise — explain is diagnostic
            error = f"{type(ex).__name__}: {ex}"
    used_index = any(s.get("path") == "index" for s in steps)
    return {
        "engine": eng.value,
        "index_policy": index_policy,
        "resident_indexes": resident["name"].tolist() if len(resident) else [],
        "steps": list(steps),
        "used_index": used_index,
        "error": error,
    }
