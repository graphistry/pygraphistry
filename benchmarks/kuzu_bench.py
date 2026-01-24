from __future__ import annotations

import os
import shutil
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

try:
    import kuzu  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    kuzu = None


@dataclass(frozen=True)
class KuzuResult:
    dataset: str
    scenario: str
    median_ms: Optional[float]
    p90_ms: Optional[float]
    std_ms: Optional[float]


@dataclass(frozen=True)
class KuzuQuery:
    name: str
    query: str


def kuzu_available() -> bool:
    return kuzu is not None


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def _summarize_times(times: List[float]) -> Tuple[float, float, float]:
    ordered = sorted(times)
    median_ms = statistics.median(ordered)
    p90_ms = _percentile(ordered, 0.9)
    std_ms = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return median_ms, p90_ms, std_ms


def _time_query(
    conn,
    query: str,
    runs: int,
    warmup: int,
    max_total_s: Optional[float] = None,
    max_call_s: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    total_start = time.perf_counter()
    for _ in range(warmup):
        start = time.perf_counter()
        conn.execute(query)
        elapsed = time.perf_counter() - start
        if max_call_s is not None and elapsed > max_call_s:
            return None
        if max_total_s is not None and (time.perf_counter() - total_start) > max_total_s:
            return None
    times: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        conn.execute(query)
        elapsed = time.perf_counter() - start
        if max_call_s is not None and elapsed > max_call_s:
            return None
        times.append(elapsed * 1000)
        if max_total_s is not None and (time.perf_counter() - total_start) > max_total_s:
            return None
    return _summarize_times(times)


def _reset_path(path: str) -> None:
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def _extract_domain(value: str) -> str:
    if isinstance(value, str) and "@" in value:
        return value.split("@", 1)[1]
    return value


def _write_redteam_csvs(staging_dir: str) -> Tuple[str, str]:
    edges = pd.read_csv(
        "demos/data/graphistry_redteam50k.csv",
        usecols=[
            "src_domain",
            "dst_domain",
            "src_computer",
            "dst_computer",
            "auth_type",
            "success_or_failure",
            "authentication_orientation",
            "logontype",
        ],
    )
    edges = edges.rename(columns={"src_computer": "src", "dst_computer": "dst"})
    nodes_src = edges[["src", "src_domain"]].rename(
        columns={"src": "id", "src_domain": "domain"}
    )
    nodes_dst = edges[["dst", "dst_domain"]].rename(
        columns={"dst": "id", "dst_domain": "domain"}
    )
    nodes = pd.concat([nodes_src, nodes_dst], ignore_index=True).dropna(subset=["id"])
    nodes["domain"] = nodes["domain"].map(_extract_domain)
    nodes = nodes.groupby("id", as_index=False).first()

    edges_out = edges[
        [
            "src",
            "dst",
            "auth_type",
            "success_or_failure",
            "authentication_orientation",
            "logontype",
        ]
    ].copy()

    node_csv = os.path.join(staging_dir, "redteam_nodes.csv")
    edge_csv = os.path.join(staging_dir, "redteam_edges.csv")
    nodes.to_csv(node_csv, index=False, header=False)
    edges_out.to_csv(edge_csv, index=False, header=False)
    return node_csv, edge_csv


def _marker_path(db_path: str, is_dir: bool) -> str:
    if is_dir:
        return os.path.join(db_path, ".loaded")
    return f"{db_path}.loaded"


def _ensure_redteam_db_path(
    db_path: str,
    is_dir: bool,
    staging_dir: str,
    rebuild: bool,
) -> "kuzu.Connection":
    marker = _marker_path(db_path, is_dir)
    if rebuild:
        _reset_path(db_path)
        _reset_path(marker)

    base_dir = db_path if is_dir else os.path.dirname(db_path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(marker):
        node_csv, edge_csv = _write_redteam_csvs(staging_dir)
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        conn.execute("CREATE NODE TABLE Computer(id STRING, domain STRING, PRIMARY KEY (id))")
        conn.execute(
            "CREATE REL TABLE Auth(FROM Computer TO Computer, auth_type STRING, "
            "success_or_failure STRING, authentication_orientation STRING, logontype STRING)"
        )
        conn.execute(f'COPY Computer FROM "{node_csv}"')
        conn.execute(f'COPY Auth FROM "{edge_csv}"')
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("loaded\n")
        return conn

    db = kuzu.Database(db_path)
    return kuzu.Connection(db)


def _ensure_redteam_db(
    dataset_name: str,
    db_root: str,
    staging_dir: str,
    rebuild: bool,
) -> "kuzu.Connection":
    candidates = [
        (os.path.join(db_root, dataset_name), True),
        (os.path.join(db_root, f"{dataset_name}.kuzu"), False),
    ]
    last_error: Optional[Exception] = None
    for db_path, is_dir in candidates:
        try:
            return _ensure_redteam_db_path(db_path, is_dir, staging_dir, rebuild)
        except RuntimeError as exc:
            last_error = exc
            msg = str(exc).lower()
            if "cannot be a directory" in msg or "cannot be a file" in msg:
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Failed to initialize Kuzu database.")


def _redteam_queries() -> List[KuzuQuery]:
    base = (
        "MATCH (a:Computer)-[e1:Auth]->(b:Computer)<-[e2:Auth]-(c:Computer) "
        "WHERE e1.auth_type = 'Kerberos' AND e2.authentication_orientation = 'LogOn' "
    )
    return [
        KuzuQuery("kerberos_fanin_simple", f"{base}RETURN COUNT(*)"),
        KuzuQuery("kerberos_domain_match", f"{base}AND a.domain = c.domain RETURN COUNT(*)"),
        KuzuQuery("kerberos_domain_mismatch", f"{base}AND a.domain <> c.domain RETURN COUNT(*)"),
    ]


def run_kuzu_comparisons(
    dataset_name: str,
    runs: int,
    warmup: int,
    db_root: str,
    rebuild: bool,
    scenario_filters: Optional[Iterable[str]] = None,
    max_total_s: Optional[float] = None,
    max_call_s: Optional[float] = None,
) -> Tuple[List[KuzuResult], Optional[str]]:
    if kuzu is None:
        return [], "Kuzu Python package not installed; skipping comparisons."
    if dataset_name != "redteam50k":
        return [], f"Kuzu comparisons not yet implemented for dataset {dataset_name}."

    db_path = os.path.join(db_root, dataset_name)
    staging_dir = os.path.join(db_root, f"{dataset_name}_staging")
    os.makedirs(staging_dir, exist_ok=True)
    conn = _ensure_redteam_db(dataset_name, db_root, staging_dir, rebuild)

    filters = [f for f in (scenario_filters or []) if f]
    queries = _redteam_queries()
    if filters:
        queries = [q for q in queries if any(f in q.name for f in filters)]

    results: List[KuzuResult] = []
    for query in queries:
        stats = _time_query(
            conn,
            query.query,
            runs,
            warmup,
            max_total_s=max_total_s,
            max_call_s=max_call_s,
        )
        if stats is None:
            median_ms = p90_ms = std_ms = None
        else:
            median_ms, p90_ms, std_ms = stats
        results.append(
            KuzuResult(
                dataset=dataset_name,
                scenario=query.name,
                median_ms=median_ms,
                p90_ms=p90_ms,
                std_ms=std_ms,
            )
        )
    return results, None
