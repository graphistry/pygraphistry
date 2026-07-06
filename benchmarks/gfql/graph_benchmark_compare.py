#!/usr/bin/env python3
"""Render a GFQL CPU/GPU vs external-engine q1-q9 comparison table.

Supports one or more comparator engines (Neo4j, Memgraph, Kuzu). Each
comparator JSON uses the ``results[q]["median_ms"]`` shape emitted by the
``graph_benchmark_*_q1_q9.py`` runners; Kuzu's nested ``results["q1-q9"][q]``
shape is normalized automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

QUERY_ORDER = [f"q{i}" for i in range(1, 10)]


def _normalize(payload: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Flatten the Kuzu ``results = {"q1-q9": {q1: ...}}`` nesting into the
    standard ``results = {q1: ...}`` shape used by the other runners."""
    if payload is None:
        return None
    results = payload.get("results")
    if isinstance(results, dict) and set(results.keys()) & {"q1-q9"}:
        flat: dict = {}
        for group in results.values():
            if isinstance(group, dict):
                flat.update(group)
        payload = dict(payload)
        payload["results"] = flat
    return payload


def _load(path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if path is None:
        return None
    return _normalize(json.loads(path.read_text()))


def _metric_ms(payload: Optional[Mapping[str, Any]], query: str, timing: str) -> Optional[float]:
    if payload is None:
        return None
    row = payload.get("results", {}).get(query)
    if not isinstance(row, dict):
        return None
    if timing == "query":
        value = row.get("median_ms")
    elif timing == "with-preindex":
        value = row.get("median_ms_with_preindex", row.get("median_ms"))
    else:
        raise ValueError(f"Unsupported timing mode: {timing}")
    return float(value) if value is not None else None


def _has_preindex_metric(payload: Optional[Mapping[str, Any]]) -> bool:
    if payload is None:
        return False
    rows = payload.get("results", {})
    return any(isinstance(row, dict) and "median_ms_with_preindex" in row for row in rows.values())


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value / 1000.0:.2f}s"
    return f"{value:.2f}ms"


def _fmt_speedup(baseline_ms: Optional[float], candidate_ms: Optional[float]) -> str:
    if baseline_ms is None or candidate_ms is None or candidate_ms == 0:
        return "-"
    return f"{baseline_ms / candidate_ms:.2f}x"


def _engine_label(payload: Optional[Mapping[str, Any]], fallback: str) -> str:
    if payload is None:
        return fallback
    engine = payload.get("engine") or fallback
    mode = payload.get("mode")
    if mode:
        return f"{engine} ({mode})"
    return str(engine)


def _fastest_comparator_ms(
    comparators: List[Tuple[str, Optional[Mapping[str, Any]]]], query: str
) -> Optional[float]:
    values = [
        _metric_ms(payload, query, "query")
        for _, payload in comparators
        if payload is not None
    ]
    values = [v for v in values if v is not None]
    return min(values) if values else None


def render_markdown(
    gfql_cpu: Optional[Mapping[str, Any]],
    gfql_gpu: Optional[Mapping[str, Any]],
    comparators: List[Tuple[str, Optional[Mapping[str, Any]]]],
    *,
    gfql_timing: str = "both",
) -> str:
    cpu_label = _engine_label(gfql_cpu, "GFQL CPU")
    gpu_label = _engine_label(gfql_gpu, "GFQL GPU")
    present = [(label, _engine_label(payload, label)) for label, payload in comparators if payload is not None]
    comp_headers = " | ".join(disp for _, disp in present)
    multi = len(present) > 1
    speed_ref = "fastest comparator" if multi else (present[0][1] if present else "comparator")
    show_preindex = gfql_timing == "both" and (_has_preindex_metric(gfql_cpu) or _has_preindex_metric(gfql_gpu))

    lines: List[str] = ["# Graph Benchmark q1-q9 Comparison", ""]

    def _comp_cells(query: str) -> str:
        cells = [_fmt_ms(_metric_ms(payload, query, "query")) for label, payload in comparators if payload is not None]
        return " | ".join(cells)

    if show_preindex:
        header = (
            f"| Query | {cpu_label} query | {cpu_label} + preindex | "
            f"{gpu_label} query | {gpu_label} + preindex | {comp_headers} | "
            f"CPU query speedup | GPU query speedup |"
        )
        sep = "|---|" + "---:|" * (4 + len(present) + 2)
        lines.extend([header, sep])
        for query in QUERY_ORDER:
            cpu_query = _metric_ms(gfql_cpu, query, "query")
            cpu_with_preindex = _metric_ms(gfql_cpu, query, "with-preindex")
            gpu_query = _metric_ms(gfql_gpu, query, "query")
            gpu_with_preindex = _metric_ms(gfql_gpu, query, "with-preindex")
            ref = _fastest_comparator_ms(comparators, query)
            lines.append(
                f"| {query} | {_fmt_ms(cpu_query)} | {_fmt_ms(cpu_with_preindex)} | "
                f"{_fmt_ms(gpu_query)} | {_fmt_ms(gpu_with_preindex)} | {_comp_cells(query)} | "
                f"{_fmt_speedup(ref, cpu_query)} | {_fmt_speedup(ref, gpu_query)} |"
            )
    else:
        timing = "with-preindex" if gfql_timing == "with-preindex" else "query"
        timing_label = "query + preindex" if timing == "with-preindex" else "query"
        header = (
            f"| Query | {cpu_label} ({timing_label}) | {gpu_label} ({timing_label}) | "
            f"{comp_headers} | CPU speedup | GPU speedup |"
        )
        sep = "|---|" + "---:|" * (2 + len(present) + 2)
        lines.extend([header, sep])
        for query in QUERY_ORDER:
            cpu = _metric_ms(gfql_cpu, query, timing)
            gpu = _metric_ms(gfql_gpu, query, timing)
            ref = _fastest_comparator_ms(comparators, query)
            lines.append(
                f"| {query} | {_fmt_ms(cpu)} | {_fmt_ms(gpu)} | {_comp_cells(query)} | "
                f"{_fmt_speedup(ref, cpu)} | {_fmt_speedup(ref, gpu)} |"
            )

    lines.extend([
        "",
        "Notes:",
        "- Query columns are warm query medians from each input JSON.",
        "- `+ preindex` columns add GFQL per-query preindex build time when present.",
        "- GFQL JSON inputs include `query_policies` for effective per-query policy accounting.",
        f"- Speedup columns divide the {speed_ref} median by the GFQL query median; values above 1.0x mean GFQL is faster.",
        "- Comparator load/index time is reported separately in each JSON `load` object and is not folded into query medians.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfql-cpu", type=Path, default=None)
    parser.add_argument("--gfql-gpu", type=Path, default=None)
    parser.add_argument("--neo4j", type=Path, default=None)
    parser.add_argument("--memgraph", type=Path, default=None)
    parser.add_argument("--kuzu", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--gfql-timing", choices=["query", "with-preindex", "both"], default="both")
    args = parser.parse_args()

    comparators: List[Tuple[str, Optional[Mapping[str, Any]]]] = [
        ("Neo4j", _load(args.neo4j)),
        ("Memgraph", _load(args.memgraph)),
        ("Kuzu", _load(args.kuzu)),
    ]

    output = render_markdown(
        _load(args.gfql_cpu),
        _load(args.gfql_gpu),
        comparators,
        gfql_timing=args.gfql_timing,
    )
    print(output, end="")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(output)


if __name__ == "__main__":
    main()
