from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResidualLane:
    run_dir: Path
    run_dir_name: str
    probe_path: Path
    config_path: Optional[str]
    remote_command: Optional[str]
    backend: str
    status: str
    issue_refs: Tuple[str, ...]
    suite: Optional[str]
    upstream_query: Optional[str]
    query_id: Optional[str]
    semantic_scope: Optional[str]
    notes: Tuple[str, ...]
    latency_ms: Optional[float]
    adapter_overhead_latency_ms: Optional[float]
    rows_returned: Optional[int]

    @property
    def lane_key(self) -> str:
        upstream = self.upstream_query or "unknown-upstream"
        query_id = self.query_id or "unknown-query-id"
        return f"{upstream}:{query_id}:{self.backend}"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_str_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return tuple()
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
    return tuple(out)


def _load_manifest_config_path(run_dir: Path) -> Optional[str]:
    manifest_path = run_dir / "run-manifest.json"
    if not manifest_path.exists():
        return None
    payload = _read_json(manifest_path)
    if not isinstance(payload, Mapping):
        return None
    config_path = payload.get("config_path")
    return config_path if isinstance(config_path, str) else None


def _load_remote_command(run_dir: Path) -> Optional[str]:
    remote_path = run_dir / "remote-execution.json"
    if not remote_path.exists():
        return None
    payload = _read_json(remote_path)
    if not isinstance(payload, Mapping):
        return None
    command = payload.get("command")
    return command if isinstance(command, str) else None


def iter_probe_result_paths(runs_dir: Path) -> Iterable[Path]:
    for probe_path in sorted(runs_dir.glob("*/probe-results.json")):
        if probe_path.is_file():
            yield probe_path


def load_residual_lanes(runs_dir: Path) -> List[ResidualLane]:
    lanes: List[ResidualLane] = []
    for probe_path in iter_probe_result_paths(runs_dir):
        run_dir = probe_path.parent
        run_dir_name = run_dir.name
        config_path = _load_manifest_config_path(run_dir)
        remote_command = _load_remote_command(run_dir)
        payload = _read_json(probe_path)
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            backend = entry.get("backend")
            status = entry.get("status")
            if not isinstance(backend, str) or not isinstance(status, str):
                continue
            lanes.append(
                ResidualLane(
                    run_dir=run_dir,
                    run_dir_name=run_dir_name,
                    probe_path=probe_path,
                    config_path=config_path,
                    remote_command=remote_command,
                    backend=backend,
                    status=status,
                    issue_refs=_coerce_str_list(entry.get("issue_refs")),
                    suite=entry.get("suite") if isinstance(entry.get("suite"), str) else None,
                    upstream_query=entry.get("upstream_query") if isinstance(entry.get("upstream_query"), str) else None,
                    query_id=entry.get("query_id") if isinstance(entry.get("query_id"), str) else None,
                    semantic_scope=entry.get("semantic_scope") if isinstance(entry.get("semantic_scope"), str) else None,
                    notes=_coerce_str_list(entry.get("notes")),
                    latency_ms=_coerce_float(entry.get("latency_ms")),
                    adapter_overhead_latency_ms=_coerce_float(entry.get("adapter_overhead_latency_ms")),
                    rows_returned=_coerce_int(entry.get("rows_returned")),
                )
            )
    return lanes


def filter_issue_residuals(
    lanes: Sequence[ResidualLane],
    issue_ref: str,
    backend: str = "gfql",
    status: str = "partial",
) -> List[ResidualLane]:
    return [
        lane
        for lane in lanes
        if lane.backend == backend and lane.status == status and issue_ref in lane.issue_refs
    ]


def latest_per_lane_key(lanes: Sequence[ResidualLane]) -> List[ResidualLane]:
    latest: Dict[str, ResidualLane] = {}
    for lane in lanes:
        prior = latest.get(lane.lane_key)
        if prior is None or lane.probe_path.stat().st_mtime > prior.probe_path.stat().st_mtime:
            latest[lane.lane_key] = lane
    return sorted(latest.values(), key=lambda lane: lane.lane_key)


def classify_residual_bucket(lane: ResidualLane) -> str:
    haystack = " ".join([lane.semantic_scope or "", *lane.notes]).lower()
    if "ancestor-post" in haystack or "ancestor post" in haystack:
        return "recursive_ancestor_row_join"
    if "shortest" in haystack:
        return "path_distance_plus_multi_table_join"
    if "co-occurrence" in haystack or "cooccurrence" in haystack:
        return "tag_cooccurrence_join_aggregation"
    if "reply" in haystack and "author" in haystack:
        return "reply_author_row_shaping_join"
    if "employment" in haystack or "company" in haystack:
        return "employment_company_row_join"
    if "join" in haystack and "aggregate" in haystack:
        return "joined_row_aggregation"
    if "join" in haystack:
        return "joined_row_projection"
    return "unclassified_residual"


def bucket_residuals(lanes: Sequence[ResidualLane]) -> Dict[str, List[ResidualLane]]:
    buckets: Dict[str, List[ResidualLane]] = {}
    for lane in lanes:
        bucket = classify_residual_bucket(lane)
        buckets.setdefault(bucket, []).append(lane)
    for values in buckets.values():
        values.sort(key=lambda lane: lane.lane_key)
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def _format_issue_template_title(bucket: str) -> str:
    return f"[FEA] GFQL residual row-bindings: {bucket.replace('_', ' ')}"


def render_markdown_report(
    lanes: Sequence[ResidualLane],
    issue_ref: str,
    include_child_issue_templates: bool = True,
) -> str:
    rows: List[str] = []
    rows.append(f"# Residual Triage For `{issue_ref}`")
    rows.append("")
    rows.append(f"- Residual lane count: **{len(lanes)}**")
    rows.append("- Filter: backend=`gfql`, status=`partial`, matching issue ref")
    rows.append("")
    rows.append("## Lane Matrix")
    rows.append("")
    rows.append("| upstream_query | query_id | semantic_scope | latency_ms | run_dir |")
    rows.append("|---|---|---|---:|---|")
    for lane in lanes:
        rows.append(
            "| {upstream} | {query_id} | {scope} | {latency} | `{run_dir}` |".format(
                upstream=lane.upstream_query or "-",
                query_id=lane.query_id or "-",
                scope=lane.semantic_scope or "-",
                latency=f"{lane.latency_ms:.3f}" if lane.latency_ms is not None else "-",
                run_dir=lane.run_dir_name,
            )
        )

    buckets = bucket_residuals(lanes)
    rows.append("")
    rows.append("## Proposed Child-Issue Split")
    rows.append("")
    rows.append("| bucket | lane_count | upstream_queries |")
    rows.append("|---|---:|---|")
    for bucket, bucket_lanes in buckets.items():
        upstreams = sorted({lane.upstream_query or "-" for lane in bucket_lanes})
        rows.append(f"| `{bucket}` | {len(bucket_lanes)} | {', '.join(upstreams)} |")

    if include_child_issue_templates:
        rows.append("")
        rows.append("## Child Issue Drafts")
        rows.append("")
        for bucket, bucket_lanes in buckets.items():
            rows.append(f"### {_format_issue_template_title(bucket)}")
            rows.append("")
            rows.append("Scope:")
            rows.append("- Residual workaround-backed GFQL partial lanes grouped by shared benchmark semantics.")
            rows.append("- Split from umbrella `{}` to keep root causes narrow and reproducible.".format(issue_ref))
            rows.append("")
            rows.append("Repro Evidence:")
            for lane in bucket_lanes:
                rows.append(
                    "- `{upstream}` / `{query_id}` from `{run_dir}` (`probe-results.json`)".format(
                        upstream=lane.upstream_query or "-",
                        query_id=lane.query_id or "-",
                        run_dir=lane.run_dir_name,
                    )
                )
                if lane.config_path:
                    rows.append(f"  - config: `{lane.config_path}`")
                if lane.remote_command:
                    rows.append(f"  - command: `{lane.remote_command}`")
                if lane.notes:
                    rows.append(f"  - workaround note: {lane.notes[-1]}")
            rows.append("")
            rows.append("Done when:")
            rows.append("- Query lanes no longer require adapter workaround rows/joins under benchmark runs.")
            rows.append(f"- `issue_refs` in benchmark probes no longer need `{issue_ref}` for these lanes.")
            rows.append("")
    return "\n".join(rows).rstrip() + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate #880 residual triage report from pyg-bench run artifacts."
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Path to pyg-bench results/runs directory. If omitted, auto-detects common local locations.",
    )
    parser.add_argument(
        "--issue-ref",
        default="graphistry/pygraphistry#880",
        help="Issue reference to filter on",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output markdown path",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include all matching lanes instead of only latest per lane key",
    )
    return parser


def _resolve_runs_dir(runs_dir_arg: Optional[str]) -> Path:
    if runs_dir_arg:
        return Path(runs_dir_arg).expanduser()

    candidates = [
        Path.cwd() / "results" / "runs",
        Path.cwd().parent / "pyg-bench" / "results" / "runs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect pyg-bench runs directory. Pass --runs-dir explicitly."
    )


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    runs_dir = _resolve_runs_dir(args.runs_dir)
    lanes = load_residual_lanes(runs_dir)
    lanes = filter_issue_residuals(lanes, issue_ref=args.issue_ref)
    if not args.all_runs:
        lanes = latest_per_lane_key(lanes)
    report = render_markdown_report(lanes, issue_ref=args.issue_ref)
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
