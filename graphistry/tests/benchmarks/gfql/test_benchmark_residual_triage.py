from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, cast


def _load_triage_module() -> Any:
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmarks" / "gfql" / "benchmark_residual_triage.py"
    spec = importlib.util.spec_from_file_location("benchmark_residual_triage", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TRIAGE = _load_triage_module()
ResidualLane = _TRIAGE.ResidualLane
bucket_residuals = _TRIAGE.bucket_residuals
filter_issue_residuals = _TRIAGE.filter_issue_residuals
generate_issue_residual_report = _TRIAGE.generate_issue_residual_report
latest_per_lane_key = _TRIAGE.latest_per_lane_key
load_residual_lanes = _TRIAGE.load_residual_lanes
render_markdown_report = _TRIAGE.render_markdown_report
resolve_runs_dir = _TRIAGE.resolve_runs_dir


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_filter_and_latest_selection(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    issue = "graphistry/pygraphistry#880"
    query_id = "new-topics"
    upstream_query = "interactive-complex-4"

    run_a = runs_dir / "run-a"
    run_b = runs_dir / "run-b"

    _write_json(
        run_a / "probe-results.json",
        [
            {
                "backend": "gfql",
                "status": "partial",
                "issue_refs": [issue],
                "suite": "snb-interactive",
                "upstream_query": upstream_query,
                "query_id": query_id,
                "semantic_scope": "adapter_join_aggregation_workaround",
                "notes": ["Adapter workaround joins and aggregates locally."],
                "latency_ms": 12.0,
            },
            {
                "backend": "cypher",
                "status": "ok",
                "issue_refs": [],
                "suite": "snb-interactive",
                "upstream_query": upstream_query,
                "query_id": query_id,
            },
        ],
    )
    _write_json(run_a / "run-manifest.json", {"config_path": "configs/suites/ic4.yaml"})
    _write_json(run_a / "remote-execution.json", {"command": "uv run python scripts/run_suite.py ..."})

    _write_json(
        run_b / "probe-results.json",
        [
            {
                "backend": "gfql",
                "status": "partial",
                "issue_refs": [issue],
                "suite": "snb-interactive",
                "upstream_query": upstream_query,
                "query_id": query_id,
                "semantic_scope": "adapter_join_aggregation_workaround",
                "notes": ["Adapter workaround joins and aggregates locally."],
                "latency_ms": 10.0,
            }
        ],
    )
    _write_json(run_b / "run-manifest.json", {"config_path": "configs/suites/ic4.yaml"})

    # Ensure run-b is newer for latest-per-key selection.
    older = run_a / "probe-results.json"
    newer = run_b / "probe-results.json"
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    lanes = load_residual_lanes(runs_dir)
    assert len(lanes) == 3

    residuals = filter_issue_residuals(lanes, issue_ref=issue)
    assert len(residuals) == 2
    assert all(lane.backend == "gfql" for lane in residuals)
    assert all(lane.status == "partial" for lane in residuals)
    assert all(issue in lane.issue_refs for lane in residuals)

    latest = latest_per_lane_key(residuals)
    assert len(latest) == 1
    assert latest[0].run_dir_name == "run-b"


def test_bucket_classification() -> None:
    lane_join = cast(Any, ResidualLane)(
        run_dir=Path("/tmp/run1"),
        run_dir_name="run1",
        probe_path=Path("/tmp/run1/probe-results.json"),
        config_path=None,
        remote_command=None,
        backend="gfql",
        status="partial",
        issue_refs=("graphistry/pygraphistry#880",),
        suite="snb-interactive",
        upstream_query="interactive-complex-4",
        query_id="new-topics",
        semantic_scope="adapter_join_aggregation_workaround",
        notes=("Adapter workaround joins rows and aggregates locally.",),
        latency_ms=1.0,
        adapter_overhead_latency_ms=1.0,
        rows_returned=10,
    )
    lane_ancestor = cast(Any, ResidualLane)(
        run_dir=Path("/tmp/run2"),
        run_dir_name="run2",
        probe_path=Path("/tmp/run2/probe-results.json"),
        config_path=None,
        remote_command=None,
        backend="gfql",
        status="partial",
        issue_refs=("graphistry/pygraphistry#880",),
        suite="snb-interactive",
        upstream_query="interactive-short-6",
        query_id="message-forum",
        semantic_scope="adapter_join_workaround",
        notes=("Adapter workaround resolves ancestor-post plus forum/moderator rows locally.",),
        latency_ms=2.0,
        adapter_overhead_latency_ms=2.0,
        rows_returned=20,
    )

    buckets = bucket_residuals([lane_join, lane_ancestor])
    assert "joined_row_aggregation" in buckets
    assert "recursive_ancestor_row_join" in buckets
    assert buckets["joined_row_aggregation"][0].query_id == "new-topics"
    assert buckets["recursive_ancestor_row_join"][0].query_id == "message-forum"


def test_render_markdown_report_contains_evidence_blocks() -> None:
    lane = cast(Any, ResidualLane)(
        run_dir=Path("/tmp/runx"),
        run_dir_name="runx",
        probe_path=Path("/tmp/runx/probe-results.json"),
        config_path="configs/suites/snb-interactive-ic4-conformance-sf1.yaml",
        remote_command="uv run python scripts/run_suite.py --suite snb-interactive ...",
        backend="gfql",
        status="partial",
        issue_refs=("graphistry/pygraphistry#880",),
        suite="snb-interactive",
        upstream_query="interactive-complex-4",
        query_id="new-topics",
        semantic_scope="adapter_join_aggregation_workaround",
        notes=("Adapter workaround joins rows and aggregates locally.",),
        latency_ms=123.456,
        adapter_overhead_latency_ms=120.0,
        rows_returned=10,
    )

    report = render_markdown_report([lane], issue_ref="graphistry/pygraphistry#880")
    assert "Residual Triage For `graphistry/pygraphistry#880`" in report
    assert "interactive-complex-4" in report
    assert "Child Issue Drafts" in report
    assert "configs/suites/snb-interactive-ic4-conformance-sf1.yaml" in report
    assert "uv run python scripts/run_suite.py" in report


def test_resolve_runs_dir_autodetect_and_explicit(tmp_path: Path, monkeypatch) -> None:
    wd = tmp_path / "repo"
    wd.mkdir(parents=True, exist_ok=True)
    sibling_runs = tmp_path / "pyg-bench" / "results" / "runs"
    sibling_runs.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(wd)
    resolved = resolve_runs_dir(None)
    assert resolved == sibling_runs

    explicit = tmp_path / "custom" / "runs"
    explicit.mkdir(parents=True, exist_ok=True)
    resolved_explicit = resolve_runs_dir(str(explicit))
    assert resolved_explicit == explicit


def test_generate_issue_residual_report_smoke(tmp_path: Path) -> None:
    issue_ref = "graphistry/pygraphistry#880"
    run_a = tmp_path / "runs" / "run-a"
    _write_json(
        run_a / "probe-results.json",
        [
            {
                "backend": "gfql",
                "status": "partial",
                "issue_refs": [issue_ref],
                "upstream_query": "interactive-complex-8",
                "query_id": "recent-replies",
                "semantic_scope": "adapter_join_workaround",
                "notes": ["Adapter workaround shapes reply-author rows locally."],
            }
        ],
    )
    report = generate_issue_residual_report(
        runs_dir_arg=str(tmp_path / "runs"),
        issue_ref=issue_ref,
        include_all_runs=False,
        include_child_issue_templates=False,
    )
    assert "interactive-complex-8" in report
