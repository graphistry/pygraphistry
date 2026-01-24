#!/usr/bin/env python3
"""
Run a focused matrix of WHERE scenarios across opt profiles.

Profiles map to env var settings (value mode, domain semijoin, auto, etc).
Groups map to scenario filters that cover multiple opt types without duplication.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class Profile:
    name: str
    env: Dict[str, str]
    note: str


@dataclass(frozen=True)
class ScenarioGroup:
    name: str
    kind: str  # "synthetic" | "realdata"
    args: List[str]
    profiles: Optional[List[str]] = None
    note: str = ""


ENV_KEYS = [
    "GRAPHISTRY_NON_ADJ_WHERE_MODE",
    "GRAPHISTRY_NON_ADJ_WHERE_STRATEGY",
    "GRAPHISTRY_NON_ADJ_WHERE_ORDER",
    "GRAPHISTRY_NON_ADJ_WHERE_BOUNDS",
    "GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS",
    "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX",
    "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS",
    "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX",
    "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_PAIR_MAX",
    "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN",
    "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO",
    "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX",
    "GRAPHISTRY_EDGE_WHERE_SEMIJOIN",
    "GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO",
    "GRAPHISTRY_EDGE_WHERE_SEMIJOIN_PAIR_MAX",
]


PROFILES = {
    "baseline": Profile(
        name="baseline",
        env={"GRAPHISTRY_NON_ADJ_WHERE_MODE": "baseline"},
        note="No opt flags (baseline behavior).",
    ),
    "auto": Profile(
        name="auto",
        env={
            "GRAPHISTRY_NON_ADJ_WHERE_MODE": "auto",
            "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO": "1",
            "GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO": "1",
        },
        note="Auto value/domain mode + edge semijoin auto.",
    ),
    "value_low_ndv": Profile(
        name="value_low_ndv",
        env={
            "GRAPHISTRY_NON_ADJ_WHERE_MODE": "value",
            "GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS": "==,!=",  # low-card equality/inequality
            "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX": "10",
            "GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO": "1",
        },
        note="Value mode for low NDV equality/inequality.",
    ),
    "domain_semijoin": Profile(
        name="domain_semijoin",
        env={
            "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO": "1",
            "GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO": "1",
        },
        note="Domain semijoin auto (high NDV equality/inequality).",
    ),
    "bounds_only": Profile(
        name="bounds_only",
        env={"GRAPHISTRY_NON_ADJ_WHERE_BOUNDS": "1"},
        note="Inequality bounds prefiltering.",
    ),
    "edge_semijoin": Profile(
        name="edge_semijoin",
        env={"GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO": "1"},
        note="Edge-edge semijoin auto for adjacent edge predicates.",
    ),
    "vector": Profile(
        name="vector",
        env={
            "GRAPHISTRY_NON_ADJ_WHERE_STRATEGY": "vector",
            "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS": "2",
            "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX": "100",
            "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_PAIR_MAX": "50000",
        },
        note="Vector strategy (opt-in) for multi-clause cases.",
    ),
}


GROUPS = [
    ScenarioGroup(
        name="synthetic_low_ndv",
        kind="synthetic",
        args=[
            "--graph-filter",
            "medium_dense,large_dense",
            "--scenario-filter",
            "nonadj_eq_lowcard,nonadj_neq_lowcard",
        ],
        profiles=["baseline", "value_low_ndv", "auto"],
        note="Low-card non-adj equality/inequality.",
    ),
    ScenarioGroup(
        name="synthetic_multi_clause",
        kind="synthetic",
        args=[
            "--graph-filter",
            "medium_dense,large_dense",
            "--scenario-filter",
            "nonadj_multi,nonadj_multi_eq,3hop_where_nonadj_multi_eq",
        ],
        profiles=["baseline", "auto", "vector"],
        note="Dense multi-clause/multi-eq stress.",
    ),
    ScenarioGroup(
        name="synthetic_adjacent",
        kind="synthetic",
        args=[
            "--graph-filter",
            "medium_dense,large_dense",
            "--scenario-filter",
            "where_adj",
        ],
        profiles=["baseline", "auto"],
        note="Adjacent clause sanity check.",
    ),
    ScenarioGroup(
        name="realdata_redteam_domain",
        kind="realdata",
        args=[
            "--datasets",
            "redteam50k",
            "--skip-chain",
            "--where-filter",
            "kerberos_domain",
        ],
        profiles=["baseline", "domain_semijoin", "auto"],
        note="High-NDV domain equality/inequality on redteam.",
    ),
    ScenarioGroup(
        name="realdata_ndv_probes",
        kind="realdata",
        args=[
            "--datasets",
            "redteam50k,transactions",
            "--skip-chain",
            "--ndv-probes",
            "--where-filter",
            "ndv_",
        ],
        profiles=["baseline", "value_low_ndv", "domain_semijoin", "auto"],
        note="Low/high NDV probes.",
    ),
    ScenarioGroup(
        name="realdata_transactions_edge",
        kind="realdata",
        args=[
            "--datasets",
            "transactions",
            "--skip-chain",
            "--where-filter",
            "amount_drop,tainted_",
        ],
        profiles=["baseline", "edge_semijoin", "auto"],
        note="Edge-edge inequality + node equality on transactions.",
    ),
    ScenarioGroup(
        name="realdata_degree_inequality",
        kind="realdata",
        args=[
            "--datasets",
            "facebook_combined,twitter_demo,lesmiserables,twitter_congress",
            "--skip-chain",
            "--where-filter",
            "degree_drop,weight_drop",
        ],
        profiles=["baseline", "bounds_only", "auto"],
        note="Node/edge inequality pruning.",
    ),
]


def _parse_filters(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _reset_env(env: Dict[str, str]) -> None:
    for key in ENV_KEYS:
        env[key] = ""


def _build_command(kind: str, args: List[str], output_path: str, runs: int, warmup: int, engine: str,
                   max_scenario_seconds: Optional[float], opt_max_call_ms: Optional[float]) -> List[str]:
    if kind == "synthetic":
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "benchmarks", "run_chain_vs_samepath.py"),
            "--runs",
            str(runs),
            "--warmup",
            str(warmup),
            "--engine",
            engine,
        ]
        if output_path:
            cmd.extend(["--output", output_path])
        cmd.extend(args)
        return cmd
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "benchmarks", "run_realdata_benchmarks.py"),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--engine",
        engine,
    ]
    if output_path:
        cmd.extend(["--output", output_path])
    if max_scenario_seconds is not None:
        cmd.extend(["--max-scenario-seconds", str(max_scenario_seconds)])
    if opt_max_call_ms is not None:
        cmd.extend(["--opt-max-call-ms", str(opt_max_call_ms)])
    cmd.extend(args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a WHERE opt benchmark matrix.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument(
        "--output-dir",
        default=os.path.join("plans", "pr-886-where", "benchmarks", "opt-matrix"),
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names (default: all).",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names (default: all).",
    )
    parser.add_argument(
        "--max-scenario-seconds",
        type=float,
        default=20.0,
        help="Scenario timeout (real-data runner).",
    )
    parser.add_argument(
        "--opt-max-call-ms",
        type=float,
        default=None,
        help="Opt per-call cap in ms (real-data runner).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile_filters = _parse_filters(args.profiles)
    group_filters = _parse_filters(args.groups)

    selected_profiles = [
        profile for name, profile in PROFILES.items()
        if not profile_filters or name in profile_filters
    ]
    selected_groups = [
        group for group in GROUPS
        if not group_filters or group.name in group_filters
    ]

    if not selected_profiles:
        raise SystemExit("No matching profiles.")
    if not selected_groups:
        raise SystemExit("No matching groups.")

    os.makedirs(args.output_dir, exist_ok=True)
    max_scenario_seconds = (
        None if args.max_scenario_seconds is None or args.max_scenario_seconds <= 0
        else args.max_scenario_seconds
    )
    opt_max_call_ms = (
        None if args.opt_max_call_ms is None or args.opt_max_call_ms <= 0
        else args.opt_max_call_ms
    )

    for group in selected_groups:
        profile_names = group.profiles or [p.name for p in selected_profiles]
        for profile in selected_profiles:
            if profile.name not in profile_names:
                continue
            output_path = os.path.join(args.output_dir, f"{group.name}-{profile.name}.md")
            cmd = _build_command(
                group.kind,
                group.args,
                output_path,
                args.runs,
                args.warmup,
                args.engine,
                max_scenario_seconds,
                opt_max_call_ms,
            )
            env = dict(os.environ)
            _reset_env(env)
            env.update(profile.env)
            env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            print(f"[{group.name}] profile={profile.name} -> {output_path}")
            print("  ", " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
