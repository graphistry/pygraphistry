#!/usr/bin/env python3
"""Helpers for presenting the filter -> PageRank benchmark from saved JSON results.

This module is intentionally presentation-only: it reads existing DGX benchmark
artifacts and renders tables/charts for docs and notebooks. It does not rerun
benchmarks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "plans" / "gfql-gpu-pagerank-benchmark" / "results"
NEO4J_GPLUS_LOWER_BOUND_S = 187.0
COLORS = {
    "Graphistry GPU": "#14866d",
    "Graphistry CPU": "#425466",
    "Neo4j": "#8f3b2f",
    "Load": "#6c8fb3",
    "Shaping": "#d17c2f",
    "Bind": "#cbd5e1",
    "Search 1": "#557a95",
    "PageRank": "#d17c2f",
    "Search 2": "#14866d",
}


def setup_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#0F172A",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.9,
        "font.size": 11,
        "legend.frameon": False,
        "figure.dpi": 144,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.24,
    })


def _json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text())


def gplus_summary() -> Dict:
    return _json(RESULTS_DIR / "gplus_q995_pr9995_summary.json")


def twitter_cpu_gpu() -> Dict:
    return _json(RESULTS_DIR / "twitter_cpu_gpu_q99_pr99.json")


def twitter_neo4j() -> Dict:
    return _json(RESULTS_DIR / "twitter_neo4j_tracked_q99_pr99.json")


def twitter_load() -> Dict:
    return _json(RESULTS_DIR / "twitter_load_prepare_infer.json")


def gplus_load() -> Dict:
    return _json(RESULTS_DIR / "gplus_load_prepare_infer.json")


def twitter_three_way_df() -> pd.DataFrame:
    cpu_gpu = {row["engine"]: row for row in twitter_cpu_gpu()["results"]}
    neo = twitter_neo4j()
    rows = [
        {"system": "Graphistry GPU\n(cudf + cugraph)", "family": "Graphistry GPU", "seconds": cpu_gpu["cudf"]["pipeline_total_median_s"]},
        {"system": "Graphistry CPU\n(pandas + igraph)", "family": "Graphistry CPU", "seconds": cpu_gpu["pandas"]["pipeline_total_median_s"]},
        {"system": "Neo4j + GDS", "family": "Neo4j", "seconds": neo["pipeline_total_median_s"]},
    ]
    df = pd.DataFrame(rows)
    fastest = float(df["seconds"].min())
    df["vs_fastest_x"] = (df["seconds"] / fastest).round(2)
    return df


def twitter_stage_df() -> pd.DataFrame:
    cpu_gpu = {row["engine"]: row for row in twitter_cpu_gpu()["results"]}
    neo = twitter_neo4j()
    rows = []
    stages = [
        ("Search 1", "gfql_filter1_median_s"),
        ("PageRank", "pagerank_median_s"),
        ("Search 2", "gfql_filter2_median_s"),
    ]
    systems = [
        ("Graphistry GPU\n(cudf + cugraph)", cpu_gpu["cudf"]),
        ("Graphistry CPU\n(pandas + igraph)", cpu_gpu["pandas"]),
        ("Neo4j + GDS", neo),
    ]
    for system, payload in systems:
        for stage, key in stages:
            rows.append({"system": system, "stage": stage, "seconds": payload[key]})
    return pd.DataFrame(rows)


def load_breakdown_df() -> pd.DataFrame:
    rows = []
    datasets = [("Twitter", twitter_load()), ("GPlus", gplus_load())]
    for dataset, payload in datasets:
        by_engine = {row["engine"]: row for row in payload["results"]}
        for engine, label in [("cudf", "Graphistry GPU"), ("pandas", "Graphistry CPU")]:
            row = by_engine[engine]
            rows.extend([
                {"dataset": dataset, "system": label, "stage": "Load", "seconds": row["load_edges_median_s"]},
                {"dataset": dataset, "system": label, "stage": "Shaping", "seconds": row["build_nodes_median_s"]},
                {"dataset": dataset, "system": label, "stage": "Bind", "seconds": row["bind_graph_median_s"]},
            ])
    return pd.DataFrame(rows)


def gplus_lifecycle_df() -> pd.DataFrame:
    summary = gplus_summary()
    load = {row["engine"]: row for row in gplus_load()["results"]}
    rows = []
    for key, engine, label in [("gpu", "cudf", "Graphistry GPU"), ("cpu", "pandas", "Graphistry CPU")]:
        row = summary[key]
        rows.extend([
            {"system": label, "phase": "Load + shape", "seconds": load[engine]["total_prepare_median_s"]},
            {"system": label, "phase": "Search + analytics pipeline", "seconds": row["pipeline_total_median_s"]},
        ])
    return pd.DataFrame(rows)


def pipeline_overview_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Phase": "Data loading", "What happens": "Read a cached SNAP edge list into a dataframe", "Why it matters": "Shows dataframe-native ingest cost, not just graph runtime"},
        {"Phase": "Data shaping", "What happens": "Compute node degree once and keep it as node metadata", "Why it matters": "Benchmarks the columnar prep that later GFQL stages query directly"},
        {"Phase": "Graph search", "What happens": "Run local Cypher `GRAPH { MATCH ... }` subgraph extraction", "Why it matters": "Measures graph-preserving search directly on dataframe-backed graphs"},
        {"Phase": "Graph analytics", "What happens": "Run local Cypher `CALL graphistry.{igraph,cugraph}.pagerank.write()`", "Why it matters": "Shows backend graph algorithm acceleration while keeping the enriched graph resident"},
        {"Phase": "Downstream use", "What happens": "Keep the final enriched subgraph for visualization or follow-on analysis", "Why it matters": "No external DB required; it stays in Python dataframes/graph objects"},
    ])


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", length=0)


def _top_with_headroom(values: Sequence[float], *, pad_ratio: float = 0.16, min_pad: float = 0.08) -> float:
    maxv = max(values) if values else 1.0
    return maxv + max(maxv * pad_ratio, min_pad)


def _label_box(ax, x: float, y: float, text: str, *, fontsize: int = 10) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#0F172A",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "linewidth": 0.9,
        },
    )


def plot_twitter_three_way(ax=None):
    setup_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
    else:
        fig = ax.figure
    df = twitter_three_way_df()
    bars = ax.bar(df["system"], df["seconds"], color=[COLORS[f] for f in df["family"]], width=0.58)
    ax.set_title("Exact Twitter pipeline wall time")
    ax.set_ylabel("Warm median seconds")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="x", pad=10)
    top = _top_with_headroom(df["seconds"].tolist(), pad_ratio=0.22, min_pad=0.18)
    ax.set_ylim(0, top)
    for bar, seconds, mult in zip(bars, df["seconds"], df["vs_fastest_x"]):
        _label_box(ax, bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02, f"{seconds:.2f}s\n{mult:.2f}x vs best")
    fig.tight_layout(pad=1.4)
    return fig, ax


def plot_twitter_stage_breakdown(ax=None):
    setup_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.6, 5.7))
    else:
        fig = ax.figure
    df = twitter_stage_df()
    systems = list(df["system"].drop_duplicates())
    stage_order = ["Search 1", "PageRank", "Search 2"]
    bottoms = [0.0] * len(systems)
    for stage in stage_order:
        subset = df[df["stage"] == stage].set_index("system").reindex(systems)
        vals = subset["seconds"].tolist()
        ax.bar(systems, vals, bottom=bottoms, label=stage, color=COLORS[stage], width=0.58)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax.set_title("Where the Twitter pipeline spends time")
    ax.set_ylabel("Warm median seconds")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="x", pad=10)
    top = _top_with_headroom(bottoms, pad_ratio=0.2, min_pad=0.2)
    ax.set_ylim(0, top)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for i, total in enumerate(bottoms):
        _label_box(ax, i, total + top * 0.018, f"{total:.2f}s")
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.4)
    return fig, ax


def plot_load_breakdown():
    setup_matplotlib()
    df = load_breakdown_df()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    datasets = ["Twitter", "GPlus"]
    stage_order = ["Load", "Shaping", "Bind"]
    global_totals = []
    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        for system in ["Graphistry GPU", "Graphistry CPU"]:
            global_totals.append(float(subset[subset["system"] == system]["seconds"].sum()))
    top = _top_with_headroom(global_totals, pad_ratio=0.18, min_pad=0.12)
    for ax, dataset in zip(axes, datasets):
        subset = df[df["dataset"] == dataset]
        systems = ["Graphistry GPU", "Graphistry CPU"]
        bottoms = [0.0] * len(systems)
        for stage in stage_order:
            vals = subset[subset["stage"] == stage].set_index("system").reindex(systems)["seconds"].tolist()
            ax.bar(systems, vals, bottom=bottoms, label=stage, color=COLORS[stage], width=0.58)
            bottoms = [a + b for a, b in zip(bottoms, vals)]
        ax.set_title(f"{dataset}: cached load + shape")
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
        _style_axes(ax)
        ax.tick_params(axis="x", pad=10)
        ax.set_ylim(0, top)
        for i, total in enumerate(bottoms):
            _label_box(ax, i, total + top * 0.018, f"{total:.2f}s")
    axes[0].set_ylabel("Median seconds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Data loading and shaping are part of the story", y=0.99, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=1.4)
    return fig, axes


def plot_gplus_lifecycle(ax=None):
    setup_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.6, 5.8))
    else:
        fig = ax.figure
    df = gplus_lifecycle_df()
    systems = ["Graphistry GPU", "Graphistry CPU"]
    phases = ["Load + shape", "Search + analytics pipeline"]
    bottoms = [0.0] * len(systems)
    phase_colors = {"Load + shape": COLORS["Load"], "Search + analytics pipeline": COLORS["PageRank"]}
    for phase in phases:
        vals = df[df["phase"] == phase].set_index("system").reindex(systems)["seconds"].tolist()
        ax.bar(systems, vals, bottom=bottoms, label=phase, color=phase_colors[phase], width=0.58)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    top = max(_top_with_headroom(bottoms, pad_ratio=0.2, min_pad=6.0), NEO4J_GPLUS_LOWER_BOUND_S + 22)
    ax.set_ylim(0, top)
    ax.axhline(NEO4J_GPLUS_LOWER_BOUND_S, color=COLORS["Neo4j"], linestyle="--", linewidth=2.2)
    ax.annotate(
        "Neo4j lower bound > 187s",
        xy=(1.25, NEO4J_GPLUS_LOWER_BOUND_S),
        xytext=(1.33, NEO4J_GPLUS_LOWER_BOUND_S + 12),
        ha="right",
        va="bottom",
        fontsize=10,
        color=COLORS["Neo4j"],
        arrowprops={"arrowstyle": "-", "color": COLORS["Neo4j"], "lw": 1.4},
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.9},
        clip_on=False,
    )
    ax.set_title("GPlus lifecycle on a larger graph")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="x", pad=10)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    for i, total in enumerate(bottoms):
        _label_box(ax, i, total + top * 0.012, f"{total:.1f}s")
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.4)
    return fig, ax


def render_all(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "twitter_three_way.svg": plot_twitter_three_way()[0],
        "twitter_stage_breakdown.svg": plot_twitter_stage_breakdown()[0],
        "load_breakdown.svg": plot_load_breakdown()[0],
        "gplus_lifecycle.svg": plot_gplus_lifecycle()[0],
    }
    for name, fig in figures.items():
        fig.savefig(output_dir / name, format="svg")
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    render_all(args.output_dir)
