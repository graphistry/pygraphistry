#\!/usr/bin/env python3
"""Helpers for presenting the GFQL Cypher benchmark from saved JSON results.

Presentation-only: reads existing DGX benchmark artifacts and renders
charts for docs and notebooks. Does not rerun benchmarks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "plans" / "gfql-gpu-pagerank-benchmark" / "results"
NEO4J_GPLUS_LOWER_BOUND_S = 187.0

COLORS = {
    "Neo4j": "#8f3b2f",
    "GFQL CPU": "#425466",
    "GFQL GPU": "#14866d",
    "ETL": "#6c8fb3",
    "Search": "#557a95",
    "Analytics": "#d17c2f",
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


def pipeline_overview_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Phase": "Data loading (ETL)", "What happens": "Read a cached SNAP edge list into a dataframe", "Why it matters": "Shows dataframe-native ingest cost, not just graph runtime"},
        {"Phase": "Data shaping (ETL)", "What happens": "Compute node degree as reusable node metadata", "Why it matters": "Benchmarks the columnar prep that later search stages query"},
        {"Phase": "Graph search", "What happens": "Run local Cypher GRAPH { MATCH ... } subgraph extraction", "Why it matters": "Measures graph-preserving search on dataframe-backed graphs"},
        {"Phase": "Graph analytics", "What happens": "Run local Cypher CALL graphistry.*.pagerank.write()", "Why it matters": "Shows backend graph algorithm acceleration"},
        {"Phase": "Graph search (2nd pass)", "What happens": "Keep high-PageRank core and neighborhood", "Why it matters": "Second graph-preserving search on enriched graph"},
    ])


# ---------------------------------------------------------------------------
# Data helpers for the stacked workload-phase charts
# ---------------------------------------------------------------------------

def twitter_lifecycle_df() -> pd.DataFrame:
    """3-engine Twitter lifecycle: ETL, Search+Enrich (search1+pagerank), Final Search."""
    cpu_gpu = {r["engine"]: r for r in twitter_cpu_gpu()["results"]}
    neo = twitter_neo4j()
    load = {r["engine"]: r for r in twitter_load()["results"]}

    rows = []
    # Neo4j
    neo_etl = neo["db_import_s"] + neo["prepare_s"]
    neo_search = neo["gfql_filter1_median_s"] + neo["gfql_filter2_median_s"]
    neo_analytics = neo["pagerank_median_s"]
    rows.extend([
        {"system": "Neo4j + GDS", "family": "Neo4j", "phase": "ETL", "seconds": neo_etl},
        {"system": "Neo4j + GDS", "family": "Neo4j", "phase": "Search", "seconds": neo_search},
        {"system": "Neo4j + GDS", "family": "Neo4j", "phase": "Analytics", "seconds": neo_analytics},
    ])
    # GFQL CPU
    cpu = cpu_gpu["pandas"]
    cpu_etl = load["pandas"]["total_prepare_median_s"]
    cpu_search = cpu["search_enrich_median_s"] - _approx_pagerank_fraction(cpu) + cpu["gfql_filter2_median_s"]
    cpu_analytics = _approx_pagerank_fraction(cpu)
    rows.extend([
        {"system": "GFQL CPU\n(pandas + igraph)", "family": "GFQL CPU", "phase": "ETL", "seconds": cpu_etl},
        {"system": "GFQL CPU\n(pandas + igraph)", "family": "GFQL CPU", "phase": "Search", "seconds": cpu_search},
        {"system": "GFQL CPU\n(pandas + igraph)", "family": "GFQL CPU", "phase": "Analytics", "seconds": cpu_analytics},
    ])
    # GFQL GPU
    gpu = cpu_gpu["cudf"]
    gpu_etl = load["cudf"]["total_prepare_median_s"]
    gpu_search = gpu["search_enrich_median_s"] - _approx_pagerank_fraction(gpu) + gpu["gfql_filter2_median_s"]
    gpu_analytics = _approx_pagerank_fraction(gpu)
    rows.extend([
        {"system": "GFQL GPU\n(cudf + cugraph)", "family": "GFQL GPU", "phase": "ETL", "seconds": gpu_etl},
        {"system": "GFQL GPU\n(cudf + cugraph)", "family": "GFQL GPU", "phase": "Search", "seconds": gpu_search},
        {"system": "GFQL GPU\n(cudf + cugraph)", "family": "GFQL GPU", "phase": "Analytics", "seconds": gpu_analytics},
    ])
    return pd.DataFrame(rows)


def _approx_pagerank_fraction(result: Dict) -> float:
    """Estimate pagerank time from the compound search_enrich timing.

    We use the ratio from the previous 3-stage run where search and pagerank
    were timed separately.  Twitter igraph pagerank was ~45% of search_enrich,
    cugraph ~13%.  This is approximate but honest enough for the stacked chart.
    """
    if result["engine"] == "pandas":
        return result["search_enrich_median_s"] * 0.45
    return result["search_enrich_median_s"] * 0.13


def gplus_lifecycle_df() -> pd.DataFrame:
    """3-engine GPlus lifecycle with Neo4j lower bound."""
    summary = gplus_summary()
    load = {r["engine"]: r for r in gplus_load()["results"]}

    rows = [
        # Neo4j: only have a pipeline lower bound, no phase split
        {"system": "Neo4j + GDS\n(lower bound)", "family": "Neo4j", "phase": "Pipeline (lower bound)", "seconds": NEO4J_GPLUS_LOWER_BOUND_S},
    ]
    for key, engine, label in [("cpu", "pandas", "GFQL CPU"), ("gpu", "cudf", "GFQL GPU")]:
        r = summary[key]
        rows.extend([
            {"system": label, "family": label, "phase": "ETL", "seconds": load[engine]["total_prepare_median_s"]},
            {"system": label, "family": label, "phase": "Search + Analytics", "seconds": r["pipeline_total_median_s"]},
        ])
    return pd.DataFrame(rows)


def twitter_three_way_df() -> pd.DataFrame:
    """Simple total-time comparison for the summary table."""
    cpu_gpu = {r["engine"]: r for r in twitter_cpu_gpu()["results"]}
    neo = twitter_neo4j()
    rows = [
        {"system": "Neo4j + GDS", "family": "Neo4j", "seconds": neo["pipeline_total_median_s"]},
        {"system": "GFQL CPU\n(pandas + igraph)", "family": "GFQL CPU", "seconds": cpu_gpu["pandas"]["pipeline_total_median_s"]},
        {"system": "GFQL GPU\n(cudf + cugraph)", "family": "GFQL GPU", "seconds": cpu_gpu["cudf"]["pipeline_total_median_s"]},
    ]
    df = pd.DataFrame(rows)
    neo4j_s = float(df["seconds"].iloc[0])
    df["speedup_vs_neo4j"] = (neo4j_s / df["seconds"]).round(1)
    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

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
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fontsize,
            color="#0F172A", clip_on=False,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": "#CBD5E1", "linewidth": 0.9})


# ---------------------------------------------------------------------------
# Main charts
# ---------------------------------------------------------------------------

def plot_twitter_lifecycle(ax=None):
    """Money shot: Twitter 3-engine stacked by ETL / Search / Analytics."""
    setup_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 6.2))
    else:
        fig = ax.figure
    df = twitter_lifecycle_df()
    systems = list(df["system"].drop_duplicates())
    phase_order = ["ETL", "Search", "Analytics"]
    bottoms = [0.0] * len(systems)
    for phase in phase_order:
        subset = df[df["phase"] == phase].set_index("system").reindex(systems)
        vals = subset["seconds"].fillna(0).tolist()
        ax.bar(systems, vals, bottom=bottoms, label=phase, color=COLORS[phase], width=0.56)
        bottoms = [a + b for a, b in zip(bottoms, vals)]

    top = _top_with_headroom(bottoms, pad_ratio=0.22, min_pad=1.2)
    ax.set_ylim(0, top)
    ax.set_title("GFQL Cypher Benchmark: Twitter end-to-end")
    ax.set_ylabel("Seconds (lower is better →)")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="x", pad=10)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18))

    neo4j_total = bottoms[0]
    for i, total in enumerate(bottoms):
        if i == 0:
            _label_box(ax, i, total + top * 0.015, f"{total:.1f}s")
        else:
            speedup = neo4j_total / total
            _label_box(ax, i, total + top * 0.015, f"{total:.2f}s\n{speedup:.0f}x vs Neo4j")

    fig.tight_layout(rect=(0, 0.12, 1, 0.97), pad=1.4)
    return fig, ax


def plot_gplus_lifecycle(ax=None):
    """GPlus 3-engine comparison with Neo4j lower bound."""
    setup_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 6.2))
    else:
        fig = ax.figure
    df = gplus_lifecycle_df()
    systems = list(df["system"].drop_duplicates())
    # Neo4j has a single "Pipeline (lower bound)" phase; others have ETL + Search+Analytics
    all_phases = ["Pipeline (lower bound)", "ETL", "Search + Analytics"]
    phase_colors = {
        "Pipeline (lower bound)": COLORS["Neo4j"],
        "ETL": COLORS["ETL"],
        "Search + Analytics": COLORS["Analytics"],
    }
    bottoms = [0.0] * len(systems)
    drawn_labels = set()
    for phase in all_phases:
        subset = df[df["phase"] == phase].set_index("system").reindex(systems)
        vals = subset["seconds"].fillna(0).tolist()
        label = phase if phase not in drawn_labels else None
        ax.bar(systems, vals, bottom=bottoms, label=label, color=phase_colors[phase], width=0.56)
        drawn_labels.add(phase)
        bottoms = [a + b for a, b in zip(bottoms, vals)]

    top = _top_with_headroom(bottoms, pad_ratio=0.2, min_pad=12.0)
    ax.set_ylim(0, top)
    ax.set_title("GFQL Cypher Benchmark: GPlus end-to-end")
    ax.set_ylabel("Seconds (lower is better →)")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.tick_params(axis="x", pad=10)

    legend_elements = [
        Patch(facecolor=COLORS["Neo4j"], label="Neo4j pipeline (lower bound)"),
        Patch(facecolor=COLORS["ETL"], label="ETL (load + shape)"),
        Patch(facecolor=COLORS["Analytics"], label="Search + Analytics"),
    ]
    ax.legend(handles=legend_elements, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18))

    neo4j_total = bottoms[0]
    for i, total in enumerate(bottoms):
        if i == 0:
            _label_box(ax, i, total + top * 0.012, f">{total:.0f}s")
        else:
            speedup = neo4j_total / total
            _label_box(ax, i, total + top * 0.012, f"{total:.1f}s\n{speedup:.0f}x vs Neo4j")

    fig.tight_layout(rect=(0, 0.12, 1, 0.97), pad=1.4)
    return fig, ax


def render_all(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "twitter_lifecycle.svg": plot_twitter_lifecycle()[0],
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
