#!/usr/bin/env python3
"""Benchmark cached SNAP load/prep time for CPU and GPU GFQL paths.

This complements the warm pipeline benchmark by isolating the local cached-file
read and graph preparation cost:
- edge-list read (`pandas.read_csv` / `cudf.read_csv`)
- node materialization (degree table + seed flag)
- Graphistry graph binding

Network download time is intentionally excluded from benchmark timings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import graphistry

from benchmarks.gfql.filter_pagerank_pipeline_cpu_gpu import dataset_spec, download


def ensure_cached(dataset: str, data_dir: Path) -> Path:
    spec = dataset_spec(dataset)
    path = data_dir / str(spec["filename"])
    download(str(spec["url"]), path)
    return path


def load_edges(engine: str, dataset: str, data_dir: Path):
    spec = dataset_spec(dataset)
    path = ensure_cached(dataset, data_dir)
    kwargs = {
        "sep": str(spec["sep"]),
        "names": ["src", "dst"],
        "compression": "infer",
        "skiprows": int(spec["skiprows"]),
    }

    if engine == "cudf":
        import cudf  # type: ignore
        return cudf.read_csv(path, **kwargs)

    import pandas as pd
    return pd.read_csv(path, **kwargs)


def build_nodes(engine: str, edges, degree_quantile: float):
    src_degree = edges["src"].value_counts().rename("src_degree").reset_index()
    src_degree.columns = ["id", "src_degree"]
    dst_degree = edges["dst"].value_counts().rename("dst_degree").reset_index()
    dst_degree.columns = ["id", "dst_degree"]
    degree_df = src_degree.merge(dst_degree, on="id", how="outer")
    degree_df = degree_df.fillna(0)
    degree_df["degree"] = degree_df["src_degree"].astype("int64") + degree_df["dst_degree"].astype("int64")
    nodes = degree_df[["id", "degree"]].copy()
    cutoff = nodes["degree"].quantile(degree_quantile)
    if engine == "cudf" and hasattr(cutoff, "to_pandas"):
        cutoff = cutoff.to_pandas()
    cutoff_f = float(cutoff)
    nodes["seed_keep"] = nodes["degree"] >= cutoff_f
    return nodes, cutoff_f


def count_rows(df) -> int:
    return int(len(df))


def measure_once(dataset: str, engine: str, degree_quantile: float, data_dir: Path) -> Dict[str, float | int | str]:
    ensure_cached(dataset, data_dir)

    t0 = time.perf_counter()
    edges = load_edges(engine, dataset, data_dir)
    t1 = time.perf_counter()

    nodes, degree_cutoff = build_nodes(engine, edges, degree_quantile)
    t2 = time.perf_counter()

    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    t3 = time.perf_counter()

    return {
        "dataset": dataset,
        "engine": engine,
        "degree_quantile": degree_quantile,
        "load_edges_s": t1 - t0,
        "build_nodes_s": t2 - t1,
        "bind_graph_s": t3 - t2,
        "total_prepare_s": t3 - t0,
        "degree_cutoff": degree_cutoff,
        "full_nodes": count_rows(g._nodes),
        "full_edges": count_rows(g._edges),
    }


def median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def benchmark(dataset: str, engine: str, degree_quantile: float, data_dir: Path, runs: int, warmup: int):
    ensure_cached(dataset, data_dir)
    for _ in range(warmup):
        measure_once(dataset, engine, degree_quantile, data_dir)

    rows = [measure_once(dataset, engine, degree_quantile, data_dir) for _ in range(runs)]
    first = rows[0]

    return {
        "dataset": dataset,
        "engine": engine,
        "degree_quantile": degree_quantile,
        "warmup_runs": warmup,
        "benchmark_runs": runs,
        "download_included": False,
        "load_edges_median_s": round(median([float(r["load_edges_s"]) for r in rows]), 4),
        "build_nodes_median_s": round(median([float(r["build_nodes_s"]) for r in rows]), 4),
        "bind_graph_median_s": round(median([float(r["bind_graph_s"]) for r in rows]), 4),
        "total_prepare_median_s": round(median([float(r["total_prepare_s"]) for r in rows]), 4),
        "load_edges_runs_s": [round(float(r["load_edges_s"]), 4) for r in rows],
        "build_nodes_runs_s": [round(float(r["build_nodes_s"]), 4) for r in rows],
        "bind_graph_runs_s": [round(float(r["bind_graph_s"]), 4) for r in rows],
        "total_prepare_runs_s": [round(float(r["total_prepare_s"]), 4) for r in rows],
        "degree_cutoff": float(first["degree_cutoff"]),
        "full_nodes": int(first["full_nodes"]),
        "full_edges": int(first["full_edges"]),
    }


def summarize(cpu: Dict, gpu: Dict) -> Dict[str, float]:
    return {
        "load_edges_speedup_x": round(cpu["load_edges_median_s"] / gpu["load_edges_median_s"], 2),
        "build_nodes_speedup_x": round(cpu["build_nodes_median_s"] / gpu["build_nodes_median_s"], 2),
        "bind_graph_speedup_x": round(cpu["bind_graph_median_s"] / gpu["bind_graph_median_s"], 2),
        "total_prepare_speedup_x": round(cpu["total_prepare_median_s"] / gpu["total_prepare_median_s"], 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["twitter", "gplus", "orkut"], default="twitter")
    ap.add_argument("--engine", choices=["pandas", "cudf", "both"], default="both")
    ap.add_argument("--degree-quantile", type=float, default=0.99)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=Path("plans/gfql-gpu-pagerank-benchmark/data"))
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    engines = [args.engine] if args.engine != "both" else ["pandas", "cudf"]
    results = []
    by_engine = {}
    for engine in engines:
        print(
            f"[run] dataset={args.dataset} engine={engine} degree_q={args.degree_quantile} warmup={args.warmup} runs={args.runs}",
            flush=True,
        )
        result = benchmark(args.dataset, engine, args.degree_quantile, args.data_dir, args.runs, args.warmup)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        results.append(result)
        by_engine[engine] = result

    payload = {"results": results}
    if "pandas" in by_engine and "cudf" in by_engine:
        payload["speedup_summary"] = summarize(by_engine["pandas"], by_engine["cudf"])
        print(json.dumps({"speedup_summary": payload["speedup_summary"]}, indent=2, sort_keys=True), flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
