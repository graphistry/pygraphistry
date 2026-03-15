#!/usr/bin/env python3
"""Benchmark a local Cypher graph-search -> CALL write -> graph-search pipeline on CPU and GPU backends.

Designed for DGX-style runs where the graph is loaded once, then the main pipeline is
benchmarked warm on the resident in-memory graph.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

import graphistry

def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"[download] {url} -> {path}", flush=True)
    urllib.request.urlretrieve(url, path)


def dataset_spec(dataset: str) -> Dict[str, str | int]:
    if dataset == "twitter":
        return {
            "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
            "filename": "twitter_combined.txt.gz",
            "sep": " ",
            "compression": "gzip",
            "skiprows": 0,
        }
    if dataset == "gplus":
        return {
            "url": "https://snap.stanford.edu/data/gplus_combined.txt.gz",
            "filename": "gplus_combined.txt.gz",
            "sep": " ",
            "compression": "gzip",
            "skiprows": 0,
        }
    if dataset == "orkut":
        return {
            "url": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
            "filename": "com-orkut.ungraph.txt.gz",
            "sep": "\t",
            "compression": "gzip",
            "skiprows": 5,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def load_edges(engine: str, dataset: str, data_dir: Path):
    spec = dataset_spec(dataset)
    path = data_dir / str(spec["filename"])
    download(str(spec["url"]), path)
    kwargs = {
        "sep": str(spec["sep"]),
        "names": ["src", "dst"],
        "compression": str(spec["compression"]),
        "skiprows": int(spec["skiprows"]),
    }
    if engine == "cudf":
        import cudf  # type: ignore

        return cudf.read_csv(path, **kwargs)
    import pandas as pd

    return pd.read_csv(path, **kwargs)


def build_nodes(edges, degree_quantile: float):
    src_degree = edges["src"].value_counts().rename("src_degree").reset_index()
    src_degree.columns = ["id", "src_degree"]
    dst_degree = edges["dst"].value_counts().rename("dst_degree").reset_index()
    dst_degree.columns = ["id", "dst_degree"]
    degree_df = src_degree.merge(dst_degree, on="id", how="outer").fillna(0)
    degree_df["degree"] = degree_df["src_degree"].astype("int64") + degree_df["dst_degree"].astype("int64")
    nodes = degree_df[["id", "degree"]].copy()
    cutoff = nodes["degree"].quantile(degree_quantile)
    if hasattr(cutoff, "to_pandas"):
        cutoff = cutoff.to_pandas()
    return nodes, float(cutoff)


def from_engine_scalar(value) -> float:
    if hasattr(value, "to_pandas"):
        value = value.to_pandas()
    return float(value)


def count_rows(df) -> int:
    return int(len(df))


def prepare_graph(dataset: str, engine: str, degree_quantile: float, data_dir: Path):
    t0 = time.perf_counter()
    edges = load_edges(engine, dataset, data_dir)
    nodes, degree_cutoff = build_nodes(edges, degree_quantile)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    t1 = time.perf_counter()
    prep = {
        "load_prepare_s": round(t1 - t0, 4),
        "degree_cutoff": degree_cutoff,
        "full_nodes": count_rows(g._nodes),
        "full_edges": count_rows(g._edges),
    }
    return g, prep


def backend_name(engine: str) -> str:
    return "igraph" if engine == "pandas" else "cugraph"


def return_graph_search_query(*, metric: str) -> str:
    if metric == "degree":
        return (
            "MATCH (seed)-[reach]-(nbr) "
            "WHERE seed.degree >= $cutoff "
            "RETURN GRAPH"
        )
    if metric == "pagerank":
        return (
            "MATCH (core)-[halo]-(nbr) "
            "WHERE core.pagerank >= $cutoff "
            "RETURN GRAPH"
        )
    raise ValueError(f"Unknown search metric: {metric}")


def pagerank_call(engine: str) -> str:
    return f"CALL graphistry.{backend_name(engine)}.pagerank.write()"


def run_pipeline_once(g, engine: str, degree_cutoff: float, pagerank_quantile: float):
    t1 = time.perf_counter()
    g1 = g.gfql(
        return_graph_search_query(metric="degree"),
        params={"cutoff": degree_cutoff},
        engine=engine,
    )
    t2 = time.perf_counter()

    g2 = g1.gfql(pagerank_call(engine), engine=engine)
    t3 = time.perf_counter()

    pr_cutoff = from_engine_scalar(g2._nodes["pagerank"].quantile(pagerank_quantile))
    g3 = g2.gfql(
        return_graph_search_query(metric="pagerank"),
        params={"cutoff": pr_cutoff},
        engine=engine,
    )
    t4 = time.perf_counter()

    return {
        "gfql_filter1_s": t2 - t1,
        "pagerank_s": t3 - t2,
        "gfql_filter2_s": t4 - t3,
        "pipeline_total_s": t4 - t1,
        "pagerank_cutoff": pr_cutoff,
        "stage1_nodes": count_rows(g1._nodes),
        "stage1_edges": count_rows(g1._edges),
        "stage2_nodes": count_rows(g2._nodes),
        "stage2_edges": count_rows(g2._edges),
        "stage3_nodes": count_rows(g3._nodes),
        "stage3_edges": count_rows(g3._edges),
    }


def median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def benchmark(dataset: str, engine: str, degree_quantile: float, pagerank_quantile: float, data_dir: Path, runs: int, warmup: int):
    g, prep = prepare_graph(dataset, engine, degree_quantile, data_dir)

    for _ in range(warmup):
        run_pipeline_once(g, engine, prep["degree_cutoff"], pagerank_quantile)

    rows = [run_pipeline_once(g, engine, prep["degree_cutoff"], pagerank_quantile) for _ in range(runs)]
    first = rows[0]

    return {
        "dataset": dataset,
        "engine": engine,
        "backend": backend_name(engine),
        "degree_quantile": degree_quantile,
        "pagerank_quantile": pagerank_quantile,
        **prep,
        "pagerank_cutoff": first["pagerank_cutoff"],
        "warm_runs": runs,
        "warmup_runs": warmup,
        "cold_total_first_s": round(prep["load_prepare_s"] + first["pipeline_total_s"], 4),
        "pipeline_total_first_s": round(first["pipeline_total_s"], 4),
        "gfql_filter1_median_s": round(median([r["gfql_filter1_s"] for r in rows]), 4),
        "pagerank_median_s": round(median([r["pagerank_s"] for r in rows]), 4),
        "gfql_filter2_median_s": round(median([r["gfql_filter2_s"] for r in rows]), 4),
        "pipeline_total_median_s": round(median([r["pipeline_total_s"] for r in rows]), 4),
        "pipeline_total_runs_s": [round(r["pipeline_total_s"], 4) for r in rows],
        "stage1_nodes": first["stage1_nodes"],
        "stage1_edges": first["stage1_edges"],
        "stage2_nodes": first["stage2_nodes"],
        "stage2_edges": first["stage2_edges"],
        "stage3_nodes": first["stage3_nodes"],
        "stage3_edges": first["stage3_edges"],
    }


def summarize(cpu: Dict, gpu: Dict) -> Dict[str, float]:
    return {
        "cold_speedup_x": round(cpu["cold_total_first_s"] / gpu["cold_total_first_s"], 2),
        "warm_speedup_x": round(cpu["pipeline_total_median_s"] / gpu["pipeline_total_median_s"], 2),
        "gfql_filter1_speedup_x": round(cpu["gfql_filter1_median_s"] / gpu["gfql_filter1_median_s"], 2),
        "pagerank_speedup_x": round(cpu["pagerank_median_s"] / gpu["pagerank_median_s"], 2),
        "gfql_filter2_speedup_x": round(cpu["gfql_filter2_median_s"] / gpu["gfql_filter2_median_s"], 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["twitter", "gplus", "orkut"], default="twitter")
    ap.add_argument("--engine", choices=["pandas", "cudf", "both"], default="both")
    ap.add_argument("--degree-quantile", type=float, default=0.99)
    ap.add_argument("--pagerank-quantile", type=float, default=0.99)
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
            f"[run] dataset={args.dataset} engine={engine} degree_q={args.degree_quantile} pagerank_q={args.pagerank_quantile} warmup={args.warmup} runs={args.runs}",
            flush=True,
        )
        result = benchmark(args.dataset, engine, args.degree_quantile, args.pagerank_quantile, args.data_dir, args.runs, args.warmup)
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
