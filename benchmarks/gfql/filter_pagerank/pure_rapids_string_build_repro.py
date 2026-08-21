#!/usr/bin/env python3
"""Pure cuDF/cuGraph reproducer for sparse string-ID graph build behavior.

No PyGraphistry imports. This script is intended for issue filing and should run
directly inside official RAPIDS base images.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List


GRAPH_CHOICES = [
    "synthetic_offset",
    "synthetic_string_offset",
    "synthetic_string_gplus_shape",
]

GPLUS_SHAPE_VERTICES = 107_614


def count_rows(df: Any) -> int:
    return int(len(df))


def expected_unique_nodes(edges: Any) -> int:
    import cudf  # type: ignore

    ids = cudf.concat([edges["src"], edges["dst"]], ignore_index=True)
    return int(ids.nunique())


def build_edges(graph_kind: str, edge_count: int):
    import cudf  # type: ignore
    import cupy as cp  # type: ignore

    if graph_kind == "synthetic_string_gplus_shape":
        start = 1_000_000
        base = cudf.Series(cp.arange(edge_count, dtype=cp.int32) % cp.int32(GPLUS_SHAPE_VERTICES))
        src = (base + start).astype("int32").astype("str")
        dst = ((base + 1) % GPLUS_SHAPE_VERTICES + start).astype("int32").astype("str")
        return cudf.DataFrame({"src": src, "dst": dst})

    start = 1_000_000
    src = cudf.Series(range(start, start + edge_count), dtype="int32")
    dst = cudf.Series(range(start + 1, start + edge_count + 1), dtype="int32")

    if graph_kind == "synthetic_offset":
        return cudf.DataFrame({"src": src, "dst": dst})
    if graph_kind == "synthetic_string_offset":
        return cudf.DataFrame({"src": src.astype("str"), "dst": dst.astype("str")})

    raise ValueError(f"Unsupported graph kind: {graph_kind}")


def measure_once(edges: Any, *, store_transposed: bool) -> Dict[str, Any]:
    import cugraph  # type: ignore

    t0 = time.perf_counter()
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(
        edges,
        source="src",
        destination="dst",
        store_transposed=store_transposed,
    )
    t1 = time.perf_counter()
    pr = cugraph.pagerank(G)
    t2 = time.perf_counter()
    return {
        "build_s": t1 - t0,
        "pagerank_s": t2 - t1,
        "total_s": t2 - t0,
        "graph_vertices": int(G.number_of_vertices()),
        "pagerank_rows": int(len(pr)),
    }


def median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def run_case(graph_kind: str, edge_count: int, *, store_transposed: bool, runs: int, warmup: int) -> Dict[str, Any]:
    edges = build_edges(graph_kind, edge_count)

    for _ in range(warmup):
        measure_once(edges, store_transposed=store_transposed)

    rows = [measure_once(edges, store_transposed=store_transposed) for _ in range(runs)]
    first = rows[0]
    expected_nodes = expected_unique_nodes(edges)

    return {
        "graph_kind": graph_kind,
        "edge_count": count_rows(edges),
        "expected_unique_nodes": expected_nodes,
        "src_dtype": str(edges["src"].dtype),
        "dst_dtype": str(edges["dst"].dtype),
        "store_transposed": store_transposed,
        "warmup_runs": warmup,
        "benchmark_runs": runs,
        "build_median_s": round(median([float(r["build_s"]) for r in rows]), 4),
        "pagerank_median_s": round(median([float(r["pagerank_s"]) for r in rows]), 4),
        "total_median_s": round(median([float(r["total_s"]) for r in rows]), 4),
        "build_runs_s": [round(float(r["build_s"]), 4) for r in rows],
        "pagerank_runs_s": [round(float(r["pagerank_s"]), 4) for r in rows],
        "total_runs_s": [round(float(r["total_s"]), 4) for r in rows],
        "graph_vertices": int(first["graph_vertices"]),
        "pagerank_rows": int(first["pagerank_rows"]),
        "expected_vertex_match": int(first["graph_vertices"]) == expected_nodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=GRAPH_CHOICES, default="synthetic_string_gplus_shape")
    parser.add_argument("--synthetic-edges", type=int, default=10_000_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--store-transposed", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_case(
        args.graph,
        args.synthetic_edges,
        store_transposed=args.store_transposed,
        runs=args.runs,
        warmup=args.warmup,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
