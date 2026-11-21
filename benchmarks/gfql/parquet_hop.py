"""Matched public GFQL and native Polars Parquet hop benchmark on Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
from statistics import median
import subprocess
import sys
from time import perf_counter
from typing import Callable, TypeVar
from unittest.mock import patch

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.gfql.index import gfql_index_edges
import polars as pl

from graphistry.compute.gfql.parquet import ParquetHopLimits


def proc_io() -> dict[str, int]:
    return {key: int(value) for key, value in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


_T = TypeVar("_T")


def measure(fn: Callable[[], _T]) -> tuple[_T, dict[str, float | int]]:
    before = proc_io()
    start = perf_counter()
    data = fn()
    seconds = perf_counter() - start
    after = proc_io()
    return data, {**{key: after[key] - before[key] for key in ("read_bytes", "write_bytes", "rchar")},
                  "seconds": seconds, "process_high_water_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def prepare(args: argparse.Namespace) -> None:
    nodes_path, edges_path = args.data / "nodes.parquet", args.data / "edges.parquet"
    report: dict[str, object] = {"kind": "setup"}
    if args.edges is None:
        if nodes_path.exists() or edges_path.exists():
            raise ValueError("Use a new --data directory when generating or importing source files")

        def ingest() -> None:
            if args.snap_gzip is not None:
                edges = pl.read_csv(
                    args.snap_gzip, separator="\t", comment_prefix="#", has_header=False,
                    schema={"src": pl.Int64, "dst": pl.Int64},
                ).with_row_index("edge_id").with_columns(pl.col("edge_id").cast(pl.Int64))
            else:
                # Deterministic chain with a second parallel edge at each source and interleaved rows.
                src = pl.Series("src", range(args.synthetic_edges // 2), dtype=pl.Int64)
                distinct = pl.DataFrame({"src": src, "dst": src + 1})
                edges = pl.concat([distinct.reverse(), distinct]).with_row_index("edge_id")
            ids = pl.concat([edges["src"], edges["dst"]]).unique().sort().alias("id")
            nodes = ids.to_frame().with_columns((pl.col("id") % 137).alias("score"))
            edges = edges.with_columns((pl.col("edge_id") % 59).alias("weight"))
            nodes.write_parquet(nodes_path, row_group_size=args.row_group_size)
            edges.write_parquet(edges_path, row_group_size=args.row_group_size)

        _, report["ingestion"] = measure(ingest)
    else:
        nodes_path, edges_path = args.nodes, args.edges
    original_edges = edges_path
    if args.cluster:
        edges_path = args.data / "edges_by_source.parquet"
        if edges_path.exists():
            raise ValueError("Use a new --data directory when building a clustered layout")

        def cluster() -> None:
            pl.scan_parquet(original_edges).sort("src").sink_parquet(
                edges_path, row_group_size=args.row_group_size, compression="zstd",
            )

        _, report["cluster_build"] = measure(cluster)
    report.update(nodes=str(nodes_path), edges=str(edges_path), source_edge_bytes=original_edges.stat().st_size,
                  layout_edge_bytes=edges_path.stat().st_size, node_bytes=nodes_path.stat().st_size)
    print(json.dumps(report), flush=True)


def digest(nodes: pl.DataFrame, edges: pl.DataFrame) -> str:
    import io

    h = hashlib.sha256()
    for frame, key in ((nodes, "id"), (edges, "edge_id")):
        output = io.BytesIO()
        frame.sort(key).rechunk().write_ipc(output)
        h.update(output.getvalue())
    return h.hexdigest()


def _native_selection(key: str, ids: pl.Series) -> pl.Expr:
    column = pl.col(key)
    return column.is_between(
        pl.lit(ids.min(), dtype=ids.dtype), pl.lit(ids.max(), dtype=ids.dtype),
    ) & column.is_in(ids.implode())


def _native_properties(scan: pl.LazyFrame, schema: pl.Schema, key: str, ids: pl.Series) -> pl.DataFrame:
    if ids.is_empty():
        return pl.DataFrame(schema=schema)
    return (
        scan.filter(_native_selection(key, ids))
        .with_columns(pl.selectors.float().fill_nan(None))
        .collect(engine="streaming")
    )


def native(nodes: pl.LazyFrame, edges: pl.LazyFrame, starts: pl.Series, hops: int) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """Public Polars scan reference with Python sets for traversal state."""
    node_schema, edge_schema = nodes.collect_schema(), edges.collect_schema()

    def lookup(ids: set[int]) -> set[int]:
        if not ids:
            return set()
        keys = pl.Series("id", sorted(ids), dtype=starts.dtype)
        matched = nodes.filter(_native_selection("id", keys)).select("id").collect(engine="streaming")
        return set(matched["id"])

    frontier = lookup(set(starts.drop_nulls()))
    visited: set[int] = set()
    visited_edges: list[int] = []
    for _ in range(hops):
        if not frontier:
            break
        src_ids = pl.Series("src", sorted(frontier), dtype=starts.dtype)
        pairs = (
            edges.filter(_native_selection("src", src_ids) & pl.col("dst").is_not_null())
            .select("src", "dst", "edge_id").collect(engine="streaming")
        )
        destinations = lookup(set(pairs["dst"]))
        valid_dst = pl.Series("dst", sorted(destinations), dtype=starts.dtype)
        matches = pairs.filter(pl.col("dst").is_in(valid_dst.implode()))
        visited.update(matches["src"])
        frontier = destinations - visited
        visited.update(destinations)
        visited_edges.extend(matches["edge_id"])

    start = perf_counter()
    node_ids = pl.Series("id", sorted(visited), dtype=starts.dtype)
    edge_ids = pl.Series("edge_id", visited_edges, dtype=edge_schema["edge_id"])
    out_edges = _native_properties(edges, edge_schema, "edge_id", edge_ids)
    out_nodes = _native_properties(nodes, node_schema, "id", node_ids)
    return out_nodes, out_edges, perf_counter() - start


def _bounded_query(
    graph: Plottable, seeds: pl.DataFrame, args: argparse.Namespace,
) -> Callable[[], tuple[pl.DataFrame, pl.DataFrame, float]]:
    from graphistry.compute.gfql.lazy.engine.polars import hop_parquet

    rehydrate_fn = hop_parquet._rehydrate
    rehydrate_seconds = 0.

    def timed_rehydrate(
        source: pl.LazyFrame, schema: pl.Schema, key: str, ids: pl.Series,
        limits: ParquetHopLimits, deadline: float,
    ) -> pl.DataFrame:
        nonlocal rehydrate_seconds
        start = perf_counter()
        result = rehydrate_fn(source, schema, key, ids, limits, deadline)
        rehydrate_seconds += perf_counter() - start
        return result

    def run() -> tuple[pl.DataFrame, pl.DataFrame, float]:
        nonlocal rehydrate_seconds
        rehydrate_seconds = 0.
        with patch.object(hop_parquet, "_rehydrate", timed_rehydrate):
            out = graph.hop(seeds, hops=args.hops, engine="polars", parquet_limits=ParquetHopLimits(
                timeout_seconds=args.timeout, max_edges=args.max_edges, max_nodes=args.max_nodes,
            ))
        return out._nodes, out._edges, rehydrate_seconds

    return run


def worker(args: argparse.Namespace) -> None:
    if args.address_space_mib:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cap = args.address_space_mib * (1 << 20)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    route, cache = args.worker
    cold_advice = None
    if cache == "cold-advised":
        for path in (args.nodes, args.edges):
            with path.open("rb") as source:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        cold_advice = True

    def bind() -> tuple[pl.LazyFrame, pl.LazyFrame, Plottable]:
        nodes, edges = pl.scan_parquet(args.nodes), pl.scan_parquet(args.edges)
        return nodes, edges, graphistry.nodes(nodes, "id").edges(edges, "src", "dst", "edge_id")

    (nodes, edges, g), setup = measure(bind)
    starts, seed_setup = measure(lambda: pl.Series("id", args.seeds, dtype=nodes.collect_schema()["id"]))
    seed_frame = starts.to_frame()
    index_build = None
    eager_load = None
    if route == "resident-index":
        graph, eager_load = measure(lambda: graphistry.nodes(nodes.collect(), "id").edges(
            edges.collect(), "src", "dst", "edge_id",
        ))
        g, index_build = measure(lambda: gfql_index_edges(graph, direction="forward", engine="polars"))

    bounded = _bounded_query(g, seed_frame, args) if route == "bounded" else None

    def run() -> tuple[pl.DataFrame, pl.DataFrame, float | None]:
        if route == "native":
            return native(nodes, edges, starts, args.hops)
        if bounded is not None:
            return bounded()
        out = g.hop(seed_frame, hops=args.hops, engine="polars")
        return out._nodes, out._edges, None

    if cache == "warm":
        run()
    (out_nodes, out_edges, rehydrate_seconds), query = measure(run)
    print(json.dumps({
        "kind": "query", "method": route, "cache": cache, "setup": setup, "index_build": index_build,
        "eager_load": eager_load, "seed_setup": seed_setup, "cold_advice": cold_advice,
        "query": query, "rehydrate_seconds": rehydrate_seconds,
        "returned_nodes": out_nodes.height, "returned_edges": out_edges.height, "digest": digest(out_nodes, out_edges),
    }), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Scratch directory for generated files")
    parser.add_argument("--snap-gzip", type=Path, help="Local small SNAP edge list, kept in raw orientation")
    parser.add_argument("--nodes", type=Path, help="Existing node Parquet with id")
    parser.add_argument("--edges", type=Path, help="Existing edge Parquet with src, dst, edge_id")
    parser.add_argument("--row-group-size", type=int, default=16_384)
    parser.add_argument("--synthetic-edges", type=int, default=1_000_000)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument("--max-edges", type=int, default=100_000)
    parser.add_argument("--address-space-mib", type=int, default=0, help="Optional subprocess RLIMIT_AS")
    parser.add_argument("--routes", nargs="+", choices=["eager", "bounded", "native", "resident-index"],
                        default=["eager", "bounded", "native", "resident-index"])
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for option in ("row_group_size", "synthetic_edges", "hops", "repeats"):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.synthetic_edges % 2:
        parser.error("--synthetic-edges must be even")
    if (args.nodes is None) != (args.edges is None) or (args.edges is not None and args.snap_gzip is not None):
        parser.error("provide both --nodes and --edges, or provide --snap-gzip, or use the synthetic graph")
    return args


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare(args)
        return
    if args.worker:
        worker(args)
        return
    args.data.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "benchmarks.gfql.parquet_hop"]
    env = dict(os.environ, POLARS_MAX_THREADS=os.environ.get("POLARS_MAX_THREADS", "4"))
    setup = json.loads(subprocess.check_output([*base, *sys.argv[1:], "--prepare"], env=env, text=True))
    print(json.dumps({
        **setup, "polars": pl.__version__, "graphistry": graphistry.__version__, "threads": int(env["POLARS_MAX_THREADS"]),
        "hops": args.hops, "seeds": args.seeds, "row_group_size": args.row_group_size, "command": sys.argv[1:],
    }), flush=True)
    source = ["--data", str(args.data), "--nodes", setup["nodes"], "--edges", setup["edges"]]
    query = ["--hops", str(args.hops), "--timeout", str(args.timeout), "--max-edges", str(args.max_edges),
             "--max-nodes", str(args.max_nodes), "--address-space-mib", str(args.address_space_mib),
             "--seeds", *(str(seed) for seed in args.seeds)]
    rows = []
    rng = random.Random(42)
    for rep in range(args.repeats):
        for cache in ("cold-advised", "warm"):
            order = args.routes.copy()
            rng.shuffle(order)
            for route in order:
                try:
                    result = subprocess.run([*base, *source, *query, "--worker", route, cache], env=env,
                                            text=True, capture_output=True, check=True, timeout=args.timeout * 2 + 10)
                    row = json.loads(result.stdout)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    row = {"kind": "query", "method": route, "cache": cache, "error": str(error),
                           "stderr": str(error.stderr or "")[-3000:]}
                row["repeat"] = rep
                rows.append(row)
                print(json.dumps(row), flush=True)
    completed = [row for row in rows if "digest" in row]
    if len({row["digest"] for row in completed}) != 1:
        raise SystemExit("Completed results disagree or no query completed")
    if {"bounded", "native"}.issubset(args.routes) and any(
        "error" in row and row["method"] in {"bounded", "native"} for row in rows
    ):
        raise SystemExit("Incomplete comparison: every bounded and native attempt must complete")
    for cache in ("cold-advised", "warm"):
        for route in args.routes:
            matched = [r for r in completed if r["method"] == route and r["cache"] == cache]
            if matched:
                first_use = [sum(r[phase]["seconds"] for phase in
                                 ("setup", "seed_setup", "eager_load", "index_build", "query") if r[phase]) for r in matched]
                print(json.dumps({"kind": "summary", "method": route, "cache": cache, "completed": len(matched),
                                  "query_seconds": median(r["query"]["seconds"] for r in matched),
                                  "setup_plus_query_seconds": median(first_use),
                                  "peak_rss_mib": median(r["query"]["process_high_water_rss_mib"] for r in matched),
                                  "read_bytes": median(r["query"]["read_bytes"] for r in matched)}))


if __name__ == "__main__":
    main()
