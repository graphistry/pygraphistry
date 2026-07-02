#!/usr/bin/env python3
"""
bench_graphframes.py — GFQL vs Apache Spark GraphFrames benchmark harness.

Compares GFQL (graphistry's dataframe-native graph query language) against
Spark GraphFrames on large SNAP graphs across three tasks: attribute/degree
filter, 1- and 2-hop neighborhood traversal, and full-graph PageRank.

DESIGN GOALS
------------
- Single file, no deps beyond graphistry / pyspark / graphframes / pandas / polars.
- Every (system, task) is *guarded*: an error/OOM in one cell records
  {"status": "error", ...} and the run continues. Missing GraphFrames /
  pyspark / GPU is skipped with a clear message, never aborts.
- Timing: `--warmups` warmup iterations (default 2) then `--iters` timed
  iterations (default 5); we report the *median* wall-clock ms. Cold load
  (parquet/txt -> in-memory graph) is timed once per system.
- Results stream to JSONL, one line per (system, task, dataset).

This script is meant to be *reviewed and then run on the benchmark box*
(datasets live at ~/data/snap). It does not download or install anything.

Task definitions and fairness caveats are documented in DESIGN.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each dataset has a parquet file (fast path) and a gzipped SNAP txt fallback.
# SNAP edge-list format: tab-separated "src<TAB>dst", comment lines start '#',
# undirected. Parquet column names are configurable via --src-col / --dst-col
# because SNAP-derived parquet files in the wild use either ('src','dst') or
# the raw SNAP header names; we default to ('src','dst') and auto-detect below.
DATASETS: Dict[str, Dict[str, str]] = {
    "lj": {
        "parquet": "com-lj.ungraph.txt.gz.parquet",
        "txt": "com-lj.ungraph.txt.gz",
        "approx_edges": "35M",
    },
    "orkut": {
        "parquet": "com-orkut.ungraph.txt.gz.parquet",
        "txt": "com-orkut.ungraph.txt.gz",
        "approx_edges": "117M",
    },
    "friendster": {
        # No prebuilt parquet shipped for friendster; txt (~1.8B edges) only.
        "parquet": "com-friendster.ungraph.txt.gz.parquet",
        "txt": "com-friendster.ungraph.txt.gz",
        "approx_edges": "1.8B",
    },
}

ALL_SYSTEMS = ["gfql-polars", "gfql-polars-gpu", "graphframes"]
ALL_TASKS = ["filter", "hop1", "hop2", "pagerank"]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class Result:
    system: str
    task: str
    dataset: str
    n_edges: Optional[int] = None
    n_nodes: Optional[int] = None
    median_ms: Optional[float] = None
    cold_load_ms: Optional[float] = None
    iters: int = 0
    warmups: int = 0
    result_size: Optional[int] = None  # rows materialized (for parity checks)
    status: str = "ok"
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _time_once(fn: Callable[[], Any]) -> Tuple[float, Any]:
    """Run fn(), returning (elapsed_ms, return_value)."""
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out


def timed_median(
    fn: Callable[[], Any],
    warmups: int,
    iters: int,
) -> Tuple[float, Optional[int]]:
    """
    Warm up `warmups` times (untimed), then time `iters` runs.
    Returns (median_ms, last_result_size).

    `fn` must return something we can size for a parity check: we accept an int
    directly, or fall back to len(). None -> result_size None.
    """
    for _ in range(warmups):
        fn()
        gc.collect()
    samples: List[float] = []
    last_size: Optional[int] = None
    for _ in range(iters):
        ms, out = _time_once(fn)
        samples.append(ms)
        last_size = _sizeof(out)
        gc.collect()
    return statistics.median(samples), last_size


def _sizeof(out: Any) -> Optional[int]:
    if out is None:
        return None
    if isinstance(out, int):
        return out
    try:
        return len(out)
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# GFQL system adapter
# ---------------------------------------------------------------------------
class GFQLSystem:
    """
    GFQL adapter. `engine` is 'polars' or 'polars-gpu' (see graphistry.Engine).

    Cold load: read parquet/txt -> pandas/polars edge frame -> build a
    graphistry Plottable with a precomputed node table carrying `degree`
    (so the filter task is a pure node-attribute WHERE, symmetric with the
    GraphFrames `gf.degrees` filter). The degree precompute is part of cold
    load for both systems.
    """

    def __init__(self, engine: str, src_col: str, dst_col: str,
                 filter_threshold: Optional[int] = None):
        self.engine = engine  # 'polars' | 'polars-gpu'
        self.src_col = src_col
        self.dst_col = dst_col
        self._threshold_override = filter_threshold
        self.g = None
        self.n_edges: Optional[int] = None
        self.n_nodes: Optional[int] = None
        self._seeds: List[Any] = []
        self._filter_threshold: Optional[int] = None

    # -- cold load ----------------------------------------------------------
    def load(self, edges_path: str, is_parquet: bool) -> None:
        import pandas as pd
        import graphistry

        if is_parquet:
            edf = pd.read_parquet(edges_path)
            edf = self._normalize_cols(edf)
        else:
            # SNAP gz txt: tab-separated, comments start with '#'.
            edf = pd.read_csv(
                edges_path,
                sep="\t",
                comment="#",
                header=None,
                names=[self.src_col, self.dst_col],
                compression="gzip",
                dtype="int64",
            )

        # Node table with degree (undirected: count endpoints on both sides).
        deg = (
            pd.concat([edf[self.src_col], edf[self.dst_col]])
            .value_counts()
            .rename_axis("id")
            .reset_index(name="degree")
        )
        self.n_edges = int(len(edf))
        self.n_nodes = int(len(deg))

        g = graphistry.edges(edf, self.src_col, self.dst_col)
        g = g.nodes(deg, "id")
        self.g = g

        # Filter threshold: shared override (for cross-system parity) else
        # ~top-decile degree. Seeds: highest-degree nodes.
        self._filter_threshold = (
            self._threshold_override
            if self._threshold_override is not None
            else int(deg["degree"].quantile(0.90))
        )
        self._seeds = deg.sort_values("degree", ascending=False)["id"].head(50).tolist()

    def _normalize_cols(self, edf):
        """Map the parquet's actual columns onto (src_col, dst_col)."""
        cols = list(edf.columns)
        if self.src_col in cols and self.dst_col in cols:
            return edf
        # Fall back to positional: first two columns are src, dst.
        if len(cols) >= 2:
            return edf.rename(columns={cols[0]: self.src_col, cols[1]: self.dst_col})
        raise ValueError(f"parquet has too few columns: {cols}")

    # -- task fns (each returns a size for parity) --------------------------
    def filter_fn(self) -> Callable[[], int]:
        from graphistry.compute.ast import n
        from graphistry.compute.predicates.numeric import ge

        g, engine, thr = self.g, self.engine, self._filter_threshold

        def run() -> int:
            out = g.gfql([n(filter_dict={"degree": ge(thr)})], engine=engine)
            return int(len(out._nodes))  # materialize

        return run

    def hop_fn(self, hops: int) -> Callable[[], int]:
        from graphistry.compute.ast import n, e_undirected
        from graphistry.compute.predicates.is_in import IsIn

        g, engine, seeds = self.g, self.engine, self._seeds

        def run() -> int:
            out = g.gfql(
                [
                    n({"id": IsIn(options=seeds)}),
                    e_undirected(to_fixed_point=False, hops=hops),
                    n(),
                ],
                engine=engine,
            )
            # Return the k-ball NODE count only, so it is directly comparable to
            # the GraphFrames `visited.count()` (union of seeds + up-to-k-hop
            # neighbors). Edges are still materialized by the traversal; we just
            # do not fold them into the parity size.
            return int(len(out._nodes))

        return run

    def pagerank_fn(self) -> Callable[[], int]:
        """
        PageRank API (found in repo):
          - CPU (polars / pandas): g.compute_igraph('pagerank')
              graphistry/plugins/igraph.py:339
          - GPU: g.compute_cugraph('pagerank')
              graphistry/plugins/cugraph.py:423
        polars has no native PageRank; the polars engine routes PageRank
        through igraph (pandas conversion under the hood).
        """
        g = self.g
        use_gpu = self.engine == "polars-gpu"

        def run() -> int:
            if use_gpu:
                out = g.compute_cugraph("pagerank")
            else:
                out = g.compute_igraph("pagerank")
            return int(len(out._nodes))  # materialize the pagerank column

        return run


# ---------------------------------------------------------------------------
# GraphFrames system adapter
# ---------------------------------------------------------------------------
class GraphFramesSystem:
    """
    Spark GraphFrames adapter. SparkSession is local[*] (multicore single node)
    with configurable driver memory. Imports are guarded so a box without
    pyspark/graphframes skips gracefully.
    """

    def __init__(self, src_col: str, dst_col: str, spark_mem: str, pagerank_iters: int,
                 spark_jars: Optional[str] = None, filter_threshold: Optional[int] = None):
        self.src_col = src_col
        self.dst_col = dst_col
        self.spark_mem = spark_mem
        self.pagerank_iters = pagerank_iters
        self._threshold_override = filter_threshold
        # Path(s) to the graphframes assembly jar. Without it the JVM has no
        # GraphFrame classes and the first Spark action fails. Falls back to the
        # GRAPHFRAMES_JAR env var so the documented run command can stay short.
        self.spark_jars = spark_jars or os.environ.get("GRAPHFRAMES_JAR")
        self.spark = None
        self.gf = None
        self.n_edges: Optional[int] = None
        self.n_nodes: Optional[int] = None
        self._seeds: List[Any] = []
        self._filter_threshold: Optional[int] = None

    # -- cold load ----------------------------------------------------------
    def load(self, edges_path: str, is_parquet: bool) -> None:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from graphframes import GraphFrame  # guarded by caller

        builder = (
            SparkSession.builder.appName("gfql-vs-graphframes")
            .master("local[*]")
            .config("spark.driver.memory", self.spark_mem)
            .config("spark.sql.shuffle.partitions", str(os.cpu_count() or 8))
            # GraphFrames connected-components / pageRank checkpointing:
            .config("spark.sql.adaptive.enabled", "true")
        )
        if self.spark_jars:
            builder = builder.config("spark.jars", self.spark_jars)
        self.spark = builder.getOrCreate()
        # Checkpoint dir is required by some GraphFrames algorithms.
        self.spark.sparkContext.setCheckpointDir(
            os.path.join(os.path.expanduser("~"), ".spark-checkpoints")
        )

        if is_parquet:
            edf = self.spark.read.parquet(edges_path)
            cols = edf.columns
            if not (self.src_col in cols and self.dst_col in cols):
                edf = edf.withColumnRenamed(cols[0], self.src_col).withColumnRenamed(
                    cols[1], self.dst_col
                )
        else:
            # SNAP gz txt: tab-separated with '#' comments.
            edf = (
                self.spark.read.option("sep", "\t")
                .option("comment", "#")
                .csv(edges_path)
            )
            edf = (
                edf.withColumnRenamed("_c0", self.src_col)
                .withColumnRenamed("_c1", self.dst_col)
                .select(
                    F.col(self.src_col).cast("long"),
                    F.col(self.dst_col).cast("long"),
                )
            )

        # GraphFrames wants columns named 'src','dst' on edges and 'id' on nodes.
        edges = edf.select(
            F.col(self.src_col).alias("src"), F.col(self.dst_col).alias("dst")
        ).cache()
        vertices = (
            edges.select(F.col("src").alias("id"))
            .union(edges.select(F.col("dst").alias("id")))
            .distinct()
            .cache()
        )
        self.gf = GraphFrame(vertices, edges)
        self.n_edges = edges.count()   # force materialization -> honest load
        self.n_nodes = vertices.count()

        # Degree threshold + seeds computed from gf.degrees (materialized once).
        degrees = self.gf.degrees.cache()
        # ~top-decile degree via approxQuantile (exact quantile is expensive).
        # A shared override (--filter-threshold) is preferred so the filter task
        # is bit-identical across systems for a clean parity check.
        if self._threshold_override is not None:
            self._filter_threshold = self._threshold_override
        else:
            thr = degrees.approxQuantile("degree", [0.90], 0.01)
            self._filter_threshold = int(thr[0]) if thr else 1
        top = (
            degrees.orderBy(F.col("degree").desc()).limit(50).select("id").collect()
        )
        self._seeds = [r["id"] for r in top]
        self._degrees = degrees

    # -- task fns -----------------------------------------------------------
    def filter_fn(self) -> Callable[[], int]:
        thr = self._filter_threshold
        degrees = self._degrees

        def run() -> int:
            # WHERE on the degree column; .count() forces materialization.
            return degrees.filter(degrees["degree"] >= thr).count()

        return run

    def hop_fn(self, hops: int) -> Callable[[], int]:
        """
        k-hop neighborhood from seeds. GraphFrames has no direct k-hop
        neighborhood op (bfs finds shortest paths between predicates, motif
        `find` matches fixed patterns), so we expand via iterated edge joins
        against the (undirected) edge frame — still pure Spark, honest timing.
        """
        from pyspark.sql import functions as F

        spark = self.spark
        edges = self.gf.edges
        seeds = self._seeds

        # Undirected adjacency: both directions.
        adj = edges.select("src", "dst").union(
            edges.select(F.col("dst").alias("src"), F.col("src").alias("dst"))
        ).cache()

        def run() -> int:
            frontier = spark.createDataFrame([(s,) for s in seeds], ["id"]).distinct()
            visited = frontier
            for _ in range(hops):
                nxt = (
                    frontier.join(adj, frontier["id"] == adj["src"])
                    .select(F.col("dst").alias("id"))
                    .distinct()
                )
                visited = visited.union(nxt).distinct()
                frontier = nxt
            return visited.count()  # materialize the neighborhood

        return run

    def pagerank_fn(self) -> Callable[[], int]:
        gf = self.gf
        iters = self.pagerank_iters

        def run() -> int:
            # resetProbability = 1 - damping (0.85) = 0.15
            res = gf.pageRank(resetProbability=0.15, maxIter=iters)
            return res.vertices.count()  # materialize

        return run

    def close(self) -> None:
        if self.spark is not None:
            try:
                self.spark.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def resolve_edges_path(data_dir: str, dataset: str) -> Tuple[str, bool]:
    """Return (path, is_parquet). Prefer parquet, fall back to gz txt."""
    d = DATASETS[dataset]
    pq = os.path.join(data_dir, d["parquet"])
    txt = os.path.join(data_dir, d["txt"])
    if os.path.exists(pq):
        return pq, True
    if os.path.exists(txt):
        return txt, False
    # Return the parquet path anyway; loader will raise a clear FileNotFound.
    return pq, True


def run_system(
    system: str,
    tasks: List[str],
    dataset: str,
    edges_path: str,
    is_parquet: bool,
    args: argparse.Namespace,
    out_fh,
) -> None:
    """Load one system, run each task guarded, stream JSONL results."""
    # --- construct + guarded import/availability ---------------------------
    adapter = None
    cold_ms: Optional[float] = None
    n_edges = n_nodes = None
    load_error: Optional[str] = None

    try:
        if system in ("gfql-polars", "gfql-polars-gpu"):
            engine = "polars" if system == "gfql-polars" else "polars-gpu"
            import graphistry  # noqa: F401  (fail fast if missing)
            adapter = GFQLSystem(engine, args.src_col, args.dst_col,
                                 filter_threshold=args.filter_threshold)
        elif system == "graphframes":
            # Guarded imports: absence -> skip cleanly.
            import pyspark  # noqa: F401
            from graphframes import GraphFrame  # noqa: F401
            adapter = GraphFramesSystem(
                args.src_col, args.dst_col, args.spark_mem, args.pagerank_iters,
                spark_jars=args.spark_jars, filter_threshold=args.filter_threshold,
            )
        else:
            raise ValueError(f"unknown system {system}")

        cold_ms, _ = _time_once(lambda: adapter.load(edges_path, is_parquet))
        n_edges = adapter.n_edges
        n_nodes = adapter.n_nodes
    except ImportError as e:
        load_error = f"skipped (import failed): {e}"
        print(f"[SKIP] {system}: {load_error}", file=sys.stderr)
    except Exception as e:
        load_error = f"cold-load failed: {e}\n{traceback.format_exc()}"
        print(f"[ERROR] {system} cold load: {e}", file=sys.stderr)

    # If load failed, emit one error row per requested task and return.
    if adapter is None or load_error is not None:
        for task in tasks:
            r = Result(
                system=system, task=task, dataset=dataset,
                cold_load_ms=cold_ms, iters=0, warmups=args.warmups,
                status="error", error=load_error or "unavailable",
            )
            out_fh.write(r.to_jsonl() + "\n")
            out_fh.flush()
        return

    # --- run each task, guarded -------------------------------------------
    for task in tasks:
        r = Result(
            system=system, task=task, dataset=dataset,
            n_edges=n_edges, n_nodes=n_nodes, cold_load_ms=cold_ms,
            iters=args.iters, warmups=args.warmups,
        )
        try:
            fn = build_task_fn(adapter, task)
            median_ms, size = timed_median(fn, args.warmups, args.iters)
            r.median_ms = median_ms
            r.result_size = size
            r.status = "ok"
            print(
                f"[OK] {system}/{task}/{dataset}: "
                f"median={median_ms:.1f}ms size={size}",
                file=sys.stderr,
            )
        except Exception as e:
            r.status = "error"
            r.error = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] {system}/{task}: {e}", file=sys.stderr)
        out_fh.write(r.to_jsonl() + "\n")
        out_fh.flush()

    # --- teardown ----------------------------------------------------------
    if isinstance(adapter, GraphFramesSystem):
        adapter.close()


def build_task_fn(adapter, task: str) -> Callable[[], int]:
    if task == "filter":
        return adapter.filter_fn()
    if task == "hop1":
        return adapter.hop_fn(1)
    if task == "hop2":
        return adapter.hop_fn(2)
    if task == "pagerank":
        return adapter.pagerank_fn()
    raise ValueError(f"unknown task {task}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GFQL vs Spark GraphFrames benchmark harness (SNAP graphs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--data-dir", default="~/data/snap")
    p.add_argument(
        "--systems", default=",".join(ALL_SYSTEMS),
        help="comma list from: " + ", ".join(ALL_SYSTEMS),
    )
    p.add_argument(
        "--tasks", default=",".join(ALL_TASKS),
        help="comma list from: " + ", ".join(ALL_TASKS),
    )
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--spark-mem", default="200g", help="Spark driver memory")
    p.add_argument(
        "--pagerank-iters", type=int, default=20,
        help="maxIter for GraphFrames pageRank (fixed-iteration for parity)",
    )
    p.add_argument(
        "--filter-threshold", type=int, default=None,
        help="shared degree>=T for the filter task (identical across systems "
             "for a clean parity check); default: each computes its own p90",
    )
    p.add_argument(
        "--spark-jars", default=None,
        help="path to graphframes assembly jar (else GRAPHFRAMES_JAR env)",
    )
    p.add_argument("--src-col", default="src", help="edge source column name")
    p.add_argument("--dst-col", default="dst", help="edge destination column name")
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the plan and exit without executing",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    data_dir = os.path.expanduser(args.data_dir)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    bad_sys = [s for s in systems if s not in ALL_SYSTEMS]
    bad_task = [t for t in tasks if t not in ALL_TASKS]
    if bad_sys:
        print(f"unknown systems: {bad_sys}", file=sys.stderr)
        return 2
    if bad_task:
        print(f"unknown tasks: {bad_task}", file=sys.stderr)
        return 2

    edges_path, is_parquet = resolve_edges_path(data_dir, args.dataset)

    # -- plan summary -------------------------------------------------------
    print("=" * 70, file=sys.stderr)
    print("GFQL vs GraphFrames benchmark plan", file=sys.stderr)
    print(f"  dataset     : {args.dataset} (~{DATASETS[args.dataset]['approx_edges']} edges)", file=sys.stderr)
    print(f"  edges_path  : {edges_path} (parquet={is_parquet})", file=sys.stderr)
    print(f"  cols        : src={args.src_col} dst={args.dst_col}", file=sys.stderr)
    print(f"  systems     : {systems}", file=sys.stderr)
    print(f"  tasks       : {tasks}", file=sys.stderr)
    print(f"  warmups     : {args.warmups}   iters(median of): {args.iters}", file=sys.stderr)
    print(f"  spark_mem   : {args.spark_mem}  pagerank_iters: {args.pagerank_iters}", file=sys.stderr)
    print(f"  out         : {args.out}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    if args.dry_run:
        print("[DRY RUN] no execution; exiting.", file=sys.stderr)
        return 0

    if not os.path.exists(edges_path):
        print(
            f"[WARN] edges path not found: {edges_path} — systems will record errors.",
            file=sys.stderr,
        )

    with open(args.out, "w") as out_fh:
        for system in systems:
            print(f"\n### SYSTEM: {system} ###", file=sys.stderr)
            run_system(system, tasks, args.dataset, edges_path, is_parquet, args, out_fh)

    print(f"\nDone. Results -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
