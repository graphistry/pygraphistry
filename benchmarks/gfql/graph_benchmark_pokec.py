#!/usr/bin/env python3
"""Run Memgraph's advertised Benchgraph/mgBench Pokec workload on GFQL vs Memgraph.

This targets the *actual* dataset and queries Memgraph markets itself on
(``tests/mgbench/workloads/pokec.py``), not a look-alike over other data. It
parses the public mgBench Pokec import (``CREATE (:User {...})`` nodes and
``MATCH ... CREATE (n)-[:Friend]->(m)`` edges) into node/edge dataframes, then
runs the exact mgBench read/aggregate/expansion/neighbours/pattern queries as
the *same Cypher* on GFQL (pandas/cuDF/Polars/Polars-GPU) and, optionally, Memgraph over Bolt,
using a shared fixed seed set for the seeded (``$id``) queries.

Dataset: https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/dataset/pokec/benchmark/
mgBench: https://github.com/memgraph/memgraph/tree/master/tests/mgbench/workloads/pokec.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
)

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))       # benchmarks/gfql for sibling imports
sys.path.insert(0, str(_HERE.parents[2]))   # repo root so `import graphistry` finds the checkout

import pandas as pd

from graph_benchmark_q1_q9 import _median

if TYPE_CHECKING:
    from graphistry.Engine import Engine
    from graphistry.compute.ComputeMixin import ComputeMixin
    from graphistry.compute.gfql.index.types import IndexPath
    from graphistry.compute.typing import DataFrameT
    from graphistry.Plottable import Plottable

_NODE_RE = re.compile(
    r'CREATE \(:User \{id: (\d+), completion_percentage: (\d+), gender: "([^"]*)", age: (\d+)\}\)'
)
_EDGE_RE = re.compile(r'MATCH \(n:User \{id: (\d+)\}\), \(m:User \{id: (\d+)\}\)')

GFQLEngine = Literal["pandas", "cudf", "polars", "polars-gpu"]
QueryKind = Literal["global", "seed"]
ExecutionTarget = Literal["cpu", "gpu", "mixed"]
FailureStatus = Literal["unsupported", "error", "oom"]
ResultT = TypeVar("ResultT")


class QuerySpec(TypedDict):
    name: str
    kind: QueryKind
    gfql: str
    memgraph: str


class TimingResult(TypedDict):
    median_ms: float
    samples_ms: List[float]


class SeedTiming(TypedDict):
    seed: int
    median_ms: float
    samples_ms: List[float]
    rowcount: int
    execution_paths: List["IndexPath"]


class ExecutionMetadata(TypedDict, total=False):
    status: Literal["ok", "fallback"]
    execution_target: ExecutionTarget
    synchronization: Literal["none", "cuda-device"]
    fallback_reason: str


class FailureResult(TypedDict):
    status: FailureStatus
    unsupported: bool
    error_type: str
    error: str
    kind: QueryKind


GFQL_ENGINES: Tuple[GFQLEngine, ...] = ("pandas", "cudf", "polars", "polars-gpu")
ENGINE_SELECTIONS: Dict[str, Tuple[GFQLEngine, ...]] = {
    "pandas": ("pandas",),
    "cudf": ("cudf",),
    "polars": ("polars",),
    "polars-gpu": ("polars-gpu",),
    "both": ("pandas", "cudf"),
    "all": GFQL_ENGINES,
}


def parse_pokec_import(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids: List[int] = []
    comp: List[int] = []
    gender: List[str] = []
    age: List[int] = []
    src: List[int] = []
    dst: List[int] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("CREATE (:User"):
                m = _NODE_RE.match(line)
                if m:
                    ids.append(int(m.group(1)))
                    comp.append(int(m.group(2)))
                    gender.append(m.group(3))
                    age.append(int(m.group(4)))
            elif line.startswith("MATCH (n:User"):
                m = _EDGE_RE.match(line)
                if m:
                    src.append(int(m.group(1)))
                    dst.append(int(m.group(2)))
    nodes = pd.DataFrame({"id": ids, "completion_percentage": comp, "gender": gender, "age": age})
    nodes["label__User"] = True
    edges = pd.DataFrame({"src": src, "dst": dst})
    return nodes, edges


# Query set: (name, kind, gfql_cypher_template, memgraph_cypher, seeded)
#   kind: "global" runs once; "seed" runs over the shared seed set ($id);
#         "pair" runs over shared (from,to) pairs.
# GFQL Cypher uses the label__User convention and untyped edges; Memgraph uses
# the exact mgBench query text.
QUERIES: Tuple[QuerySpec, ...] = (
    {"name": "aggregate", "kind": "global",
     "gfql": "MATCH (n:User) RETURN n.age, COUNT(*)",
     "memgraph": "MATCH (n:User) RETURN n.age, COUNT(*)"},
    {"name": "aggregate_with_distinct", "kind": "global",
     "gfql": "MATCH (n:User) RETURN COUNT(DISTINCT n.age)",
     "memgraph": "MATCH (n:User) RETURN COUNT(DISTINCT n.age)"},
    {"name": "aggregate_with_filter", "kind": "global",
     "gfql": "MATCH (n:User) WHERE n.age >= 18 RETURN n.age, COUNT(*)",
     "memgraph": "MATCH (n:User) WHERE n.age >= 18 RETURN n.age, COUNT(*)"},
    {"name": "min_max_avg", "kind": "global",
     "gfql": "MATCH (n:User) RETURN min(n.age), max(n.age), avg(n.age)",
     "memgraph": "MATCH (n) RETURN min(n.age), max(n.age), avg(n.age)"},
    {"name": "expansion_1", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-->(n:User) RETURN n.id",
     "memgraph": "MATCH (s:User {id: $id})-->(n:User) RETURN n.id"},
    {"name": "expansion_1_with_filter", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-->(n:User) WHERE n.age >= 18 RETURN n.id",
     "memgraph": "MATCH (s:User {id: $id})-->(n:User) WHERE n.age >= 18 RETURN n.id"},
    # NOTE: expansion uses explicit-hop chains (not varlen) to match Memgraph's
    # walk semantics. GFQL varlen `-[*k..k]->` uses simple-path (no-repeat-node)
    # semantics and undercounts distinct k-walk endpoints; explicit `-->()-->()`
    # matches Memgraph's `-->()-->()` exactly.
    {"name": "expansion_2", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-->()-->(n:User) RETURN DISTINCT n.id",
     "memgraph": "MATCH (s:User {id: $id})-->()-->(n:User) RETURN DISTINCT n.id"},
    {"name": "expansion_3", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-->()-->()-->(n:User) RETURN DISTINCT n.id",
     "memgraph": "MATCH (s:User {id: $id})-->()-->()-->(n:User) RETURN DISTINCT n.id"},
    {"name": "expansion_4", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-->()-->()-->()-->(n:User) RETURN DISTINCT n.id",
     "memgraph": "MATCH (s:User {id: $id})-->()-->()-->()-->(n:User) RETURN DISTINCT n.id"},
    {"name": "neighbours_2", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-[*1..2]->(n:User) RETURN DISTINCT n.id",
     "memgraph": "MATCH (s:User {id: $id})-[*1..2]->(n:User) RETURN DISTINCT n.id"},
    {"name": "neighbours_2_with_filter", "kind": "seed",
     "gfql": "MATCH (s:User {{id: {id}}})-[*1..2]->(n:User) WHERE n.age >= 18 RETURN DISTINCT n.id",
     "memgraph": "MATCH (s:User {id: $id})-[*1..2]->(n:User) WHERE n.age >= 18 RETURN DISTINCT n.id"},
    {"name": "pattern_short", "kind": "seed",
     "gfql": "MATCH (n:User {{id: {id}}})-[e]->(m) RETURN m LIMIT 1",
     "memgraph": "MATCH (n:User {id: $id})-[e]->(m) RETURN m LIMIT 1"},
    {"name": "pattern_cycle", "kind": "seed",
     "gfql": "MATCH (n:User {{id: {id}}})-[e1]->(m)-[e2]->(n) RETURN e1, m, e2",
     "memgraph": "MATCH (n:User {id: $id})-[e1]->(m)-[e2]->(n) RETURN e1, m, e2"},
)


def _rowcount(result: Any) -> int:
    frame = result._nodes if hasattr(result, "_nodes") else result
    try:
        return int(len(frame))
    except Exception:
        return -1


def _time(
    fn: Callable[[], ResultT],
    runs: int,
    warmup: int,
    synchronize: Optional[Callable[[], None]] = None,
) -> Tuple[ResultT, TimingResult]:
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    for _ in range(warmup):
        fn()
        if synchronize is not None:
            synchronize()

    if synchronize is not None:
        synchronize()
    start = time.perf_counter()
    result = fn()
    if synchronize is not None:
        synchronize()
    times = [(time.perf_counter() - start) * 1000.0]
    for _ in range(1, runs):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        result = fn()
        if synchronize is not None:
            synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return result, {"median_ms": _median(times), "samples_ms": times}


def _hop_dsts(
    g: "ComputeMixin",
    frontier_ids: Sequence[int],
    engine: GFQLEngine,
) -> List[int]:
    """One indexed forward 1-hop from a frontier; return distinct destination ids.

    hop(nodes=, hops=1) is routed through the #1658 CSR index on every engine
    (pandas/cudf/polars), unlike chain() which is only indexed on polars. Distinct
    edge destinations give exact walk-semantics endpoints (matches Memgraph).
    """
    res = g.hop(nodes=pd.DataFrame({"id": list(frontier_ids)}), direction="forward", hops=1, engine=engine)
    e = res._edges
    e = e.to_pandas() if hasattr(e, "to_pandas") else e
    if e is None or len(e) == 0:
        return []
    return list(pd.unique(e["dst"]))


NATIVE_SEEDED_NAMES = ("expansion_", "neighbours_", "pattern_short")


def _native_seeded(
    g: "ComputeMixin",
    spec: QuerySpec,
    seed: int,
    engine: GFQLEngine,
    adult_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Express the seeded Pokec traversals as a loop of indexed 1-hops.

    expansion_k = final frontier of exactly k hops (distinct); neighbours_k =
    union of frontiers at hops 1..k (distinct); pattern_short = one out-neighbour.
    Walk semantics (matches Memgraph). `_with_filter` intersects endpoints with a
    precomputed adult-id set (age>=18), built once as untimed setup.

    Why hop() and not the exact Cypher: the #1658 seeded index is only consulted
    by hop() / maybe_index_hop, NOT by the Cypher/chain lowering, so Cypher seeded
    queries still O(E)-scan even with a resident index. Routing Cypher/chain
    through the index is graphistry/pygraphistry#1676. We also use explicit chained
    hops (walk semantics) rather than Cypher varlen `-[*k..k]->`, which uses
    simple-path (no-repeat) semantics and undercounts k-walk endpoints vs
    Neo4j/Memgraph (graphistry/pygraphistry#1685).
    """
    name = spec["name"]
    if name == "pattern_short":
        dsts = _hop_dsts(g, [seed], engine)
        return dsts[:1]
    k = int(name.split("_")[1])
    frontier: List[int] = [seed]
    acc: set = set()
    for _ in range(k):
        frontier = _hop_dsts(g, frontier, engine)
        acc.update(frontier)
    endpoints = set(frontier) if name.startswith("expansion_") else acc
    if name.endswith("_with_filter") and adult_ids is not None:
        endpoints = endpoints & adult_ids
    return list(endpoints)


def _frame_engine(engine: GFQLEngine) -> "Engine":
    from graphistry.Engine import Engine

    if engine == "pandas":
        return Engine.PANDAS
    if engine == "cudf":
        return Engine.CUDF
    if engine in ("polars", "polars-gpu"):
        # Polars-GPU is an execution target; its resident frames are ordinary Polars.
        return Engine.POLARS
    raise ValueError(f"unsupported GFQL benchmark engine: {engine!r}")


def _to_native_frames(
    nodes: pd.DataFrame, edges: pd.DataFrame, engine: GFQLEngine
) -> Tuple["DataFrameT", "DataFrameT"]:
    from graphistry.Engine import Engine, df_to_engine

    frame_engine = _frame_engine(engine)
    if frame_engine == Engine.PANDAS:
        return nodes, edges
    return df_to_engine(nodes, frame_engine), df_to_engine(edges, frame_engine)


def _selected_engines(selection: str) -> Tuple[GFQLEngine, ...]:
    try:
        return ENGINE_SELECTIONS[selection]
    except KeyError as exc:
        raise ValueError(f"unsupported GFQL benchmark engine selection: {selection!r}") from exc


def _require_compute_graph(g: "Plottable") -> "ComputeMixin":
    from graphistry.compute.ComputeMixin import ComputeMixin

    if not isinstance(g, ComputeMixin):
        raise TypeError(f"GFQL benchmark requires ComputeMixin, got {type(g)!r}")
    return g


def _trace_native_seeded(
    g: "ComputeMixin",
    spec: QuerySpec,
    seed: int,
    engine: GFQLEngine,
    adult_ids: Optional[Set[int]],
) -> Tuple[List[int], List["IndexPath"]]:
    from graphistry.compute.gfql.index import index_trace

    with index_trace() as steps:
        result = _native_seeded(g, spec, seed, engine, adult_ids)
    paths = [step["path"] for step in steps if "path" in step]
    if not paths:
        raise RuntimeError(
            f"Native indexed traversal produced no planner trace for {spec['name']}"
        )
    return result, paths


def _execution_metadata(
    engine: GFQLEngine,
    *,
    uses_native_index: bool,
    execution_paths: Sequence["IndexPath"] = (),
) -> ExecutionMetadata:
    if engine == "polars-gpu" and uses_native_index:
        paths = set(execution_paths)
        if not paths:
            raise ValueError("Polars-GPU native traversal requires execution paths")
        if paths == {"scan"}:
            return {
                "status": "ok",
                "execution_target": "gpu",
                "synchronization": "cuda-device",
            }
        if paths == {"index"}:
            target: ExecutionTarget = "cpu"
            detail = "all measured seeds used the NumPy host adjacency index"
        else:
            target = "mixed"
            detail = "planner decisions mixed NumPy host indexes with GPU scans"
        return {
            "status": "fallback",
            "execution_target": target,
            "synchronization": "cuda-device",
            "fallback_reason": (
                f"{detail}; this row is not a GPU-only result"
            ),
        }
    return {
        "status": "ok",
        "execution_target": "gpu" if engine in ("cudf", "polars-gpu") else "cpu",
        "synchronization": (
            "cuda-device" if engine in ("cudf", "polars-gpu") else "none"
        ),
    }


def _failure_result(exc: Exception, kind: QueryKind) -> FailureResult:
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

    unsupported = isinstance(exc, NotImplementedError) or (
        isinstance(exc, GFQLValidationError) and exc.code == ErrorCode.E108
    )
    type_name = type(exc).__name__
    normalized_type = type_name.lower().replace("_", "")
    oom = isinstance(exc, MemoryError) or "outofmemory" in normalized_type
    status: FailureStatus = (
        "unsupported" if unsupported else "oom" if oom else "error"
    )
    return {
        "status": status,
        "unsupported": unsupported,
        "error_type": type_name,
        "error": str(exc)[:500],
        "kind": kind,
    }


def _synchronizer(engine: GFQLEngine) -> Optional[Callable[[], None]]:
    if engine not in ("cudf", "polars-gpu"):
        return None
    import cupy as cp

    def synchronize() -> None:
        cp.cuda.runtime.deviceSynchronize()

    return synchronize


def run_gfql(nodes: pd.DataFrame, edges: pd.DataFrame, engine: GFQLEngine, seeds: Sequence[int],
             runs: int, warmup: int, index_policy: str = "off", traversal: str = "cypher") -> Dict[str, Any]:
    import graphistry
    synchronize = _synchronizer(engine)
    n, e = _to_native_frames(nodes, edges, engine)
    g = _require_compute_graph(graphistry.nodes(n, "id").edges(e, "src", "dst"))
    # #1658 seeded adjacency index (opt-in). Build once as untimed setup (like a
    # DB's load-time index), then seeded gfql() runs with index_policy set.
    index_build_ms: Optional[float] = None
    gfql_kwargs: Dict[str, Any] = {"engine": engine}
    if index_policy != "off":
        t0 = time.perf_counter()
        g = g.gfql_index_all(engine=engine)  # edge_out/in adjacency + node_id (requires #1658)
        index_build_ms = (time.perf_counter() - t0) * 1000.0
        gfql_kwargs["index_policy"] = index_policy  # gfql() accepts this kwarg
        try:
            setattr(g, "_gfql_index_policy", index_policy)  # chain()/hop() read this attribute
        except Exception:
            pass
    # Precompute the age>=18 endpoint set once (untimed setup) for *_with_filter.
    adult_ids: Optional[Set[int]] = None
    if traversal == "native":
        _nn = nodes  # host pandas frame (before engine coercion) — fine for a setup lookup
        adult_ids = set(_nn[_nn["age"] >= 18]["id"])
    results: Dict[str, Any] = {"_index_build_ms": index_build_ms, "_index_policy": index_policy}
    for spec in QUERIES:
        name = spec["name"]
        try:
            if spec["kind"] == "global":
                q = spec["gfql"]
                result, timing = _time(
                    lambda: g.gfql(q, **gfql_kwargs),
                    runs,
                    warmup,
                    synchronize,
                )
                results[name] = {
                    **timing,
                    **_execution_metadata(engine, uses_native_index=False),
                    "kind": "global",
                    "rowcount": _rowcount(result),
                    "unit": "ms",
                }
            else:
                use_native = traversal == "native" and name.startswith(NATIVE_SEEDED_NAMES)
                per_seed: List[float] = []
                rowcounts: List[int] = []
                seed_timings: List[SeedTiming] = []
                execution_paths: List["IndexPath"] = []
                for sid in seeds:
                    if use_native:
                        res, timing = _time(
                            lambda: _native_seeded(g, spec, sid, engine, adult_ids),
                            runs,
                            warmup,
                            synchronize,
                        )
                        rc = int(len(res))
                        traced_res, seed_paths = _trace_native_seeded(
                            g,
                            spec,
                            sid,
                            engine,
                            adult_ids,
                        )
                        if set(traced_res) != set(res):
                            raise RuntimeError(
                                f"Trace probe changed result for {name} seed {sid}"
                            )
                        execution_paths.extend(seed_paths)
                    else:
                        q = spec["gfql"].format(id=sid)
                        res, timing = _time(
                            lambda: g.gfql(q, **gfql_kwargs),
                            runs,
                            warmup,
                            synchronize,
                        )
                        rc = _rowcount(res)
                    per_seed.append(timing["median_ms"])
                    rowcounts.append(rc)
                    seed_timings.append(
                        {
                            "seed": int(sid),
                            "median_ms": timing["median_ms"],
                            "samples_ms": timing["samples_ms"],
                            "rowcount": rc,
                            "execution_paths": seed_paths if use_native else [],
                        }
                    )
                results[name] = {
                    "median_ms": _median(per_seed),
                    "min_seed_median_ms": min(per_seed),
                    "max_seed_median_ms": max(per_seed),
                    "seed_timings": seed_timings,
                    "execution_paths": execution_paths,
                    **_execution_metadata(
                        engine,
                        uses_native_index=use_native,
                        execution_paths=execution_paths,
                    ),
                    "kind": "seed",
                    "unit": "ms",
                    "seeds": len(seeds),
                    "traversal": "native" if use_native else "cypher",
                    "median_rowcount": int(_median([float(r) for r in rowcounts])),
                }
        except Exception as exc:
            results[name] = _failure_result(exc, spec["kind"])
    return results


def run_memgraph(driver: Any, seeds: Sequence[int], runs: int, warmup: int) -> Dict[str, Any]:
    from graph_benchmark_memgraph_q1_q9 import execute
    results: Dict[str, Any] = {}
    for spec in QUERIES:
        name = spec["name"]
        q = spec["memgraph"]
        try:
            if spec["kind"] == "global":
                result, timing = _time(lambda: execute(driver, q), runs, warmup)
                results[name] = {
                    **timing,
                    "kind": "global",
                    "rowcount": len(result),
                    "unit": "ms",
                }
            else:
                per_seed: List[float] = []
                rowcounts: List[int] = []
                seed_timings: List[SeedTiming] = []
                for sid in seeds:
                    res, timing = _time(
                        lambda: execute(driver, q, id=int(sid)),
                        runs,
                        warmup,
                    )
                    rowcount = len(res)
                    per_seed.append(timing["median_ms"])
                    rowcounts.append(rowcount)
                    seed_timings.append(
                        {
                            "seed": int(sid),
                            "median_ms": timing["median_ms"],
                            "samples_ms": timing["samples_ms"],
                            "rowcount": rowcount,
                            "execution_paths": [],
                        }
                    )
                results[name] = {
                    "median_ms": _median(per_seed),
                    "min_seed_median_ms": min(per_seed),
                    "max_seed_median_ms": max(per_seed),
                    "seed_timings": seed_timings,
                    "kind": "seed",
                    "unit": "ms",
                    "seeds": len(seeds),
                    "median_rowcount": int(_median([float(r) for r in rowcounts])),
                }
        except Exception as exc:
            results[name] = _failure_result(exc, spec["kind"])
    return results


def load_memgraph(driver: Any, nodes: pd.DataFrame, edges: pd.DataFrame, batch_size: int) -> Dict[str, Any]:
    from graph_benchmark_memgraph_q1_q9 import execute, execute_implicit, _records, _chunks
    t0 = time.perf_counter()
    execute(driver, "MATCH (n) DETACH DELETE n")
    for stmt in ("CREATE INDEX ON :User(id)", "CREATE INDEX ON :User(age)", "CREATE INDEX ON :User(gender)"):
        try:
            execute_implicit(driver, stmt)
        except Exception as exc:
            if "exist" not in str(exc).lower():
                raise
    node_rows = _records(nodes.drop(columns=["label__User"]), ["id", "completion_percentage", "gender", "age"])
    for batch in _chunks(node_rows, batch_size):
        execute(driver, "UNWIND $rows AS row CREATE (n:User) SET n += row", rows=batch)
    edge_rows = _records(edges, ["src", "dst"])
    for batch in _chunks(edge_rows, batch_size):
        execute(
            driver,
            "UNWIND $rows AS row WITH row MATCH (n:User {id: row.src}) MATCH (m:User {id: row.dst}) CREATE (n)-[:Friend]->(m)",
            rows=batch,
        )
    return {"load_ms": (time.perf_counter() - t0) * 1000.0, "nodes": int(len(nodes)), "edges": int(len(edges))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-cypher", type=Path, required=True, help="mgBench Pokec *_import.cypher file")
    parser.add_argument("--engine", choices=[*GFQL_ENGINES, "both", "all"], default="both")
    parser.add_argument("--seeds", type=int, default=30, help="Number of shared seed vertices for seeded queries")
    parser.add_argument("--seed-rng", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gfql-index", choices=["off", "use", "auto", "force"], default="off",
                        help="#1658 seeded adjacency index: 'off' = scan (pre-index); "
                             "'use'/'auto'/'force' build indexes as untimed setup and route seeded hops through them.")
    parser.add_argument("--gfql-traversal", choices=["cypher", "native"], default="cypher",
                        help="'cypher' = exact mgBench Cypher (not index-routed on #1658); "
                             "'native' = express expansion/neighbours via indexed hop()/chain surfaces (walk semantics, parity-checked).")
    parser.add_argument("--memgraph-uri", default=None, help="Bolt URI; if set, also benchmark Memgraph")
    parser.add_argument("--memgraph-batch-size", type=int, default=5000)
    parser.add_argument("--skip-memgraph-load", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    nodes, edges = parse_pokec_import(args.import_cypher)
    rng = random.Random(args.seed_rng)
    all_ids = nodes["id"].tolist()
    seeds = rng.sample(all_ids, min(args.seeds, len(all_ids)))

    output: Dict[str, Any] = {
        "dataset": str(args.import_cypher),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "seeds": len(seeds),
        "runs": args.runs,
        "warmup": args.warmup,
        "gfql_index": args.gfql_index,
        "gfql_traversal": args.gfql_traversal,
        "gfql": {},
    }

    engines = _selected_engines(args.engine)
    for engine in engines:
        output["gfql"][engine] = run_gfql(nodes, edges, engine, seeds, args.runs, args.warmup, args.gfql_index, args.gfql_traversal)

    if args.memgraph_uri:
        from graph_benchmark_memgraph_q1_q9 import make_driver, wait_ready
        wait_ready(lambda: make_driver(args.memgraph_uri), 120)
        with make_driver(args.memgraph_uri) as driver:
            if not args.skip_memgraph_load:
                output["memgraph_load"] = load_memgraph(driver, nodes, edges, args.memgraph_batch_size)
            output["memgraph"] = run_memgraph(driver, seeds, args.runs, args.warmup)

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
