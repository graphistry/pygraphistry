#!/usr/bin/env python3
"""Run graph-benchmark q1-q9 against Neo4j.

Reuses the shared Cypher q1-q9 query set, bolt loaders, and timing/output
shape from ``graph_benchmark_memgraph_q1_q9.py`` (both speak Bolt via the
``neo4j`` driver and standard Cypher). Only the container, index DDL syntax,
and auth differ between the two engines, so this module overrides just those.

Writes the same JSON timing shape as ``graph_benchmark_q1_q9.py`` /
``graph_benchmark_memgraph_q1_q9.py`` so ``graph_benchmark_compare.py`` can
render a GFQL CPU/GPU vs Neo4j column.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_benchmark_q1_q9 import DEFAULT_ROOT, EDGE_FILES, NODE_FILES, _load_edges, _load_nodes
from graph_benchmark_memgraph_q1_q9 import (
    _load_edge_type,
    _load_node_label,
    execute,
    make_driver,
    run_queries,
    stop_container,
    wait_ready,
    _run,
)

DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_CONTAINER = "gfql-bench-neo4j"
DEFAULT_IMAGE = "neo4j:5"
NODE_LABELS = tuple(NODE_FILES.keys())


def start_container(container_name: str, image: str, port: int) -> None:
    """Start a Neo4j container with auth disabled and a generous heap/pagecache."""
    stop_container(container_name)
    _run([
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{port}:7687",
        "-e",
        "NEO4J_AUTH=none",
        "-e",
        "NEO4J_server_memory_heap_max__size=4G",
        "-e",
        "NEO4J_server_memory_pagecache_size=2G",
        "-v",
        "/tmp:/tmp",
        image,
    ])


def _create_indexes(driver: Any) -> None:
    # Neo4j 5 range-index DDL: CREATE INDEX <name> IF NOT EXISTS FOR (n:Label) ON (n.prop)
    specs = [
        *((label, "node_id") for label in NODE_LABELS),
        ("Person", "age"),
        ("Person", "gender_lc"),
        ("City", "city"),
        ("City", "country"),
        ("State", "country"),
        ("Interest", "interest_lc"),
    ]
    for label, prop in specs:
        name = f"idx_{label.lower()}_{prop}"
        execute(driver, f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
    # Block until indexes are ONLINE so query timings are not skewed by population.
    execute(driver, "CALL db.awaitIndexes(300)")


def load_graph_benchmark_bolt(
    driver: Any,
    nodes_df: Any,
    edges_df: Any,
    *,
    batch_size: int,
    create_indexes: bool,
    clear: bool,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if clear:
        execute(driver, "MATCH (n) DETACH DELETE n")
    if create_indexes:
        _create_indexes(driver)
    node_counts = {label: _load_node_label(driver, nodes_df, label, batch_size) for label in NODE_LABELS}
    edge_counts: Dict[str, int] = {}
    for _, edge_type, src_label, dst_label in EDGE_FILES:
        edge_counts[edge_type] = _load_edge_type(driver, edges_df, edge_type, src_label, dst_label, batch_size)
    return {
        "load_ms": (time.perf_counter() - t0) * 1000.0,
        "load_method": "bolt",
        "nodes": node_counts,
        "edges": edge_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--skip-load", action="store_true", help="Use the currently loaded Neo4j database.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the database before loading.")
    parser.add_argument("--no-indexes", action="store_true", help="Do not create indexes before loading.")
    parser.add_argument("--start-container", action="store_true", help="Start a local Docker Neo4j container before running.")
    parser.add_argument("--keep-container", action="store_true", help="Keep the Docker container after the run.")
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER)
    parser.add_argument("--neo4j-image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-port", type=int, default=7687)
    parser.add_argument("--ready-timeout", type=int, default=180)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.start_container:
        start_container(args.container_name, args.neo4j_image, args.container_port)

    def _driver_factory() -> Any:
        return make_driver(args.uri, args.user, args.password)

    wait_ready(_driver_factory, args.ready_timeout)

    load_stats: Optional[Dict[str, Any]] = None
    try:
        with _driver_factory() as driver:
            if not args.skip_load:
                nodes_path = args.graph_benchmark_root / "data" / "output" / "nodes"
                edges_path = args.graph_benchmark_root / "data" / "output" / "edges"
                if not nodes_path.exists() or not edges_path.exists():
                    raise FileNotFoundError(
                        f"Missing data at {nodes_path} or {edges_path}. Run generate_data.sh in graph-benchmark first."
                    )
                nodes_df, offsets = _load_nodes(nodes_path)
                edges_df = _load_edges(edges_path, offsets)
                load_stats = load_graph_benchmark_bolt(
                    driver,
                    nodes_df,
                    edges_df,
                    batch_size=args.batch_size,
                    create_indexes=not args.no_indexes,
                    clear=not args.no_clear,
                )
            results = run_queries(driver, args.runs, args.warmup)
    finally:
        if args.start_container and not args.keep_container:
            stop_container(args.container_name)

    output: Dict[str, Any] = {
        "engine": "neo4j",
        "backend": "neo4j",
        "uri": args.uri,
        "runs": args.runs,
        "warmup": args.warmup,
        "load": load_stats,
        "results": results,
    }
    if args.start_container:
        output["container_image"] = args.neo4j_image
        output["container_name"] = args.container_name

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
