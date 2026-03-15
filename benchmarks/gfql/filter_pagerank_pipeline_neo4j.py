#!/usr/bin/env python3
"""Benchmark a GFQL-like filter -> PageRank -> filter pipeline in Neo4j + GDS.

Manual-only benchmark script intended for DGX/host runs where Docker is available.
It measures warm pipeline runtime separately from one-time CSV import and DB startup.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = REPO_ROOT / "plans" / "gfql-gpu-pagerank-benchmark"
DEFAULT_DATA_DIR = PLAN_DIR / "data"
DEFAULT_WORKSPACE_DIR = PLAN_DIR / "neo4j"
DEFAULT_CONTAINER = "gfql-bench-neo4j"
DEFAULT_IMAGE = "neo4j:2026.02.2"
DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_AUTH = ("neo4j", "testtesttest")
DEFAULT_HEAP_GB = 16
DEFAULT_PAGECACHE_GB = 16
DEFAULT_TX_GB = 32


def dataset_spec(dataset: str) -> Dict[str, str | int]:
    if dataset == "twitter":
        return {
            "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
            "filename": "twitter_combined.txt.gz",
            "sep": " ",
            "skiprows": 0,
        }
    if dataset == "gplus":
        return {
            "url": "https://snap.stanford.edu/data/gplus_combined.txt.gz",
            "filename": "gplus_combined.txt.gz",
            "sep": " ",
            "skiprows": 0,
        }
    raise ValueError(f"unknown dataset: {dataset}")


def run(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def download_if_needed(dataset: str, data_dir: Path) -> Path:
    spec = dataset_spec(dataset)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / str(spec["filename"])
    if not path.exists():
        print(f"[download] {spec[url]} -> {path}", flush=True)
        urllib.request.urlretrieve(str(spec["url"]), path)
    return path


def ensure_import_csvs(dataset: str, data_dir: Path, workspace_dir: Path) -> Tuple[Path, Path]:
    src_path = download_if_needed(dataset, data_dir)
    dataset_dir = workspace_dir / "import" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = dataset_dir / "nodes.csv"
    rels_csv = dataset_dir / "rels.csv"
    if nodes_csv.exists() and rels_csv.exists():
        return nodes_csv, rels_csv

    print(f"[prepare-import] building {nodes_csv} and {rels_csv}", flush=True)
    seen = set()
    spec = dataset_spec(dataset)
    sep = str(spec["sep"])
    with gzip.open(src_path, "rt", encoding="utf-8") as src, rels_csv.open("w", newline="") as rels_f:
        rels_writer = csv.writer(rels_f)
        rels_writer.writerow(["src:START_ID(Node)", "dst:END_ID(Node)"])
        skiprows = int(spec["skiprows"])
        for i, line in enumerate(src):
            if i < skiprows:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep) if sep != "\t" else line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            src_id, dst_id = parts[0], parts[1]
            rels_writer.writerow([src_id, dst_id])
            seen.add(src_id)
            seen.add(dst_id)

    with nodes_csv.open("w", newline="") as nodes_f:
        nodes_writer = csv.writer(nodes_f)
        nodes_writer.writerow(["id:ID(Node)"])
        for node_id in sorted(seen, key=int):
            nodes_writer.writerow([node_id])

    return nodes_csv, rels_csv


def stop_container(container_name: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True)


def driver_config() -> Dict[str, object]:
    from neo4j import NotificationDisabledClassification, NotificationMinimumSeverity

    return {
        "notifications_min_severity": NotificationMinimumSeverity.OFF,
        "notifications_disabled_classifications": (NotificationDisabledClassification.DEPRECATION,),
    }


def make_driver(uri: str, auth: Tuple[str, str]):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=auth, **driver_config())


def import_database(
    dataset: str,
    data_dir: Path,
    workspace_dir: Path,
    image: str,
    container_name: str,
    *,
    force: bool = False,
) -> float:
    state_dir = workspace_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f"{dataset}.import_complete.json"
    if marker.exists() and not force:
        return 0.0

    stop_container(container_name)
    dataset_data_dir = workspace_dir / "data" / dataset
    dataset_logs_dir = workspace_dir / "logs" / dataset
    dataset_plugins_dir = workspace_dir / "plugins"
    dataset_import_dir = workspace_dir / "import" / dataset
    dataset_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_plugins_dir.mkdir(parents=True, exist_ok=True)
    dataset_import_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv, rels_csv = ensure_import_csvs(dataset, data_dir, workspace_dir)
    report_file = dataset_data_dir / "import.report"

    t0 = time.perf_counter()
    run([
        "docker",
        "run",
        "--rm",
        "-v",
        f"{dataset_data_dir}:/data",
        "-v",
        f"{dataset_import_dir}:/import",
        image,
        "neo4j-admin",
        "database",
        "import",
        "full",
        "--overwrite-destination=true",
        "--id-type=string",
        f"--nodes=Node=/import/{nodes_csv.name}",
        f"--relationships=LINK=/import/{rels_csv.name}",
        f"--report-file=/data/{report_file.name}",
        "neo4j",
    ], capture_output=True)
    elapsed = time.perf_counter() - t0
    marker.write_text(json.dumps({"dataset": dataset, "import_s": round(elapsed, 4)}) + "\n")
    return elapsed


def start_container(
    dataset: str,
    workspace_dir: Path,
    image: str,
    container_name: str,
    *,
    heap_gb: int,
    pagecache_gb: int,
    tx_gb: int,
) -> None:
    stop_container(container_name)
    dataset_data_dir = workspace_dir / "data" / dataset
    dataset_logs_dir = workspace_dir / "logs" / dataset
    dataset_plugins_dir = workspace_dir / "plugins"
    dataset_import_dir = workspace_dir / "import" / dataset
    dataset_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_plugins_dir.mkdir(parents=True, exist_ok=True)
    dataset_import_dir.mkdir(parents=True, exist_ok=True)

    run([
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        "7474:7474",
        "-p",
        "7687:7687",
        "-e",
        f"NEO4J_AUTH={DEFAULT_AUTH[0]}/{DEFAULT_AUTH[1]}",
        "-e",
        'NEO4J_PLUGINS=["graph-data-science"]',
        "-e",
        f"NEO4J_server_memory_heap_initial__size={heap_gb}G",
        "-e",
        f"NEO4J_server_memory_heap_max__size={heap_gb}G",
        "-e",
        f"NEO4J_server_memory_pagecache_size={pagecache_gb}G",
        "-e",
        f"NEO4J_dbms_memory_transaction_total_max={tx_gb}G",
        "-e",
        "NEO4J_dbms_security_procedures_unrestricted=gds.*",
        "-v",
        f"{dataset_data_dir}:/data",
        "-v",
        f"{dataset_logs_dir}:/logs",
        "-v",
        f"{dataset_plugins_dir}:/plugins",
        "-v",
        f"{dataset_import_dir}:/import",
        image,
    ], capture_output=True)


def wait_ready(uri: str, auth: Tuple[str, str], timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with make_driver(uri, auth) as driver:
                records, _, _ = driver.execute_query("RETURN 1 AS ok", database_="neo4j")
                if records and records[0]["ok"] == 1:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError("neo4j did not become ready")


def q(driver, cypher: str, **params):
    return driver.execute_query(cypher, parameters_=params, database_="neo4j")


def prepare_db(driver, degree_quantile: float) -> Dict[str, float]:
    t0 = time.perf_counter()
    q(driver, "MATCH (n) REMOVE n.degree, n.seed_keep, n.sub1_keep, n.core_keep, n.final_keep, n.pagerank")
    q(driver, "MATCH (n:Node) WITH n, COUNT { (n)--() } AS degree SET n.degree = degree")
    records, _, _ = q(driver, "MATCH (n:Node) RETURN percentileCont(n.degree, $q) AS cutoff", q=degree_quantile)
    cutoff = records[0]["cutoff"]
    t1 = time.perf_counter()
    full_nodes = q(driver, "MATCH (n:Node) RETURN count(n) AS c")[0][0]["c"]
    full_edges = q(driver, "MATCH ()-[r:LINK]->() RETURN count(r) AS c")[0][0]["c"]
    return {
        "degree_cutoff": float(cutoff),
        "prepare_s": round(t1 - t0, 4),
        "full_nodes": int(full_nodes),
        "full_edges": int(full_edges),
    }


def drop_graph_if_exists(driver, name: str) -> None:
    try:
        q(driver, "CALL gds.graph.drop($name, false)", name=name)
    except Exception:
        pass


def batched(xs: Sequence[int], size: int) -> Iterator[List[int]]:
    for i in range(0, len(xs), size):
        yield list(xs[i : i + size])


def run_pipeline_once(driver, degree_cutoff: float, pagerank_quantile: float, batch_size: int) -> Dict[str, float]:
    drop_graph_if_exists(driver, "sub1")
    q(
        driver,
        "MATCH (n:Node) SET n.seed_keep = n.degree >= $cutoff, n.sub1_keep = false, n.core_keep = false, n.final_keep = false REMOVE n.pagerank",
        cutoff=degree_cutoff,
    )
    q(driver, "MATCH ()-[r:LINK]->() REMOVE r.sub1_keep, r.final_keep")

    t1 = time.perf_counter()
    q(driver, "MATCH (n:Node) WHERE n.seed_keep SET n.sub1_keep = true")
    seed_records = q(driver, "MATCH (s:Node) WHERE s.seed_keep RETURN id(s) AS id")[0]
    seed_ids = [record["id"] for record in seed_records]
    for batch in batched(seed_ids, batch_size):
        q(
            driver,
            "UNWIND $seed_ids AS sid MATCH (s:Node) WHERE id(s) = sid MATCH (s)-[r:LINK]-(nbr:Node) SET nbr.sub1_keep = true, r.sub1_keep = true",
            seed_ids=batch,
        )
    stage1_nodes = q(driver, "MATCH (n:Node) WHERE n.sub1_keep RETURN count(n) AS c")[0][0]["c"]
    stage1_edges = q(driver, "MATCH ()-[r:LINK]->() WHERE r.sub1_keep RETURN count(r) AS c")[0][0]["c"]
    t2 = time.perf_counter()

    node_query = "MATCH (n:Node) WHERE n.sub1_keep RETURN id(n) AS id"
    rel_query = (
        "MATCH (a:Node)-[r:LINK]->(b:Node) "
        "WHERE r.sub1_keep "
        "RETURN id(a) AS source, id(b) AS target "
        "UNION ALL "
        "MATCH (a:Node)-[r:LINK]->(b:Node) "
        "WHERE r.sub1_keep "
        "RETURN id(b) AS source, id(a) AS target"
    )
    q(driver, "CALL gds.graph.project.cypher('sub1', $node_query, $rel_query)", node_query=node_query, rel_query=rel_query)
    q(driver, "CALL gds.pageRank.write('sub1', {writeProperty: 'pagerank'})")
    t3 = time.perf_counter()

    pagerank_cutoff = q(
        driver,
        "MATCH (n:Node) WHERE n.sub1_keep RETURN percentileCont(n.pagerank, $q) AS cutoff",
        q=pagerank_quantile,
    )[0][0]["cutoff"]
    q(
        driver,
        "MATCH (n:Node) SET n.core_keep = coalesce(n.sub1_keep, false) AND coalesce(n.pagerank, 0.0) >= $cutoff, n.final_keep = false",
        cutoff=pagerank_cutoff,
    )
    q(driver, "MATCH (n:Node) WHERE n.core_keep SET n.final_keep = true")
    core_records = q(driver, "MATCH (n:Node) WHERE n.core_keep RETURN id(n) AS id")[0]
    core_ids = [record["id"] for record in core_records]
    for batch in batched(core_ids, batch_size):
        q(
            driver,
            "UNWIND $core_ids AS cid MATCH (c:Node) WHERE id(c) = cid MATCH (c)-[r:LINK]-(nbr:Node) SET nbr.final_keep = true, r.final_keep = true",
            core_ids=batch,
        )
    stage3_nodes = q(driver, "MATCH (n:Node) WHERE n.final_keep RETURN count(n) AS c")[0][0]["c"]
    stage3_edges = q(driver, "MATCH ()-[r:LINK]->() WHERE r.final_keep RETURN count(r) AS c")[0][0]["c"]
    t4 = time.perf_counter()

    return {
        "gfql_filter1_s": t2 - t1,
        "pagerank_s": t3 - t2,
        "gfql_filter2_s": t4 - t3,
        "pipeline_total_s": t4 - t1,
        "pagerank_cutoff": float(pagerank_cutoff),
        "stage1_nodes": int(stage1_nodes),
        "stage1_edges": int(stage1_edges),
        "stage2_nodes": int(stage1_nodes),
        "stage2_edges": int(stage1_edges),
        "stage3_nodes": int(stage3_nodes),
        "stage3_edges": int(stage3_edges),
    }


def median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def benchmark(
    dataset: str,
    degree_quantile: float,
    pagerank_quantile: float,
    *,
    warmup: int,
    runs: int,
    force_import: bool,
    data_dir: Path,
    workspace_dir: Path,
    image: str,
    container_name: str,
    batch_size: int,
    keep_container: bool,
    heap_gb: int,
    pagecache_gb: int,
    tx_gb: int,
) -> Dict[str, object]:
    import_s = import_database(dataset, data_dir, workspace_dir, image, container_name, force=force_import)
    start_container(
        dataset,
        workspace_dir,
        image,
        container_name,
        heap_gb=heap_gb,
        pagecache_gb=pagecache_gb,
        tx_gb=tx_gb,
    )
    wait_ready(DEFAULT_URI, DEFAULT_AUTH)
    try:
        with make_driver(DEFAULT_URI, DEFAULT_AUTH) as driver:
            prep = prepare_db(driver, degree_quantile)
            for _ in range(warmup):
                run_pipeline_once(driver, prep["degree_cutoff"], pagerank_quantile, batch_size)
            rows = [run_pipeline_once(driver, prep["degree_cutoff"], pagerank_quantile, batch_size) for _ in range(runs)]
    finally:
        if not keep_container:
            stop_container(container_name)

    first = rows[0]
    return {
        "dataset": dataset,
        "engine": "neo4j",
        "backend": "neo4j+gds",
        "container_image": image,
        "container_name": container_name,
        "degree_quantile": degree_quantile,
        "pagerank_quantile": pagerank_quantile,
        "db_import_s": round(import_s, 4),
        **prep,
        "pagerank_cutoff": first["pagerank_cutoff"],
        "warmup_runs": warmup,
        "warm_runs": runs,
        "batch_size": batch_size,
        "heap_gb": heap_gb,
        "pagecache_gb": pagecache_gb,
        "transaction_gb": tx_gb,
        "pipeline_total_first_s": round(first["pipeline_total_s"], 4),
        "pipeline_total_median_s": round(median([row["pipeline_total_s"] for row in rows]), 4),
        "pipeline_total_runs_s": [round(row["pipeline_total_s"], 4) for row in rows],
        "gfql_filter1_median_s": round(median([row["gfql_filter1_s"] for row in rows]), 4),
        "pagerank_median_s": round(median([row["pagerank_s"] for row in rows]), 4),
        "gfql_filter2_median_s": round(median([row["gfql_filter2_s"] for row in rows]), 4),
        "stage1_nodes": first["stage1_nodes"],
        "stage1_edges": first["stage1_edges"],
        "stage2_nodes": first["stage2_nodes"],
        "stage2_edges": first["stage2_edges"],
        "stage3_nodes": first["stage3_nodes"],
        "stage3_edges": first["stage3_edges"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["twitter", "gplus"], default="twitter")
    parser.add_argument("--degree-quantile", type=float, default=0.99)
    parser.add_argument("--pagerank-quantile", type=float, default=0.99)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workspace-dir", type=Path, default=DEFAULT_WORKSPACE_DIR)
    parser.add_argument("--neo4j-image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--container-name", type=str, default=DEFAULT_CONTAINER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heap-gb", type=int, default=DEFAULT_HEAP_GB)
    parser.add_argument("--pagecache-gb", type=int, default=DEFAULT_PAGECACHE_GB)
    parser.add_argument("--transaction-gb", type=int, default=DEFAULT_TX_GB)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print(
        f"[run] dataset={args.dataset} degree_q={args.degree_quantile} pagerank_q={args.pagerank_quantile} warmup={args.warmup} runs={args.runs} image={args.neo4j_image}",
        flush=True,
    )
    result = benchmark(
        args.dataset,
        args.degree_quantile,
        args.pagerank_quantile,
        warmup=args.warmup,
        runs=args.runs,
        force_import=args.force_import,
        data_dir=args.data_dir,
        workspace_dir=args.workspace_dir,
        image=args.neo4j_image,
        container_name=args.container_name,
        batch_size=args.batch_size,
        keep_container=args.keep_container,
        heap_gb=args.heap_gb,
        pagecache_gb=args.pagecache_gb,
        tx_gb=args.transaction_gb,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
