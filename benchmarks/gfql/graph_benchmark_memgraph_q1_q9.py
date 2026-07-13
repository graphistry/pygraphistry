#!/usr/bin/env python3
"""Run graph-benchmark q1-q9 against Memgraph.

The script uses the same generated graph-benchmark parquet data as
``graph_benchmark_q1_q9.py`` and writes a compatible JSON timing shape so
GFQL CPU/GPU results can be compared with Memgraph query timings.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from graph_benchmark_q1_q9 import DEFAULT_ROOT, EDGE_FILES, NODE_FILES, _load_edges, _load_nodes, _median

DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_CONTAINER = "gfql-bench-memgraph"
DEFAULT_IMAGE = "memgraph/memgraph-mage"
NODE_LABELS = tuple(NODE_FILES.keys())
EDGE_TYPES = tuple(edge_type for _, edge_type, _, _ in EDGE_FILES)


def _import_neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Memgraph benchmark requires the optional neo4j Python driver. "
            "Run with `uv run --with neo4j python benchmarks/gfql/graph_benchmark_memgraph_q1_q9.py ...`."
        ) from exc
    return GraphDatabase


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), check=True, text=True, capture_output=True)


def stop_container(container_name: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True)


def start_container(container_name: str, image: str, port: int) -> None:
    stop_container(container_name)
    _run([
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{port}:7687",
        "-v",
        "/tmp:/tmp",
        image,
    ])


def make_driver(uri: str, user: str = "", password: str = "") -> Any:
    GraphDatabase = _import_neo4j_driver()
    if user:
        return GraphDatabase.driver(uri, auth=(user, password))
    return GraphDatabase.driver(uri)


def execute(driver: Any, query: str, **params: Any) -> List[Any]:
    records, _, _ = driver.execute_query(query, parameters_=params)
    return list(records)


def execute_implicit(driver: Any, query: str, **params: Any) -> List[Any]:
    with driver.session() as session:
        return list(session.run(query, **params))


def wait_ready(driver_factory: Callable[[], Any], timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with driver_factory() as driver:
                rows = execute(driver, "RETURN 1 AS ok")
                if rows and rows[0]["ok"] == 1:
                    return
        except Exception as exc:  # service may still be booting
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Memgraph did not become ready within {timeout_s}s: {last_error}")


def _value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _records(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in df[list(columns)].to_dict(orient="records"):
        clean = {str(k): _value(v) for k, v in row.items()}
        out.append({k: v for k, v in clean.items() if v is not None})
    return out


def _chunks(rows: Sequence[Mapping[str, Any]], size: int) -> Iterable[List[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _create_indexes(driver: Any) -> None:
    statements = [
        *(f"CREATE INDEX ON :{label}(node_id)" for label in NODE_LABELS),
        "CREATE INDEX ON :Person(age)",
        "CREATE INDEX ON :Person(gender_lc)",
        "CREATE INDEX ON :City(city)",
        "CREATE INDEX ON :City(country)",
        "CREATE INDEX ON :State(country)",
        "CREATE INDEX ON :Interest(interest_lc)",
    ]
    for statement in statements:
        try:
            execute_implicit(driver, statement)
        except Exception as exc:
            # Memgraph raises if the index already exists; reuse is fine for benchmarks.
            if "exist" not in str(exc).lower():
                raise


def _load_node_label(driver: Any, nodes_df: pd.DataFrame, label: str, batch_size: int) -> int:
    subset = nodes_df[nodes_df["node_type"] == label].copy()
    if subset.empty:
        return 0
    columns = [c for c in subset.columns if c != "node_type"]
    rows = _records(subset, columns)
    query = f"UNWIND $rows AS row CREATE (n:{label}) SET n += row"
    for batch in _chunks(rows, batch_size):
        execute(driver, query, rows=batch)
    return len(rows)


def _load_edge_type(driver: Any, edges_df: pd.DataFrame, edge_type: str, src_label: str, dst_label: str, batch_size: int) -> int:
    subset = edges_df[edges_df["rel"] == edge_type]
    if subset.empty:
        return 0
    rows = _records(subset, ["src", "dst"])
    query = (
        f"UNWIND $rows AS row "
        f"MATCH (src:{src_label} {{node_id: row.src}}) "
        f"MATCH (dst:{dst_label} {{node_id: row.dst}}) "
        f"CREATE (src)-[:{edge_type}]->(dst)"
    )
    for batch in _chunks(rows, batch_size):
        execute(driver, query, rows=batch)
    return len(rows)


def load_graph_benchmark_bolt(
    driver: Any,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
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
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "load_ms": elapsed_ms,
        "nodes": node_counts,
        "edges": edge_counts,
    }


def _csv_path(path: Path) -> str:
    return str(path).replace('"', '\\"')


def _write_label_csv(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> int:
    existing = [col for col in columns if col in df.columns]
    df.loc[:, existing].to_csv(path, index=False)
    return len(df)


def _write_edge_csv_chunks(csv_dir: Path, edge_type: str, edges: pd.DataFrame, batch_size: int) -> List[Path]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths: List[Path] = []
    for start in range(0, len(edges), batch_size):
        path = csv_dir / f"edges_{edge_type}_{start // batch_size:06d}.csv"
        edges.iloc[start : start + batch_size].to_csv(path, index=False)
        paths.append(path)
    return paths


def _prepare_csv_imports(
    csv_dir: Path,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    rebuild: bool,
    batch_size: int,
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    if rebuild and csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    node_paths: Dict[str, Path] = {}
    label_columns: Dict[str, Sequence[str]] = {
        "Person": ("node_id", "name", "gender_lc", "age"),
        "City": ("node_id", "city", "state", "country"),
        "State": ("node_id", "state", "country"),
        "Country": ("node_id", "country"),
        "Interest": ("node_id", "interest_lc"),
    }
    for label, columns in label_columns.items():
        path = csv_dir / f"nodes_{label}.csv"
        subset = nodes_df[nodes_df["node_type"] == label]
        _write_label_csv(path, subset, columns)
        node_paths[label] = path

    edge_paths: Dict[str, List[Path]] = {}
    for _, edge_type, _, _ in EDGE_FILES:
        subset = edges_df.loc[edges_df["rel"] == edge_type, ["src", "dst"]]
        edge_paths[edge_type] = _write_edge_csv_chunks(csv_dir, edge_type, subset, batch_size)
    return node_paths, edge_paths


def load_graph_benchmark_csv(
    driver: Any,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    csv_dir: Path,
    create_indexes: bool,
    clear: bool,
    rebuild_csv: bool,
    batch_size: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if clear:
        execute(driver, "MATCH (n) DETACH DELETE n")
    if create_indexes:
        _create_indexes(driver)

    node_paths, edge_paths = _prepare_csv_imports(csv_dir, nodes_df, edges_df, rebuild_csv, batch_size)

    node_queries = {
        "Person": (
            f'LOAD CSV FROM "{_csv_path(node_paths["Person"])}" WITH HEADER AS row '
            'CREATE (:Person {node_id: ToInteger(row.node_id), name: row.name, gender_lc: row.gender_lc, age: ToInteger(row.age)})'
        ),
        "City": (
            f'LOAD CSV FROM "{_csv_path(node_paths["City"])}" WITH HEADER AS row '
            'CREATE (:City {node_id: ToInteger(row.node_id), city: row.city, state: row.state, country: row.country})'
        ),
        "State": (
            f'LOAD CSV FROM "{_csv_path(node_paths["State"])}" WITH HEADER AS row '
            'CREATE (:State {node_id: ToInteger(row.node_id), state: row.state, country: row.country})'
        ),
        "Country": (
            f'LOAD CSV FROM "{_csv_path(node_paths["Country"])}" WITH HEADER AS row '
            'CREATE (:Country {node_id: ToInteger(row.node_id), country: row.country})'
        ),
        "Interest": (
            f'LOAD CSV FROM "{_csv_path(node_paths["Interest"])}" WITH HEADER AS row '
            'CREATE (:Interest {node_id: ToInteger(row.node_id), interest_lc: row.interest_lc})'
        ),
    }
    for query in node_queries.values():
        execute(driver, query)

    for _, edge_type, src_label, dst_label in EDGE_FILES:
        for edge_path in edge_paths[edge_type]:
            path = _csv_path(edge_path)
            execute(
                driver,
                f'LOAD CSV FROM "{path}" WITH HEADER AS row '
                'WITH ToInteger(row.src) AS src_id, ToInteger(row.dst) AS dst_id '
                f'MATCH (src:{src_label} {{node_id: src_id}}) '
                f'MATCH (dst:{dst_label} {{node_id: dst_id}}) '
                f'CREATE (src)-[:{edge_type}]->(dst)',
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "load_ms": elapsed_ms,
        "load_method": "csv",
        "csv_dir": str(csv_dir),
        "edge_csv_chunks": {edge_type: len(paths) for edge_type, paths in edge_paths.items()},
        "nodes": {label: int((nodes_df["node_type"] == label).sum()) for label in NODE_LABELS},
        "edges": {edge_type: int((edges_df["rel"] == edge_type).sum()) for _, edge_type, _, _ in EDGE_FILES},
    }


QuerySpec = Tuple[str, str, Dict[str, Any]]


QUERIES: Tuple[QuerySpec, ...] = (
    (
        "q1",
        """
        MATCH (:Person)-[:FOLLOWS]->(p:Person)
        RETURN p.node_id AS node_id, p.name AS name, count(*) AS numFollowers
        ORDER BY numFollowers DESC
        LIMIT 3
        """,
        {},
    ),
    (
        "q2",
        """
        MATCH (:Person)-[:FOLLOWS]->(p:Person)
        WITH p, count(*) AS numFollowers
        ORDER BY numFollowers DESC
        LIMIT 1
        MATCH (p)-[:LIVES_IN]->(city:City)
        RETURN p.name AS name, city.city AS city, city.state AS state, city.country AS country
        """,
        {},
    ),
    (
        "q3",
        """
        MATCH (p:Person)-[:LIVES_IN]->(city:City)-[:CITY_IN]->(:State)-[:STATE_IN]->(country:Country {country: $country})
        RETURN city.city AS city, avg(p.age) AS averageAge
        ORDER BY averageAge
        LIMIT 5
        """,
        {"country": "United States"},
    ),
    (
        "q4",
        """
        MATCH (p:Person)-[:LIVES_IN]->(:City)-[:CITY_IN]->(:State)-[:STATE_IN]->(country:Country)
        WHERE p.age >= $age_lower AND p.age <= $age_upper
        RETURN country.country AS country, count(*) AS personCounts
        ORDER BY personCounts DESC
        LIMIT 3
        """,
        {"age_lower": 30, "age_upper": 40},
    ),
    (
        "q5",
        """
        MATCH (p:Person {gender_lc: $gender})-[:HAS_INTEREST]->(:Interest {interest_lc: $interest})
        MATCH (p)-[:LIVES_IN]->(:City {city: $city, country: $country})
        RETURN count(DISTINCT p) AS numPersons
        """,
        {
            "gender": "male",
            "city": "London",
            "country": "United Kingdom",
            "interest": "fine dining",
        },
    ),
    (
        "q6",
        """
        MATCH (p:Person {gender_lc: $gender})-[:HAS_INTEREST]->(:Interest {interest_lc: $interest})
        MATCH (p)-[:LIVES_IN]->(city:City)
        RETURN city.city AS city, city.country AS country, count(DISTINCT p) AS numPersons
        ORDER BY numPersons DESC
        LIMIT 5
        """,
        {"gender": "female", "interest": "tennis"},
    ),
    (
        "q7",
        """
        MATCH (p:Person)-[:HAS_INTEREST]->(:Interest {interest_lc: $interest})
        MATCH (p)-[:LIVES_IN]->(:City)-[:CITY_IN]->(state:State {country: $country})
        WHERE p.age >= $age_lower AND p.age <= $age_upper
        RETURN state.state AS state, state.country AS country, count(DISTINCT p) AS numPersons
        ORDER BY numPersons DESC
        LIMIT 1
        """,
        {"country": "United States", "age_lower": 23, "age_upper": 30, "interest": "photography"},
    ),
    (
        "q8",
        """
        MATCH (b:Person)
        OPTIONAL MATCH (a:Person)-[:FOLLOWS]->(b)
        WITH b, count(a) AS indeg
        OPTIONAL MATCH (b)-[:FOLLOWS]->(c:Person)
        WITH indeg, count(c) AS outdeg
        RETURN sum(indeg * outdeg) AS numPaths
        """,
        {},
    ),
    (
        "q9",
        """
        MATCH (b:Person)
        WHERE b.age < $age_1
        OPTIONAL MATCH (a:Person)-[:FOLLOWS]->(b)
        WITH b, count(a) AS indeg
        OPTIONAL MATCH (b)-[:FOLLOWS]->(c:Person)
        WHERE c.age > $age_2
        WITH indeg, count(c) AS outdeg
        RETURN sum(indeg * outdeg) AS numPaths
        """,
        {"age_1": 50, "age_2": 25},
    ),
)


def _summarize_times(times: Sequence[float]) -> Dict[str, Any]:
    return {
        "median_ms": _median(times),
        "runs": list(times),
    }


def _record_preview(rows: Sequence[Any], limit: int = 3) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for row in list(rows)[:limit]:
        preview.append({key: _value(row[key]) for key in row.keys()})
    return preview


def run_queries(driver: Any, runs: int, warmup: int) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, query, params in QUERIES:
        for _ in range(warmup):
            execute(driver, query, **params)
        times: List[float] = []
        last_rows: List[Any] = []
        for _ in range(runs):
            start = time.perf_counter()
            last_rows = execute(driver, query, **params)
            times.append((time.perf_counter() - start) * 1000.0)
        result = _summarize_times(times)
        result["result_preview"] = _record_preview(last_rows)
        results[name] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--load-method", choices=["csv", "bolt"], default="csv")
    parser.add_argument("--csv-dir", type=Path, default=Path("/tmp/gfql_memgraph_import"))
    parser.add_argument("--no-csv-rebuild", action="store_true", help="Reuse existing CSV staging files for --load-method csv.")
    parser.add_argument("--skip-load", action="store_true", help="Use the currently loaded Memgraph database.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the database before loading.")
    parser.add_argument("--no-indexes", action="store_true", help="Do not create Memgraph indexes before loading.")
    parser.add_argument("--start-container", action="store_true", help="Start a local Docker Memgraph container before running.")
    parser.add_argument("--keep-container", action="store_true", help="Keep the Docker container after the run.")
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER)
    parser.add_argument("--memgraph-image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-port", type=int, default=7687)
    parser.add_argument("--ready-timeout", type=int, default=90)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.start_container:
        start_container(args.container_name, args.memgraph_image, args.container_port)

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
                if args.load_method == "csv":
                    load_stats = load_graph_benchmark_csv(
                        driver,
                        nodes_df,
                        edges_df,
                        csv_dir=args.csv_dir,
                        create_indexes=not args.no_indexes,
                        clear=not args.no_clear,
                        rebuild_csv=not args.no_csv_rebuild,
                        batch_size=args.batch_size,
                    )
                else:
                    load_stats = load_graph_benchmark_bolt(
                        driver,
                        nodes_df,
                        edges_df,
                        batch_size=args.batch_size,
                        create_indexes=not args.no_indexes,
                        clear=not args.no_clear,
                    )
                    load_stats["load_method"] = "bolt"
            results = run_queries(driver, args.runs, args.warmup)
    finally:
        if args.start_container and not args.keep_container:
            stop_container(args.container_name)

    output: Dict[str, Any] = {
        "engine": "memgraph",
        "backend": "memgraph",
        "uri": args.uri,
        "runs": args.runs,
        "warmup": args.warmup,
        "load": load_stats,
        "results": results,
    }
    if args.start_container:
        output["container_image"] = args.memgraph_image
        output["container_name"] = args.container_name

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
