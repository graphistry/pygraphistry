#!/usr/bin/env python3
# Run graph-benchmark q1-q9 and neighbors_2/3/4 against Kuzu.
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from graph_benchmark_q1_q9 import DEFAULT_ROOT, EDGE_FILES, NODE_FILES, _load_edges, _load_nodes, _median
from graph_benchmark_neighbors_probe import SOURCE_AGE_LOWER, SOURCE_AGE_UPPER, SOURCE_GENDER, TARGET_AGE_LOWER

NODE_LABELS = tuple(NODE_FILES.keys())
DEFAULT_DEPTHS = (2, 3, 4)
CSV_DELIMITER = '|'
COPY_OPTIONS = "(HEADER=true, DELIM='|')"


def _import_kuzu() -> Any:
    try:
        import kuzu  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Kuzu benchmark requires the optional kuzu Python package. "
            "Run with `uv run --with kuzu python benchmarks/gfql/graph_benchmark_kuzu.py ...`."
        ) from exc
    return kuzu


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "\\'")


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    return str(value)


def _parse_depths(value: str) -> Tuple[int, ...]:
    depths: List[int] = []
    for part in value.split(','):
        stripped = part.strip()
        if not stripped:
            continue
        depth = int(stripped)
        if depth not in DEFAULT_DEPTHS:
            raise argparse.ArgumentTypeError(f"depth must be one of {DEFAULT_DEPTHS}: {depth}")
        depths.append(depth)
    if not depths:
        raise argparse.ArgumentTypeError('at least one depth is required')
    return tuple(dict.fromkeys(depths))


def _execute_df(conn: Any, query: str) -> pd.DataFrame:
    result = conn.execute(query)
    if hasattr(result, 'get_as_df'):
        return result.get_as_df()
    rows: List[Any] = []
    while result.has_next():
        rows.append(result.get_next())
    return pd.DataFrame(rows)


def _records_preview(df: pd.DataFrame, limit: int = 3) -> List[Dict[str, Any]]:
    preview = df.head(limit).to_dict(orient='records')
    clean: List[Dict[str, Any]] = []
    for row in preview:
        clean.append({str(k): (v.item() if hasattr(v, 'item') else v) for k, v in row.items()})
    return clean


def _summarize_times(times: Sequence[float]) -> Dict[str, Any]:
    sorted_times = sorted(times)
    p95 = sorted_times[int(round((len(sorted_times) - 1) * 0.95))] if sorted_times else 0.0
    return {'median_ms': _median(times), 'p95_ms': p95, 'runs_ms': list(times)}


def _write_csv(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> int:
    existing = [column for column in columns if column in df.columns]
    out = df.loc[:, existing].copy()
    for column in {'node_id', 'age'} & set(out.columns):
        out[column] = pd.to_numeric(out[column], errors='coerce').astype('Int64')
    out.to_csv(path, index=False, sep=CSV_DELIMITER)
    return len(df)


def prepare_csv_imports(csv_dir: Path, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, rebuild: bool) -> Dict[str, Any]:
    if rebuild and csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    node_columns: Dict[str, Sequence[str]] = {
        'Person': ('node_id', 'name', 'gender_lc', 'age'),
        'City': ('node_id', 'city', 'state', 'country'),
        'State': ('node_id', 'state', 'country'),
        'Country': ('node_id', 'country'),
        'Interest': ('node_id', 'interest_lc'),
    }
    node_paths: Dict[str, Path] = {}
    for label, columns in node_columns.items():
        path = csv_dir / f'nodes_{label}.csv'
        _write_csv(path, nodes_df[nodes_df['node_type'] == label], columns)
        node_paths[label] = path
    edge_paths: Dict[str, Path] = {}
    for _, edge_type, _, _ in EDGE_FILES:
        path = csv_dir / f'edges_{edge_type}.csv'
        subset = edges_df.loc[edges_df['rel'] == edge_type, ['src', 'dst']].rename(columns={'src': 'from', 'dst': 'to'})
        subset = subset.astype({'from': 'int64', 'to': 'int64'})
        subset.to_csv(path, index=False, sep=CSV_DELIMITER)
        edge_paths[edge_type] = path
    return {'nodes': node_paths, 'edges': edge_paths}


def create_schema(conn: Any) -> None:
    statements = [
        'CREATE NODE TABLE Person(node_id INT64, name STRING, gender_lc STRING, age INT64, PRIMARY KEY(node_id))',
        'CREATE NODE TABLE City(node_id INT64, city STRING, state STRING, country STRING, PRIMARY KEY(node_id))',
        'CREATE NODE TABLE State(node_id INT64, state STRING, country STRING, PRIMARY KEY(node_id))',
        'CREATE NODE TABLE Country(node_id INT64, country STRING, PRIMARY KEY(node_id))',
        'CREATE NODE TABLE Interest(node_id INT64, interest_lc STRING, PRIMARY KEY(node_id))',
        'CREATE REL TABLE FOLLOWS(FROM Person TO Person)',
        'CREATE REL TABLE LIVES_IN(FROM Person TO City)',
        'CREATE REL TABLE HAS_INTEREST(FROM Person TO Interest)',
        'CREATE REL TABLE CITY_IN(FROM City TO State)',
        'CREATE REL TABLE STATE_IN(FROM State TO Country)',
    ]
    for statement in statements:
        conn.execute(statement)


def load_graph_benchmark_kuzu(
    conn: Any, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, *, csv_dir: Path, rebuild_csv: bool
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    paths = prepare_csv_imports(csv_dir, nodes_df, edges_df, rebuild_csv)
    create_schema(conn)
    for label, path in paths['nodes'].items():
        conn.execute(f"COPY {label} FROM '{_quote_path(path)}' {COPY_OPTIONS}")
    for _, edge_type, _, _ in EDGE_FILES:
        path = paths['edges'][edge_type]
        conn.execute(f"COPY {edge_type} FROM '{_quote_path(path)}' {COPY_OPTIONS}")
    return {
        'load_ms': (time.perf_counter() - t0) * 1000.0,
        'load_method': 'csv',
        'csv_dir': str(csv_dir),
        'nodes': {label: int((nodes_df['node_type'] == label).sum()) for label in NODE_LABELS},
        'edges': {edge_type: int((edges_df['rel'] == edge_type).sum()) for _, edge_type, _, _ in EDGE_FILES},
    }


QuerySpec = Tuple[str, str, Dict[str, Any]]

Q1_Q9: Tuple[QuerySpec, ...] = (
    ('q1', '''MATCH (:Person)-[:FOLLOWS]->(p:Person) RETURN p.node_id AS node_id, p.name AS name, count(*) AS numFollowers ORDER BY numFollowers DESC LIMIT 3''', {}),
    ('q2', '''MATCH (:Person)-[:FOLLOWS]->(p:Person) WITH p, count(*) AS numFollowers ORDER BY numFollowers DESC LIMIT 1 MATCH (p)-[:LIVES_IN]->(city:City) RETURN p.name AS name, city.city AS city, city.state AS state, city.country AS country''', {}),
    ('q3', '''MATCH (p:Person)-[:LIVES_IN]->(city:City)-[:CITY_IN]->(:State)-[:STATE_IN]->(country:Country) WHERE country.country = $country RETURN city.city AS city, avg(p.age) AS averageAge ORDER BY averageAge LIMIT 5''', {'country': 'United States'}),
    ('q4', '''MATCH (p:Person)-[:LIVES_IN]->(:City)-[:CITY_IN]->(:State)-[:STATE_IN]->(country:Country) WHERE p.age >= $age_lower AND p.age <= $age_upper RETURN country.country AS country, count(*) AS personCounts ORDER BY personCounts DESC LIMIT 3''', {'age_lower': 30, 'age_upper': 40}),
    ('q5', '''MATCH (p:Person)-[:HAS_INTEREST]->(i:Interest) MATCH (p)-[:LIVES_IN]->(city:City) WHERE p.gender_lc = $gender AND i.interest_lc = $interest AND city.city = $city AND city.country = $country RETURN count(DISTINCT p) AS numPersons''', {'gender': 'male', 'city': 'London', 'country': 'United Kingdom', 'interest': 'fine dining'}),
    ('q6', '''MATCH (p:Person)-[:HAS_INTEREST]->(i:Interest) MATCH (p)-[:LIVES_IN]->(city:City) WHERE p.gender_lc = $gender AND i.interest_lc = $interest RETURN city.city AS city, city.country AS country, count(DISTINCT p) AS numPersons ORDER BY numPersons DESC LIMIT 5''', {'gender': 'female', 'interest': 'tennis'}),
    ('q7', '''MATCH (p:Person)-[:HAS_INTEREST]->(i:Interest) MATCH (p)-[:LIVES_IN]->(:City)-[:CITY_IN]->(state:State) WHERE i.interest_lc = $interest AND state.country = $country AND p.age >= $age_lower AND p.age <= $age_upper RETURN state.state AS state, state.country AS country, count(DISTINCT p) AS numPersons ORDER BY numPersons DESC LIMIT 1''', {'country': 'United States', 'age_lower': 23, 'age_upper': 30, 'interest': 'photography'}),
    ('q8', '''MATCH (b:Person) OPTIONAL MATCH (a:Person)-[:FOLLOWS]->(b) WITH b, count(a) AS indeg OPTIONAL MATCH (b)-[:FOLLOWS]->(c:Person) WITH indeg, count(c) AS outdeg RETURN sum(indeg * outdeg) AS numPaths''', {}),
    ('q9', '''MATCH (b:Person) WHERE b.age < $age_1 OPTIONAL MATCH (a:Person)-[:FOLLOWS]->(b) WITH b, count(a) AS indeg OPTIONAL MATCH (b)-[:FOLLOWS]->(c:Person) WHERE c.age > $age_2 WITH indeg, count(c) AS outdeg RETURN sum(indeg * outdeg) AS numPaths''', {'age_1': 50, 'age_2': 25}),
)


def _apply_params(query: str, params: Dict[str, Any]) -> str:
    out = query
    for key, value in params.items():
        out = out.replace(f'${key}', _literal(value))
    return out


def run_query_specs(conn: Any, specs: Sequence[QuerySpec], runs: int, warmup: int) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, query, params in specs:
        rendered = _apply_params(query, params)
        for _ in range(warmup):
            _execute_df(conn, rendered)
        times: List[float] = []
        last_df = pd.DataFrame()
        for _ in range(runs):
            start = time.perf_counter()
            last_df = _execute_df(conn, rendered)
            times.append((time.perf_counter() - start) * 1000.0)
        result = _summarize_times(times)
        result['result_preview'] = _records_preview(last_df)
        result['query'] = ' '.join(line.strip() for line in rendered.strip().splitlines())
        results[name] = result
    return results


def _prefix_match_clauses(depth: int) -> str:
    if depth not in DEFAULT_DEPTHS:
        raise ValueError(f'unsupported depth: {depth}')
    clauses = ['MATCH (s:Person)']
    previous = 's'
    for idx in range(1, depth):
        current = 'mid' if idx == depth - 1 else f'n{idx}'
        clauses.append(f'MATCH ({previous})-[:FOLLOWS]->({current}:Person)')
        previous = current
    return '\n    '.join(clauses)


def neighbors_factorized_query(depth: int) -> str:
    return f'''
    {_prefix_match_clauses(depth)}
    WHERE s.gender_lc = {SOURCE_GENDER!r}
      AND s.age >= {SOURCE_AGE_LOWER}
      AND s.age <= {SOURCE_AGE_UPPER}
    WITH mid, count(*) AS prefixCount
    MATCH (mid)-[:FOLLOWS]->(t:Person)
    MATCH (t)-[:LIVES_IN]->(c:City)
    WHERE t.age > {TARGET_AGE_LOWER}
    WITH c.city AS city, c.country AS country, sum(prefixCount) AS pathCount
    RETURN city, country, pathCount
    ORDER BY pathCount DESC, city ASC, country ASC
    LIMIT 10
    '''


def run_neighbors(conn: Any, depths: Sequence[int], runs: int, warmup: int) -> Dict[str, Dict[str, Any]]:
    specs: List[QuerySpec] = [(f'neighbors_{depth}', neighbors_factorized_query(depth), {}) for depth in depths]
    results = run_query_specs(conn, specs, runs, warmup)
    for depth in depths:
        key = f'neighbors_{depth}'
        results[key]['depth'] = depth
        results[key]['source_to_target_hops'] = depth
        results[key]['query_strategy'] = 'factorized'
    return results


def open_database(db_path: Path, fresh: bool) -> Tuple[Any, Any]:
    kuzu = _import_kuzu()
    if fresh and db_path.exists():
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = kuzu.Database(str(db_path))
    connection = kuzu.Connection(database)
    return database, connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--db-path', type=Path, default=Path('/tmp/gfql_kuzu_graph_benchmark'))
    parser.add_argument('--csv-dir', type=Path, default=Path('/tmp/gfql_kuzu_import'))
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--workload', choices=['q1-q9', 'neighbors', 'all'], default='all')
    parser.add_argument('--depths', type=_parse_depths, default=DEFAULT_DEPTHS)
    parser.add_argument('--skip-load', action='store_true')
    parser.add_argument('--reuse-db', action='store_true', help='Do not delete an existing Kuzu database before loading.')
    parser.add_argument('--no-csv-rebuild', action='store_true', help='Reuse existing CSV staging files.')
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()

    _, conn = open_database(args.db_path, fresh=not args.reuse_db and not args.skip_load)
    load_stats: Optional[Dict[str, Any]] = None
    if not args.skip_load:
        nodes_path = args.graph_benchmark_root / 'data' / 'output' / 'nodes'
        edges_path = args.graph_benchmark_root / 'data' / 'output' / 'edges'
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f'Missing data at {nodes_path} or {edges_path}. Run generate_data.sh first.')
        nodes_df, offsets = _load_nodes(nodes_path)
        edges_df = _load_edges(edges_path, offsets)
        load_stats = load_graph_benchmark_kuzu(conn, nodes_df, edges_df, csv_dir=args.csv_dir, rebuild_csv=not args.no_csv_rebuild)

    results: Dict[str, Any] = {}
    if args.workload in {'q1-q9', 'all'}:
        results['q1-q9'] = run_query_specs(conn, Q1_Q9, args.runs, args.warmup)
    if args.workload in {'neighbors', 'all'}:
        results['neighbors'] = run_neighbors(conn, args.depths, args.runs, args.warmup)

    output: Dict[str, Any] = {
        'engine': 'kuzu',
        'backend': 'kuzu',
        'graph_benchmark_root': str(args.graph_benchmark_root),
        'db_path': str(args.db_path),
        'runs': args.runs,
        'warmup': args.warmup,
        'load': load_stats,
        'workload': args.workload,
        'results': results,
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + '\n')


if __name__ == '__main__':
    main()
