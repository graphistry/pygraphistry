#!/usr/bin/env python3
"""Run Benchgraph-like neighbors_2/3/4 against Memgraph.

This uses the same generated graph-benchmark parquet data and workload
semantics as ``graph_benchmark_neighbors_probe.py`` so GFQL internal
neighbors timings can be compared against a same-query service run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_benchmark_memgraph_q1_q9 import (
    DEFAULT_CONTAINER,
    DEFAULT_IMAGE,
    DEFAULT_URI,
    execute,
    load_graph_benchmark_bolt,
    load_graph_benchmark_csv,
    make_driver,
    start_container,
    stop_container,
    wait_ready,
)
from graph_benchmark_q1_q9 import DEFAULT_ROOT, _load_edges, _load_nodes, _median
from graph_benchmark_neighbors_probe import (
    SOURCE_AGE_LOWER,
    SOURCE_AGE_UPPER,
    SOURCE_GENDER,
    TARGET_AGE_LOWER,
)


DEFAULT_DEPTHS = (2, 3, 4)


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


def _walk_match_clauses(depth: int) -> str:
    if depth not in DEFAULT_DEPTHS:
        raise ValueError(f"unsupported depth: {depth}")
    clauses = ["MATCH (s:Person)"]
    previous = "s"
    for idx in range(1, depth):
        current = f"n{idx}"
        clauses.append(f"MATCH ({previous})-[:FOLLOWS]->({current}:Person)")
        previous = current
    clauses.append(f"MATCH ({previous})-[:FOLLOWS]->(t:Person)")
    clauses.append("MATCH (t)-[:LIVES_IN]->(c:City)")
    return "\n    ".join(clauses)


def _neighbors_query(depth: int) -> str:
    return f"""
    {_walk_match_clauses(depth)}
    WHERE s.gender_lc = $source_gender
      AND s.age >= $source_age_lower
      AND s.age <= $source_age_upper
      AND t.age > $target_age_lower
    RETURN c.city AS city, c.country AS country, count(*) AS pathCount
    ORDER BY pathCount DESC, city ASC, country ASC
    LIMIT 10
    """


def _prefix_match_clauses(depth: int) -> str:
    if depth not in DEFAULT_DEPTHS:
        raise ValueError(f"unsupported depth: {depth}")
    clauses = ["MATCH (s:Person)"]
    previous = "s"
    for idx in range(1, depth):
        current = "mid" if idx == depth - 1 else f"n{idx}"
        clauses.append(f"MATCH ({previous})-[:FOLLOWS]->({current}:Person)")
        previous = current
    return "\n    ".join(clauses)


def _neighbors_factorized_query(depth: int) -> str:
    return f"""
    {_prefix_match_clauses(depth)}
    WHERE s.gender_lc = $source_gender
      AND s.age >= $source_age_lower
      AND s.age <= $source_age_upper
    WITH mid, count(*) AS prefixCount
    MATCH (mid)-[:FOLLOWS]->(t:Person)
    MATCH (t)-[:LIVES_IN]->(c:City)
    WHERE t.age > $target_age_lower
    WITH c.city AS city, c.country AS country, sum(prefixCount) AS pathCount
    RETURN city, country, pathCount
    ORDER BY pathCount DESC, city ASC, country ASC
    LIMIT 10
    """


def _query_for_strategy(depth: int, query_strategy: str) -> str:
    if query_strategy == 'walk':
        return _neighbors_query(depth)
    if query_strategy == 'factorized':
        return _neighbors_factorized_query(depth)
    raise ValueError(f"unsupported query strategy: {query_strategy}")


def _record_to_dict(row: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if hasattr(value, 'item'):
            value = value.item()
        out[key] = value
    return out


def _summarize_times(times: Sequence[float]) -> Dict[str, Any]:
    sorted_times = sorted(times)
    p95 = sorted_times[int(round((len(sorted_times) - 1) * 0.95))] if sorted_times else 0.0
    return {
        'median_ms': _median(times),
        'p95_ms': p95,
        'runs_ms': list(times),
    }


def run_neighbors(
    driver: Any, depths: Sequence[int], runs: int, warmup: int, query_strategy: str
) -> Dict[str, Dict[str, Any]]:
    params = {
        'source_gender': SOURCE_GENDER,
        'source_age_lower': SOURCE_AGE_LOWER,
        'source_age_upper': SOURCE_AGE_UPPER,
        'target_age_lower': TARGET_AGE_LOWER,
    }
    results: Dict[str, Dict[str, Any]] = {}
    for depth in depths:
        query = _query_for_strategy(depth, query_strategy)
        for _ in range(warmup):
            execute(driver, query, **params)
        times: List[float] = []
        rows: List[Any] = []
        for _ in range(runs):
            start = time.perf_counter()
            rows = execute(driver, query, **params)
            times.append((time.perf_counter() - start) * 1000.0)
        key = f'neighbors_{depth}'
        result = _summarize_times(times)
        result.update(
            {
                'depth': depth,
                'source_to_target_hops': depth,
                'query': ' '.join(line.strip() for line in query.strip().splitlines()),
                'query_strategy': query_strategy,
                'top10': [_record_to_dict(row) for row in rows],
            }
        )
        results[key] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--uri', default=DEFAULT_URI)
    parser.add_argument('--user', default='')
    parser.add_argument('--password', default='')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--depths', type=_parse_depths, default=DEFAULT_DEPTHS)
    parser.add_argument('--query-strategy', choices=['walk', 'factorized'], default='walk')
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--load-method', choices=['csv', 'bolt'], default='csv')
    parser.add_argument('--csv-dir', type=Path, default=Path('/tmp/gfql_memgraph_neighbors_import'))
    parser.add_argument('--no-csv-rebuild', action='store_true', help='Reuse existing CSV staging files for --load-method csv.')
    parser.add_argument('--skip-load', action='store_true', help='Use the currently loaded Memgraph database.')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear the database before loading.')
    parser.add_argument('--no-indexes', action='store_true', help='Do not create Memgraph indexes before loading.')
    parser.add_argument('--start-container', action='store_true', help='Start a local Docker Memgraph container before running.')
    parser.add_argument('--keep-container', action='store_true', help='Keep the Docker container after the run.')
    parser.add_argument('--container-name', default=DEFAULT_CONTAINER)
    parser.add_argument('--memgraph-image', default=DEFAULT_IMAGE)
    parser.add_argument('--container-port', type=int, default=7687)
    parser.add_argument('--ready-timeout', type=int, default=90)
    parser.add_argument('--output-json', type=Path, default=None)
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
                nodes_path = args.graph_benchmark_root / 'data' / 'output' / 'nodes'
                edges_path = args.graph_benchmark_root / 'data' / 'output' / 'edges'
                if not nodes_path.exists() or not edges_path.exists():
                    raise FileNotFoundError(
                        f'Missing data at {nodes_path} or {edges_path}. Run generate_data.sh in graph-benchmark first.'
                    )
                nodes_df, offsets = _load_nodes(nodes_path)
                edges_df = _load_edges(edges_path, offsets)
                if args.load_method == 'csv':
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
                    load_stats['load_method'] = 'bolt'
            results = run_neighbors(driver, args.depths, args.runs, args.warmup, args.query_strategy)
    finally:
        if args.start_container and not args.keep_container:
            stop_container(args.container_name)

    output: Dict[str, Any] = {
        'engine': 'memgraph',
        'backend': 'memgraph',
        'uri': args.uri,
        'graph_benchmark_root': str(args.graph_benchmark_root),
        'runs': args.runs,
        'warmup': args.warmup,
        'load': load_stats,
        'workload': {
            'name': 'neighbors_with_data_and_filter',
            'source_gender': SOURCE_GENDER,
            'source_age_lower': SOURCE_AGE_LOWER,
            'source_age_upper': SOURCE_AGE_UPPER,
            'target_age_lower': TARGET_AGE_LOWER,
            'result': 'top city/country by fixed-depth walk count',
            'query_strategy': args.query_strategy,
            'depths': list(args.depths),
        },
        'results': results,
    }
    if args.start_container:
        output['container_image'] = args.memgraph_image
        output['container_name'] = args.container_name

    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + '\n')


if __name__ == '__main__':
    main()
