#!/usr/bin/env python3
"""Shortest-path/BFS-style probe over graph-benchmark FOLLOWS."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandas.testing import assert_frame_equal

import graphistry
from benchmarks.gfql.graph_benchmark_q1_q9 import _edges_by_rel, _load_edges, _load_nodes, _maybe_to_cudf
from graphistry.Engine import Engine
from graphistry.compute.gfql.same_path.native_shortest_path import try_native_shortest_path


DEFAULT_ROOT = Path('/tmp/graph-benchmark-gfql-memgraph')
DEFAULT_MAX_HOPS = 15
DEFAULT_SOURCE_COUNT = 2
DEFAULT_TARGET_COUNT = 4


def _timed(fn: Callable[[], Any], runs: int, warmup: int) -> Tuple[Any, List[float]]:
    for _ in range(warmup):
        fn()
    times: List[float] = []
    result: Any = None
    for _ in range(runs):
        start = perf_counter()
        result = fn()
        times.append((perf_counter() - start) * 1000.0)
    return result, times


def _median(values: Iterable[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _to_pandas(df: Any) -> pd.DataFrame:
    if hasattr(df, 'to_pandas'):
        return df.to_pandas()
    return pd.DataFrame(df)


def _adjacency(follows: pd.DataFrame) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = defaultdict(list)
    for src, dst in follows[['src', 'dst']].itertuples(index=False):
        adj[int(src)].append(int(dst))
    for src in adj:
        adj[src].sort()
    return dict(adj)


def _distances_from(adj: Dict[int, List[int]], source: int, max_hops: int) -> Dict[int, int]:
    seen = {int(source): 0}
    q: deque[int] = deque([int(source)])
    while q:
        cur = q.popleft()
        hop = seen[cur]
        if hop >= max_hops:
            continue
        for nxt in adj.get(cur, []):
            if nxt in seen:
                continue
            seen[nxt] = hop + 1
            q.append(nxt)
    return {node: dist for node, dist in seen.items() if dist > 0}


def _select_sources_targets(
    follows: pd.DataFrame,
    source_count: int,
    target_count: int,
    max_hops: int,
    source_ids: Optional[Sequence[int]],
    target_ids: Optional[Sequence[int]],
) -> Tuple[List[int], List[int]]:
    if source_ids:
        sources = [int(v) for v in source_ids]
    else:
        counts = follows.groupby('src').size().reset_index(name='outDegree')
        top = counts.sort_values(['outDegree', 'src'], ascending=[False, True]).head(source_count)
        sources = [int(v) for v in top['src'].tolist()]

    if target_ids:
        targets = [int(v) for v in target_ids]
    else:
        adj = _adjacency(follows)
        candidates: Dict[int, Tuple[int, int]] = {}
        for source in sources:
            dists = _distances_from(adj, source, max_hops)
            for node, dist in dists.items():
                if node in sources:
                    continue
                prev = candidates.get(node)
                if prev is None or dist > prev[0]:
                    candidates[node] = (dist, node)
        ranked = sorted(candidates.values(), key=lambda pair: (-pair[0], pair[1]))
        targets = [node for _, node in ranked[:target_count]]
        if len(targets) < target_count:
            fallback = follows['dst'].drop_duplicates().sort_values().tolist()
            for node in fallback:
                node_i = int(node)
                if node_i not in targets and node_i not in sources:
                    targets.append(node_i)
                if len(targets) >= target_count:
                    break
    return sources, targets


def _expected_distances_from_adj(
    adj: Dict[int, List[int]],
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    for source in sources:
        dists = _distances_from(adj, int(source), max_hops)
        for target in targets:
            target_i = int(target)
            rows.append({'source': int(source), 'target': target_i, 'distance': int(dists.get(target_i, -1))})
    return pd.DataFrame(rows).sort_values(['source', 'target']).reset_index(drop=True)


def _expected_distances(follows: pd.DataFrame, sources: Sequence[int], targets: Sequence[int], max_hops: int) -> pd.DataFrame:
    return _expected_distances_from_adj(_adjacency(follows), sources, targets, max_hops)


def _normalize_result(result: Any) -> pd.DataFrame:
    out = _to_pandas(result._nodes if hasattr(result, '_nodes') else result).copy()
    if out.empty:
        return pd.DataFrame({'source': [], 'target': [], 'distance': []}).astype(
            {'source': 'int64', 'target': 'int64', 'distance': 'int64'}
        )
    out = out[['source', 'target', 'distance']]
    out['distance'] = out['distance'].fillna(-1).astype('int64')
    out['source'] = out['source'].astype('int64')
    out['target'] = out['target'].astype('int64')
    return out.sort_values(['source', 'target']).reset_index(drop=True)


def _normalize_native_result(result: Any) -> pd.DataFrame:
    out = _to_pandas(result).copy()
    if out.empty:
        return pd.DataFrame({'source': [], 'target': [], 'distance': []}).astype(
            {'source': 'int64', 'target': 'int64', 'distance': 'int64'}
        )
    out = out.rename(
        columns={
            '__sp_source__': 'source',
            '__sp_target__': 'target',
            '__sp_hops__': 'distance',
        }
    )[['source', 'target', 'distance']]
    out['distance'] = out['distance'].fillna(-1).astype('int64')
    out['source'] = out['source'].astype('int64')
    out['target'] = out['target'].astype('int64')
    return out.sort_values(['source', 'target']).reset_index(drop=True)


def _run_direct_native(
    follows_pd: pd.DataFrame,
    engine: str,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
    direction: str,
    shortest_path_backend: str,
    *,
    cache: Optional[Dict[Any, Any]] = None,
    cache_key: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    if direction != 'forward':
        raise ValueError('direct native probe currently supports forward paths only')
    pairs = [(int(source), int(target)) for source in sources for target in targets]
    step_pairs_pd = follows_pd.rename(columns={'src': '__from__', 'dst': '__to__'})[['__from__', '__to__']]
    source_values = [source for source, _target in pairs]
    target_values = [target for _source, target in pairs]
    engine_value = Engine.CUDF if engine == 'cudf' else Engine.PANDAS
    if engine == 'cudf':
        step_pairs = _maybe_to_cudf('cudf', step_pairs_pd)
    else:
        step_pairs = step_pairs_pd
    result = try_native_shortest_path(
        step_pairs,
        source_values,
        target_values,
        min_hops=1,
        max_hops=max_hops,
        directed=True,
        engine=engine_value,
        backend=shortest_path_backend,
        cache=cache,
        cache_key=cache_key,
    )
    if result is None:
        return None
    return _normalize_native_result(result)


def _time_direct_native(
    follows_pd: pd.DataFrame,
    engine: str,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
    direction: str,
    shortest_path_backend: str,
    expected: pd.DataFrame,
    runs: int,
    warmup: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        'available': False,
        'error': None,
        'cold_median_ms': None,
        'reuse_median_ms': None,
        'cold_runs_ms': [],
        'reuse_runs_ms': [],
    }
    try:
        cold_result, cold_times = _timed(
            lambda: _run_direct_native(
                follows_pd,
                engine,
                sources,
                targets,
                max_hops,
                direction,
                shortest_path_backend,
                cache={},
                cache_key='path_probe',
            ),
            runs,
            warmup,
        )
        if cold_result is None:
            payload['error'] = 'backend returned None'
            return payload
        assert_frame_equal(cold_result, expected, check_dtype=False)

        reusable_cache: Dict[Any, Any] = {}
        reuse_result, reuse_times = _timed(
            lambda: _run_direct_native(
                follows_pd,
                engine,
                sources,
                targets,
                max_hops,
                direction,
                shortest_path_backend,
                cache=reusable_cache,
                cache_key='path_probe',
            ),
            runs,
            warmup,
        )
        assert reuse_result is not None
        assert_frame_equal(reuse_result, expected, check_dtype=False)
        payload.update({
            'available': True,
            'cold_median_ms': _median(cold_times),
            'reuse_median_ms': _median(reuse_times),
            'cold_runs_ms': cold_times,
            'reuse_runs_ms': reuse_times,
        })
    except Exception as exc:
        payload['error'] = f'{type(exc).__name__}: {exc}'
    return payload


def _path_pattern(max_hops: int, direction: str) -> str:
    if direction == 'forward':
        return f'(person1)-[*1..{max_hops}]->(person2)'
    if direction == 'reverse':
        return f'(person1)<-[*1..{max_hops}]-(person2)'
    if direction == 'undirected':
        return f'(person1)-[*1..{max_hops}]-(person2)'
    raise ValueError(f'Unsupported direction: {direction}')


def _cypher_literal(value: int) -> str:
    return str(int(value))


def _cypher_list(values: Sequence[int]) -> str:
    return '[' + ', '.join(_cypher_literal(v) for v in values) + ']'


def _query(max_hops: int, direction: str, source: int, target: int) -> str:
    pattern = _path_pattern(max_hops, direction)
    return f"""
        MATCH
            (person1 {{id: {int(source)}}}),
            (person2 {{id: {int(target)}}}),
            path = shortestPath({pattern})
        RETURN person1.id AS source, person2.id AS target, length(path) AS distance
    """


def _batched_query(max_hops: int, direction: str, sources: Sequence[int], targets: Sequence[int]) -> str:
    pattern = _path_pattern(max_hops, direction)
    return f"""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath({pattern})
        WHERE person1.id IN {_cypher_list(sources)} AND person2.id IN {_cypher_list(targets)}
        RETURN person1.id AS source, person2.id AS target, length(path) AS distance
        ORDER BY source, target
    """


def _batched_prefiltered_query(max_hops: int, direction: str, sources: Sequence[int], targets: Sequence[int]) -> str:
    pattern = _path_pattern(max_hops, direction)
    return f"""
        MATCH (person1:Person), (person2:Person)
        WHERE person1.id IN {_cypher_list(sources)} AND person2.id IN {_cypher_list(targets)}
        WITH person1, person2
        MATCH path = shortestPath({pattern})
        RETURN person1.id AS source, person2.id AS target, length(path) AS distance
        ORDER BY source, target
    """


def _run_cypher_batch(
    nodes: Any,
    follows: Any,
    engine: str,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
    direction: str,
    shortest_path_backend: str,
    cypher_strategy: str,
) -> pd.DataFrame:
    g = graphistry.nodes(nodes, 'id').edges(follows, 's', 'd')
    if cypher_strategy in {'batched', 'batched-prefiltered'}:
        query = (
            _batched_prefiltered_query(max_hops, direction, sources, targets)
            if cypher_strategy == 'batched-prefiltered'
            else _batched_query(max_hops, direction, sources, targets)
        )
        result = g.gfql(
            query,
            engine=engine,
            shortest_path_backend=shortest_path_backend,
        )
        row = _to_pandas(result._nodes if hasattr(result, '_nodes') else result).copy()
        return row[['source', 'target', 'distance']] if not row.empty else pd.DataFrame(columns=['source', 'target', 'distance'])
    if cypher_strategy != 'loop':
        raise ValueError(f'Unsupported cypher strategy: {cypher_strategy}')

    rows: List[pd.DataFrame] = []
    for source, target in itertools.product(sources, targets):
        result = g.gfql(
            _query(max_hops, direction, int(source), int(target)),
            engine=engine,
            shortest_path_backend=shortest_path_backend,
        )
        row = _to_pandas(result._nodes if hasattr(result, '_nodes') else result).copy()
        if row.empty:
            row = pd.DataFrame({'source': [int(source)], 'target': [int(target)], 'distance': [-1]})
        rows.append(row[['source', 'target', 'distance']])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['source', 'target', 'distance'])


def _run_single(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    engine: str,
    sources: Sequence[int],
    targets: Sequence[int],
    max_hops: int,
    direction: str,
    shortest_path_backend: str,
    cypher_strategy: str,
    include_native_direct: bool,
    runs: int,
    warmup: int,
) -> Dict[str, Any]:
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    persons_pd = nodes_df[nodes_df['node_type'] == 'Person'][['node_id']].copy()
    persons_pd['id'] = persons_pd['node_id'].astype('int64')
    persons_pd['label__Person'] = True
    expected, expected_times = _timed(lambda: _expected_distances(follows_pd, sources, targets, max_hops), runs, warmup)
    reusable_adj = _adjacency(follows_pd)
    expected_reuse, expected_reuse_times = _timed(
        lambda: _expected_distances_from_adj(reusable_adj, sources, targets, max_hops),
        runs,
        warmup,
    )
    assert_frame_equal(expected_reuse, expected, check_dtype=False)

    if engine == 'cudf':
        nodes = _maybe_to_cudf('cudf', persons_pd)
        follows = _maybe_to_cudf('cudf', follows_pd.rename(columns={'src': 's', 'dst': 'd'}))
    else:
        nodes = persons_pd
        follows = follows_pd.rename(columns={'src': 's', 'dst': 'd'})

    result, cypher_times = _timed(
        lambda: _run_cypher_batch(
            nodes, follows, engine, sources, targets, max_hops, direction, shortest_path_backend, cypher_strategy
        ),
        runs,
        warmup,
    )
    actual = _normalize_result(result)
    assert_frame_equal(actual, expected, check_dtype=False)
    direct_native = (
        _time_direct_native(
            follows_pd,
            engine,
            sources,
            targets,
            max_hops,
            direction,
            shortest_path_backend,
            expected,
            runs,
            warmup,
        )
        if include_native_direct
        else None
    )
    return {
        'engine': engine,
        'shortest_path_backend': shortest_path_backend,
        'cypher_strategy': cypher_strategy,
        'direction': direction,
        'max_hops': max_hops,
        'sources': [int(v) for v in sources],
        'targets': [int(v) for v in targets],
        'pair_count': int(len(sources) * len(targets)),
        'cypher_median_ms': _median(cypher_times),
        'dataframe_bfs_median_ms': _median(expected_times),
        'dataframe_bfs_reuse_median_ms': _median(expected_reuse_times),
        'direct_native': direct_native,
        'cypher_runs_ms': cypher_times,
        'dataframe_bfs_runs_ms': expected_times,
        'dataframe_bfs_reuse_runs_ms': expected_reuse_times,
        'rows': actual.to_dict(orient='records'),
    }


def _parse_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == '':
        return None
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def run_probe(
    root: Path,
    engine: str,
    max_hops: int,
    source_count: int,
    target_count: int,
    source_ids: Optional[Sequence[int]],
    target_ids: Optional[Sequence[int]],
    direction: str,
    shortest_path_backend: str,
    cypher_strategy: str,
    include_native_direct: bool,
    runs: int,
    warmup: int,
) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(root / 'data' / 'output' / 'nodes')
    edges_df = _load_edges(root / 'data' / 'output' / 'edges', offsets)
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    sources, targets = _select_sources_targets(follows_pd, source_count, target_count, max_hops, source_ids, target_ids)
    engines = ['pandas', 'cudf'] if engine == 'both' else [engine]
    results = [
        _run_single(
            nodes_df,
            edges_df,
            selected_engine,
            sources,
            targets,
            max_hops,
            direction,
            shortest_path_backend,
            cypher_strategy,
            include_native_direct,
            runs,
            warmup,
        )
        for selected_engine in engines
    ]
    return {
        'graph_benchmark_root': str(root),
        'runs': runs,
        'warmup': warmup,
        'max_hops': max_hops,
        'source_count': len(sources),
        'target_count': len(targets),
        'direction': direction,
        'shortest_path_backend': shortest_path_backend,
        'cypher_strategy': cypher_strategy,
        'include_native_direct': include_native_direct,
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--engine', choices=['pandas', 'cudf', 'both'], default='pandas')
    parser.add_argument('--max-hops', type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument('--source-count', type=int, default=DEFAULT_SOURCE_COUNT)
    parser.add_argument('--target-count', type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument('--source-ids', default=None, help='Comma-separated source node_ids; overrides source-count')
    parser.add_argument('--target-ids', default=None, help='Comma-separated target node_ids; overrides target-count')
    parser.add_argument('--direction', choices=['forward', 'reverse', 'undirected'], default='forward')
    parser.add_argument('--shortest-path-backend', choices=['auto', 'bfs', 'igraph', 'cugraph'], default='auto')
    parser.add_argument('--cypher-strategy', choices=['loop', 'batched', 'batched-prefiltered'], default='loop')
    parser.add_argument('--include-native-direct', action='store_true', help='Probe direct native helper timing outside GFQL/Cypher row lowering.')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()

    if args.max_hops < 1:
        raise ValueError('--max-hops must be >= 1')
    if args.source_count < 1 or args.target_count < 1:
        raise ValueError('--source-count and --target-count must be >= 1')

    output = run_probe(
        args.graph_benchmark_root,
        args.engine,
        args.max_hops,
        args.source_count,
        args.target_count,
        _parse_ids(args.source_ids),
        _parse_ids(args.target_ids),
        args.direction,
        args.shortest_path_backend,
        args.cypher_strategy,
        args.include_native_direct,
        args.runs,
        args.warmup,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
