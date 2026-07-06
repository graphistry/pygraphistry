#!/usr/bin/env python3
"""Pokec-style exact-hop expansion/filter probe over graph-benchmark FOLLOWS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandas.testing import assert_frame_equal

from benchmarks.gfql.graph_benchmark_q1_q9 import _edges_by_rel, _load_edges, _load_nodes, _maybe_to_cudf


DEFAULT_ROOT = Path('/tmp/graph-benchmark-gfql-memgraph')
CPU_POLICY_FOLLOWS_MIN_ROWS = 1_000_000
DEFAULT_FILTER_AGE_LOWER = 18
DEFAULT_DEPTHS = [1, 2, 3, 4]


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


def _normalize(result: Any) -> pd.DataFrame:
    out = _to_pandas(result)[['node_id']].copy()
    return out.sort_values('node_id').reset_index(drop=True)


def _select_seed(edges_df: pd.DataFrame, seed_node_id: Optional[int]) -> int:
    if seed_node_id is not None:
        return int(seed_node_id)
    follows = edges_df[edges_df['rel'] == 'FOLLOWS']
    counts = follows.groupby('src').size().reset_index(name='outDegree')
    top = counts.sort_values(['outDegree', 'src'], ascending=[False, True]).head(1)
    return int(top['src'].iloc[0])


def _prepare_frames(nodes: Any, edges: Any) -> Tuple[Any, Any]:
    persons = nodes[nodes['node_type'] == 'Person'][['node_id', 'age']]
    follows = _edges_by_rel(edges, 'FOLLOWS')[['src', 'dst']]
    return persons, follows


def _prepare_frames_polars(nodes: Any, edges: Any) -> Tuple[Any, Any]:
    import polars as pl  # type: ignore

    persons = nodes.filter(pl.col('node_type') == 'Person').select(['node_id', 'age'])
    follows = edges.filter(pl.col('rel') == 'FOLLOWS').select(['src', 'dst'])
    return persons, follows


def _path_join_expansion(nodes: Any, edges: Any, seed_node_id: int, depth: int, filter_age_lower: Optional[int]) -> Any:
    persons, follows = _prepare_frames(nodes, edges)
    endpoints = follows[follows['src'] == seed_node_id][['dst']].rename(columns={'dst': 'node_id'})
    for _ in range(2, depth + 1):
        endpoints = endpoints.merge(follows, left_on='node_id', right_on='src')[['dst']].rename(columns={'dst': 'node_id'})
    endpoints = endpoints.drop_duplicates()
    if filter_age_lower is not None:
        targets = persons[persons['age'] >= filter_age_lower][['node_id']]
        endpoints = endpoints.merge(targets, on='node_id')
    return endpoints.sort_values('node_id').reset_index(drop=True)


def _frontier_dedup_expansion(nodes: Any, edges: Any, seed_node_id: int, depth: int, filter_age_lower: Optional[int]) -> Any:
    persons, follows = _prepare_frames(nodes, edges)
    frontier = follows[follows['src'] == seed_node_id][['dst']].drop_duplicates().rename(columns={'dst': 'node_id'})
    for _ in range(2, depth + 1):
        frontier = (
            frontier.merge(follows, left_on='node_id', right_on='src')[['dst']]
            .drop_duplicates()
            .rename(columns={'dst': 'node_id'})
        )
    if filter_age_lower is not None:
        targets = persons[persons['age'] >= filter_age_lower][['node_id']]
        frontier = frontier.merge(targets, on='node_id')
    return frontier.sort_values('node_id').reset_index(drop=True)


def _path_join_expansion_polars(
    nodes: Any,
    edges: Any,
    seed_node_id: int,
    depth: int,
    filter_age_lower: Optional[int],
) -> Any:
    import polars as pl  # type: ignore

    persons, follows = _prepare_frames_polars(nodes, edges)
    endpoints = follows.filter(pl.col('src') == seed_node_id).select(pl.col('dst').alias('node_id'))
    for _ in range(2, depth + 1):
        endpoints = endpoints.join(follows, left_on='node_id', right_on='src', how='inner').select(
            pl.col('dst').alias('node_id')
        )
    endpoints = endpoints.unique()
    if filter_age_lower is not None:
        targets = persons.filter(pl.col('age') >= filter_age_lower).select('node_id')
        endpoints = endpoints.join(targets, on='node_id', how='inner')
    return endpoints.sort('node_id')


def _frontier_dedup_expansion_polars(
    nodes: Any,
    edges: Any,
    seed_node_id: int,
    depth: int,
    filter_age_lower: Optional[int],
) -> Any:
    import polars as pl  # type: ignore

    persons, follows = _prepare_frames_polars(nodes, edges)
    frontier = follows.filter(pl.col('src') == seed_node_id).select(pl.col('dst').alias('node_id')).unique()
    for _ in range(2, depth + 1):
        frontier = (
            frontier.join(follows, left_on='node_id', right_on='src', how='inner')
            .select(pl.col('dst').alias('node_id'))
            .unique()
        )
    if filter_age_lower is not None:
        targets = persons.filter(pl.col('age') >= filter_age_lower).select('node_id')
        frontier = frontier.join(targets, on='node_id', how='inner')
    return frontier.sort('node_id')


def _cpu_policy_engine(edges_df: pd.DataFrame) -> Tuple[str, int]:
    follows_rows = int((edges_df['rel'] == 'FOLLOWS').sum())
    if follows_rows >= CPU_POLICY_FOLLOWS_MIN_ROWS:
        return 'polars', follows_rows
    return 'pandas', follows_rows


def _preview(result: Any) -> List[int]:
    out = _to_pandas(result)
    return [int(v) for v in out['node_id'].head(10).tolist()]


def _run_single(
    nodes: Any,
    edges: Any,
    engine: str,
    seed_node_id: int,
    depth: int,
    filter_age_lower: Optional[int],
    strategy: str,
    runs: int,
    warmup: int,
) -> Dict[str, Any]:
    if engine == 'polars':
        path_fn = _path_join_expansion_polars
        frontier_fn = _frontier_dedup_expansion_polars
    else:
        path_fn = _path_join_expansion
        frontier_fn = _frontier_dedup_expansion

    selected_strategy = 'frontier_dedup' if strategy == 'strategy-policy' else strategy
    path_result: Optional[Any] = None
    frontier_result: Optional[Any] = None
    path_times: List[float] = []
    frontier_times: List[float] = []

    if selected_strategy in {'both', 'path_join'}:
        path_result, path_times = _timed(
            lambda: path_fn(nodes, edges, seed_node_id, depth, filter_age_lower), runs, warmup
        )
    if selected_strategy in {'both', 'frontier_dedup'}:
        frontier_result, frontier_times = _timed(
            lambda: frontier_fn(nodes, edges, seed_node_id, depth, filter_age_lower), runs, warmup
        )

    if selected_strategy == 'both':
        assert path_result is not None
        assert frontier_result is not None
        assert_frame_equal(_normalize(path_result), _normalize(frontier_result), check_dtype=False)

    measured = {
        'path_join': _median(path_times) if path_times else None,
        'frontier_dedup': _median(frontier_times) if frontier_times else None,
    }
    present = {k: v for k, v in measured.items() if v is not None}
    best_strategy = min(present, key=present.get)
    best_ms = present[best_strategy]
    best_result = path_result if best_strategy == 'path_join' else frontier_result
    assert best_result is not None
    count = len(best_result)

    return {
        'depth': depth,
        'filter_age_lower': filter_age_lower,
        'strategy': strategy,
        'selected_strategy': selected_strategy,
        'path_join_median_ms': measured['path_join'],
        'frontier_dedup_median_ms': measured['frontier_dedup'],
        'path_join_runs_ms': path_times,
        'frontier_dedup_runs_ms': frontier_times,
        'best_strategy': best_strategy,
        'best_strategy_median_ms': best_ms,
        'count': int(count),
        'preview_node_ids': _preview(best_result),
    }


def run_probe(
    root: Path,
    engine: str,
    depths: Sequence[int],
    workload: str,
    strategy: str,
    seed_node_id: Optional[int],
    filter_age_lower: int,
    runs: int,
    warmup: int,
) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(root / 'data' / 'output' / 'nodes')
    edges_df = _load_edges(root / 'data' / 'output' / 'edges', offsets)
    selected_engine = engine
    follows_rows = int((edges_df['rel'] == 'FOLLOWS').sum())
    if engine == 'cpu-policy':
        selected_engine, follows_rows = _cpu_policy_engine(edges_df)
    seed = _select_seed(edges_df, seed_node_id)

    if selected_engine == 'polars':
        import polars as pl  # type: ignore

        nodes = pl.from_pandas(nodes_df)
        edges = pl.from_pandas(edges_df)
    else:
        nodes = _maybe_to_cudf(selected_engine, nodes_df)
        edges = _maybe_to_cudf(selected_engine, edges_df)

    workloads: List[Tuple[str, Optional[int]]] = []
    if workload in {'plain', 'all'}:
        workloads.append(('expansion', None))
    if workload in {'filtered', 'all'}:
        workloads.append(('expansion_with_filter', filter_age_lower))

    results: Dict[str, Any] = {}
    for depth in depths:
        for suffix, age_filter in workloads:
            key = f'{suffix}_{depth}'
            results[key] = _run_single(
                nodes,
                edges,
                selected_engine,
                seed,
                depth,
                age_filter,
                strategy,
                runs,
                warmup,
            )
    return {
        'engine': engine,
        'selected_engine': selected_engine,
        'follows_rows': follows_rows,
        'cpu_policy_follows_min_rows': CPU_POLICY_FOLLOWS_MIN_ROWS,
        'seed_node_id': seed,
        'workload': workload,
        'strategy': strategy,
        'results': results,
    }


def _parse_depths(text: str) -> List[int]:
    depths = [int(part.strip()) for part in text.split(',') if part.strip()]
    invalid = [depth for depth in depths if depth < 1 or depth > 5]
    if invalid:
        raise argparse.ArgumentTypeError(f'depths must be in 1..5, got {invalid}')
    return depths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--engine', choices=['pandas', 'cudf', 'polars', 'cpu-policy', 'both', 'all'], default='both')
    parser.add_argument('--depths', type=_parse_depths, default=DEFAULT_DEPTHS)
    parser.add_argument('--workload', choices=['plain', 'filtered', 'all'], default='all')
    parser.add_argument('--strategy', choices=['both', 'path-join', 'frontier-dedup', 'strategy-policy'], default='strategy-policy')
    parser.add_argument('--seed-node-id', type=int, default=None)
    parser.add_argument('--filter-age-lower', type=int, default=DEFAULT_FILTER_AGE_LOWER)
    parser.add_argument('--runs', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()

    if args.engine == 'both':
        engines = ['pandas', 'cudf']
    elif args.engine == 'all':
        engines = ['pandas', 'cudf', 'polars']
    else:
        engines = [args.engine]
    strategy = {'path-join': 'path_join', 'frontier-dedup': 'frontier_dedup'}.get(args.strategy, args.strategy)

    payload = {
        'graph_benchmark_root': str(args.graph_benchmark_root),
        'runs': args.runs,
        'warmup': args.warmup,
        'depths': args.depths,
        'workload': args.workload,
        'filter_age_lower': args.filter_age_lower,
        'strategy': args.strategy,
        'source': 'Pokec-style expansion_1..4 and expansion_*_with_filter over graph-benchmark FOLLOWS',
        'results': {},
    }
    print(
        f'root={args.graph_benchmark_root} depths={args.depths} workload={args.workload} '
        f'strategy={args.strategy} runs={args.runs} warmup={args.warmup}'
    )
    for engine in engines:
        result = run_probe(
            args.graph_benchmark_root,
            engine,
            args.depths,
            args.workload,
            strategy,
            args.seed_node_id,
            args.filter_age_lower,
            args.runs,
            args.warmup,
        )
        payload['results'][engine] = result
        selected = result['selected_engine']
        policy_suffix = f' selected={selected}' if selected != engine else ''
        for key, row in result['results'].items():
            path_ms = row['path_join_median_ms']
            frontier_ms = row['frontier_dedup_median_ms']
            path_text = f'{path_ms:.3f}ms' if path_ms is not None else 'skipped'
            frontier_text = f'{frontier_ms:.3f}ms' if frontier_ms is not None else 'skipped'
            parity_text = 'parity=pass' if row['selected_strategy'] == 'both' else 'parity=not-run'
            print(
                f'engine={engine}{policy_suffix} {key} selected_strategy={row["selected_strategy"]} '
                f'path_join={path_text} frontier_dedup={frontier_text} '
                f'best={row["best_strategy"]} {row["best_strategy_median_ms"]:.3f}ms '
                f'count={row["count"]} {parity_text}'
            )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
