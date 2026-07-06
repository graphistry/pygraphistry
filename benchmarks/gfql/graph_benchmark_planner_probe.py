#!/usr/bin/env python3
"""Planner-style microbench probes over graph-benchmark data."""
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
WORKLOADS = [
    'starts_with',
    'or_filter',
    'indexed_order_by',
    'parallel_counting',
    'bfs_expand_from_source',
]


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
    if hasattr(df, 'to_pandas_df'):
        return df.to_pandas_df()
    return pd.DataFrame(df)


def _scalar_frame(name: str, value: Any) -> pd.DataFrame:
    if hasattr(value, 'item'):
        value = value.item()
    return pd.DataFrame({name: [int(value)]})


def _normalize(workload: str, result: Any) -> pd.DataFrame:
    out = _to_pandas(result).copy()
    if workload in {'starts_with', 'or_filter', 'parallel_counting', 'bfs_expand_from_source'}:
        col = {
            'starts_with': 'personCount',
            'or_filter': 'personCount',
            'parallel_counting': 'followsCount',
            'bfs_expand_from_source': 'endpointCount',
        }[workload]
        if out.empty:
            return pd.DataFrame({col: [0]}, dtype='int64')
        out = out[[col]].copy()
        out[col] = out[col].fillna(0).astype('int64')
        return out.reset_index(drop=True)
    if workload == 'indexed_order_by':
        if out.empty:
            return pd.DataFrame({'node_id': [], 'numFollowers': []}).astype({'node_id': 'int64', 'numFollowers': 'int64'})
        out = out[['node_id', 'numFollowers']].copy()
        out['node_id'] = out['node_id'].astype('int64')
        out['numFollowers'] = out['numFollowers'].astype('int64')
        return out.sort_values(['numFollowers', 'node_id'], ascending=[False, True]).head(10).reset_index(drop=True)
    raise ValueError(f'Unsupported workload: {workload}')


def _pandas_starts_with(persons: Any, prefix: str) -> pd.DataFrame:
    return _scalar_frame('personCount', persons['name'].astype(str).str.startswith(prefix).sum())


def _pandas_or_filter(persons: Any, age_upper: int, gender: str) -> pd.DataFrame:
    mask = (persons['age'] <= age_upper) | (persons['gender_lc'] == gender)
    return _scalar_frame('personCount', mask.sum())


def _pandas_indexed_order_by(follows: Any) -> pd.DataFrame:
    counts = follows.groupby('dst').size().reset_index(name='numFollowers').rename(columns={'dst': 'node_id'})
    return counts.sort_values(['numFollowers', 'node_id'], ascending=[False, True]).head(10).reset_index(drop=True)


def _pandas_parallel_counting(follows: Any) -> pd.DataFrame:
    return _scalar_frame('followsCount', len(follows))


def _pandas_bfs_expand_from_source(follows: Any, source: int) -> pd.DataFrame:
    first = follows[follows['src'] == source][['dst']].rename(columns={'dst': 'mid'})
    second = first.merge(follows, left_on='mid', right_on='src')[['dst']].drop_duplicates()
    return _scalar_frame('endpointCount', len(second))


def _polars_starts_with(persons: Any, prefix: str) -> Any:
    import polars as pl  # type: ignore

    return persons.select(pl.col('name').cast(pl.Utf8).str.starts_with(prefix).sum().alias('personCount'))


def _polars_or_filter(persons: Any, age_upper: int, gender: str) -> Any:
    import polars as pl  # type: ignore

    return persons.select(((pl.col('age') <= age_upper) | (pl.col('gender_lc') == gender)).sum().alias('personCount'))


def _polars_indexed_order_by(follows: Any) -> Any:
    import polars as pl  # type: ignore

    return (
        follows.group_by('dst')
        .len(name='numFollowers')
        .rename({'dst': 'node_id'})
        .sort(['numFollowers', 'node_id'], descending=[True, False])
        .head(10)
    )


def _polars_parallel_counting(follows: Any) -> Any:
    import polars as pl  # type: ignore

    return pl.DataFrame({'followsCount': [follows.height]})


def _polars_bfs_expand_from_source(follows: Any, source: int) -> Any:
    import polars as pl  # type: ignore

    first = follows.filter(pl.col('src') == source).select(pl.col('dst').alias('mid'))
    second = first.join(follows, left_on='mid', right_on='src', how='inner').select('dst').unique()
    return pl.DataFrame({'endpointCount': [second.height]})


def _choose_source(follows_pd: pd.DataFrame) -> int:
    counts = follows_pd.groupby('src').size().reset_index(name='outDegree')
    top = counts.sort_values(['outDegree', 'src'], ascending=[False, True]).head(1)
    return int(top['src'].iloc[0])


def _frames_for_engine(nodes_df: pd.DataFrame, follows_pd: pd.DataFrame, engine: str) -> Tuple[Any, Any]:
    persons_pd = nodes_df[nodes_df['node_type'] == 'Person'].copy()
    if engine == 'polars':
        import polars as pl  # type: ignore

        return pl.from_pandas(persons_pd), pl.from_pandas(follows_pd)
    if engine == 'cudf':
        return _maybe_to_cudf('cudf', persons_pd), _maybe_to_cudf('cudf', follows_pd)
    return persons_pd, follows_pd


def _run_workload(
    engine: str,
    workload: str,
    persons: Any,
    follows: Any,
    prefix: str,
    age_upper: int,
    gender: str,
    source: int,
) -> Any:
    if engine == 'polars':
        return {
            'starts_with': lambda: _polars_starts_with(persons, prefix),
            'or_filter': lambda: _polars_or_filter(persons, age_upper, gender),
            'indexed_order_by': lambda: _polars_indexed_order_by(follows),
            'parallel_counting': lambda: _polars_parallel_counting(follows),
            'bfs_expand_from_source': lambda: _polars_bfs_expand_from_source(follows, source),
        }[workload]()
    return {
        'starts_with': lambda: _pandas_starts_with(persons, prefix),
        'or_filter': lambda: _pandas_or_filter(persons, age_upper, gender),
        'indexed_order_by': lambda: _pandas_indexed_order_by(follows),
        'parallel_counting': lambda: _pandas_parallel_counting(follows),
        'bfs_expand_from_source': lambda: _pandas_bfs_expand_from_source(follows, source),
    }[workload]()


def _run_engine(
    nodes_df: pd.DataFrame,
    follows_pd: pd.DataFrame,
    engine: str,
    workloads: Sequence[str],
    runs: int,
    warmup: int,
    prefix: str,
    age_upper: int,
    gender: str,
    source: int,
) -> Dict[str, Any]:
    try:
        persons, follows = _frames_for_engine(nodes_df, follows_pd, engine)
    except Exception as exc:
        return {'engine': engine, 'available': False, 'error': f'{type(exc).__name__}: {exc}', 'workloads': []}

    expected_persons, expected_follows = _frames_for_engine(nodes_df, follows_pd, 'pandas')
    results: List[Dict[str, Any]] = []
    for workload in workloads:
        expected = _normalize(
            workload,
            _run_workload('pandas', workload, expected_persons, expected_follows, prefix, age_upper, gender, source),
        )
        result, times = _timed(
            lambda: _run_workload(engine, workload, persons, follows, prefix, age_upper, gender, source),
            runs,
            warmup,
        )
        actual = _normalize(workload, result)
        assert_frame_equal(actual, expected, check_dtype=False)
        results.append({
            'workload': workload,
            'median_ms': _median(times),
            'runs_ms': times,
            'rows': actual.to_dict(orient='records'),
        })
    return {'engine': engine, 'available': True, 'error': None, 'workloads': results}


def _parse_workloads(value: str) -> List[str]:
    if value == 'all':
        return list(WORKLOADS)
    workloads = [part.strip().replace('-', '_') for part in value.split(',') if part.strip()]
    unknown = sorted(set(workloads) - set(WORKLOADS))
    if unknown:
        raise ValueError(f'Unknown workloads: {unknown}; expected {WORKLOADS}')
    return workloads


def run_probe(
    root: Path,
    engine: str,
    workloads: Sequence[str],
    runs: int,
    warmup: int,
    prefix: str,
    age_upper: int,
    gender: str,
) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(root / 'data' / 'output' / 'nodes')
    edges_df = _load_edges(root / 'data' / 'output' / 'edges', offsets)
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    source = _choose_source(follows_pd)
    engines = ['pandas', 'cudf', 'polars'] if engine == 'all' else [engine]
    return {
        'graph_benchmark_root': str(root),
        'runs': runs,
        'warmup': warmup,
        'workloads': list(workloads),
        'prefix': prefix,
        'age_upper': age_upper,
        'gender': gender,
        'source': source,
        'results': [
            _run_engine(nodes_df, follows_pd, selected_engine, workloads, runs, warmup, prefix, age_upper, gender, source)
            for selected_engine in engines
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--engine', choices=['pandas', 'cudf', 'polars', 'all'], default='pandas')
    parser.add_argument('--workloads', default='all', help='Comma-separated workloads or all')
    parser.add_argument('--prefix', default='A')
    parser.add_argument('--age-upper', type=int, default=30)
    parser.add_argument('--gender', default='female')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()
    workloads = _parse_workloads(args.workloads)
    output = run_probe(
        args.graph_benchmark_root,
        args.engine,
        workloads,
        args.runs,
        args.warmup,
        args.prefix,
        args.age_upper,
        args.gender,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
