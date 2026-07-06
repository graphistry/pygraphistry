#!/usr/bin/env python3
# Pokec-style cyclic pattern probes over graph-benchmark FOLLOWS.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandas.testing import assert_frame_equal

import graphistry
from benchmarks.gfql.graph_benchmark_q1_q9 import _edges_by_rel, _load_edges, _load_nodes, _maybe_to_cudf


DEFAULT_ROOT = Path('/tmp/graph-benchmark-gfql-memgraph')


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


def _normalize_cycle(df: Any) -> pd.DataFrame:
    out = _to_pandas(df).copy()
    if out.empty:
        return pd.DataFrame({'mid': []}).astype({'mid': 'int64'})
    out = out[['mid']].drop_duplicates().sort_values('mid').reset_index(drop=True)
    out['mid'] = out['mid'].astype('int64')
    return out


def _normalize_triangle(df: Any) -> pd.DataFrame:
    out = _to_pandas(df).copy()
    if out.empty:
        return pd.DataFrame({'seed': [], 'a': [], 'b': []}).astype({'seed': 'int64', 'a': 'int64', 'b': 'int64'})
    out = out[['seed', 'a', 'b']].drop_duplicates().sort_values(['seed', 'a', 'b']).reset_index(drop=True)
    for col in ['seed', 'a', 'b']:
        out[col] = out[col].astype('int64')
    return out


def _normalize_pattern_long(df: Any) -> pd.DataFrame:
    out = _to_pandas(df).copy()
    if out.empty:
        return pd.DataFrame({'n5': []}).astype({'n5': 'int64'})
    out = out[['n5']].drop_duplicates().sort_values('n5').head(1).reset_index(drop=True)
    out['n5'] = out['n5'].astype('int64')
    return out


def _parse_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == '':
        return None
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def _select_cycle_seed(follows: pd.DataFrame, seed_node_id: Optional[int]) -> int:
    if seed_node_id is not None:
        return int(seed_node_id)
    outgoing = follows[['src', 'dst']].rename(columns={'dst': 'mid'})
    incoming = follows[['src', 'dst']].rename(columns={'dst': 'seed', 'src': 'mid'})
    cycles = outgoing.merge(incoming, left_on=['src', 'mid'], right_on=['seed', 'mid'])
    if len(cycles) > 0:
        cycle_counts = cycles.groupby('seed').size().reset_index(name='cycleCount')
        out_counts = follows.groupby('src').size().reset_index(name='outDegree').rename(columns={'src': 'seed'})
        ranked = cycle_counts.merge(out_counts, on='seed', how='left')
        top = ranked.sort_values(['cycleCount', 'outDegree', 'seed'], ascending=[False, False, True]).head(1)
        return int(top['seed'].iloc[0])
    counts = follows.groupby('src').size().reset_index(name='outDegree')
    top = counts.sort_values(['outDegree', 'src'], ascending=[False, True]).head(1)
    return int(top['src'].iloc[0])


def _select_triangle_seeds(
    follows: pd.DataFrame,
    seed_ids: Optional[Sequence[int]],
    seed_count: int,
    seed_candidate_count: int,
) -> List[int]:
    if seed_ids:
        return [int(v) for v in seed_ids]
    counts = follows.groupby('src').size().reset_index(name='outDegree')
    candidate_count = max(seed_count, seed_candidate_count)
    candidates = counts.sort_values(['outDegree', 'src'], ascending=[False, True]).head(candidate_count)
    candidate_ids = [int(v) for v in candidates['src'].tolist()]
    if candidate_ids:
        seed_edges = follows[follows['src'].isin(candidate_ids)][['src', 'dst']].rename(
            columns={'src': 'seed', 'dst': 'a'}
        )
        twohop = seed_edges.merge(follows, left_on='a', right_on='src')[['seed', 'a', 'dst']].rename(
            columns={'dst': 'b'}
        )
        closing = follows[follows['dst'].isin(candidate_ids)][['src', 'dst']].rename(
            columns={'src': 'b', 'dst': 'seed'}
        )
        triangles = twohop.merge(closing, on=['seed', 'b'])
        if len(triangles) > 0:
            tri_counts = triangles.groupby('seed').size().reset_index(name='triangleCount')
            ranked = tri_counts.merge(candidates.rename(columns={'src': 'seed'}), on='seed', how='left')
            top = ranked.sort_values(['triangleCount', 'outDegree', 'seed'], ascending=[False, False, True]).head(seed_count)
            return [int(v) for v in top['seed'].tolist()]
    return candidate_ids[:seed_count]


def _binary_join_cycle(follows: Any, seed: int) -> Any:
    outgoing = follows[follows['src'] == seed][['dst']].rename(columns={'dst': 'mid'})
    incoming = follows[follows['dst'] == seed][['src']].rename(columns={'src': 'mid'})
    return outgoing.merge(incoming, on='mid')[['mid']].drop_duplicates().sort_values('mid').reset_index(drop=True)


def _binary_join_cycle_polars(follows: Any, seed: int) -> Any:
    import polars as pl  # type: ignore

    outgoing = follows.filter(pl.col('src') == seed).select(pl.col('dst').alias('mid'))
    incoming = follows.filter(pl.col('dst') == seed).select(pl.col('src').alias('mid'))
    return outgoing.join(incoming, on='mid', how='inner').unique().sort('mid')


def _adjacency_intersection_cycle(follows_pd: pd.DataFrame, seed: int) -> pd.DataFrame:
    outgoing: Dict[int, Set[int]] = defaultdict(set)
    incoming: Dict[int, Set[int]] = defaultdict(set)
    for src, dst in follows_pd[['src', 'dst']].itertuples(index=False):
        src_i = int(src)
        dst_i = int(dst)
        outgoing[src_i].add(dst_i)
        incoming[dst_i].add(src_i)
    mids = sorted(outgoing.get(seed, set()).intersection(incoming.get(seed, set())))
    return pd.DataFrame({'mid': mids}, dtype='int64')


def _cypher_cycle(nodes: Any, follows: Any, engine: str, seed: int) -> Any:
    g = graphistry.nodes(nodes, 'id').edges(follows, 's', 'd')
    query = (
        f"MATCH (n {{id: {int(seed)}}})-[e1]->(m)-[e2]->(n) "
        "RETURN m.id AS mid ORDER BY mid"
    )
    result = g.gfql(query, engine=engine)
    return result._nodes


def _binary_join_triangle(follows: Any, seeds: Sequence[int]) -> Any:
    seed_edges = follows[follows['src'].isin(seeds)][['src', 'dst']].rename(columns={'src': 'seed', 'dst': 'a'})
    twohop = seed_edges.merge(follows, left_on='a', right_on='src')[['seed', 'a', 'dst']].rename(columns={'dst': 'b'})
    closing = follows[follows['dst'].isin(seeds)][['src', 'dst']].rename(columns={'src': 'b', 'dst': 'seed'})
    return twohop.merge(closing, on=['seed', 'b'])[['seed', 'a', 'b']].drop_duplicates().sort_values(['seed', 'a', 'b']).reset_index(drop=True)


def _binary_join_triangle_polars(follows: Any, seeds: Sequence[int]) -> Any:
    import polars as pl  # type: ignore

    seed_edges = follows.filter(pl.col('src').is_in(list(seeds))).select([
        pl.col('src').alias('seed'),
        pl.col('dst').alias('a'),
    ])
    twohop = seed_edges.join(follows, left_on='a', right_on='src', how='inner').select([
        'seed',
        'a',
        pl.col('dst').alias('b'),
    ])
    closing = follows.filter(pl.col('dst').is_in(list(seeds))).select([
        pl.col('src').alias('b'),
        pl.col('dst').alias('seed'),
    ])
    return twohop.join(closing, on=['seed', 'b'], how='inner').select(['seed', 'a', 'b']).unique().sort(['seed', 'a', 'b'])


def _adjacency_intersection_triangle(follows_pd: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    outgoing: Dict[int, Set[int]] = defaultdict(set)
    incoming: Dict[int, Set[int]] = defaultdict(set)
    for src, dst in follows_pd[['src', 'dst']].itertuples(index=False):
        src_i = int(src)
        dst_i = int(dst)
        outgoing[src_i].add(dst_i)
        incoming[dst_i].add(src_i)
    rows: List[Dict[str, int]] = []
    for seed in seeds:
        seed_i = int(seed)
        closers = incoming.get(seed_i, set())
        for a in outgoing.get(seed_i, set()):
            for b in sorted(outgoing.get(a, set()).intersection(closers)):
                rows.append({'seed': seed_i, 'a': int(a), 'b': int(b)})
    return pd.DataFrame(rows, columns=['seed', 'a', 'b']).drop_duplicates().sort_values(['seed', 'a', 'b']).reset_index(drop=True)


def _cypher_triangle(nodes: Any, follows: Any, engine: str, seeds: Sequence[int]) -> Any:
    g = graphistry.nodes(nodes, 'id').edges(follows, 's', 'd')
    seed_list = '[' + ', '.join(str(int(v)) for v in seeds) + ']'
    query = (
        "MATCH (n)-[e1]->(a)-[e2]->(b)-[e3]->(n) "
        f"WHERE n.id IN {seed_list} "
        "RETURN n.id AS seed, a.id AS a, b.id AS b ORDER BY seed, a, b"
    )
    result = g.gfql(query, engine=engine)
    return result._nodes


def _binary_join_pattern_long(follows: Any) -> Any:
    e1 = follows[['src', 'dst']].rename(columns={'src': 'n1', 'dst': 'n2'})
    e2 = follows[['src', 'dst']].rename(columns={'src': 'n2', 'dst': 'n3'})
    e3 = follows[['src', 'dst']].rename(columns={'src': 'n3', 'dst': 'n4'})
    e4 = follows[['src', 'dst']].rename(columns={'src': 'n5', 'dst': 'n4'})
    path = e1.merge(e2, on='n2')
    path = path.merge(e3, on='n3')
    path = path.merge(e4, on='n4')
    return path[['n5']].drop_duplicates().sort_values('n5').head(1).reset_index(drop=True)


def _binary_join_pattern_long_polars(follows: Any) -> Any:
    import polars as pl  # type: ignore

    e1 = follows.select([pl.col('src').alias('n1'), pl.col('dst').alias('n2')])
    e2 = follows.select([pl.col('src').alias('n2'), pl.col('dst').alias('n3')])
    e3 = follows.select([pl.col('src').alias('n3'), pl.col('dst').alias('n4')])
    e4 = follows.select([pl.col('src').alias('n5'), pl.col('dst').alias('n4')])
    return (
        e1.join(e2, on='n2', how='inner')
        .join(e3, on='n3', how='inner')
        .join(e4, on='n4', how='inner')
        .select('n5')
        .unique()
        .sort('n5')
        .head(1)
    )


def _cypher_pattern_long(nodes: Any, follows: Any, engine: str) -> Any:
    g = graphistry.nodes(nodes, 'id').edges(follows, 's', 'd')
    query = (
        "MATCH (n1)-[e1]->(n2)-[e2]->(n3)-[e3]->(n4)<-[e4]-(n5) "
        "RETURN n5.id AS n5 ORDER BY n5 LIMIT 1"
    )
    result = g.gfql(query, engine=engine)
    return result._nodes


def _frames_for_engine(nodes_df: pd.DataFrame, follows_pd: pd.DataFrame, engine: str) -> Tuple[Any, Any, Any, Any]:
    person_nodes = nodes_df[nodes_df['node_type'] == 'Person'][['node_id']].rename(columns={'node_id': 'id'})
    cypher_edges = follows_pd.rename(columns={'src': 's', 'dst': 'd'})
    if engine == 'polars':
        import polars as pl  # type: ignore

        return pl.from_pandas(person_nodes), pl.from_pandas(follows_pd), pl.from_pandas(cypher_edges), follows_pd
    if engine == 'cudf':
        return _maybe_to_cudf('cudf', person_nodes), _maybe_to_cudf('cudf', follows_pd), _maybe_to_cudf('cudf', cypher_edges), follows_pd
    return person_nodes, follows_pd, cypher_edges, follows_pd


def _run_cycle_single(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, engine: str, seed: int, strategy: str, runs: int, warmup: int) -> Dict[str, Any]:
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    expected, expected_times = _timed(lambda: _adjacency_intersection_cycle(follows_pd, seed), runs, warmup)
    nodes, follows, follows_cypher, _ = _frames_for_engine(nodes_df, follows_pd, engine)
    binary_times: List[float] = []
    cypher_times: List[float] = []
    if strategy in {'both', 'binary_join'}:
        binary_fn = _binary_join_cycle_polars if engine == 'polars' else _binary_join_cycle
        binary_result, binary_times = _timed(lambda: binary_fn(follows, seed), runs, warmup)
        assert_frame_equal(_normalize_cycle(binary_result), _normalize_cycle(expected), check_dtype=False)
    if strategy in {'both', 'cypher'} and engine != 'polars':
        cypher_result, cypher_times = _timed(lambda: _cypher_cycle(nodes, follows_cypher, engine, seed), runs, warmup)
        assert_frame_equal(_normalize_cycle(cypher_result), _normalize_cycle(expected), check_dtype=False)
    medians = {
        'adjacency_intersection': _median(expected_times),
        'binary_join': _median(binary_times) if binary_times else None,
        'cypher': _median(cypher_times) if cypher_times else None,
    }
    present = {k: v for k, v in medians.items() if v is not None}
    best_strategy = min(present, key=present.get)
    expected_norm = _normalize_cycle(expected)
    return {
        'engine': engine,
        'seed_node_id': int(seed),
        'seed_count': 1,
        'strategy': strategy,
        'count': int(len(expected_norm)),
        'preview_rows': expected_norm.head(10).to_dict(orient='records'),
        'adjacency_intersection_median_ms': medians['adjacency_intersection'],
        'binary_join_median_ms': medians['binary_join'],
        'cypher_median_ms': medians['cypher'],
        'adjacency_intersection_runs_ms': expected_times,
        'binary_join_runs_ms': binary_times,
        'cypher_runs_ms': cypher_times,
        'best_strategy': best_strategy,
        'best_strategy_median_ms': present[best_strategy],
        'cypher_supported': medians['cypher'] is not None,
    }


def _run_triangle_single(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, engine: str, seeds: Sequence[int], strategy: str, runs: int, warmup: int) -> Dict[str, Any]:
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    expected, expected_times = _timed(lambda: _adjacency_intersection_triangle(follows_pd, seeds), runs, warmup)
    nodes, follows, follows_cypher, _ = _frames_for_engine(nodes_df, follows_pd, engine)
    binary_times: List[float] = []
    cypher_times: List[float] = []
    cypher_error: Optional[str] = None
    if strategy in {'both', 'binary_join'}:
        binary_fn = _binary_join_triangle_polars if engine == 'polars' else _binary_join_triangle
        binary_result, binary_times = _timed(lambda: binary_fn(follows, seeds), runs, warmup)
        assert_frame_equal(_normalize_triangle(binary_result), _normalize_triangle(expected), check_dtype=False)
    if strategy in {'both', 'cypher'} and engine != 'polars':
        try:
            cypher_result, cypher_times = _timed(lambda: _cypher_triangle(nodes, follows_cypher, engine, seeds), runs, warmup)
            assert_frame_equal(_normalize_triangle(cypher_result), _normalize_triangle(expected), check_dtype=False)
        except Exception as exc:
            cypher_error = f'{type(exc).__name__}: {exc}'
            if strategy == 'cypher':
                raise
    medians = {
        'adjacency_intersection': _median(expected_times),
        'binary_join': _median(binary_times) if binary_times else None,
        'cypher': _median(cypher_times) if cypher_times else None,
    }
    present = {k: v for k, v in medians.items() if v is not None}
    best_strategy = min(present, key=present.get)
    expected_norm = _normalize_triangle(expected)
    return {
        'engine': engine,
        'seed_node_ids': [int(v) for v in seeds],
        'seed_count': int(len(seeds)),
        'strategy': strategy,
        'count': int(len(expected_norm)),
        'preview_rows': expected_norm.head(10).to_dict(orient='records'),
        'adjacency_intersection_median_ms': medians['adjacency_intersection'],
        'binary_join_median_ms': medians['binary_join'],
        'cypher_median_ms': medians['cypher'],
        'adjacency_intersection_runs_ms': expected_times,
        'binary_join_runs_ms': binary_times,
        'cypher_runs_ms': cypher_times,
        'best_strategy': best_strategy,
        'best_strategy_median_ms': present[best_strategy],
        'cypher_supported': medians['cypher'] is not None,
        'cypher_error': cypher_error,
    }


def _run_pattern_long_single(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    engine: str,
    strategy: str,
    runs: int,
    warmup: int,
    max_reference_edges: int,
) -> Dict[str, Any]:
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    if max_reference_edges > 0 and len(follows_pd) > max_reference_edges:
        raise ValueError(
            f'pattern-long reference would run over {len(follows_pd)} FOLLOWS edges; '
            f'limit is {max_reference_edges}. Increase --pattern-long-max-reference-edges '
            'only for an intentional guarded run.'
        )
    expected_fn = _binary_join_pattern_long
    expected, expected_times = _timed(lambda: expected_fn(follows_pd), runs, warmup)
    nodes, follows, follows_cypher, _ = _frames_for_engine(nodes_df, follows_pd, engine)
    binary_times: List[float] = []
    cypher_times: List[float] = []
    cypher_error: Optional[str] = None
    if strategy in {'both', 'binary_join'}:
        binary_fn = _binary_join_pattern_long_polars if engine == 'polars' else _binary_join_pattern_long
        binary_result, binary_times = _timed(lambda: binary_fn(follows), runs, warmup)
        assert_frame_equal(_normalize_pattern_long(binary_result), _normalize_pattern_long(expected), check_dtype=False)
    if strategy in {'both', 'cypher'} and engine != 'polars':
        try:
            cypher_result, cypher_times = _timed(lambda: _cypher_pattern_long(nodes, follows_cypher, engine), runs, warmup)
            assert_frame_equal(_normalize_pattern_long(cypher_result), _normalize_pattern_long(expected), check_dtype=False)
        except Exception as exc:
            cypher_error = f'{type(exc).__name__}: {exc}'
            if strategy == 'cypher':
                raise
    medians = {
        'binary_join_reference': _median(expected_times),
        'binary_join': _median(binary_times) if binary_times else None,
        'cypher': _median(cypher_times) if cypher_times else None,
    }
    present = {k: v for k, v in medians.items() if v is not None}
    best_strategy = min(present, key=present.get)
    expected_norm = _normalize_pattern_long(expected)
    return {
        'engine': engine,
        'strategy': strategy,
        'count': int(len(expected_norm)),
        'preview_rows': expected_norm.head(10).to_dict(orient='records'),
        'binary_join_reference_median_ms': medians['binary_join_reference'],
        'binary_join_median_ms': medians['binary_join'],
        'cypher_median_ms': medians['cypher'],
        'binary_join_reference_runs_ms': expected_times,
        'binary_join_runs_ms': binary_times,
        'cypher_runs_ms': cypher_times,
        'best_strategy': best_strategy,
        'best_strategy_median_ms': present[best_strategy],
        'cypher_supported': medians['cypher'] is not None,
        'cypher_error': cypher_error,
    }


def run_probe(
    root: Path,
    engine: str,
    workload: str,
    seed_node_id: Optional[int],
    seed_ids: Optional[Sequence[int]],
    seed_count: int,
    seed_candidate_count: int,
    strategy: str,
    runs: int,
    warmup: int,
    pattern_long_max_reference_edges: int,
) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(root / 'data' / 'output' / 'nodes')
    edges_df = _load_edges(root / 'data' / 'output' / 'edges', offsets)
    follows_pd = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    engines = ['pandas', 'cudf', 'polars'] if engine == 'all' else [engine]
    if workload == 'cycle_2':
        seed = _select_cycle_seed(follows_pd, seed_node_id)
        results = [_run_cycle_single(nodes_df, edges_df, selected_engine, seed, strategy, runs, warmup) for selected_engine in engines]
        selected_seeds = [seed]
    elif workload == 'directed_triangle':
        selected_seeds = _select_triangle_seeds(follows_pd, seed_ids, seed_count, seed_candidate_count)
        results = [_run_triangle_single(nodes_df, edges_df, selected_engine, selected_seeds, strategy, runs, warmup) for selected_engine in engines]
    elif workload == 'pattern_long':
        selected_seeds = []
        results = [
            _run_pattern_long_single(
                nodes_df,
                edges_df,
                selected_engine,
                strategy,
                runs,
                warmup,
                pattern_long_max_reference_edges,
            )
            for selected_engine in engines
        ]
    else:
        raise ValueError(f'Unsupported workload: {workload}')
    return {
        'graph_benchmark_root': str(root),
        'workload': workload,
        'runs': runs,
        'warmup': warmup,
        'seed_node_ids': [int(v) for v in selected_seeds],
        'strategy': strategy,
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--engine', choices=['pandas', 'cudf', 'polars', 'all'], default='pandas')
    parser.add_argument('--workload', choices=['cycle-2', 'directed-triangle', 'pattern-long'], default='cycle-2')
    parser.add_argument('--seed-node-id', type=int, default=None)
    parser.add_argument('--seed-ids', default=None, help='Comma-separated seed node_ids for directed-triangle')
    parser.add_argument('--seed-count', type=int, default=3)
    parser.add_argument('--seed-candidate-count', type=int, default=100)
    parser.add_argument('--strategy', choices=['both', 'binary-join', 'cypher'], default='both')
    parser.add_argument(
        '--pattern-long-max-reference-edges',
        type=int,
        default=100_000,
        help='Safety cap for pattern-long dataframe reference over FOLLOWS edges; set 0 to disable.',
    )
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()
    if args.seed_count < 1:
        raise ValueError('--seed-count must be >= 1')
    if args.seed_candidate_count < 1:
        raise ValueError('--seed-candidate-count must be >= 1')
    strategy = args.strategy.replace('-', '_')
    workload = args.workload.replace('-', '_')
    output = run_probe(
        args.graph_benchmark_root,
        args.engine,
        workload,
        args.seed_node_id,
        _parse_ids(args.seed_ids),
        args.seed_count,
        args.seed_candidate_count,
        strategy,
        args.runs,
        args.warmup,
        args.pattern_long_max_reference_edges,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
