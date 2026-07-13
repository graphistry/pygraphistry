#!/usr/bin/env python3
"""LDBC-style analytical probes over graph-benchmark data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandas.testing import assert_frame_equal

from benchmarks.gfql.graph_benchmark_q1_q9 import _edges_by_rel, _load_edges, _load_nodes, _maybe_to_cudf


DEFAULT_ROOT = Path('/tmp/graph-benchmark-gfql-memgraph')
WORKLOADS = [
    'branch_semijoin_count',
    'interest_city_topk',
    'age_interest_state_top1',
    'country_interest_follow_topk',
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
    return pd.DataFrame(df)


def _scalar_frame(name: str, value: Any) -> pd.DataFrame:
    if hasattr(value, 'item'):
        value = value.item()
    return pd.DataFrame({name: [int(value)]})


def _normalize(workload: str, result: Any) -> pd.DataFrame:
    out = _to_pandas(result).copy()
    if workload == 'branch_semijoin_count':
        col = 'numPersons'
        if out.empty:
            return pd.DataFrame({col: [0]}, dtype='int64')
        out = out[[col]].copy()
        out[col] = out[col].fillna(0).astype('int64')
        return out.reset_index(drop=True)
    if workload == 'interest_city_topk':
        cols = ['city', 'country', 'numPersons']
        if out.empty:
            return pd.DataFrame({col: [] for col in cols}).astype({'numPersons': 'int64'})
        out = out[cols].copy()
        out['numPersons'] = out['numPersons'].astype('int64')
        return out.sort_values(['numPersons', 'city', 'country'], ascending=[False, True, True]).head(5).reset_index(drop=True)
    if workload == 'age_interest_state_top1':
        cols = ['state', 'country', 'numPersons']
        if out.empty:
            return pd.DataFrame({col: [] for col in cols}).astype({'numPersons': 'int64'})
        out = out[cols].copy()
        out['numPersons'] = out['numPersons'].astype('int64')
        return out.sort_values(['numPersons', 'state', 'country'], ascending=[False, True, True]).head(1).reset_index(drop=True)
    if workload == 'country_interest_follow_topk':
        cols = ['node_id', 'numFollowersFromSeed']
        if out.empty:
            return pd.DataFrame({col: [] for col in cols}).astype({'node_id': 'int64', 'numFollowersFromSeed': 'int64'})
        out = out[cols].copy()
        out['node_id'] = out['node_id'].astype('int64')
        out['numFollowersFromSeed'] = out['numFollowersFromSeed'].astype('int64')
        return out.sort_values(['numFollowersFromSeed', 'node_id'], ascending=[False, True]).head(10).reset_index(drop=True)
    raise ValueError(f'Unsupported workload: {workload}')


def _frames_for_engine(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, engine: str) -> Dict[str, Any]:
    persons = nodes_df[nodes_df['node_type'] == 'Person'][['node_id', 'gender_lc', 'age']]
    cities = nodes_df[nodes_df['node_type'] == 'City'][['node_id', 'city', 'country']]
    states = nodes_df[nodes_df['node_type'] == 'State'][['node_id', 'state', 'country']]
    countries = nodes_df[nodes_df['node_type'] == 'Country'][['node_id', 'country']]
    interests = nodes_df[nodes_df['node_type'] == 'Interest'][['node_id', 'interest_lc']]
    follows = _edges_by_rel(edges_df, 'FOLLOWS')[['src', 'dst']]
    lives = _edges_by_rel(edges_df, 'LIVES_IN')[['src', 'dst']]
    interested = _edges_by_rel(edges_df, 'HAS_INTEREST')[['src', 'dst']]
    city_in = _edges_by_rel(edges_df, 'CITY_IN')[['src', 'dst']]
    state_in = _edges_by_rel(edges_df, 'STATE_IN')[['src', 'dst']]
    frames = {
        'persons': persons,
        'cities': cities,
        'states': states,
        'countries': countries,
        'interests': interests,
        'follows': follows,
        'lives': lives,
        'interested': interested,
        'city_in': city_in,
        'state_in': state_in,
    }
    if engine == 'polars':
        import polars as pl  # type: ignore

        return {name: pl.from_pandas(frame) for name, frame in frames.items()}
    if engine == 'cudf':
        return {name: _maybe_to_cudf('cudf', frame) for name, frame in frames.items()}
    return frames


def _eager_branch_semijoin_count(frames: Dict[str, Any], gender: str, city: str, country: str, interest: str) -> pd.DataFrame:
    people = frames['persons'][frames['persons']['gender_lc'] == gender][['node_id']]
    interest_nodes = frames['interests'][frames['interests']['interest_lc'] == interest][['node_id']]
    city_nodes = frames['cities'][(frames['cities']['city'] == city) & (frames['cities']['country'] == country)][['node_id']]
    interest_people = frames['interested'][frames['interested']['dst'].isin(interest_nodes['node_id'])]
    interest_people = interest_people[interest_people['src'].isin(people['node_id'])][['src']].drop_duplicates()
    location_people = frames['lives'][frames['lives']['dst'].isin(city_nodes['node_id'])][['src']].drop_duplicates()
    return _scalar_frame('numPersons', len(interest_people[interest_people['src'].isin(location_people['src'])]))


def _eager_interest_city_topk(frames: Dict[str, Any], gender: str, interest: str) -> Any:
    people = frames['persons'][frames['persons']['gender_lc'] == gender][['node_id']]
    interest_nodes = frames['interests'][frames['interests']['interest_lc'] == interest][['node_id']]
    interest_people = frames['interested'][frames['interested']['dst'].isin(interest_nodes['node_id'])]
    interest_people = interest_people[interest_people['src'].isin(people['node_id'])][['src']].drop_duplicates()
    matched = frames['lives'][frames['lives']['src'].isin(interest_people['src'])]
    grouped = matched.groupby('dst').size().reset_index(name='numPersons')
    result = grouped.merge(frames['cities'], left_on='dst', right_on='node_id')
    return result[['city', 'country', 'numPersons']].sort_values(['numPersons', 'city', 'country'], ascending=[False, True, True]).head(5)


def _eager_age_interest_state_top1(frames: Dict[str, Any], country: str, age_lower: int, age_upper: int, interest: str) -> Any:
    people = frames['persons'][(frames['persons']['age'] >= age_lower) & (frames['persons']['age'] <= age_upper)][['node_id']]
    interest_nodes = frames['interests'][frames['interests']['interest_lc'] == interest][['node_id']]
    interest_people = frames['interested'][frames['interested']['dst'].isin(interest_nodes['node_id'])]
    interest_people = interest_people[interest_people['src'].isin(people['node_id'])][['src']].drop_duplicates()
    states = frames['states'][frames['states']['country'] == country][['node_id', 'state', 'country']]
    matched_lives = frames['lives'][frames['lives']['src'].isin(interest_people['src'])]
    path = matched_lives.merge(frames['city_in'], left_on='dst', right_on='src', suffixes=('_person', '_city'))
    grouped = path.groupby('dst_city').size().reset_index(name='numPersons')
    result = grouped.merge(states, left_on='dst_city', right_on='node_id')
    return result[['state', 'country', 'numPersons']].sort_values(['numPersons', 'state', 'country'], ascending=[False, True, True]).head(1)


def _eager_country_interest_follow_topk(frames: Dict[str, Any], country: str, interest: str) -> Any:
    country_nodes = frames['countries'][frames['countries']['country'] == country][['node_id']]
    state_ids = frames['state_in'][frames['state_in']['dst'].isin(country_nodes['node_id'])][['src']].rename(columns={'src': 'state_id'})
    city_ids = frames['city_in'][frames['city_in']['dst'].isin(state_ids['state_id'])][['src']].rename(columns={'src': 'city_id'})
    location_people = frames['lives'][frames['lives']['dst'].isin(city_ids['city_id'])][['src']].drop_duplicates()
    interest_nodes = frames['interests'][frames['interests']['interest_lc'] == interest][['node_id']]
    interest_people = frames['interested'][frames['interested']['dst'].isin(interest_nodes['node_id'])][['src']].drop_duplicates()
    seed_people = location_people[location_people['src'].isin(interest_people['src'])]
    followed = frames['follows'][frames['follows']['src'].isin(seed_people['src'])]
    grouped = followed.groupby('dst').size().reset_index(name='numFollowersFromSeed').rename(columns={'dst': 'node_id'})
    return grouped.sort_values(['numFollowersFromSeed', 'node_id'], ascending=[False, True]).head(10)


def _polars_branch_semijoin_count(frames: Dict[str, Any], gender: str, city: str, country: str, interest: str) -> Any:
    import polars as pl  # type: ignore

    people = frames['persons'].filter(pl.col('gender_lc') == gender).select('node_id')
    interest_nodes = frames['interests'].filter(pl.col('interest_lc') == interest).select('node_id')
    city_nodes = frames['cities'].filter((pl.col('city') == city) & (pl.col('country') == country)).select('node_id')
    interest_people = (
        frames['interested']
        .join(interest_nodes, left_on='dst', right_on='node_id', how='semi')
        .join(people, left_on='src', right_on='node_id', how='semi')
        .select('src')
        .unique()
    )
    location_people = frames['lives'].join(city_nodes, left_on='dst', right_on='node_id', how='semi').select('src').unique()
    return interest_people.join(location_people, on='src', how='semi').select(pl.len().alias('numPersons'))


def _polars_interest_city_topk(frames: Dict[str, Any], gender: str, interest: str) -> Any:
    import polars as pl  # type: ignore

    people = frames['persons'].filter(pl.col('gender_lc') == gender).select('node_id')
    interest_nodes = frames['interests'].filter(pl.col('interest_lc') == interest).select('node_id')
    interest_people = (
        frames['interested']
        .join(interest_nodes, left_on='dst', right_on='node_id', how='semi')
        .join(people, left_on='src', right_on='node_id', how='semi')
        .select('src')
        .unique()
    )
    return (
        frames['lives']
        .join(interest_people, on='src', how='semi')
        .group_by('dst')
        .len(name='numPersons')
        .join(frames['cities'], left_on='dst', right_on='node_id', how='inner')
        .select(['city', 'country', 'numPersons'])
        .sort(['numPersons', 'city', 'country'], descending=[True, False, False])
        .head(5)
    )


def _polars_age_interest_state_top1(frames: Dict[str, Any], country: str, age_lower: int, age_upper: int, interest: str) -> Any:
    import polars as pl  # type: ignore

    people = frames['persons'].filter((pl.col('age') >= age_lower) & (pl.col('age') <= age_upper)).select('node_id')
    interest_nodes = frames['interests'].filter(pl.col('interest_lc') == interest).select('node_id')
    interest_people = (
        frames['interested']
        .join(interest_nodes, left_on='dst', right_on='node_id', how='semi')
        .join(people, left_on='src', right_on='node_id', how='semi')
        .select('src')
        .unique()
    )
    states = frames['states'].filter(pl.col('country') == country).select(['node_id', 'state', 'country'])
    return (
        frames['lives']
        .join(interest_people, on='src', how='semi')
        .join(frames['city_in'], left_on='dst', right_on='src', how='inner', suffix='_city')
        .group_by('dst_city')
        .len(name='numPersons')
        .join(states, left_on='dst_city', right_on='node_id', how='inner')
        .select(['state', 'country', 'numPersons'])
        .sort(['numPersons', 'state', 'country'], descending=[True, False, False])
        .head(1)
    )


def _polars_country_interest_follow_topk(frames: Dict[str, Any], country: str, interest: str) -> Any:
    import polars as pl  # type: ignore

    country_nodes = frames['countries'].filter(pl.col('country') == country).select('node_id')
    state_ids = frames['state_in'].join(country_nodes, left_on='dst', right_on='node_id', how='semi').select(pl.col('src').alias('state_id'))
    city_ids = frames['city_in'].join(state_ids, left_on='dst', right_on='state_id', how='semi').select(pl.col('src').alias('city_id'))
    location_people = frames['lives'].join(city_ids, left_on='dst', right_on='city_id', how='semi').select('src').unique()
    interest_nodes = frames['interests'].filter(pl.col('interest_lc') == interest).select('node_id')
    interest_people = frames['interested'].join(interest_nodes, left_on='dst', right_on='node_id', how='semi').select('src').unique()
    seed_people = location_people.join(interest_people, on='src', how='semi')
    return (
        frames['follows']
        .join(seed_people, on='src', how='semi')
        .group_by('dst')
        .len(name='numFollowersFromSeed')
        .rename({'dst': 'node_id'})
        .sort(['numFollowersFromSeed', 'node_id'], descending=[True, False])
        .head(10)
    )


def _run_workload(engine: str, workload: str, frames: Dict[str, Any], args: argparse.Namespace) -> Any:
    if engine == 'polars':
        return {
            'branch_semijoin_count': lambda: _polars_branch_semijoin_count(frames, args.gender_q5, args.city_q5, args.country_q5, args.interest_q5),
            'interest_city_topk': lambda: _polars_interest_city_topk(frames, args.gender_q6, args.interest_q6),
            'age_interest_state_top1': lambda: _polars_age_interest_state_top1(frames, args.country_q7, args.age_lower_q7, args.age_upper_q7, args.interest_q7),
            'country_interest_follow_topk': lambda: _polars_country_interest_follow_topk(frames, args.country_follow, args.interest_follow),
        }[workload]()
    return {
        'branch_semijoin_count': lambda: _eager_branch_semijoin_count(frames, args.gender_q5, args.city_q5, args.country_q5, args.interest_q5),
        'interest_city_topk': lambda: _eager_interest_city_topk(frames, args.gender_q6, args.interest_q6),
        'age_interest_state_top1': lambda: _eager_age_interest_state_top1(frames, args.country_q7, args.age_lower_q7, args.age_upper_q7, args.interest_q7),
        'country_interest_follow_topk': lambda: _eager_country_interest_follow_topk(frames, args.country_follow, args.interest_follow),
    }[workload]()


def _run_engine(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    engine: str,
    workloads: Sequence[str],
    runs: int,
    warmup: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        frames = _frames_for_engine(nodes_df, edges_df, engine)
    except Exception as exc:
        return {'engine': engine, 'available': False, 'error': f'{type(exc).__name__}: {exc}', 'workloads': []}
    expected_frames = _frames_for_engine(nodes_df, edges_df, 'pandas')

    results: List[Dict[str, Any]] = []
    for workload in workloads:
        expected = _normalize(workload, _run_workload('pandas', workload, expected_frames, args))
        result, times = _timed(lambda workload=workload: _run_workload(engine, workload, frames, args), runs, warmup)
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


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(args.graph_benchmark_root / 'data' / 'output' / 'nodes')
    edges_df = _load_edges(args.graph_benchmark_root / 'data' / 'output' / 'edges', offsets)
    engines = ['pandas', 'cudf', 'polars'] if args.engine == 'all' else [args.engine]
    workloads = _parse_workloads(args.workloads)
    return {
        'graph_benchmark_root': str(args.graph_benchmark_root),
        'runs': args.runs,
        'warmup': args.warmup,
        'workloads': workloads,
        'parameters': {
            'q5': {
                'gender': args.gender_q5,
                'city': args.city_q5,
                'country': args.country_q5,
                'interest': args.interest_q5,
            },
            'q6': {'gender': args.gender_q6, 'interest': args.interest_q6},
            'q7': {
                'country': args.country_q7,
                'age_lower': args.age_lower_q7,
                'age_upper': args.age_upper_q7,
                'interest': args.interest_q7,
            },
            'follow': {'country': args.country_follow, 'interest': args.interest_follow},
        },
        'results': [
            _run_engine(nodes_df, edges_df, selected_engine, workloads, args.runs, args.warmup, args)
            for selected_engine in engines
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-benchmark-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--engine', choices=['pandas', 'cudf', 'polars', 'all'], default='pandas')
    parser.add_argument('--workloads', default='all', help='Comma-separated workloads or all')
    parser.add_argument('--gender-q5', default='male')
    parser.add_argument('--city-q5', default='London')
    parser.add_argument('--country-q5', default='United Kingdom')
    parser.add_argument('--interest-q5', default='fine dining')
    parser.add_argument('--gender-q6', default='female')
    parser.add_argument('--interest-q6', default='tennis')
    parser.add_argument('--country-q7', default='United States')
    parser.add_argument('--age-lower-q7', type=int, default=23)
    parser.add_argument('--age-upper-q7', type=int, default=30)
    parser.add_argument('--interest-q7', default='photography')
    parser.add_argument('--country-follow', default='United States')
    parser.add_argument('--interest-follow', default='photography')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output-json', type=Path, default=None)
    args = parser.parse_args()

    output = run_probe(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
