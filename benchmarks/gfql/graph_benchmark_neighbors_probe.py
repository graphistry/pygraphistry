#!/usr/bin/env python3
"""Benchgraph-like neighbors-with-data/filter probe over graph-benchmark FOLLOWS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from pandas.testing import assert_frame_equal

from benchmarks.gfql.graph_benchmark_q1_q9 import _edges_by_rel, _load_edges, _load_nodes, _maybe_to_cudf


DEFAULT_ROOT = Path("/tmp/graph-benchmark-gfql-memgraph")

SOURCE_GENDER = "female"
SOURCE_AGE_LOWER = 30
SOURCE_AGE_UPPER = 40
TARGET_AGE_LOWER = 50
CPU_POLICY_FOLLOWS_MIN_ROWS = 1_000_000
STRATEGY_POLICY_FOLLOWS_MIN_ROWS = 1_000_000


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
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df)


def _normalize(df: Any) -> pd.DataFrame:
    out = _to_pandas(df)[["city", "country", "pathCount"]].copy()
    return out.sort_values(["pathCount", "city", "country"], ascending=[False, True, True]).head(10).reset_index(drop=True)


def _group_size(df: Any, by: List[str], name: str) -> Any:
    return df.groupby(by).size().reset_index(name=name)


def _prepare_frames(nodes: Any, edges: Any) -> Tuple[Any, Any, Any, Any]:
    persons = nodes[nodes["node_type"] == "Person"][["node_id", "age", "gender_lc"]]
    cities = nodes[nodes["node_type"] == "City"][["node_id", "city", "country"]]
    follows = _edges_by_rel(edges, "FOLLOWS")[["src", "dst"]]
    lives_in = _edges_by_rel(edges, "LIVES_IN")[["src", "dst"]]
    return persons, cities, follows, lives_in


def _prepare_frames_polars(nodes: Any, edges: Any) -> Tuple[Any, Any, Any, Any]:
    import polars as pl  # type: ignore

    persons = nodes.filter(pl.col("node_type") == "Person").select(["node_id", "age", "gender_lc"])
    cities = nodes.filter(pl.col("node_type") == "City").select(["node_id", "city", "country"])
    follows = edges.filter(pl.col("rel") == "FOLLOWS").select(["src", "dst"])
    lives_in = edges.filter(pl.col("rel") == "LIVES_IN").select(["src", "dst"])
    return persons, cities, follows, lives_in


def _prefix_paths(follows: Any, sources: Any, depth: int) -> Any:
    prefix = follows.merge(sources, left_on="src", right_on="node_id")[["dst"]]
    prefix = prefix.rename(columns={"dst": "mid"})
    for _ in range(2, depth):
        prefix = prefix.merge(follows, left_on="mid", right_on="src", suffixes=("", "_next"))[["dst"]]
        prefix = prefix.rename(columns={"dst": "mid"})
    return prefix


def _target_city_suffix(follows: Any, lives_in: Any, cities: Any, targets: Any) -> Any:
    target_city = lives_in.merge(targets, left_on="src", right_on="node_id")[["src", "dst"]]
    target_city = target_city.merge(cities, left_on="dst", right_on="node_id")[["src", "city", "country"]]
    suffix = follows.merge(target_city, left_on="dst", right_on="src", suffixes=("", "_target"))
    return suffix[["src", "city", "country"]].rename(columns={"src": "mid"})


def _path_join_strategy(nodes: Any, edges: Any, depth: int) -> Any:
    persons, cities, follows, lives_in = _prepare_frames(nodes, edges)
    sources = persons[
        (persons["gender_lc"] == SOURCE_GENDER)
        & (persons["age"] >= SOURCE_AGE_LOWER)
        & (persons["age"] <= SOURCE_AGE_UPPER)
    ][["node_id"]]
    targets = persons[persons["age"] > TARGET_AGE_LOWER][["node_id"]]

    prefix = _prefix_paths(follows, sources, depth)
    suffix = _target_city_suffix(follows, lives_in, cities, targets)

    paths = prefix.merge(suffix, on="mid")
    result = _group_size(paths, ["city", "country"], "pathCount")
    return result.sort_values(["pathCount", "city", "country"], ascending=[False, True, True]).head(10)


def _prefix_paths_polars(follows: Any, sources: Any, depth: int) -> Any:
    import polars as pl  # type: ignore

    prefix = follows.join(sources, left_on="src", right_on="node_id", how="inner").select(
        pl.col("dst").alias("mid")
    )
    for _ in range(2, depth):
        prefix = prefix.join(follows, left_on="mid", right_on="src", how="inner").select(
            pl.col("dst").alias("mid")
        )
    return prefix


def _target_city_suffix_polars(follows: Any, lives_in: Any, cities: Any, targets: Any) -> Any:
    import polars as pl  # type: ignore

    target_city = (
        lives_in.join(targets, left_on="src", right_on="node_id", how="inner")
        .select(["src", "dst"])
        .join(cities, left_on="dst", right_on="node_id", how="inner")
        .select(["src", "city", "country"])
    )
    return follows.join(target_city, left_on="dst", right_on="src", how="inner").select(
        pl.col("src").alias("mid"),
        "city",
        "country",
    )


def _path_join_strategy_polars(nodes: Any, edges: Any, depth: int) -> Any:
    import polars as pl  # type: ignore

    persons, cities, follows, lives_in = _prepare_frames_polars(nodes, edges)
    sources = persons.filter(
        (pl.col("gender_lc") == SOURCE_GENDER)
        & (pl.col("age") >= SOURCE_AGE_LOWER)
        & (pl.col("age") <= SOURCE_AGE_UPPER)
    ).select("node_id")
    targets = persons.filter(pl.col("age") > TARGET_AGE_LOWER).select("node_id")

    prefix = _prefix_paths_polars(follows, sources, depth)
    suffix = _target_city_suffix_polars(follows, lives_in, cities, targets)

    return (
        prefix.join(suffix, on="mid", how="inner")
        .group_by(["city", "country"])
        .len()
        .rename({"len": "pathCount"})
        .sort(["pathCount", "city", "country"], descending=[True, False, False])
        .head(10)
    )


def _preaggregated_strategy(nodes: Any, edges: Any, depth: int) -> Any:
    persons, cities, follows, lives_in = _prepare_frames(nodes, edges)
    sources = persons[
        (persons["gender_lc"] == SOURCE_GENDER)
        & (persons["age"] >= SOURCE_AGE_LOWER)
        & (persons["age"] <= SOURCE_AGE_UPPER)
    ][["node_id"]]
    targets = persons[persons["age"] > TARGET_AGE_LOWER][["node_id"]]

    prefix = _prefix_paths(follows, sources, depth)
    suffix = _target_city_suffix(follows, lives_in, cities, targets)
    by_mid = _group_size(suffix, ["mid", "city", "country"], "pathCount")

    joined = prefix.merge(by_mid, on="mid")
    result = joined.groupby(["city", "country"])["pathCount"].sum().reset_index()
    return result.sort_values(["pathCount", "city", "country"], ascending=[False, True, True]).head(10)


def _preaggregated_strategy_polars(nodes: Any, edges: Any, depth: int) -> Any:
    import polars as pl  # type: ignore

    persons, cities, follows, lives_in = _prepare_frames_polars(nodes, edges)
    sources = persons.filter(
        (pl.col("gender_lc") == SOURCE_GENDER)
        & (pl.col("age") >= SOURCE_AGE_LOWER)
        & (pl.col("age") <= SOURCE_AGE_UPPER)
    ).select("node_id")
    targets = persons.filter(pl.col("age") > TARGET_AGE_LOWER).select("node_id")

    prefix = _prefix_paths_polars(follows, sources, depth)
    suffix = _target_city_suffix_polars(follows, lives_in, cities, targets)
    by_mid = suffix.group_by(["mid", "city", "country"]).len().rename({"len": "pathCount"})

    return (
        prefix.join(by_mid, on="mid", how="inner")
        .group_by(["city", "country"])
        .agg(pl.col("pathCount").sum())
        .sort(["pathCount", "city", "country"], descending=[True, False, False])
        .head(10)
    )


def _prefix_path_counts(follows: Any, sources: Any, depth: int) -> Any:
    counts = _group_size(
        follows.merge(sources, left_on="src", right_on="node_id")[["dst"]].rename(columns={"dst": "mid"}),
        ["mid"],
        "prefixCount",
    )
    for _ in range(2, depth):
        expanded = counts.merge(follows, left_on="mid", right_on="src")[["dst", "prefixCount"]]
        expanded = expanded.rename(columns={"dst": "mid"})
        counts = expanded.groupby("mid")["prefixCount"].sum().reset_index()
    return counts


def _prefix_path_counts_polars(follows: Any, sources: Any, depth: int) -> Any:
    import polars as pl  # type: ignore

    counts = (
        follows.join(sources, left_on="src", right_on="node_id", how="inner")
        .select(pl.col("dst").alias("mid"))
        .group_by("mid")
        .len()
        .rename({"len": "prefixCount"})
    )
    for _ in range(2, depth):
        counts = (
            counts.join(follows, left_on="mid", right_on="src", how="inner")
            .select(pl.col("dst").alias("mid"), "prefixCount")
            .group_by("mid")
            .agg(pl.col("prefixCount").sum())
        )
    return counts


def _factorized_preaggregated_strategy(nodes: Any, edges: Any, depth: int) -> Any:
    persons, cities, follows, lives_in = _prepare_frames(nodes, edges)
    sources = persons[
        (persons["gender_lc"] == SOURCE_GENDER)
        & (persons["age"] >= SOURCE_AGE_LOWER)
        & (persons["age"] <= SOURCE_AGE_UPPER)
    ][["node_id"]]
    targets = persons[persons["age"] > TARGET_AGE_LOWER][["node_id"]]

    prefix_counts = _prefix_path_counts(follows, sources, depth)
    suffix = _target_city_suffix(follows, lives_in, cities, targets)
    suffix_counts = _group_size(suffix, ["mid", "city", "country"], "suffixCount")

    joined = prefix_counts.merge(suffix_counts, on="mid")
    joined["pathCount"] = joined["prefixCount"] * joined["suffixCount"]
    result = joined.groupby(["city", "country"])["pathCount"].sum().reset_index()
    return result.sort_values(["pathCount", "city", "country"], ascending=[False, True, True]).head(10)


def _factorized_preaggregated_strategy_polars(nodes: Any, edges: Any, depth: int) -> Any:
    import polars as pl  # type: ignore

    persons, cities, follows, lives_in = _prepare_frames_polars(nodes, edges)
    sources = persons.filter(
        (pl.col("gender_lc") == SOURCE_GENDER)
        & (pl.col("age") >= SOURCE_AGE_LOWER)
        & (pl.col("age") <= SOURCE_AGE_UPPER)
    ).select("node_id")
    targets = persons.filter(pl.col("age") > TARGET_AGE_LOWER).select("node_id")

    prefix_counts = _prefix_path_counts_polars(follows, sources, depth)
    suffix = _target_city_suffix_polars(follows, lives_in, cities, targets)
    suffix_counts = suffix.group_by(["mid", "city", "country"]).len().rename({"len": "suffixCount"})

    return (
        prefix_counts.join(suffix_counts, on="mid", how="inner")
        .with_columns((pl.col("prefixCount") * pl.col("suffixCount")).alias("pathCount"))
        .group_by(["city", "country"])
        .agg(pl.col("pathCount").sum())
        .sort(["pathCount", "city", "country"], descending=[True, False, False])
        .head(10)
    )


def _cpu_policy_engine(edges_df: pd.DataFrame) -> Tuple[str, int]:
    follows_rows = int((edges_df["rel"] == "FOLLOWS").sum())
    if follows_rows >= CPU_POLICY_FOLLOWS_MIN_ROWS:
        return "polars", follows_rows
    return "pandas", follows_rows


def _strategy_policy(depth: int, follows_rows: int, selected_engine: str) -> str:
    if follows_rows >= STRATEGY_POLICY_FOLLOWS_MIN_ROWS and depth >= 3:
        return "factorized_preaggregated"
    return "path_join"


def run_probe(root: Path, engine: str, runs: int, warmup: int, depth: int, strategy: str) -> Dict[str, Any]:
    nodes_df, offsets = _load_nodes(root / "data" / "output" / "nodes")
    edges_df = _load_edges(root / "data" / "output" / "edges", offsets)
    selected_engine = engine
    follows_rows = int((edges_df["rel"] == "FOLLOWS").sum())
    if engine == "cpu-policy":
        selected_engine, follows_rows = _cpu_policy_engine(edges_df)

    if selected_engine == "polars":
        import polars as pl  # type: ignore

        nodes = pl.from_pandas(nodes_df)
        edges = pl.from_pandas(edges_df)
        path_fn = _path_join_strategy_polars
        preagg_fn = _preaggregated_strategy_polars
        factorized_preagg_fn = _factorized_preaggregated_strategy_polars
    else:
        nodes = _maybe_to_cudf(selected_engine, nodes_df)
        edges = _maybe_to_cudf(selected_engine, edges_df)
        path_fn = _path_join_strategy
        preagg_fn = _preaggregated_strategy
        factorized_preagg_fn = _factorized_preaggregated_strategy

    selected_strategy = _strategy_policy(depth, follows_rows, selected_engine) if strategy == "strategy-policy" else strategy
    path_result: Optional[Any] = None
    preagg_result: Optional[Any] = None
    factorized_preagg_result: Optional[Any] = None
    path_times: List[float] = []
    preagg_times: List[float] = []
    factorized_preagg_times: List[float] = []

    if selected_strategy in {"both", "path_join"}:
        path_result, path_times = _timed(lambda: path_fn(nodes, edges, depth), runs, warmup)
    if selected_strategy in {"both", "preaggregated"}:
        preagg_result, preagg_times = _timed(lambda: preagg_fn(nodes, edges, depth), runs, warmup)
    if selected_strategy == "factorized_preaggregated":
        factorized_preagg_result, factorized_preagg_times = _timed(
            lambda: factorized_preagg_fn(nodes, edges, depth), runs, warmup
        )

    if selected_strategy == "both":
        assert path_result is not None
        assert preagg_result is not None
        assert_frame_equal(_normalize(path_result), _normalize(preagg_result), check_dtype=False)

    path_ms = _median(path_times) if path_times else None
    preagg_ms = _median(preagg_times) if preagg_times else None
    factorized_preagg_ms = _median(factorized_preagg_times) if factorized_preagg_times else None
    measured = {
        "path_join": path_ms,
        "preaggregated": preagg_ms,
        "factorized_preaggregated": factorized_preagg_ms,
    }
    measured = {name: value for name, value in measured.items() if value is not None}
    best_strategy = min(measured, key=measured.get)
    best_strategy_ms = measured[best_strategy]
    if best_strategy == "path_join":
        top_result = path_result
    elif best_strategy == "preaggregated":
        top_result = preagg_result
    else:
        top_result = factorized_preagg_result

    return {
        "engine": engine,
        "selected_engine": selected_engine,
        "follows_rows": follows_rows,
        "depth": depth,
        "strategy": strategy,
        "selected_strategy": selected_strategy,
        "cpu_policy_follows_min_rows": CPU_POLICY_FOLLOWS_MIN_ROWS,
        "strategy_policy_follows_min_rows": STRATEGY_POLICY_FOLLOWS_MIN_ROWS,
        "path_join_median_ms": path_ms,
        "preaggregated_median_ms": preagg_ms,
        "factorized_preaggregated_median_ms": factorized_preagg_ms,
        "best_strategy": best_strategy,
        "best_strategy_median_ms": best_strategy_ms,
        "speedup": path_ms / preagg_ms if path_ms is not None and preagg_ms else 0.0,
        "path_join_runs_ms": path_times,
        "preaggregated_runs_ms": preagg_times,
        "factorized_preaggregated_runs_ms": factorized_preagg_times,
        "top10": _normalize(top_result).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--engine", choices=["pandas", "cudf", "polars", "cpu-policy", "both", "all"], default="both")
    parser.add_argument("--depth", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument(
        "--strategy",
        choices=["both", "path-join", "preaggregated", "factorized-preaggregated", "strategy-policy"],
        default="both",
    )
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.engine == "both":
        engines = ["pandas", "cudf"]
    elif args.engine == "all":
        engines = ["pandas", "cudf", "polars"]
    else:
        engines = [args.engine]
    payload = {
        "graph_benchmark_root": str(args.graph_benchmark_root),
        "runs": args.runs,
        "warmup": args.warmup,
        "workload": {
            "name": f"neighbors_{args.depth}_with_data_and_filter",
            "source_gender": SOURCE_GENDER,
            "source_age_lower": SOURCE_AGE_LOWER,
            "source_age_upper": SOURCE_AGE_UPPER,
            "target_age_lower": TARGET_AGE_LOWER,
            "result": "top city/country by path count",
            "source_to_target_hops": args.depth,
        },
        "cpu_policy": {
            "name": "pandas_below_threshold_polars_at_or_above_threshold",
            "follows_min_rows": CPU_POLICY_FOLLOWS_MIN_ROWS,
        },
        "strategy_policy": {
            "name": "path_join_for_small_or_shallow_factorized_preaggregated_for_large_depth3_plus",
            "follows_min_rows": STRATEGY_POLICY_FOLLOWS_MIN_ROWS,
            "factorized_preaggregated_min_depth": 3,
        },
        "strategy": args.strategy,
        "results": {},
    }
    print(f"root={args.graph_benchmark_root} depth={args.depth} strategy={args.strategy} runs={args.runs} warmup={args.warmup}")
    for engine in engines:
        strategy = {
            "path-join": "path_join",
            "factorized-preaggregated": "factorized_preaggregated",
        }.get(args.strategy, args.strategy)
        result = run_probe(args.graph_benchmark_root, engine, args.runs, args.warmup, args.depth, strategy)
        payload["results"][engine] = result
        selected = result.get("selected_engine", engine)
        policy_suffix = f" selected={selected}" if selected != engine else ""
        path_ms = result["path_join_median_ms"]
        preagg_ms = result["preaggregated_median_ms"]
        factorized_ms = result["factorized_preaggregated_median_ms"]
        path_text = f"{path_ms:.3f}ms" if path_ms is not None else "skipped"
        preagg_text = f"{preagg_ms:.3f}ms" if preagg_ms is not None else "skipped"
        factorized_text = f"{factorized_ms:.3f}ms" if factorized_ms is not None else "skipped"
        parity_text = "parity=pass" if result["selected_strategy"] == "both" else "parity=not-run"
        print(
            f"engine={engine}{policy_suffix} strategy={result['selected_strategy']} "
            f"path_join={path_text} preaggregated={preagg_text} "
            f"factorized_preaggregated={factorized_text} "
            f"best={result['best_strategy']} {result['best_strategy_median_ms']:.3f}ms "
            f"speedup={result['speedup']:.2f}x {parity_text}"
        )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
