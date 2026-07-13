#!/usr/bin/env python3
"""Prototype Polars CPU strategies for graph-benchmark q5-q7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd
from pandas.testing import assert_frame_equal

try:
    import polars as pl
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "Polars is required for this prototype benchmark. "
        "Run inside a RAPIDS benchmark image that includes polars."
    ) from exc


DEFAULT_ROOT = Path("/tmp/graph-benchmark-gfql-memgraph")

GENDER_Q5 = "male"
CITY_Q5 = "London"
COUNTRY_Q5 = "United Kingdom"
INTEREST_Q5 = "fine dining"
GENDER_Q6 = "female"
INTEREST_Q6 = "tennis"
COUNTRY_Q7 = "United States"
AGE_LOWER_Q7 = 23
AGE_UPPER_Q7 = 30
INTEREST_Q7 = "photography"

PandasData = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
PolarsData = Tuple[Any, Any, Any, Any, Any, Any, Any]
PolarsFlags = Dict[str, bool]


def _timed(fn: Callable[[], pd.DataFrame], runs: int, warmup: int) -> Tuple[pd.DataFrame, List[float]]:
    for _ in range(warmup):
        fn()
    times: List[float] = []
    result = pd.DataFrame()
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


def _edge_path(edges_path: Path, primary: str, fallback: str | None = None) -> Path:
    path = edges_path / primary
    if path.exists() or fallback is None:
        return path
    return edges_path / fallback


def _offsets_pandas(nodes_path: Path) -> Tuple[Dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    persons = pd.read_parquet(nodes_path / "persons.parquet")
    cities = pd.read_parquet(nodes_path / "cities.parquet")
    states = pd.read_parquet(nodes_path / "states.parquet")
    countries = pd.read_parquet(nodes_path / "countries.parquet")
    interests = pd.read_parquet(nodes_path / "interests.parquet")
    offsets = {"Person": 0, "City": int(persons["id"].max()) + 1}
    offsets["State"] = offsets["City"] + int(cities["id"].max()) + 1
    offsets["Country"] = offsets["State"] + int(states["id"].max()) + 1
    offsets["Interest"] = offsets["Country"] + int(countries["id"].max()) + 1
    return offsets, persons, cities, states, countries, interests


def load_pandas(root: Path) -> PandasData:
    nodes_path = root / "data" / "output" / "nodes"
    edges_path = root / "data" / "output" / "edges"
    offsets, persons, cities, states, _countries, interests = _offsets_pandas(nodes_path)

    persons = persons.assign(
        node_id=persons["id"].astype("int64") + offsets["Person"],
        gender_lc=persons["gender"].str.lower(),
    )
    cities = cities.assign(node_id=cities["id"].astype("int64") + offsets["City"])
    states = states.assign(node_id=states["id"].astype("int64") + offsets["State"])
    interests = interests.assign(
        node_id=interests["id"].astype("int64") + offsets["Interest"],
        interest_lc=interests["interest"].str.lower(),
    )

    lives = pd.read_parquet(edges_path / "lives_in.parquet").rename(columns={"from": "src", "to": "dst"})
    lives["src"] = lives["src"].astype("int64") + offsets["Person"]
    lives["dst"] = lives["dst"].astype("int64") + offsets["City"]

    interested_path = _edge_path(edges_path, "interested_in.parquet", "interests.parquet")
    interested = pd.read_parquet(interested_path).rename(columns={"from": "src", "to": "dst"})
    interested["src"] = interested["src"].astype("int64") + offsets["Person"]
    interested["dst"] = interested["dst"].astype("int64") + offsets["Interest"]

    city_in = pd.read_parquet(edges_path / "city_in.parquet").rename(columns={"from": "src", "to": "dst"})
    city_in["src"] = city_in["src"].astype("int64") + offsets["City"]
    city_in["dst"] = city_in["dst"].astype("int64") + offsets["State"]
    return persons, cities, states, interests, lives, interested, city_in


def load_polars(root: Path) -> PolarsData:
    nodes_path = root / "data" / "output" / "nodes"
    edges_path = root / "data" / "output" / "edges"
    persons = pl.read_parquet(nodes_path / "persons.parquet")
    cities = pl.read_parquet(nodes_path / "cities.parquet")
    states = pl.read_parquet(nodes_path / "states.parquet")
    countries = pl.read_parquet(nodes_path / "countries.parquet")
    interests = pl.read_parquet(nodes_path / "interests.parquet")

    offsets = {"Person": 0, "City": int(persons["id"].max()) + 1}
    offsets["State"] = offsets["City"] + int(cities["id"].max()) + 1
    offsets["Country"] = offsets["State"] + int(states["id"].max()) + 1
    offsets["Interest"] = offsets["Country"] + int(countries["id"].max()) + 1

    persons = persons.with_columns(
        (pl.col("id").cast(pl.Int64) + offsets["Person"]).alias("node_id"),
        pl.col("gender").str.to_lowercase().alias("gender_lc"),
    )
    cities = cities.with_columns((pl.col("id").cast(pl.Int64) + offsets["City"]).alias("node_id"))
    states = states.with_columns((pl.col("id").cast(pl.Int64) + offsets["State"]).alias("node_id"))
    interests = interests.with_columns(
        (pl.col("id").cast(pl.Int64) + offsets["Interest"]).alias("node_id"),
        pl.col("interest").str.to_lowercase().alias("interest_lc"),
    )

    lives = pl.read_parquet(edges_path / "lives_in.parquet").rename({"from": "src", "to": "dst"})
    lives = lives.with_columns(
        (pl.col("src").cast(pl.Int64) + offsets["Person"]).alias("src"),
        (pl.col("dst").cast(pl.Int64) + offsets["City"]).alias("dst"),
    )

    interested_path = _edge_path(edges_path, "interested_in.parquet", "interests.parquet")
    interested = pl.read_parquet(interested_path).rename({"from": "src", "to": "dst"})
    interested = interested.with_columns(
        (pl.col("src").cast(pl.Int64) + offsets["Person"]).alias("src"),
        (pl.col("dst").cast(pl.Int64) + offsets["Interest"]).alias("dst"),
    )

    city_in = pl.read_parquet(edges_path / "city_in.parquet").rename({"from": "src", "to": "dst"})
    city_in = city_in.with_columns(
        (pl.col("src").cast(pl.Int64) + offsets["City"]).alias("src"),
        (pl.col("dst").cast(pl.Int64) + offsets["State"]).alias("dst"),
    )
    return persons, cities, states, interests, lives, interested, city_in


def load_polars_lazy(data: PolarsData) -> PolarsData:
    return tuple(frame.lazy() for frame in data)  # type: ignore[return-value]


def _is_unique_polars(frame: Any, columns: List[str]) -> bool:
    return frame.select(columns).unique().height == frame.height


def polars_uniqueness_flags(data: PolarsData) -> PolarsFlags:
    _persons, _cities, _states, interests, lives, interested, _city_in = data
    return {
        "interest_edges_unique": _is_unique_polars(interested, ["src", "dst"]),
        "interest_lc_unique": _is_unique_polars(interests, ["interest_lc"]),
        "lives_src_unique": _is_unique_polars(lives, ["src"]),
    }


def pandas_q5(data: PandasData) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons[persons["gender_lc"] == GENDER_Q5][["node_id"]]
    interest_nodes = interests[interests["interest_lc"] == INTEREST_Q5][["node_id"]]
    city_nodes = cities[(cities["city"] == CITY_Q5) & (cities["country"] == COUNTRY_Q5)][["node_id"]]
    interest_people = interested[interested["dst"].isin(interest_nodes["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]].drop_duplicates()
    location_people = lives[lives["dst"].isin(city_nodes["node_id"])][["src"]].drop_duplicates()
    return pd.DataFrame({"numPersons": [int(len(interest_people[interest_people["src"].isin(location_people["src"])]))]})


def pandas_q6(data: PandasData) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons[persons["gender_lc"] == GENDER_Q6][["node_id"]]
    interest_nodes = interests[interests["interest_lc"] == INTEREST_Q6][["node_id"]]
    interest_people = interested[interested["dst"].isin(interest_nodes["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]].drop_duplicates()
    matched = lives.merge(interest_people.rename(columns={"src": "node_id"}), left_on="src", right_on="node_id")
    grouped = matched.groupby("dst").size().reset_index(name="numPersons")
    result = grouped.merge(cities[["node_id", "city", "country"]], left_on="dst", right_on="node_id")
    return result[["city", "country", "numPersons"]].sort_values(
        ["numPersons", "city", "country"], ascending=[False, True, True]
    ).head(5).reset_index(drop=True)


def pandas_q7(data: PandasData) -> pd.DataFrame:
    persons, _cities, states, interests, lives, interested, city_in = data
    people = persons[(persons["age"] >= AGE_LOWER_Q7) & (persons["age"] <= AGE_UPPER_Q7)][["node_id"]]
    interest_nodes = interests[interests["interest_lc"] == INTEREST_Q7][["node_id"]]
    interest_people = interested[interested["dst"].isin(interest_nodes["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]].drop_duplicates()
    state_nodes = states[states["country"] == COUNTRY_Q7][["node_id", "state", "country"]]
    path = lives.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    path = path.merge(interest_people.rename(columns={"src": "node_id"}), left_on="src_person", right_on="node_id")
    grouped = path.groupby("dst_city").size().reset_index(name="numPersons")
    result = grouped.merge(state_nodes, left_on="dst_city", right_on="node_id")
    return result[["state", "country", "numPersons"]].sort_values(
        ["numPersons", "state", "country"], ascending=[False, True, True]
    ).head(1).reset_index(drop=True)


def polars_q5(data: PolarsData) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q5).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q5).select("node_id")
    city_nodes = cities.filter((pl.col("city") == CITY_Q5) & (pl.col("country") == COUNTRY_Q5)).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi").select("src").unique()
    location_people = lives.join(city_nodes, left_on="dst", right_on="node_id", how="semi").select("src").unique()
    return pd.DataFrame({"numPersons": [interest_people.join(location_people, on="src", how="semi").height]})


def polars_q6(data: PolarsData) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q6).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q6).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id")).unique()
    result = (
        lives.join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst")
        .len()
        .rename({"len": "numPersons"})
        .join(cities.select(["node_id", "city", "country"]), left_on="dst", right_on="node_id", how="inner")
        .select(["city", "country", "numPersons"])
        .sort(["numPersons", "city", "country"], descending=[True, False, False])
        .head(5)
    )
    return result.to_pandas()


def polars_q7(data: PolarsData) -> pd.DataFrame:
    persons, _cities, states, interests, lives, interested, city_in = data
    people = persons.filter((pl.col("age") >= AGE_LOWER_Q7) & (pl.col("age") <= AGE_UPPER_Q7)).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q7).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id")).unique()
    state_nodes = states.filter(pl.col("country") == COUNTRY_Q7).select(["node_id", "state", "country"])
    result = (
        lives.join(city_in, left_on="dst", right_on="src", how="inner", suffix="_city")
        .join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst_city")
        .len()
        .rename({"len": "numPersons"})
        .join(state_nodes, left_on="dst_city", right_on="node_id", how="inner")
        .select(["state", "country", "numPersons"])
        .sort(["numPersons", "state", "country"], descending=[True, False, False])
        .head(1)
    )
    return result.to_pandas()


def polars_q5_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q5).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q5).select("node_id")
    city_nodes = cities.filter((pl.col("city") == CITY_Q5) & (pl.col("country") == COUNTRY_Q5)).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi").select("src")
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    location_people = lives.join(city_nodes, left_on="dst", right_on="node_id", how="semi").select("src")
    if not flags["lives_src_unique"]:
        location_people = location_people.unique()
    return pd.DataFrame({"numPersons": [interest_people.join(location_people, on="src", how="semi").height]})


def polars_q6_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q6).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q6).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    result = (
        lives.join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst")
        .len()
        .rename({"len": "numPersons"})
        .join(cities.select(["node_id", "city", "country"]), left_on="dst", right_on="node_id", how="inner")
        .select(["city", "country", "numPersons"])
        .sort(["numPersons", "city", "country"], descending=[True, False, False])
        .head(5)
    )
    return result.to_pandas()


def polars_q7_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, _cities, states, interests, lives, interested, city_in = data
    people = persons.filter((pl.col("age") >= AGE_LOWER_Q7) & (pl.col("age") <= AGE_UPPER_Q7)).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q7).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    state_nodes = states.filter(pl.col("country") == COUNTRY_Q7).select(["node_id", "state", "country"])
    result = (
        lives.join(city_in, left_on="dst", right_on="src", how="inner", suffix="_city")
        .join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst_city")
        .len()
        .rename({"len": "numPersons"})
        .join(state_nodes, left_on="dst_city", right_on="node_id", how="inner")
        .select(["state", "country", "numPersons"])
        .sort(["numPersons", "state", "country"], descending=[True, False, False])
        .head(1)
    )
    return result.to_pandas()


def polars_lazy_q5_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q5).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q5).select("node_id")
    city_nodes = cities.filter((pl.col("city") == CITY_Q5) & (pl.col("country") == COUNTRY_Q5)).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi").select("src")
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    location_people = lives.join(city_nodes, left_on="dst", right_on="node_id", how="semi").select("src")
    if not flags["lives_src_unique"]:
        location_people = location_people.unique()
    result = interest_people.join(location_people, on="src", how="semi").select(pl.len().alias("numPersons"))
    return result.collect().to_pandas()


def polars_lazy_q6_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, cities, _states, interests, lives, interested, _city_in = data
    people = persons.filter(pl.col("gender_lc") == GENDER_Q6).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q6).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    result = (
        lives.join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst")
        .len()
        .rename({"len": "numPersons"})
        .join(cities.select(["node_id", "city", "country"]), left_on="dst", right_on="node_id", how="inner")
        .select(["city", "country", "numPersons"])
        .sort(["numPersons", "city", "country"], descending=[True, False, False])
        .head(5)
    )
    return result.collect().to_pandas()


def polars_lazy_q7_unique_gated(data: PolarsData, flags: PolarsFlags) -> pd.DataFrame:
    persons, _cities, states, interests, lives, interested, city_in = data
    people = persons.filter((pl.col("age") >= AGE_LOWER_Q7) & (pl.col("age") <= AGE_UPPER_Q7)).select("node_id")
    interest_nodes = interests.filter(pl.col("interest_lc") == INTEREST_Q7).select("node_id")
    interest_people = interested.join(interest_nodes, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (flags["interest_edges_unique"] and flags["interest_lc_unique"]):
        interest_people = interest_people.unique()
    state_nodes = states.filter(pl.col("country") == COUNTRY_Q7).select(["node_id", "state", "country"])
    result = (
        lives.join(city_in, left_on="dst", right_on="src", how="inner", suffix="_city")
        .join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst_city")
        .len()
        .rename({"len": "numPersons"})
        .join(state_nodes, left_on="dst_city", right_on="node_id", how="inner")
        .select(["state", "country", "numPersons"])
        .sort(["numPersons", "state", "country"], descending=[True, False, False])
        .head(1)
    )
    return result.collect().to_pandas()


def _normalize(label: str, df: pd.DataFrame) -> pd.DataFrame:
    cols_by_label = {
        "q5": ["numPersons"],
        "q6": ["city", "country", "numPersons"],
        "q7": ["state", "country", "numPersons"],
    }
    return df[cols_by_label[label]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--queries", default="q5,q6,q7", help="Comma-separated subset of q5,q6,q7.")
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    query_names = [q.strip() for q in args.queries.split(",") if q.strip()]
    unknown = sorted(set(query_names) - {"q5", "q6", "q7"})
    if unknown:
        raise ValueError(f"Unknown queries: {unknown}")

    pandas_data = load_pandas(args.graph_benchmark_root)
    polars_data = load_polars(args.graph_benchmark_root)
    polars_lazy_data = load_polars_lazy(polars_data)
    polars_flags = polars_uniqueness_flags(polars_data)
    pandas_functions: Dict[str, Callable[[PandasData], pd.DataFrame]] = {
        "q5": pandas_q5,
        "q6": pandas_q6,
        "q7": pandas_q7,
    }
    polars_variants: Dict[str, Dict[str, Callable[[], pd.DataFrame]]] = {
        "q5": {
            "polars_eager": lambda: polars_q5(polars_data),
            "polars_unique_gated": lambda: polars_q5_unique_gated(polars_data, polars_flags),
            "polars_lazy_unique_gated": lambda: polars_lazy_q5_unique_gated(polars_lazy_data, polars_flags),
        },
        "q6": {
            "polars_eager": lambda: polars_q6(polars_data),
            "polars_unique_gated": lambda: polars_q6_unique_gated(polars_data, polars_flags),
            "polars_lazy_unique_gated": lambda: polars_lazy_q6_unique_gated(polars_lazy_data, polars_flags),
        },
        "q7": {
            "polars_eager": lambda: polars_q7(polars_data),
            "polars_unique_gated": lambda: polars_q7_unique_gated(polars_data, polars_flags),
            "polars_lazy_unique_gated": lambda: polars_lazy_q7_unique_gated(polars_lazy_data, polars_flags),
        },
    }

    results: Dict[str, Dict[str, Any]] = {}
    print(f"root={args.graph_benchmark_root} runs={args.runs} warmup={args.warmup}")
    print(f"polars_flags={polars_flags}")
    for label in query_names:
        pandas_fn = pandas_functions[label]
        expected, pandas_times = _timed(lambda pandas_fn=pandas_fn: pandas_fn(pandas_data), args.runs, args.warmup)
        pandas_ms = _median(pandas_times)
        variants: Dict[str, Dict[str, Any]] = {}
        best_name = ""
        best_ms = float("inf")
        for variant_name, variant_fn in polars_variants[label].items():
            actual, variant_times = _timed(variant_fn, args.runs, args.warmup)
            assert_frame_equal(_normalize(label, expected), _normalize(label, actual), check_dtype=False)
            variant_ms = _median(variant_times)
            if variant_ms < best_ms:
                best_name = variant_name
                best_ms = variant_ms
            variants[variant_name] = {
                "median_ms": variant_ms,
                "speedup_vs_pandas": pandas_ms / variant_ms if variant_ms else 0.0,
                "runs_ms": variant_times,
            }
        eager_ms = variants["polars_eager"]["median_ms"]
        results[label] = {
            "pandas_median_ms": pandas_ms,
            "polars_median_ms": eager_ms,
            "speedup": pandas_ms / eager_ms if eager_ms else 0.0,
            "pandas_runs_ms": pandas_times,
            "polars_runs_ms": variants["polars_eager"]["runs_ms"],
            "best_variant": best_name,
            "best_variant_median_ms": best_ms,
            "variants": variants,
        }
        variant_summary = " ".join(
            f"{name}={payload['median_ms']:.3f}ms"
            for name, payload in variants.items()
        )
        print(
            f"{label}: pandas={pandas_ms:.3f}ms {variant_summary} "
            f"best={best_name} speedup={pandas_ms / best_ms if best_ms else 0.0:.2f}x parity=pass"
        )

    if args.output_json is not None:
        payload = {
            "graph_benchmark_root": str(args.graph_benchmark_root),
            "runs": args.runs,
            "warmup": args.warmup,
            "polars_flags": polars_flags,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
