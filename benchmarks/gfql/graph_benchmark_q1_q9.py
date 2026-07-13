#!/usr/bin/env python3
"""Run q1-q9 from graph-benchmark on Graphistry (pandas/cudf)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse
from graphistry.compute.predicates.numeric import between


DEFAULT_ROOT = Path(os.environ.get("GRAPH_BENCHMARK_ROOT", "/home/lmeyerov/Work/graph-benchmark"))

NODE_FILES = {
    "Person": "persons.parquet",
    "City": "cities.parquet",
    "State": "states.parquet",
    "Country": "countries.parquet",
    "Interest": "interests.parquet",
}

EDGE_FILES = [
    ("follows.parquet", "FOLLOWS", "Person", "Person"),
    ("lives_in.parquet", "LIVES_IN", "Person", "City"),
    ("interested_in.parquet", "HAS_INTEREST", "Person", "Interest"),
    ("city_in.parquet", "CITY_IN", "City", "State"),
    ("state_in.parquet", "STATE_IN", "State", "Country"),
]

DEFAULT_MODE = "baseline"
Q5_POLARS_MIN_HAS_INTEREST_ROWS = 100_000
Q67_POLARS_MIN_HAS_INTEREST_ROWS = 100_000


def _is_unique_by_columns(frame: Any, columns: List[str]) -> bool:
    return len(frame) == len(frame.drop_duplicates(subset=columns))


def _load_nodes(nodes_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    persons = pd.read_parquet(nodes_path / NODE_FILES["Person"])
    cities = pd.read_parquet(nodes_path / NODE_FILES["City"])
    states = pd.read_parquet(nodes_path / NODE_FILES["State"])
    countries = pd.read_parquet(nodes_path / NODE_FILES["Country"])
    interests = pd.read_parquet(nodes_path / NODE_FILES["Interest"])

    offsets: Dict[str, int] = {}
    offsets["Person"] = 0
    offsets["City"] = int(persons["id"].max()) + 1
    offsets["State"] = offsets["City"] + int(cities["id"].max()) + 1
    offsets["Country"] = offsets["State"] + int(states["id"].max()) + 1
    offsets["Interest"] = offsets["Country"] + int(countries["id"].max()) + 1

    def _apply(df: pd.DataFrame, node_type: str) -> pd.DataFrame:
        out = df.copy()
        out["node_type"] = node_type
        out["node_id"] = out["id"].astype("int64") + offsets[node_type]
        return out

    persons = _apply(persons, "Person")
    if "birthday" in persons.columns:
        persons["birthday"] = pd.to_datetime(persons["birthday"])
    persons["gender_lc"] = persons["gender"].str.lower()

    interests = _apply(interests, "Interest")
    interests["interest_lc"] = interests["interest"].str.lower()

    cities = _apply(cities, "City")
    states = _apply(states, "State")
    countries = _apply(countries, "Country")

    nodes = pd.concat([persons, interests, cities, states, countries], ignore_index=True, sort=False)
    return nodes, offsets


def _load_edges(edges_path: Path, offsets: Dict[str, int]) -> pd.DataFrame:
    edges: List[pd.DataFrame] = []
    for filename, rel, src_type, dst_type in EDGE_FILES:
        path = edges_path / filename
        if not path.exists() and filename in {"interested_in.parquet", "interests.parquet"}:
            fallback = "interests.parquet" if filename == "interested_in.parquet" else "interested_in.parquet"
            path = edges_path / fallback
        df = pd.read_parquet(path).rename(columns={"from": "src", "to": "dst"})
        df["src"] = df["src"].astype("int64") + offsets[src_type]
        df["dst"] = df["dst"].astype("int64") + offsets[dst_type]
        df["rel"] = rel
        edges.append(df[["src", "dst", "rel"]])
    return pd.concat(edges, ignore_index=True, sort=False)


def _maybe_to_cudf(engine: str, df: pd.DataFrame) -> Any:
    if engine == "pandas":
        return df
    if engine != "cudf":
        raise ValueError(f"Unsupported engine: {engine}")
    try:
        import cudf  # type: ignore
    except Exception as exc:
        raise RuntimeError("cudf engine requested but cudf is not available") from exc
    return cudf.from_pandas(df)


def _concat_frames(engine: str, frames: List[Any]) -> Any:
    if not frames:
        return pd.DataFrame()
    if engine == "cudf":
        import cudf  # type: ignore

        return cudf.concat(frames, ignore_index=True)
    return pd.concat(frames, ignore_index=True)


def _edges_by_rel(edges: Any, rel: str) -> Any:
    return edges[edges["rel"] == rel]


def _nodes_by_type(nodes: Any, node_type: str) -> Any:
    return nodes[nodes["node_type"] == node_type]


def _build_preindexed_graphs(
    nodes: Any,
    edges: Any,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    engine: str,
    spec: Dict[str, Tuple[List[str], List[str]]],
) -> Dict[str, Any]:
    nodes_by_type = {t: _nodes_by_type(nodes, t) for t in nodes_df["node_type"].unique().tolist()}
    edges_by_rel = {r: _edges_by_rel(edges, r) for r in edges_df["rel"].unique().tolist()}

    def _graph_for(types: List[str], rels: List[str]) -> Any:
        nodes_parts = [nodes_by_type[t] for t in types]
        edges_parts = [edges_by_rel[r] for r in rels]
        g_nodes = _concat_frames(engine, nodes_parts)
        g_edges = _concat_frames(engine, edges_parts)
        return graphistry.nodes(g_nodes, "node_id").edges(g_edges, "src", "dst")

    return {name: _graph_for(types, rels) for name, (types, rels) in spec.items()}


def _timed(label: str, fn: Callable[[], Any], runs: int, warmup: int) -> Tuple[Any, List[float]]:
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
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _query1_dataframe_shortcut(g: Any) -> pd.DataFrame:
    # Direct columnar in-degree count over FOLLOWS: skip the traversal/row
    # pipeline and group the FOLLOWS destination column, then top-3 by count.
    nodes = g._nodes
    edges = _edges_by_rel(g._edges, "FOLLOWS")
    counts = edges.groupby("dst").size().reset_index(name="numFollowers")
    persons = nodes[["node_id", "name"]].drop_duplicates()
    result = counts.merge(persons, left_on="dst", right_on="node_id")
    return result.sort_values("numFollowers", ascending=False).head(3)


def _query1(g: Any, engine: str, mode: str, query_variant: str = "standard") -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q1", engine):
        return _query1_dataframe_shortcut(g)
    chain = [
        n(),
        e_forward(),
        n(),
    ] if mode == "preindexed" else [
        n({"node_type": "Person"}),
        e_forward({"rel": "FOLLOWS"}),
        n({"node_type": "Person"}),
    ]
    gq = g.gfql(chain, engine=engine)
    edges = gq._edges
    nodes = gq._nodes
    dst_col = gq._destination
    counts = edges.groupby(dst_col).size().reset_index(name="numFollowers")
    persons = nodes[["node_id", "name"]].drop_duplicates()
    result = counts.merge(persons, left_on=dst_col, right_on="node_id")
    return result.sort_values("numFollowers", ascending=False).head(3)


def _top_person_id(g: Any, engine: str, mode: str) -> int:
    if mode != "preindexed":
        top = _query1(g, engine, mode)
        top_id_value = top["node_id"].iloc[0]
    else:
        edges = _edges_by_rel(g._edges, "FOLLOWS")
        counts = edges.groupby("dst").size().reset_index(name="numFollowers")
        top = counts.sort_values("numFollowers", ascending=False).head(1)
        top_id_value = top["dst"].iloc[0]
    if hasattr(top_id_value, "item"):
        top_id_value = top_id_value.item()
    return int(top_id_value)


def _query2(g_follow: Any, g_lives: Any, engine: str, mode: str) -> pd.DataFrame:
    top_id = _top_person_id(g_follow, engine, mode)
    chain = [
        n({"node_id": top_id}),
        e_forward(),
        n(),
    ] if mode == "preindexed" else [
        n({"node_id": top_id}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
    ]
    gq = g_lives.gfql(chain, engine=engine)
    nodes = gq._nodes
    person = nodes[nodes["node_type"] == "Person"][["node_id", "name"]]
    city = nodes[nodes["node_type"] == "City"][["node_id", "city", "state", "country"]]
    edges = _edges_by_rel(gq._edges, "LIVES_IN")
    joined = edges.merge(person, left_on="src", right_on="node_id")
    joined = joined.merge(city, left_on="dst", right_on="node_id", suffixes=("_person", "_city"))
    return joined[["name", "city", "state", "country"]]


def _uses_dataframe_shortcut(query_variant: str, query_label: str, engine: str) -> bool:
    if query_variant == "dataframe-shortcut":
        return True
    if query_variant != "standard":
        return False
    return query_label in {"q1", "q3", "q4", "q5", "q6", "q7"}


def _query3_dataframe_shortcut(g: Any, country: str) -> pd.DataFrame:
    nodes = g._nodes
    edges = g._edges
    persons = nodes[nodes["node_type"] == "Person"][["node_id", "age"]].rename(columns={"node_id": "person_id"})
    cities = nodes[nodes["node_type"] == "City"][["node_id", "city"]].rename(columns={"node_id": "city_id"})
    countries = nodes[
        (nodes["node_type"] == "Country")
        & (nodes["country"] == country)
    ][["node_id"]].rename(columns={"node_id": "country_id"})

    state_in = _edges_by_rel(edges, "STATE_IN")[["src", "dst"]].rename(
        columns={"src": "state_id", "dst": "country_id"}
    )
    city_in = _edges_by_rel(edges, "CITY_IN")[["src", "dst"]].rename(
        columns={"src": "city_id", "dst": "state_id"}
    )
    lives_in = _edges_by_rel(edges, "LIVES_IN")[["src", "dst"]].rename(
        columns={"src": "person_id", "dst": "city_id"}
    )

    states = state_in.merge(countries, on="country_id")[["state_id"]]
    country_cities = city_in.merge(states, on="state_id")[["city_id"]]
    matched = lives_in.merge(country_cities, on="city_id")
    matched = matched.merge(persons, on="person_id")
    matched = matched.merge(cities, on="city_id")
    avg_age = matched.groupby("city")["age"].mean().reset_index(name="averageAge")
    return avg_age.sort_values("averageAge").head(5)


def _query4_dataframe_shortcut(g: Any, age_lower: int, age_upper: int) -> pd.DataFrame:
    nodes = g._nodes
    edges = g._edges
    persons = nodes[
        (nodes["node_type"] == "Person")
        & (nodes["age"] >= age_lower)
        & (nodes["age"] <= age_upper)
    ][["node_id"]].rename(columns={"node_id": "person_id"})
    countries = nodes[nodes["node_type"] == "Country"][["node_id", "country"]].rename(
        columns={"node_id": "country_id"}
    )

    lives_in = _edges_by_rel(edges, "LIVES_IN")[["src", "dst"]].rename(
        columns={"src": "person_id", "dst": "city_id"}
    )
    city_in = _edges_by_rel(edges, "CITY_IN")[["src", "dst"]].rename(
        columns={"src": "city_id", "dst": "state_id"}
    )
    state_in = _edges_by_rel(edges, "STATE_IN")[["src", "dst"]].rename(
        columns={"src": "state_id", "dst": "country_id"}
    )

    path = lives_in.merge(persons, on="person_id")
    path = path.merge(city_in, on="city_id")
    path = path.merge(state_in, on="state_id")
    counts = path.groupby("country_id").size().reset_index(name="personCounts")
    result = counts.merge(countries, on="country_id")
    return result[["country", "personCounts"]].sort_values("personCounts", ascending=False).head(3)


def _query3(g: Any, engine: str, mode: str, country: str, query_variant: str) -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q3", engine):
        return _query3_dataframe_shortcut(g, country)

    if query_variant == "reverse-seeded":
        chain = [
            n({"country": country}),
            e_reverse(),
            n(),
            e_reverse(),
            n(),
            e_reverse(),
            n(),
        ] if mode == "preindexed" else [
            n({"node_type": "Country", "country": country}),
            e_reverse({"rel": "STATE_IN"}),
            n({"node_type": "State"}),
            e_reverse({"rel": "CITY_IN"}),
            n({"node_type": "City"}),
            e_reverse({"rel": "LIVES_IN"}),
            n({"node_type": "Person"}),
        ]
    else:
        chain = [
            n(),
            e_forward(),
            n(),
            e_forward(),
            n(),
            e_forward(),
            n({"country": country}),
        ] if mode == "preindexed" else [
            n({"node_type": "Person"}),
            e_forward({"rel": "LIVES_IN"}),
            n({"node_type": "City"}),
            e_forward({"rel": "CITY_IN"}),
            n({"node_type": "State"}),
            e_forward({"rel": "STATE_IN"}),
            n({"node_type": "Country", "country": country}),
        ]
    gq = g.gfql(chain, engine=engine)
    nodes = gq._nodes
    edges = gq._edges
    persons = nodes[nodes["node_type"] == "Person"][["node_id", "age"]]
    cities = nodes[nodes["node_type"] == "City"][["node_id", "city"]]
    lives_in = _edges_by_rel(edges, "LIVES_IN")
    merged = lives_in.merge(persons, left_on="src", right_on="node_id")
    merged = merged.merge(cities, left_on="dst", right_on="node_id", suffixes=("_person", "_city"))
    avg_age = merged.groupby("city")["age"].mean().reset_index(name="averageAge")
    return avg_age.sort_values("averageAge").head(5)


def _query4(
    g: Any,
    engine: str,
    mode: str,
    age_lower: int,
    age_upper: int,
    query_variant: str,
) -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q4", engine):
        return _query4_dataframe_shortcut(g, age_lower, age_upper)

    chain = [
        n({"age": between(age_lower, age_upper)}),
        e_forward(),
        n(),
        e_forward(),
        n(),
        e_forward(),
        n(),
    ] if mode == "preindexed" else [
        n({"node_type": "Person", "age": between(age_lower, age_upper)}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
        e_forward({"rel": "CITY_IN"}),
        n({"node_type": "State"}),
        e_forward({"rel": "STATE_IN"}),
        n({"node_type": "Country"}),
    ]
    gq = g.gfql(chain, engine=engine)
    nodes = gq._nodes
    edges = gq._edges
    countries = nodes[nodes["node_type"] == "Country"][["node_id", "country"]]
    lives_in = _edges_by_rel(edges, "LIVES_IN")
    city_in = _edges_by_rel(edges, "CITY_IN")
    state_in = _edges_by_rel(edges, "STATE_IN")

    path = lives_in.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    path = path.merge(state_in, left_on="dst_city", right_on="src", suffixes=("", "_state"))
    counts = path.groupby("dst").size().reset_index(name="personCounts")
    result = counts.merge(countries, left_on="dst", right_on="node_id")
    return result[["country", "personCounts"]].sort_values("personCounts", ascending=False).head(3)


def _query5_dataframe_shortcut(
    g_interest: Any,
    g_location: Any,
    gender: str,
    city: str,
    country: str,
    interest: str,
) -> pd.DataFrame:
    interest_nodes = g_interest._nodes
    interest_edges = _edges_by_rel(g_interest._edges, "HAS_INTEREST")[["src", "dst"]]
    people = interest_nodes[
        (interest_nodes["node_type"] == "Person")
        & (interest_nodes["gender_lc"] == gender.lower())
    ][["node_id"]]
    interests = interest_nodes[
        (interest_nodes["node_type"] == "Interest")
        & (interest_nodes["interest_lc"] == interest.lower())
    ][["node_id"]]

    interest_people = interest_edges[interest_edges["dst"].isin(interests["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]].drop_duplicates()

    location_nodes = g_location._nodes
    location_edges = _edges_by_rel(g_location._edges, "LIVES_IN")[["src", "dst"]]
    cities = location_nodes[
        (location_nodes["node_type"] == "City")
        & (location_nodes["city"] == city)
        & (location_nodes["country"] == country)
    ][["node_id"]]
    location_people = location_edges[location_edges["dst"].isin(cities["node_id"])][["src"]].drop_duplicates()

    matched = interest_people[interest_people["src"].isin(location_people["src"])]
    return pd.DataFrame({"numPersons": [len(matched)]})


def _is_unique_polars(frame: Any, columns: List[str]) -> bool:
    return frame.select(columns).unique().height == frame.height


def _query5_polars_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    cities_frame: Any,
    lives_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    lives_in_src_unique: bool,
    gender: str,
    city: str,
    country: str,
    interest: str,
) -> pd.DataFrame:
    import polars as pl  # type: ignore

    people = persons.filter(pl.col("gender_lc") == gender.lower()).select("node_id")
    interests = interests_frame.filter(pl.col("interest_lc") == interest.lower()).select("node_id")
    interest_people = interest_edges.join(interests, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi").select("src")
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.unique()

    cities = cities_frame.filter(
        (pl.col("city") == city)
        & (pl.col("country") == country)
    ).select("node_id")
    location_people = lives_in.join(cities, left_on="dst", right_on="node_id", how="semi").select("src")
    if not lives_in_src_unique:
        location_people = location_people.unique()

    result = location_people.join(interest_people, on="src", how="semi").select(pl.len().alias("numPersons"))
    return result.collect().to_pandas()


def _maybe_build_q5_polars_context(g_interest: Any, g_location: Any) -> Optional[Tuple[Any, Any, Any, Any, Any, bool, bool, bool]]:
    interest_edges_pd = _edges_by_rel(g_interest._edges, "HAS_INTEREST")[["src", "dst"]]
    if len(interest_edges_pd) < Q5_POLARS_MIN_HAS_INTEREST_ROWS:
        return None
    try:
        import polars as pl  # type: ignore
    except Exception:
        return None

    interest_nodes = pd.DataFrame(g_interest._nodes)
    location_nodes = pd.DataFrame(g_location._nodes)
    location_edges = pd.DataFrame(g_location._edges)
    persons = pl.from_pandas(interest_nodes[interest_nodes["node_type"] == "Person"][["node_id", "gender_lc", "age"]])
    interests_frame = pl.from_pandas(interest_nodes[interest_nodes["node_type"] == "Interest"][["node_id", "interest_lc"]])
    interest_edges = pl.from_pandas(pd.DataFrame(interest_edges_pd))
    cities = pl.from_pandas(location_nodes[location_nodes["node_type"] == "City"][["node_id", "city", "country"]])
    lives_in = pl.from_pandas(pd.DataFrame(_edges_by_rel(location_edges, "LIVES_IN")[["src", "dst"]]))
    return (
        persons.lazy(),
        interests_frame.lazy(),
        interest_edges.lazy(),
        cities.lazy(),
        lives_in.lazy(),
        _is_unique_polars(interest_edges, ["src", "dst"]),
        _is_unique_polars(interests_frame, ["interest_lc"]),
        _is_unique_polars(lives_in, ["src"]),
    )


def _query5_cudf_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    cities_frame: Any,
    lives_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    lives_in_src_unique: bool,
    gender: str,
    city: str,
    country: str,
    interest: str,
) -> pd.DataFrame:
    people = persons[persons["gender_lc"] == gender.lower()][["node_id"]]
    interests = interests_frame[interests_frame["interest_lc"] == interest.lower()][["node_id"]]
    interest_people = interest_edges[interest_edges["dst"].isin(interests["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]]
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.drop_duplicates()

    cities = cities_frame[
        (cities_frame["city"] == city)
        & (cities_frame["country"] == country)
    ][["node_id"]]
    location_people = lives_in[lives_in["dst"].isin(cities["node_id"])][["src"]]
    if not lives_in_src_unique:
        location_people = location_people.drop_duplicates()

    matched = interest_people[interest_people["src"].isin(location_people["src"])]
    return pd.DataFrame({"numPersons": [len(matched)]})


def _maybe_build_q5_cudf_context(g_interest: Any, g_location: Any) -> Optional[Tuple[Any, Any, Any, Any, Any, bool, bool, bool]]:
    interest_edges = _edges_by_rel(g_interest._edges, "HAS_INTEREST")[["src", "dst"]]
    if len(interest_edges) < Q5_POLARS_MIN_HAS_INTEREST_ROWS:
        return None
    interest_nodes = g_interest._nodes
    interest_frame = interest_nodes[interest_nodes["node_type"] == "Interest"][["node_id", "interest_lc"]]
    location_nodes = g_location._nodes
    location_edges = g_location._edges
    lives_in = _edges_by_rel(location_edges, "LIVES_IN")[["src", "dst"]]
    return (
        interest_nodes[interest_nodes["node_type"] == "Person"][["node_id", "gender_lc", "age"]],
        interest_frame,
        interest_edges,
        location_nodes[location_nodes["node_type"] == "City"][["node_id", "city", "country"]],
        lives_in,
        _is_unique_by_columns(interest_edges, ["src", "dst"]),
        _is_unique_by_columns(interest_frame, ["interest_lc"]),
        _is_unique_by_columns(lives_in, ["src"]),
    )


def _should_use_q5_cudf_policy(engine: str, mode: str, query_variant: str) -> bool:
    return engine == "cudf" and mode == "preindexed" and query_variant == "standard"


def _should_use_q5_polars_policy(engine: str, mode: str, query_variant: str) -> bool:
    return engine == "pandas" and mode == "preindexed" and query_variant == "standard"


def _query5(
    g_interest: Any,
    g_location: Any,
    engine: str,
    mode: str,
    gender: str,
    city: str,
    country: str,
    interest: str,
    query_variant: str,
) -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q5", engine):
        return _query5_dataframe_shortcut(g_interest, g_location, gender, city, country, interest)

    if query_variant == "reverse-seeded":
        chain_interest = [
            n({"interest_lc": interest.lower()}),
            e_reverse(),
            n({"gender_lc": gender.lower()}),
        ] if mode == "preindexed" else [
            n({"node_type": "Interest", "interest_lc": interest.lower()}),
            e_reverse({"rel": "HAS_INTEREST"}),
            n({"node_type": "Person", "gender_lc": gender.lower()}),
        ]
    else:
        chain_interest = [
            n({"gender_lc": gender.lower()}),
            e_forward(),
            n({"interest_lc": interest.lower()}),
        ] if mode == "preindexed" else [
            n({"node_type": "Person", "gender_lc": gender.lower()}),
            e_forward({"rel": "HAS_INTEREST"}),
            n({"node_type": "Interest", "interest_lc": interest.lower()}),
        ]
    g_interest = g_interest.gfql(chain_interest, engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    if query_variant == "reverse-seeded":
        chain_location = [
            n({"city": city, "country": country}),
            e_reverse(),
            n(),
        ] if mode == "preindexed" else [
            n({"node_type": "City", "city": city, "country": country}),
            e_reverse({"rel": "LIVES_IN"}),
            n({"node_type": "Person"}),
        ]
    else:
        chain_location = [
            n(),
            e_forward(),
            n({"city": city, "country": country}),
        ] if mode == "preindexed" else [
            n({"node_type": "Person"}),
            e_forward({"rel": "LIVES_IN"}),
            n({"node_type": "City", "city": city, "country": country}),
        ]
    g_location = g_location.gfql(chain_location, engine=engine)
    location_edges = _edges_by_rel(g_location._edges, "LIVES_IN")
    location_people = location_edges[["src"]].rename(columns={"src": "node_id"}).drop_duplicates()

    matched = interest_people.merge(location_people, on="node_id")
    return pd.DataFrame({"numPersons": [len(matched)]})


def _query6_dataframe_shortcut(
    g_interest: Any,
    g_location: Any,
    gender: str,
    interest: str,
) -> pd.DataFrame:
    interest_nodes = g_interest._nodes
    interest_edges = _edges_by_rel(g_interest._edges, "HAS_INTEREST")
    people = interest_nodes[
        (interest_nodes["node_type"] == "Person")
        & (interest_nodes["gender_lc"] == gender.lower())
    ][["node_id"]]
    interests = interest_nodes[
        (interest_nodes["node_type"] == "Interest")
        & (interest_nodes["interest_lc"] == interest.lower())
    ][["node_id"]]

    interest_people = interest_edges.merge(people, left_on="src", right_on="node_id")
    interest_people = interest_people.merge(
        interests,
        left_on="dst",
        right_on="node_id",
        suffixes=("_person", "_interest"),
    )
    interest_people = interest_people[["src"]].rename(columns={"src": "node_id"}).drop_duplicates()

    location_nodes = g_location._nodes
    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_nodes = location_nodes[location_nodes["node_type"] == "City"][["node_id", "city", "country"]]

    matched = lives_in.merge(interest_people, left_on="src", right_on="node_id")
    grouped = matched.groupby("dst").size().reset_index(name="numPersons")
    result = grouped.merge(city_nodes, left_on="dst", right_on="node_id")
    return result.sort_values(["numPersons", "city", "country"], ascending=[False, True, True]).head(5)


def _query7_dataframe_shortcut(
    g_interest: Any,
    g_location: Any,
    country: str,
    age_lower: int,
    age_upper: int,
    interest: str,
) -> pd.DataFrame:
    interest_nodes = g_interest._nodes
    interest_edges = _edges_by_rel(g_interest._edges, "HAS_INTEREST")
    people = interest_nodes[
        (interest_nodes["node_type"] == "Person")
        & (interest_nodes["age"] >= age_lower)
        & (interest_nodes["age"] <= age_upper)
    ][["node_id"]]
    interests = interest_nodes[
        (interest_nodes["node_type"] == "Interest")
        & (interest_nodes["interest_lc"] == interest.lower())
    ][["node_id"]]

    interest_people = interest_edges.merge(people, left_on="src", right_on="node_id")
    interest_people = interest_people.merge(
        interests,
        left_on="dst",
        right_on="node_id",
        suffixes=("_person", "_interest"),
    )
    interest_people = interest_people[["src"]].rename(columns={"src": "node_id"}).drop_duplicates()

    location_nodes = g_location._nodes
    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_in = _edges_by_rel(g_location._edges, "CITY_IN")
    state_nodes = location_nodes[
        (location_nodes["node_type"] == "State")
        & (location_nodes["country"] == country)
    ][["node_id", "state", "country"]]

    path = lives_in.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    path = path.merge(interest_people, left_on="src_person", right_on="node_id")
    grouped = path.groupby("dst_city").size().reset_index(name="numPersons")
    result = grouped.merge(state_nodes, left_on="dst_city", right_on="node_id")
    return result.sort_values(["numPersons", "state", "country"], ascending=[False, True, True]).head(1)


def _query6_polars_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    cities_frame: Any,
    lives_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    lives_in_src_unique: bool,
    gender: str,
    interest: str,
) -> pd.DataFrame:
    del lives_in_src_unique
    import polars as pl  # type: ignore

    people = persons.filter(pl.col("gender_lc") == gender.lower()).select("node_id")
    interests = interests_frame.filter(pl.col("interest_lc") == interest.lower()).select("node_id")
    interest_people = interest_edges.join(interests, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.unique()

    result = (
        interest_people.join(lives_in, left_on="person_id", right_on="src", how="inner")
        .group_by("dst")
        .len()
        .rename({"len": "numPersons"})
        .join(cities_frame, left_on="dst", right_on="node_id", how="inner")
        .select(["city", "country", "numPersons"])
        .sort(["numPersons", "city", "country"], descending=[True, False, False])
        .head(5)
    )
    return result.collect().to_pandas()


def _maybe_build_q6_polars_index_context(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Optional[Tuple[Any, Any, Any, Dict[str, int], Dict[str, Any], bool, bool]]:
    interest_edges_pd = _edges_by_rel(edges_df, "HAS_INTEREST")[["src", "dst"]]
    if len(interest_edges_pd) < Q67_POLARS_MIN_HAS_INTEREST_ROWS:
        return None
    try:
        import polars as pl  # type: ignore
    except Exception:
        return None

    persons = pl.from_pandas(nodes_df[nodes_df["node_type"] == "Person"][["node_id", "gender_lc", "age"]])
    interests = pl.from_pandas(nodes_df[nodes_df["node_type"] == "Interest"][["node_id", "interest_lc"]])
    interest_lc_unique = _is_unique_polars(interests, ["interest_lc"])
    if not interest_lc_unique:
        return None
    interest_lookup = dict(zip(interests["interest_lc"].to_list(), interests["node_id"].to_list()))
    people_by_gender = {
        gender: persons.filter(pl.col("gender_lc") == gender).select("node_id").lazy()
        for gender in persons.select("gender_lc").unique().to_series().to_list()
    }
    interest_edges = pl.from_pandas(pd.DataFrame(interest_edges_pd))
    cities = pl.from_pandas(nodes_df[nodes_df["node_type"] == "City"][["node_id", "city", "country"]])
    lives_in = pl.from_pandas(pd.DataFrame(_edges_by_rel(edges_df, "LIVES_IN")[["src", "dst"]]))
    return (
        interest_edges.lazy(),
        cities.lazy(),
        lives_in.lazy(),
        interest_lookup,
        people_by_gender,
        _is_unique_polars(interest_edges, ["src", "dst"]),
        interest_lc_unique,
    )


def _query6_polars_index_shortcut(
    interest_edges: Any,
    cities_frame: Any,
    lives_in: Any,
    interest_lookup: Dict[str, int],
    people_by_gender: Dict[str, Any],
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    gender: str,
    interest: str,
) -> pd.DataFrame:
    del interest_lc_unique
    import polars as pl  # type: ignore

    interest_id = interest_lookup[interest.lower()]
    people = people_by_gender[gender.lower()]
    interest_people = (
        interest_edges
        .filter(pl.col("dst") == interest_id)
        .join(people, left_on="src", right_on="node_id", how="semi")
        .select(pl.col("src").alias("person_id"))
    )
    if not interest_edges_unique:
        interest_people = interest_people.unique()

    result = (
        lives_in.join(interest_people, left_on="src", right_on="person_id", how="inner")
        .group_by("dst")
        .len()
        .rename({"len": "numPersons"})
        .join(cities_frame.select(["node_id", "city", "country"]), left_on="dst", right_on="node_id", how="inner")
        .select(["city", "country", "numPersons"])
        .sort(["numPersons", "city", "country"], descending=[True, False, False])
        .head(5)
    )
    return result.collect().to_pandas()


def _query7_polars_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    states_frame: Any,
    lives_in: Any,
    city_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    country: str,
    age_lower: int,
    age_upper: int,
    interest: str,
) -> pd.DataFrame:
    import polars as pl  # type: ignore

    people = persons.filter(
        (pl.col("age") >= age_lower)
        & (pl.col("age") <= age_upper)
    ).select("node_id")
    interests = interests_frame.filter(pl.col("interest_lc") == interest.lower()).select("node_id")
    interest_people = interest_edges.join(interests, left_on="dst", right_on="node_id", how="semi")
    interest_people = interest_people.join(people, left_on="src", right_on="node_id", how="semi")
    interest_people = interest_people.select(pl.col("src").alias("person_id"))
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.unique()

    states = states_frame.filter(pl.col("country") == country).select(["node_id", "state", "country"])
    city_in_country = city_in.join(states.select("node_id"), left_on="dst", right_on="node_id", how="semi")
    result = (
        lives_in.join(interest_people, left_on="src", right_on="person_id", how="semi")
        .join(city_in_country, left_on="dst", right_on="src", how="inner", suffix="_city")
        .group_by("dst_city")
        .len()
        .rename({"len": "numPersons"})
        .join(states, left_on="dst_city", right_on="node_id", how="inner")
        .select(["state", "country", "numPersons"])
        .sort(["numPersons", "state", "country"], descending=[True, False, False])
        .head(1)
    )
    return result.collect().to_pandas()


def _maybe_build_q7_polars_context(g_interest: Any, g_location: Any) -> Optional[Tuple[Any, Any, Any, Any, Any, Any, bool, bool]]:
    interest_edges_pd = _edges_by_rel(g_interest._edges, "HAS_INTEREST")[["src", "dst"]]
    if len(interest_edges_pd) < Q67_POLARS_MIN_HAS_INTEREST_ROWS:
        return None
    try:
        import polars as pl  # type: ignore
    except Exception:
        return None

    interest_nodes = pd.DataFrame(g_interest._nodes)
    location_nodes = pd.DataFrame(g_location._nodes)
    location_edges = pd.DataFrame(g_location._edges)
    persons = pl.from_pandas(interest_nodes[interest_nodes["node_type"] == "Person"][["node_id", "age"]])
    interests_frame = pl.from_pandas(interest_nodes[interest_nodes["node_type"] == "Interest"][["node_id", "interest_lc"]])
    interest_edges = pl.from_pandas(pd.DataFrame(interest_edges_pd))
    states = pl.from_pandas(location_nodes[location_nodes["node_type"] == "State"][["node_id", "state", "country"]])
    lives_in = pl.from_pandas(pd.DataFrame(_edges_by_rel(location_edges, "LIVES_IN")[["src", "dst"]]))
    city_in = pl.from_pandas(pd.DataFrame(_edges_by_rel(location_edges, "CITY_IN")[["src", "dst"]]))
    return (
        persons.lazy(),
        interests_frame.lazy(),
        interest_edges.lazy(),
        states.lazy(),
        lives_in.lazy(),
        city_in.lazy(),
        _is_unique_polars(interest_edges, ["src", "dst"]),
        _is_unique_polars(interests_frame, ["interest_lc"]),
    )


def _query6_cudf_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    cities_frame: Any,
    lives_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    lives_in_src_unique: bool,
    gender: str,
    interest: str,
) -> pd.DataFrame:
    del lives_in_src_unique
    people = persons[persons["gender_lc"] == gender.lower()][["node_id"]]
    interests = interests_frame[interests_frame["interest_lc"] == interest.lower()][["node_id"]]
    interest_people = interest_edges[interest_edges["dst"].isin(interests["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]]
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.drop_duplicates()

    matched = lives_in[lives_in["src"].isin(interest_people["src"])]
    grouped = matched.groupby("dst").size().reset_index(name="numPersons")
    result = grouped.merge(cities_frame, left_on="dst", right_on="node_id")
    return result[["city", "country", "numPersons"]].sort_values(
        ["numPersons", "city", "country"],
        ascending=[False, True, True],
    ).head(5).to_pandas().reset_index(drop=True)


def _query7_cudf_shortcut(
    persons: Any,
    interests_frame: Any,
    interest_edges: Any,
    states_frame: Any,
    lives_in: Any,
    city_in: Any,
    interest_edges_unique: bool,
    interest_lc_unique: bool,
    country: str,
    age_lower: int,
    age_upper: int,
    interest: str,
) -> pd.DataFrame:
    people = persons[
        (persons["age"] >= age_lower)
        & (persons["age"] <= age_upper)
    ][["node_id"]]
    interests = interests_frame[interests_frame["interest_lc"] == interest.lower()][["node_id"]]
    interest_people = interest_edges[interest_edges["dst"].isin(interests["node_id"])]
    interest_people = interest_people[interest_people["src"].isin(people["node_id"])][["src"]]
    if not (interest_edges_unique and interest_lc_unique):
        interest_people = interest_people.drop_duplicates()

    states = states_frame[states_frame["country"] == country][["node_id", "state", "country"]]
    matched_lives = lives_in[lives_in["src"].isin(interest_people["src"])]
    path = matched_lives.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    grouped = path.groupby("dst_city").size().reset_index(name="numPersons")
    result = grouped.merge(states, left_on="dst_city", right_on="node_id")
    return result[["state", "country", "numPersons"]].sort_values(
        ["numPersons", "state", "country"],
        ascending=[False, True, True],
    ).head(1).to_pandas().reset_index(drop=True)


def _maybe_build_q7_cudf_context(g_interest: Any, g_location: Any) -> Optional[Tuple[Any, Any, Any, Any, Any, Any, bool, bool]]:
    interest_edges = _edges_by_rel(g_interest._edges, "HAS_INTEREST")[["src", "dst"]]
    if len(interest_edges) < Q67_POLARS_MIN_HAS_INTEREST_ROWS:
        return None
    interest_nodes = g_interest._nodes
    interest_frame = interest_nodes[interest_nodes["node_type"] == "Interest"][["node_id", "interest_lc"]]
    location_nodes = g_location._nodes
    location_edges = g_location._edges
    return (
        interest_nodes[interest_nodes["node_type"] == "Person"][["node_id", "age"]],
        interest_frame,
        interest_edges,
        location_nodes[location_nodes["node_type"] == "State"][["node_id", "state", "country"]],
        _edges_by_rel(location_edges, "LIVES_IN")[["src", "dst"]],
        _edges_by_rel(location_edges, "CITY_IN")[["src", "dst"]],
        _is_unique_by_columns(interest_edges, ["src", "dst"]),
        _is_unique_by_columns(interest_frame, ["interest_lc"]),
    )


def _should_use_q67_cudf_policy(engine: str, mode: str, query_variant: str) -> bool:
    return engine == "cudf" and mode == "preindexed" and query_variant == "standard"


def _should_use_q67_polars_policy(engine: str, mode: str, query_variant: str) -> bool:
    return engine == "pandas" and mode == "preindexed" and query_variant == "standard"


def _query6(
    g_interest: Any,
    g_location: Any,
    engine: str,
    mode: str,
    gender: str,
    interest: str,
    query_variant: str,
) -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q6", engine):
        return _query6_dataframe_shortcut(g_interest, g_location, gender, interest)

    chain_interest = [
        n({"gender_lc": gender.lower()}),
        e_forward(),
        n({"interest_lc": interest.lower()}),
    ] if mode == "preindexed" else [
        n({"node_type": "Person", "gender_lc": gender.lower()}),
        e_forward({"rel": "HAS_INTEREST"}),
        n({"node_type": "Interest", "interest_lc": interest.lower()}),
    ]
    g_interest = g_interest.gfql(chain_interest, engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    chain_location = [
        n(),
        e_forward(),
        n(),
    ] if mode == "preindexed" else [
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
    ]
    g_location = g_location.gfql(chain_location, engine=engine)
    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_nodes = g_location._nodes
    city_nodes = city_nodes[city_nodes["node_type"] == "City"][["node_id", "city", "country"]]

    matched = lives_in.merge(interest_people, left_on="src", right_on="node_id")
    grouped = matched.groupby("dst").size().reset_index(name="numPersons")
    result = grouped.merge(city_nodes, left_on="dst", right_on="node_id")
    return result.sort_values(["numPersons", "city", "country"], ascending=[False, True, True]).head(5)


def _query7(
    g_interest: Any,
    g_location: Any,
    engine: str,
    mode: str,
    country: str,
    age_lower: int,
    age_upper: int,
    interest: str,
    query_variant: str,
) -> pd.DataFrame:
    if _uses_dataframe_shortcut(query_variant, "q7", engine):
        return _query7_dataframe_shortcut(g_interest, g_location, country, age_lower, age_upper, interest)

    chain_interest = [
        n({"age": between(age_lower, age_upper)}),
        e_forward(),
        n({"interest_lc": interest.lower()}),
    ] if mode == "preindexed" else [
        n({"node_type": "Person", "age": between(age_lower, age_upper)}),
        e_forward({"rel": "HAS_INTEREST"}),
        n({"node_type": "Interest", "interest_lc": interest.lower()}),
    ]
    g_interest = g_interest.gfql(chain_interest, engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    chain_location = [
        n(),
        e_forward(),
        n(),
        e_forward(),
        n({"country": country}),
    ] if mode == "preindexed" else [
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
        e_forward({"rel": "CITY_IN"}),
        n({"node_type": "State", "country": country}),
    ]
    g_location = g_location.gfql(chain_location, engine=engine)

    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_in = _edges_by_rel(g_location._edges, "CITY_IN")
    state_nodes = g_location._nodes
    state_nodes = state_nodes[state_nodes["node_type"] == "State"][["node_id", "state", "country"]]

    path = lives_in.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    path = path.merge(interest_people, left_on="src_person", right_on="node_id")
    grouped = path.groupby("dst_city").size().reset_index(name="numPersons")
    result = grouped.merge(state_nodes, left_on="dst_city", right_on="node_id")
    return result.sort_values(["numPersons", "state", "country"], ascending=[False, True, True]).head(1)


def _query8(g: Any) -> pd.DataFrame:
    edges = _edges_by_rel(g._edges, "FOLLOWS")
    indeg = edges.groupby("dst").size().rename("indeg")
    outdeg = edges.groupby("src").size().rename("outdeg")
    degrees = indeg.to_frame().merge(outdeg.to_frame(), left_index=True, right_index=True, how="inner")
    degrees["paths"] = degrees["indeg"] * degrees["outdeg"]
    return pd.DataFrame({"numPaths": [int(degrees["paths"].sum())]})


def _query9(g: Any, age_1: int, age_2: int) -> pd.DataFrame:
    nodes = g._nodes
    persons = nodes[nodes["node_type"] == "Person"][["node_id", "age"]]
    edges = _edges_by_rel(g._edges, "FOLLOWS")

    b_nodes = persons[persons["age"] < age_1][["node_id"]]
    c_nodes = persons[persons["age"] > age_2][["node_id"]]

    in_edges = edges.merge(b_nodes, left_on="dst", right_on="node_id")
    out_edges = edges.merge(c_nodes, left_on="dst", right_on="node_id")
    indeg = in_edges.groupby("dst").size().rename("indeg")
    outdeg = out_edges.groupby("src").size().rename("outdeg")
    degrees = indeg.to_frame().merge(outdeg.to_frame(), left_index=True, right_index=True, how="inner")
    degrees["paths"] = degrees["indeg"] * degrees["outdeg"]
    return pd.DataFrame({"numPaths": [int(degrees["paths"].sum())]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--engine", choices=["pandas", "cudf"], default="pandas")
    parser.add_argument("--mode", choices=["baseline", "preindexed", "presorted"], default=DEFAULT_MODE)
    parser.add_argument(
        "--include-preindex",
        action="store_true",
        help="For preindexed mode, report per-query medians including preindex build time.",
    )
    parser.add_argument(
        "--query-variant",
        choices=["standard", "reverse-seeded", "dataframe-shortcut"],
        default="standard",
        help=(
            "Query implementation variant. 'standard' applies the simple benchmark policy "
            "(direct dataframe shortcuts for q1/q3/q4/q5/q6/q7, plus scoped large-q5/q6/q7 "
            "lazy Polars CPU and typed cudf GPU semijoin/groupby paths for preindexed runs when available); "
            "'dataframe-shortcut' forces direct dataframe aggregations for q1/q3/q4/q5/q6/q7."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    nodes_path = args.graph_benchmark_root / "data" / "output" / "nodes"
    edges_path = args.graph_benchmark_root / "data" / "output" / "edges"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Missing data at {nodes_path} or {edges_path}. Run generate_data.sh in graph-benchmark first."
        )

    nodes_df, offsets = _load_nodes(nodes_path)
    edges_df = _load_edges(edges_path, offsets)

    nodes = _maybe_to_cudf(args.engine, nodes_df)
    edges = _maybe_to_cudf(args.engine, edges_df)

    if args.include_preindex and args.mode != "preindexed":
        raise ValueError("--include-preindex requires --mode preindexed")

    if args.mode == "presorted":
        nodes = nodes.sort_values(["node_type", "node_id"])
        edges = edges.sort_values(["rel", "src", "dst"])

    g_full = graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")

    results: Dict[str, Dict[str, Any]] = {}
    preindex_ms_by_query: Dict[str, float] = {}
    preindex_total_ms: Optional[float] = None

    def _run(label: str, fn: Callable[[], pd.DataFrame]) -> None:
        _, times = _timed(label, fn, runs=args.runs, warmup=args.warmup)
        median_ms = _median(times)
        result = {
            "median_ms": median_ms,
            "runs": times,
        }
        if args.include_preindex and label in preindex_ms_by_query:
            preindex_ms = preindex_ms_by_query[label]
            result["preindex_ms"] = preindex_ms
            result["median_ms_with_preindex"] = median_ms + preindex_ms
        results[label] = result

    if args.mode == "preindexed":
        preindex_graphs: Dict[str, Tuple[List[str], List[str]]] = {
            "g_q1": (["Person"], ["FOLLOWS"]),
            "g_q2_lives": (["Person", "City"], ["LIVES_IN"]),
            "g_q3": (["Person", "City", "State", "Country"], ["LIVES_IN", "CITY_IN", "STATE_IN"]),
            "g_q5_interest": (["Person", "Interest"], ["HAS_INTEREST"]),
            "g_q5_location": (["Person", "City"], ["LIVES_IN"]),
            "g_q7_interest": (["Person", "Interest"], ["HAS_INTEREST"]),
            "g_q7_location": (["Person", "City", "State"], ["LIVES_IN", "CITY_IN"]),
        }
        preindex_by_query: Dict[str, List[str]] = {
            "q1": ["g_q1"],
            "q2": ["g_q1", "g_q2_lives"],
            "q3": ["g_q3"],
            "q4": ["g_q3"],
            "q5": ["g_q5_interest", "g_q5_location"],
            "q6": ["g_q5_interest", "g_q5_location"],
            "q7": ["g_q7_interest", "g_q7_location"],
            "q8": ["g_q1"],
            "q9": ["g_q1"],
        }

        if args.include_preindex:
            for label, graph_names in preindex_by_query.items():
                spec = {name: preindex_graphs[name] for name in graph_names}
                start = perf_counter()
                label_graphs = _build_preindexed_graphs(nodes, edges, nodes_df, edges_df, args.engine, spec)
                if (
                    label == "q5"
                    and _should_use_q5_polars_policy(args.engine, args.mode, args.query_variant)
                    and "g_q5_interest" in label_graphs
                    and "g_q5_location" in label_graphs
                ):
                    _maybe_build_q5_polars_context(label_graphs["g_q5_interest"], label_graphs["g_q5_location"])
                if (
                    label == "q5"
                    and _should_use_q5_cudf_policy(args.engine, args.mode, args.query_variant)
                    and "g_q5_interest" in label_graphs
                    and "g_q5_location" in label_graphs
                ):
                    _maybe_build_q5_cudf_context(label_graphs["g_q5_interest"], label_graphs["g_q5_location"])
                if (
                    label == "q6"
                    and _should_use_q67_polars_policy(args.engine, args.mode, args.query_variant)
                ):
                    _maybe_build_q6_polars_index_context(nodes_df, edges_df)
                if (
                    label == "q6"
                    and _should_use_q67_cudf_policy(args.engine, args.mode, args.query_variant)
                    and "g_q5_interest" in label_graphs
                    and "g_q5_location" in label_graphs
                ):
                    _maybe_build_q5_cudf_context(label_graphs["g_q5_interest"], label_graphs["g_q5_location"])
                if (
                    label == "q7"
                    and _should_use_q67_polars_policy(args.engine, args.mode, args.query_variant)
                    and "g_q7_interest" in label_graphs
                    and "g_q7_location" in label_graphs
                ):
                    _maybe_build_q7_polars_context(label_graphs["g_q7_interest"], label_graphs["g_q7_location"])
                if (
                    label == "q7"
                    and _should_use_q67_cudf_policy(args.engine, args.mode, args.query_variant)
                    and "g_q7_interest" in label_graphs
                    and "g_q7_location" in label_graphs
                ):
                    _maybe_build_q7_cudf_context(label_graphs["g_q7_interest"], label_graphs["g_q7_location"])
                preindex_ms_by_query[label] = (perf_counter() - start) * 1000.0

        start = perf_counter()
        all_graphs = _build_preindexed_graphs(
            nodes,
            edges,
            nodes_df,
            edges_df,
            args.engine,
            preindex_graphs,
        )
        preindex_total_ms = (perf_counter() - start) * 1000.0

        g_q1 = all_graphs["g_q1"]
        g_q2_follow = g_q1
        g_q2_lives = all_graphs["g_q2_lives"]
        g_q3 = all_graphs["g_q3"]
        g_q4 = g_q3
        g_q5_interest = all_graphs["g_q5_interest"]
        g_q5_location = all_graphs["g_q5_location"]
        q5_polars_context = (
            _maybe_build_q5_polars_context(g_q5_interest, g_q5_location)
            if _should_use_q5_polars_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        q5_cudf_context = (
            _maybe_build_q5_cudf_context(g_q5_interest, g_q5_location)
            if _should_use_q5_cudf_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        g_q6_interest = g_q5_interest
        g_q6_location = g_q5_location
        q6_polars_context = (
            _maybe_build_q6_polars_index_context(nodes_df, edges_df)
            if _should_use_q67_polars_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        q6_cudf_context = (
            q5_cudf_context
            if _should_use_q67_cudf_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        g_q7_interest = all_graphs["g_q7_interest"]
        g_q7_location = all_graphs["g_q7_location"]
        q7_polars_context = (
            _maybe_build_q7_polars_context(g_q7_interest, g_q7_location)
            if _should_use_q67_polars_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        q7_cudf_context = (
            _maybe_build_q7_cudf_context(g_q7_interest, g_q7_location)
            if _should_use_q67_cudf_policy(args.engine, args.mode, args.query_variant)
            else None
        )
        g_q8 = g_q1
        g_q9 = g_q8
    else:
        g_q1 = g_full
        g_q2_follow = g_full
        g_q2_lives = g_full
        g_q3 = g_full
        g_q4 = g_full
        g_q5_interest = g_full
        g_q5_location = g_full
        q5_polars_context = None
        q5_cudf_context = None
        g_q6_interest = g_full
        g_q6_location = g_full
        q6_polars_context = None
        q6_cudf_context = None
        g_q7_interest = g_full
        g_q7_location = g_full
        q7_polars_context = None
        q7_cudf_context = None
        g_q8 = g_full
        g_q9 = g_full

    _run("q1", lambda: _query1(g_q1, args.engine, args.mode, args.query_variant))
    _run("q2", lambda: _query2(g_q2_follow, g_q2_lives, args.engine, args.mode))
    _run("q3", lambda: _query3(g_q3, args.engine, args.mode, country="United States", query_variant=args.query_variant))
    _run(
        "q4",
        lambda: _query4(
            g_q4,
            args.engine,
            args.mode,
            age_lower=30,
            age_upper=40,
            query_variant=args.query_variant,
        ),
    )
    if q5_cudf_context is not None:
        _run(
            "q5",
            lambda: _query5_cudf_shortcut(
                *q5_cudf_context,
                gender="male",
                city="London",
                country="United Kingdom",
                interest="fine dining",
            ),
        )
    elif q5_polars_context is not None:
        _run(
            "q5",
            lambda: _query5_polars_shortcut(
                *q5_polars_context,
                gender="male",
                city="London",
                country="United Kingdom",
                interest="fine dining",
            ),
        )
    else:
        _run(
            "q5",
            lambda: _query5(
                g_q5_interest,
                g_q5_location,
                args.engine,
                args.mode,
                gender="male",
                city="London",
                country="United Kingdom",
                interest="fine dining",
                query_variant=args.query_variant,
            ),
        )
    if q6_cudf_context is not None:
        _run(
            "q6",
            lambda: _query6_cudf_shortcut(
                *q6_cudf_context,
                gender="female",
                interest="tennis",
            ),
        )
    elif q6_polars_context is not None:
        _run(
            "q6",
            lambda: _query6_polars_index_shortcut(
                *q6_polars_context,
                gender="female",
                interest="tennis",
            ),
        )
    else:
        _run(
            "q6",
            lambda: _query6(
                g_q6_interest,
                g_q6_location,
                args.engine,
                args.mode,
                gender="female",
                interest="tennis",
                query_variant=args.query_variant,
            ),
        )
    if q7_cudf_context is not None:
        _run(
            "q7",
            lambda: _query7_cudf_shortcut(
                *q7_cudf_context,
                country="United States",
                age_lower=23,
                age_upper=30,
                interest="photography",
            ),
        )
    elif q7_polars_context is not None:
        _run(
            "q7",
            lambda: _query7_polars_shortcut(
                *q7_polars_context,
                country="United States",
                age_lower=23,
                age_upper=30,
                interest="photography",
            ),
        )
    else:
        _run(
            "q7",
            lambda: _query7(
                g_q7_interest,
                g_q7_location,
                args.engine,
                args.mode,
                country="United States",
                age_lower=23,
                age_upper=30,
                interest="photography",
                query_variant=args.query_variant,
            ),
        )
    _run("q8", lambda: _query8(g_q8))
    _run("q9", lambda: _query9(g_q9, age_1=50, age_2=25))

    q5_policy = "gfql_dataframe_shortcut"
    if q5_cudf_context is not None:
        q5_policy = "cudf_semijoin_count_large_has_interest"
    elif q5_polars_context is not None:
        q5_policy = "polars_lazy_semijoin_count_large_has_interest"

    q6_policy = "gfql_dataframe_shortcut"
    if q6_cudf_context is not None:
        q6_policy = "cudf_semijoin_groupby_large_has_interest"
    elif q6_polars_context is not None:
        q6_policy = "polars_interest_id_gender_index_groupby_large_has_interest"

    q7_policy = "gfql_dataframe_shortcut"
    if q7_cudf_context is not None:
        q7_policy = "cudf_country_pruned_semijoin_groupby_large_has_interest"
    elif q7_polars_context is not None:
        q7_policy = "polars_country_pruned_semijoin_groupby_large_has_interest"

    output = {
        "engine": args.engine,
        "mode": args.mode,
        "preindex_total_ms": preindex_total_ms,
        "query_policies": {
            "q1": (
                "dataframe_shortcut_follow_indegree"
                if _uses_dataframe_shortcut(args.query_variant, "q1", args.engine)
                else "gfql_follow_indegree"
            ),
            "q2": "top_id_then_location_lookup",
            "q3": "dataframe_shortcut_country_city_age_avg",
            "q4": "dataframe_shortcut_age_country_counts",
            "q5": q5_policy,
            "q6": q6_policy,
            "q7": q7_policy,
            "q8": "dataframe_degree_product",
            "q9": "dataframe_filtered_degree_product",
        },
        "q5_polars_min_has_interest_rows": Q5_POLARS_MIN_HAS_INTEREST_ROWS,
        "q5_polars_policy_active": q5_polars_context is not None,
        "q5_polars_interest_edges_unique": q5_polars_context[5] if q5_polars_context is not None else None,
        "q5_polars_interest_lc_unique": q5_polars_context[6] if q5_polars_context is not None else None,
        "q5_polars_lives_in_src_unique": q5_polars_context[7] if q5_polars_context is not None else None,
        "q5_cudf_policy_active": q5_cudf_context is not None,
        "q5_cudf_interest_edges_unique": q5_cudf_context[5] if q5_cudf_context is not None else None,
        "q5_cudf_interest_lc_unique": q5_cudf_context[6] if q5_cudf_context is not None else None,
        "q5_cudf_lives_in_src_unique": q5_cudf_context[7] if q5_cudf_context is not None else None,
        "q67_polars_min_has_interest_rows": Q67_POLARS_MIN_HAS_INTEREST_ROWS,
        "q6_polars_policy_active": q6_polars_context is not None,
        "q6_polars_interest_edges_unique": q6_polars_context[5] if q6_polars_context is not None else None,
        "q6_polars_interest_lc_unique": q6_polars_context[6] if q6_polars_context is not None else None,
        "q6_polars_index_context": q6_polars_context is not None,
        "q7_polars_policy_active": q7_polars_context is not None,
        "q7_polars_interest_edges_unique": q7_polars_context[6] if q7_polars_context is not None else None,
        "q7_polars_interest_lc_unique": q7_polars_context[7] if q7_polars_context is not None else None,
        "q6_cudf_policy_active": q6_cudf_context is not None,
        "q7_cudf_policy_active": q7_cudf_context is not None,
        "q6_cudf_interest_edges_unique": q6_cudf_context[5] if q6_cudf_context is not None else None,
        "q6_cudf_interest_lc_unique": q6_cudf_context[6] if q6_cudf_context is not None else None,
        "q7_cudf_interest_edges_unique": q7_cudf_context[6] if q7_cudf_context is not None else None,
        "q7_cudf_interest_lc_unique": q7_cudf_context[7] if q7_cudf_context is not None else None,
        "query_variant": args.query_variant,
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
