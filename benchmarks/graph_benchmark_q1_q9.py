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
from graphistry.compute.ast import n, e_forward
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
    ("interests.parquet", "HAS_INTEREST", "Person", "Interest"),
    ("city_in.parquet", "CITY_IN", "City", "State"),
    ("state_in.parquet", "STATE_IN", "State", "Country"),
]


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
        df = pd.read_parquet(edges_path / filename).rename(columns={"from": "src", "to": "dst"})
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


def _edges_by_rel(edges: Any, rel: str) -> Any:
    return edges[edges["rel"] == rel]


def _nodes_by_type(nodes: Any, node_type: str) -> Any:
    return nodes[nodes["node_type"] == node_type]


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


def _query1(g: Any, engine: str) -> pd.DataFrame:
    gq = g.gfql([
        n({"node_type": "Person"}),
        e_forward({"rel": "FOLLOWS"}),
        n({"node_type": "Person"}),
    ], engine=engine)
    edges = gq._edges
    nodes = gq._nodes
    dst_col = gq._destination
    counts = edges.groupby(dst_col).size().reset_index(name="numFollowers")
    persons = nodes[["node_id", "name"]].drop_duplicates()
    result = counts.merge(persons, left_on=dst_col, right_on="node_id")
    return result.sort_values("numFollowers", ascending=False).head(3)


def _query2(g: Any, engine: str) -> pd.DataFrame:
    top = _query1(g, engine)
    top_id = int(top.iloc[0]["node_id"])
    gq = g.gfql([
        n({"node_id": top_id}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
    ], engine=engine)
    nodes = gq._nodes
    person = nodes[nodes["node_type"] == "Person"][["node_id", "name"]]
    city = nodes[nodes["node_type"] == "City"][["node_id", "city", "state", "country"]]
    edges = _edges_by_rel(gq._edges, "LIVES_IN")
    joined = edges.merge(person, left_on="src", right_on="node_id")
    joined = joined.merge(city, left_on="dst", right_on="node_id", suffixes=("_person", "_city"))
    return joined[["name", "city", "state", "country"]]


def _query3(g: Any, engine: str, country: str) -> pd.DataFrame:
    gq = g.gfql([
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
        e_forward({"rel": "CITY_IN"}),
        n({"node_type": "State"}),
        e_forward({"rel": "STATE_IN"}),
        n({"node_type": "Country", "country": country}),
    ], engine=engine)
    nodes = gq._nodes
    edges = gq._edges
    persons = nodes[nodes["node_type"] == "Person"][["node_id", "age"]]
    cities = nodes[nodes["node_type"] == "City"][["node_id", "city"]]
    lives_in = _edges_by_rel(edges, "LIVES_IN")
    merged = lives_in.merge(persons, left_on="src", right_on="node_id")
    merged = merged.merge(cities, left_on="dst", right_on="node_id", suffixes=("_person", "_city"))
    avg_age = merged.groupby("city")["age"].mean().reset_index(name="averageAge")
    return avg_age.sort_values("averageAge").head(5)


def _query4(g: Any, engine: str, age_lower: int, age_upper: int) -> pd.DataFrame:
    gq = g.gfql([
        n({"node_type": "Person", "age": between(age_lower, age_upper)}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
        e_forward({"rel": "CITY_IN"}),
        n({"node_type": "State"}),
        e_forward({"rel": "STATE_IN"}),
        n({"node_type": "Country"}),
    ], engine=engine)
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


def _query5(g: Any, engine: str, gender: str, city: str, country: str, interest: str) -> pd.DataFrame:
    g_interest = g.gfql([
        n({"node_type": "Person", "gender_lc": gender.lower()}),
        e_forward({"rel": "HAS_INTEREST"}),
        n({"node_type": "Interest", "interest_lc": interest.lower()}),
    ], engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    g_location = g.gfql([
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City", "city": city, "country": country}),
    ], engine=engine)
    location_edges = _edges_by_rel(g_location._edges, "LIVES_IN")
    location_people = location_edges[["src"]].rename(columns={"src": "node_id"}).drop_duplicates()

    matched = interest_people.merge(location_people, on="node_id")
    return pd.DataFrame({"numPersons": [len(matched)]})


def _query6(g: Any, engine: str, gender: str, interest: str) -> pd.DataFrame:
    g_interest = g.gfql([
        n({"node_type": "Person", "gender_lc": gender.lower()}),
        e_forward({"rel": "HAS_INTEREST"}),
        n({"node_type": "Interest", "interest_lc": interest.lower()}),
    ], engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    g_location = g.gfql([
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
    ], engine=engine)
    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_nodes = g_location._nodes
    city_nodes = city_nodes[city_nodes["node_type"] == "City"][["node_id", "city", "country"]]

    matched = lives_in.merge(interest_people, left_on="src", right_on="node_id")
    grouped = matched.groupby("dst").size().reset_index(name="numPersons")
    result = grouped.merge(city_nodes, left_on="dst", right_on="node_id")
    return result.sort_values("numPersons", ascending=False).head(5)


def _query7(
    g: Any, engine: str, country: str, age_lower: int, age_upper: int, interest: str
) -> pd.DataFrame:
    g_interest = g.gfql([
        n({"node_type": "Person", "age": between(age_lower, age_upper)}),
        e_forward({"rel": "HAS_INTEREST"}),
        n({"node_type": "Interest", "interest_lc": interest.lower()}),
    ], engine=engine)
    interest_people = g_interest._nodes
    interest_people = interest_people[interest_people["node_type"] == "Person"][["node_id"]]

    g_location = g.gfql([
        n({"node_type": "Person"}),
        e_forward({"rel": "LIVES_IN"}),
        n({"node_type": "City"}),
        e_forward({"rel": "CITY_IN"}),
        n({"node_type": "State", "country": country}),
    ], engine=engine)

    lives_in = _edges_by_rel(g_location._edges, "LIVES_IN")
    city_in = _edges_by_rel(g_location._edges, "CITY_IN")
    state_nodes = g_location._nodes
    state_nodes = state_nodes[state_nodes["node_type"] == "State"][["node_id", "state", "country"]]

    path = lives_in.merge(city_in, left_on="dst", right_on="src", suffixes=("_person", "_city"))
    path = path.merge(interest_people, left_on="src_person", right_on="node_id")
    grouped = path.groupby("dst_city").size().reset_index(name="numPersons")
    result = grouped.merge(state_nodes, left_on="dst_city", right_on="node_id")
    return result.sort_values("numPersons", ascending=False).head(1)


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

    g = graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")

    results: Dict[str, Dict[str, Any]] = {}

    def _run(label: str, fn: Callable[[], pd.DataFrame]) -> None:
        _, times = _timed(label, fn, runs=args.runs, warmup=args.warmup)
        results[label] = {
            "median_ms": _median(times),
            "runs": times,
        }

    _run("q1", lambda: _query1(g, args.engine))
    _run("q2", lambda: _query2(g, args.engine))
    _run("q3", lambda: _query3(g, args.engine, country="United States"))
    _run("q4", lambda: _query4(g, args.engine, age_lower=30, age_upper=40))
    _run(
        "q5",
        lambda: _query5(
            g,
            args.engine,
            gender="male",
            city="London",
            country="United Kingdom",
            interest="fine dining",
        ),
    )
    _run("q6", lambda: _query6(g, args.engine, gender="female", interest="tennis"))
    _run(
        "q7",
        lambda: _query7(
            g,
            args.engine,
            country="United States",
            age_lower=23,
            age_upper=30,
            interest="photography",
        ),
    )
    _run("q8", lambda: _query8(g))
    _run("q9", lambda: _query9(g, age_1=50, age_2=25))

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
