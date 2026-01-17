#!/usr/bin/env python3
"""
Run GFQL chain benchmarks on real datasets (no WHERE predicates).

This is intended for hop/chain performance sanity checks on medium-scale data.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import n, e_forward, e_reverse


@dataclass(frozen=True)
class Scenario:
    name: str
    chain: List


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[Engine], graphistry.Plottable]
    scenarios: List[Scenario]


@dataclass
class TimingStats:
    median_ms: float
    p90_ms: float
    std_ms: float


@dataclass
class ResultRow:
    dataset: str
    scenario: str
    median_ms: Optional[float]
    p90_ms: Optional[float]
    std_ms: Optional[float]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight


def _summarize_times(times: List[float]) -> TimingStats:
    ordered = sorted(times)
    median_ms = statistics.median(ordered)
    p90_ms = _percentile(ordered, 0.9)
    std_ms = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return TimingStats(median_ms=median_ms, p90_ms=p90_ms, std_ms=std_ms)


def _time_call(fn, runs: int, warmup: int) -> TimingStats:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return _summarize_times(times)


def _as_engine(engine_label: str) -> Engine:
    return Engine.CUDF if engine_label == "cudf" else Engine.PANDAS


def _maybe_to_cudf(df: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    if engine == Engine.CUDF:
        import cudf  # type: ignore

        return cudf.from_pandas(df)
    return df


def _extract_domain(value: str) -> str:
    if isinstance(value, str) and "@" in value:
        return value.split("@", 1)[1]
    return value


def _degree_nodes(edges: pd.DataFrame, src_col: str, dst_col: str, threshold: int) -> pd.DataFrame:
    degree = edges[src_col].value_counts().add(edges[dst_col].value_counts(), fill_value=0)
    nodes = pd.DataFrame({"id": degree.index, "degree": degree.values.astype(int)})
    nodes["high_degree"] = nodes["degree"] >= threshold
    return nodes


def load_redteam(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/graphistry_redteam50k.csv")
    edges = edges.rename(columns={"src_computer": "src", "dst_computer": "dst"})
    edges["src_domain_parsed"] = edges["src_domain"].map(_extract_domain)
    edges["dst_domain_parsed"] = edges["dst_domain"].map(_extract_domain)

    nodes_src = edges[["src", "src_domain_parsed"]].rename(
        columns={"src": "id", "src_domain_parsed": "domain"}
    )
    nodes_dst = edges[["dst", "dst_domain_parsed"]].rename(
        columns={"dst": "id", "dst_domain_parsed": "domain"}
    )
    nodes = pd.concat([nodes_src, nodes_dst], ignore_index=True).dropna(subset=["id"])
    nodes = nodes.groupby("id", as_index=False).first()

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_transactions(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/transactions.csv", lineterminator="\r")
    edges = edges.rename(
        columns={
            "Amount $": "amount",
            "Date": "date",
            "Destination": "dst",
            "Source": "src",
            "Transaction ID": "tx_id",
            "isTainted": "is_tainted",
        }
    )
    edges["is_tainted"] = edges["is_tainted"].astype("int64")
    nodes = pd.DataFrame({"id": pd.unique(pd.concat([edges["src"], edges["dst"]]))})
    tainted_in = edges.loc[edges["is_tainted"] == 5, "dst"].unique()
    nodes["tainted_in"] = nodes["id"].isin(tainted_in)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_facebook(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv(
        "demos/data/facebook_combined.txt",
        sep=" ",
        header=None,
        names=["src", "dst"],
    )
    nodes = _degree_nodes(edges, "src", "dst", threshold=50)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_honeypot(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/honeypot.csv")
    edges = edges.rename(columns={"attackerIP": "src", "victimIP": "dst"})
    edges["victimPort"] = edges["victimPort"].astype("int64")
    edges["count"] = edges["count"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=2)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_twitter_demo(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/twitterDemo.csv")
    edges = edges.rename(columns={"srcAccount": "src", "dstAccount": "dst"})
    nodes = _degree_nodes(edges, "src", "dst", threshold=5)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_lesmiserables(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/lesmiserables.csv")
    edges = edges.rename(columns={"source": "src", "target": "dst"})
    edges["value"] = edges["value"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=5)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_twitter_congress(engine: Engine) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/twitter_congress_edges_weighted.csv.gz")
    edges = edges.rename(columns={"from": "src", "to": "dst"})
    edges["weight"] = edges["weight"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=10)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def build_specs() -> List[DatasetSpec]:
    redteam_scenarios = [
        Scenario(
            "kerberos_logon_fanin",
            [
                n({"domain": "DOM1"}, name="a"),
                e_forward(
                    {"auth_type": "Kerberos", "success_or_failure": "Success"},
                    name="e1",
                ),
                n(name="hub"),
                e_reverse({"authentication_orientation": "LogOn"}, name="e2"),
                n(name="c"),
            ],
        ),
        Scenario(
            "ntlm_network_chain",
            [
                n(),
                e_forward({"auth_type": "NTLM"}, name="e1"),
                n(name="mid"),
                e_forward({"logontype": "Network"}, name="e2"),
                n(name="dst"),
            ],
        ),
        Scenario(
            "kerberos_fanin_simple",
            [
                n(name="a"),
                e_forward({"auth_type": "Kerberos"}, name="e1"),
                n(name="b"),
                e_reverse({"authentication_orientation": "LogOn"}, name="e2"),
                n(name="c"),
            ],
        ),
    ]

    transactions_scenarios = [
        Scenario(
            "tainted_fanin",
            [
                n(),
                e_forward({"is_tainted": 5}, name="e1"),
                n(name="hub"),
                e_reverse({"is_tainted": 0}, name="e2"),
                n(),
            ],
        ),
        Scenario(
            "large_to_small",
            [
                n(),
                e_forward(edge_query="amount > 10000", name="e1"),
                n(name="mid"),
                e_forward(edge_query="amount < 10", name="e2"),
                n(),
            ],
        ),
        Scenario(
            "tainted_fanin_seeded",
            [
                n({"tainted_in": True}, name="a"),
                e_forward({"is_tainted": 5}, name="e1"),
                n(name="b"),
                e_reverse({"is_tainted": 0}, name="e2"),
                n(name="c"),
            ],
        ),
    ]

    facebook_scenarios = [
        Scenario(
            "high_degree_fanin",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="hub"),
                e_reverse(name="e2"),
                n(),
            ],
        ),
        Scenario(
            "two_hop",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="mid"),
                e_forward(name="e2"),
                n(),
            ],
        ),
        Scenario(
            "high_degree_fanin_rev",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_reverse(name="e2"),
                n({"high_degree": True}, name="c"),
            ],
        ),
    ]

    honeypot_scenarios = [
        Scenario(
            "smb_fanin",
            [
                n(),
                e_forward({"victimPort": 139}, name="e1"),
                n(name="hub"),
                e_reverse({"victimPort": 139}, name="e2"),
                n(),
            ],
        ),
        Scenario(
            "vuln_chain",
            [
                n({"high_degree": True}, name="a"),
                e_forward({"vulnName": "MS08067 (NetAPI)"}, name="e1"),
                n(name="mid"),
                e_forward(edge_query="count >= 3", name="e2"),
                n(),
            ],
        ),
    ]

    twitter_demo_scenarios = [
        Scenario(
            "fan_in",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="hub"),
                e_reverse(name="e2"),
                n(),
            ],
        ),
        Scenario(
            "two_hop",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="mid"),
                e_forward(name="e2"),
                n(),
            ],
        ),
    ]

    lesmiserables_scenarios = [
        Scenario(
            "weighted_fanin",
            [
                n(),
                e_forward(edge_query="value >= 5", name="e1"),
                n(name="hub"),
                e_reverse(edge_query="value >= 5", name="e2"),
                n(),
            ],
        ),
        Scenario(
            "high_degree_two_hop",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="mid"),
                e_forward(name="e2"),
                n(),
            ],
        ),
    ]

    twitter_congress_scenarios = [
        Scenario(
            "weighted_fanin",
            [
                n(),
                e_forward(edge_query="weight >= 2", name="e1"),
                n(name="hub"),
                e_reverse(edge_query="weight >= 2", name="e2"),
                n(),
            ],
        ),
        Scenario(
            "high_degree_two_hop",
            [
                n({"high_degree": True}, name="a"),
                e_forward(name="e1"),
                n(name="mid"),
                e_forward(name="e2"),
                n(),
            ],
        ),
    ]

    return [
        DatasetSpec("redteam50k", load_redteam, redteam_scenarios),
        DatasetSpec("transactions", load_transactions, transactions_scenarios),
        DatasetSpec("facebook_combined", load_facebook, facebook_scenarios),
        DatasetSpec("honeypot", load_honeypot, honeypot_scenarios),
        DatasetSpec("twitter_demo", load_twitter_demo, twitter_demo_scenarios),
        DatasetSpec("lesmiserables", load_lesmiserables, lesmiserables_scenarios),
        DatasetSpec("twitter_congress", load_twitter_congress, twitter_congress_scenarios),
    ]


def run_scenarios(
    dataset: DatasetSpec, engine_label: str, runs: int, warmup: int
) -> Iterable[ResultRow]:
    engine = _as_engine(engine_label)
    g = dataset.loader(engine)

    for scenario in dataset.scenarios:
        def _call() -> None:
            g.gfql(scenario.chain, engine=engine_label)

        stats = _time_call(_call, runs, warmup)
        yield ResultRow(
            dataset=dataset.name,
            scenario=scenario.name,
            median_ms=stats.median_ms,
            p90_ms=stats.p90_ms,
            std_ms=stats.std_ms,
        )


def write_markdown(results: Iterable[ResultRow], output_path: str) -> None:
    header = [
        "# Real-Data Benchmark Results",
        "",
        "Notes:",
        "- No WHERE predicates; uses chain-style GFQL only.",
        "- Datasets are loaded from `demos/data/`.",
        "- Values are median over runs; p90 and std columns show variability.",
        "",
        "| Dataset | Scenario | Median | P90 | Std |",
        "|---------|----------|--------|-----|-----|",
    ]
    lines = header + [
        f"| {row.dataset} | {row.scenario} | {row.median_ms:.2f}ms | {row.p90_ms:.2f}ms | {row.std_ms:.2f}ms |"
        for row in results
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data GFQL benchmarks (no WHERE).")
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated list: redteam50k,transactions,facebook_combined,honeypot,twitter_demo,lesmiserables,twitter_congress,all",
    )
    args = parser.parse_args()

    dataset_filter = {d.strip() for d in args.datasets.split(",")} if args.datasets else {"all"}
    specs = build_specs()
    if "all" not in dataset_filter:
        specs = [s for s in specs if s.name in dataset_filter]

    results: List[ResultRow] = []
    for dataset in specs:
        results.extend(run_scenarios(dataset, args.engine, args.runs, args.warmup))

    if args.output:
        write_markdown(results, args.output)

    print("| Dataset | Scenario | Median | P90 | Std |")
    print("|---------|----------|--------|-----|-----|")
    for row in results:
        print(
            f"| {row.dataset} | {row.scenario} | {row.median_ms:.2f}ms |"
            f" {row.p90_ms:.2f}ms | {row.std_ms:.2f}ms |"
        )


if __name__ == "__main__":
    main()
