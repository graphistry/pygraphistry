#!/usr/bin/env python3
"""
Run GFQL chain benchmarks on real datasets (no WHERE predicates).

This is intended for hop/chain performance sanity checks on medium-scale data.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import n, e_forward, e_reverse
from graphistry.compute.gfql.df_executor import execute_same_path_chain
from graphistry.compute.gfql.same_path_types import WhereComparison, col, compare
from otel_setup import setup_tracer


@dataclass(frozen=True)
class Scenario:
    name: str
    chain: List


@dataclass(frozen=True)
class WhereScenario:
    name: str
    chain: List
    where: List[WhereComparison]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[Engine], graphistry.Plottable]
    scenarios: List[Scenario]
    where_scenarios: List[WhereScenario]


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


def _time_call(
    fn,
    runs: int,
    warmup: int,
    max_total_s: Optional[float] = None,
    max_call_s: Optional[float] = None,
) -> Optional[TimingStats]:
    total_start = time.perf_counter()
    for _ in range(warmup):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        if max_call_s is not None and elapsed > max_call_s:
            return None
        if max_total_s is not None and (time.perf_counter() - total_start) > max_total_s:
            return None
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        if max_call_s is not None and elapsed > max_call_s:
            return None
        times.append(elapsed * 1000)
        if max_total_s is not None and (time.perf_counter() - total_start) > max_total_s:
            return None
    return _summarize_times(times)


def _as_engine(engine_label: str) -> Engine:
    return Engine.CUDF if engine_label == "cudf" else Engine.PANDAS


def _parse_filters(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def _add_ndv_probe_columns(
    nodes: pd.DataFrame,
    id_col: str = "id",
    buckets: int = 3,
) -> pd.DataFrame:
    if buckets <= 0:
        buckets = 3
    ids = nodes[id_col].astype(str)
    hashed = pd.util.hash_pandas_object(ids, index=False)
    nodes = nodes.copy()
    nodes["ndv_hi"] = hashed
    nodes["ndv_lo"] = (hashed % buckets).astype("int64")
    return nodes


def _log_ndv(label: str, nodes: pd.DataFrame, cols: Iterable[str]) -> None:
    stats = {}
    for col in cols:
        if col in nodes.columns:
            stats[col] = int(nodes[col].nunique(dropna=True))
    if stats:
        summary = ", ".join(f"{key}={value}" for key, value in stats.items())
        print(f"NDV[{label}]: {summary}")


def load_redteam(
    engine: Engine,
    domain_categorical: bool = False,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
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
    if domain_categorical:
        nodes["domain"] = nodes["domain"].astype("category")
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["domain"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("redteam50k", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_transactions(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
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
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["tainted_in"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("transactions", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_facebook(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
    edges = pd.read_csv(
        "demos/data/facebook_combined.txt",
        sep=" ",
        header=None,
        names=["src", "dst"],
    )
    nodes = _degree_nodes(edges, "src", "dst", threshold=50)
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["degree", "high_degree"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("facebook_combined", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_honeypot(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/honeypot.csv")
    edges = edges.rename(columns={"attackerIP": "src", "victimIP": "dst"})
    edges["victimPort"] = edges["victimPort"].astype("int64")
    edges["count"] = edges["count"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=2)
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["degree", "high_degree"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("honeypot", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_twitter_demo(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/twitterDemo.csv")
    edges = edges.rename(columns={"srcAccount": "src", "dstAccount": "dst"})
    nodes = _degree_nodes(edges, "src", "dst", threshold=5)
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["degree", "high_degree"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("twitter_demo", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_lesmiserables(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/lesmiserables.csv")
    edges = edges.rename(columns={"source": "src", "target": "dst"})
    edges["value"] = edges["value"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=5)
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["degree", "high_degree"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("lesmiserables", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def load_twitter_congress(
    engine: Engine,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> graphistry.Plottable:
    edges = pd.read_csv("demos/data/twitter_congress_edges_weighted.csv.gz")
    edges = edges.rename(columns={"from": "src", "to": "dst"})
    edges["weight"] = edges["weight"].astype("int64")
    nodes = _degree_nodes(edges, "src", "dst", threshold=10)
    if ndv_probes:
        nodes = _add_ndv_probe_columns(nodes, "id", ndv_probe_buckets)
    if ndv_log:
        cols = ["degree", "high_degree"]
        if ndv_probes:
            cols.extend(["ndv_lo", "ndv_hi"])
        _log_ndv("twitter_congress", nodes, cols)

    edges = _maybe_to_cudf(edges, engine)
    nodes = _maybe_to_cudf(nodes, engine)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


def build_specs(
    redteam_domain_categorical: bool = False,
    ndv_probes: bool = False,
    ndv_probe_buckets: int = 3,
    ndv_log: bool = False,
) -> List[DatasetSpec]:
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
    redteam_two_hop_chain = [
        n(name="a"),
        e_forward({"auth_type": "Kerberos"}, name="e1"),
        n(name="b"),
        e_reverse({"authentication_orientation": "LogOn"}, name="e2"),
        n(name="c"),
    ]
    redteam_where_scenarios = [
        WhereScenario(
            "kerberos_domain_match",
            redteam_two_hop_chain,
            [compare(col("a", "domain"), "==", col("c", "domain"))],
        ),
        WhereScenario(
            "kerberos_domain_mismatch",
            redteam_two_hop_chain,
            [compare(col("a", "domain"), "!=", col("c", "domain"))],
        ),
    ]
    if ndv_probes:
        redteam_where_scenarios.extend(
            [
                WhereScenario(
                    "kerberos_ndv_lo_match",
                    redteam_two_hop_chain,
                    [compare(col("a", "ndv_lo"), "==", col("c", "ndv_lo"))],
                ),
                WhereScenario(
                    "kerberos_ndv_hi_match",
                    redteam_two_hop_chain,
                    [compare(col("a", "ndv_hi"), "==", col("c", "ndv_hi"))],
                ),
                WhereScenario(
                    "kerberos_ndv_lo_mismatch",
                    redteam_two_hop_chain,
                    [compare(col("a", "ndv_lo"), "!=", col("c", "ndv_lo"))],
                ),
                WhereScenario(
                    "kerberos_ndv_hi_mismatch",
                    redteam_two_hop_chain,
                    [compare(col("a", "ndv_hi"), "!=", col("c", "ndv_hi"))],
                ),
            ]
        )

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
    transactions_two_hop_chain = [
        n(name="a"),
        e_forward(name="e1"),
        n(name="b"),
        e_forward(name="e2"),
        n(name="c"),
    ]
    transactions_where_scenarios = [
        WhereScenario(
            "amount_drop_two_hop",
            transactions_two_hop_chain,
            [compare(col("e1", "amount"), ">", col("e2", "amount"))],
        ),
        WhereScenario(
            "tainted_match_two_hop",
            transactions_two_hop_chain,
            [compare(col("a", "tainted_in"), "==", col("c", "tainted_in"))],
        ),
        WhereScenario(
            "tainted_mismatch_two_hop",
            transactions_two_hop_chain,
            [compare(col("a", "tainted_in"), "!=", col("c", "tainted_in"))],
        ),
    ]
    if ndv_probes:
        transactions_where_scenarios.extend(
            [
                WhereScenario(
                    "ndv_lo_match_two_hop",
                    transactions_two_hop_chain,
                    [compare(col("a", "ndv_lo"), "==", col("c", "ndv_lo"))],
                ),
                WhereScenario(
                    "ndv_hi_match_two_hop",
                    transactions_two_hop_chain,
                    [compare(col("a", "ndv_hi"), "==", col("c", "ndv_hi"))],
                ),
                WhereScenario(
                    "ndv_lo_mismatch_two_hop",
                    transactions_two_hop_chain,
                    [compare(col("a", "ndv_lo"), "!=", col("c", "ndv_lo"))],
                ),
                WhereScenario(
                    "ndv_hi_mismatch_two_hop",
                    transactions_two_hop_chain,
                    [compare(col("a", "ndv_hi"), "!=", col("c", "ndv_hi"))],
                ),
            ]
        )

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
    facebook_where_scenarios = [
        WhereScenario(
            "degree_drop_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("a", "degree"), ">=", col("c", "degree"))],
        ),
        WhereScenario(
            "high_degree_match_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("a", "high_degree"), "==", col("c", "high_degree"))],
        ),
        WhereScenario(
            "high_degree_mismatch_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("a", "high_degree"), "!=", col("c", "high_degree"))],
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
    honeypot_where_scenarios = [
        WhereScenario(
            "port_match_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("e1", "victimPort"), "==", col("e2", "victimPort"))],
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
    twitter_demo_where_scenarios = [
        WhereScenario(
            "degree_drop_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("a", "degree"), ">=", col("c", "degree"))],
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
    lesmiserables_where_scenarios = [
        WhereScenario(
            "weight_drop_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("e1", "value"), ">=", col("e2", "value"))],
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
    twitter_congress_where_scenarios = [
        WhereScenario(
            "weight_drop_two_hop",
            [
                n(name="a"),
                e_forward(name="e1"),
                n(name="b"),
                e_forward(name="e2"),
                n(name="c"),
            ],
            [compare(col("e1", "weight"), ">=", col("e2", "weight"))],
        ),
    ]

    loader_kwargs = {
        "ndv_probes": ndv_probes,
        "ndv_probe_buckets": ndv_probe_buckets,
        "ndv_log": ndv_log,
    }
    redteam_loader = partial(
        load_redteam,
        domain_categorical=redteam_domain_categorical,
        **loader_kwargs,
    )
    transactions_loader = partial(load_transactions, **loader_kwargs)
    facebook_loader = partial(load_facebook, **loader_kwargs)
    honeypot_loader = partial(load_honeypot, **loader_kwargs)
    twitter_demo_loader = partial(load_twitter_demo, **loader_kwargs)
    lesmiserables_loader = partial(load_lesmiserables, **loader_kwargs)
    twitter_congress_loader = partial(load_twitter_congress, **loader_kwargs)

    return [
        DatasetSpec(
            "redteam50k",
            redteam_loader,
            redteam_scenarios,
            redteam_where_scenarios,
        ),
        DatasetSpec(
            "transactions",
            transactions_loader,
            transactions_scenarios,
            transactions_where_scenarios,
        ),
        DatasetSpec(
            "facebook_combined",
            facebook_loader,
            facebook_scenarios,
            facebook_where_scenarios,
        ),
        DatasetSpec("honeypot", honeypot_loader, honeypot_scenarios, honeypot_where_scenarios),
        DatasetSpec(
            "twitter_demo",
            twitter_demo_loader,
            twitter_demo_scenarios,
            twitter_demo_where_scenarios,
        ),
        DatasetSpec(
            "lesmiserables",
            lesmiserables_loader,
            lesmiserables_scenarios,
            lesmiserables_where_scenarios,
        ),
        DatasetSpec(
            "twitter_congress",
            twitter_congress_loader,
            twitter_congress_scenarios,
            twitter_congress_where_scenarios,
        ),
    ]


def run_chain_scenarios(
    g: graphistry.Plottable,
    dataset_name: str,
    scenarios: Iterable[Scenario],
    engine_label: str,
    runs: int,
    warmup: int,
    max_total_s: Optional[float] = None,
    max_call_s: Optional[float] = None,
) -> Iterable[ResultRow]:
    for scenario in scenarios:
        def _call() -> None:
            g.gfql(scenario.chain, engine=engine_label)

        stats = _time_call(_call, runs, warmup, max_total_s=max_total_s, max_call_s=max_call_s)
        yield ResultRow(
            dataset=dataset_name,
            scenario=scenario.name,
            median_ms=stats.median_ms if stats else None,
            p90_ms=stats.p90_ms if stats else None,
            std_ms=stats.std_ms if stats else None,
        )


def run_where_scenarios(
    g: graphistry.Plottable,
    dataset_name: str,
    scenarios: Iterable[WhereScenario],
    engine: Engine,
    runs: int,
    warmup: int,
    max_total_s: Optional[float] = None,
    max_call_s: Optional[float] = None,
) -> Iterable[ResultRow]:
    for scenario in scenarios:
        def _call() -> None:
            execute_same_path_chain(g, scenario.chain, scenario.where, engine, include_paths=False)

        stats = _time_call(_call, runs, warmup, max_total_s=max_total_s, max_call_s=max_call_s)
        yield ResultRow(
            dataset=dataset_name,
            scenario=scenario.name,
            median_ms=stats.median_ms if stats else None,
            p90_ms=stats.p90_ms if stats else None,
            std_ms=stats.std_ms if stats else None,
        )


def _fmt_ms(value: Optional[float]) -> str:
    return "TIMEOUT" if value is None else f"{value:.2f}ms"


def _table_lines(title: str, results: Iterable[ResultRow]) -> List[str]:
    rows = list(results)
    if not rows:
        return []
    lines = [
        f"## {title}",
        "",
        "| Dataset | Scenario | Median | P90 | Std |",
        "|---------|----------|--------|-----|-----|",
    ]
    lines.extend(
        f"| {row.dataset} | {row.scenario} | {_fmt_ms(row.median_ms)} | {_fmt_ms(row.p90_ms)} | {_fmt_ms(row.std_ms)} |"
        for row in rows
    )
    valid_medians = [row.median_ms for row in rows if row.median_ms is not None]
    score = statistics.median(valid_medians) if valid_medians else None
    lines.append("")
    lines.append(
        f"Score (median of medians): {_fmt_ms(score)}"
    )
    return lines


def write_markdown(
    chain_results: Iterable[ResultRow],
    where_results: Iterable[ResultRow],
    output_path: str,
    notes_extra: Optional[List[str]] = None,
) -> None:
    header = [
        "# Real-Data Benchmark Results",
        "",
        "Notes:",
        "- Chain results use GFQL (no WHERE).",
        "- WHERE results use the df_executor same-path engine.",
        "- Datasets are loaded from `demos/data/`.",
        "- Values are median over runs; p90 and std columns show variability.",
    ]
    if notes_extra:
        for note in notes_extra:
            header.append(f"- {note}")
    header.append("")
    lines = header
    lines.extend(_table_lines("Chain-only (GFQL)", chain_results))
    lines.append("")
    lines.extend(_table_lines("WHERE (df_executor)", where_results))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data GFQL benchmarks (no WHERE).")
    parser.add_argument("--engine", default="pandas", choices=["pandas", "cudf"])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--max-scenario-seconds",
        type=float,
        default=20.0,
        help="Total time budget per scenario (seconds). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-call-seconds",
        type=float,
        default=None,
        help="Per-call time budget (seconds). Defaults to max-scenario-seconds.",
    )
    parser.add_argument(
        "--opt-max-call-ms",
        type=float,
        default=200.0,
        help="Per-call budget for opt WHERE runs (milliseconds). Use 0 to disable.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated list: redteam50k,transactions,facebook_combined,honeypot,twitter_demo,lesmiserables,twitter_congress,all",
    )
    parser.add_argument(
        "--skip-chain",
        action="store_true",
        help="Skip chain-only scenarios.",
    )
    parser.add_argument(
        "--skip-where",
        action="store_true",
        help="Skip WHERE scenarios.",
    )
    parser.add_argument(
        "--chain-filter",
        default="",
        help="Comma-separated substrings to select chain scenario names.",
    )
    parser.add_argument(
        "--where-filter",
        default="",
        help="Comma-separated substrings to select WHERE scenario names.",
    )
    parser.add_argument(
        "--redteam-domain-categorical",
        action="store_true",
        help="Cast redteam node domain column to categorical (pandas only).",
    )
    parser.add_argument(
        "--ndv-probes",
        action="store_true",
        help="Add ndv_lo/ndv_hi node columns and extra WHERE scenarios for NDV sensitivity.",
    )
    parser.add_argument(
        "--ndv-probe-buckets",
        type=int,
        default=3,
        help="Bucket count for ndv_lo when --ndv-probes is enabled.",
    )
    parser.add_argument(
        "--ndv-log",
        action="store_true",
        help="Print NDV summaries for selected node columns.",
    )
    parser.add_argument(
        "--non-adj-mode",
        default="",
        help="Set GRAPHISTRY_NON_ADJ_WHERE_MODE (baseline/prefilter/value/value_prefilter).",
    )
    parser.add_argument(
        "--non-adj-strategy",
        default="",
        help="Set GRAPHISTRY_NON_ADJ_WHERE_STRATEGY (vector).",
    )
    parser.add_argument(
        "--non-adj-value-ops",
        default="",
        help="Set GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS (comma-separated).",
    )
    parser.add_argument(
        "--non-adj-value-card-max",
        type=int,
        default=None,
        help="Set GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX.",
    )
    parser.add_argument(
        "--non-adj-order",
        default="",
        help="Set GRAPHISTRY_NON_ADJ_WHERE_ORDER (selectivity/size).",
    )
    parser.add_argument(
        "--non-adj-bounds",
        action="store_true",
        help="Enable GRAPHISTRY_NON_ADJ_WHERE_BOUNDS for inequality prefiltering.",
    )
    parser.add_argument(
        "--non-adj-vector-max-hops",
        type=int,
        default=None,
        help="Set GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS.",
    )
    parser.add_argument(
        "--non-adj-vector-label-max",
        type=int,
        default=None,
        help="Set GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX.",
    )
    parser.add_argument(
        "--non-adj-vector-pair-max",
        type=int,
        default=None,
        help="Set GRAPHISTRY_NON_ADJ_WHERE_VECTOR_PAIR_MAX.",
    )
    parser.add_argument(
        "--non-adj-domain-semijoin",
        action="store_true",
        help="Enable GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN.",
    )
    parser.add_argument(
        "--non-adj-domain-semijoin-auto",
        action="store_true",
        help="Enable GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO.",
    )
    parser.add_argument(
        "--non-adj-domain-semijoin-pair-max",
        type=int,
        default=None,
        help="Set GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX.",
    )
    parser.add_argument(
        "--edge-where-semijoin",
        action="store_true",
        help="Enable GRAPHISTRY_EDGE_WHERE_SEMIJOIN.",
    )
    parser.add_argument(
        "--edge-where-semijoin-auto",
        action="store_true",
        help="Enable GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO.",
    )
    parser.add_argument(
        "--edge-where-semijoin-pair-max",
        type=int,
        default=None,
        help="Set GRAPHISTRY_EDGE_WHERE_SEMIJOIN_PAIR_MAX.",
    )
    args = parser.parse_args()

    if args.non_adj_mode:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_MODE"] = args.non_adj_mode
    if args.non_adj_strategy:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_STRATEGY"] = args.non_adj_strategy
    if args.non_adj_value_ops:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS"] = args.non_adj_value_ops
    if args.non_adj_value_card_max is not None:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX"] = str(args.non_adj_value_card_max)
    if args.non_adj_order:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_ORDER"] = args.non_adj_order
    if args.non_adj_bounds:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_BOUNDS"] = "1"
    if args.non_adj_vector_max_hops is not None:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS"] = str(args.non_adj_vector_max_hops)
    if args.non_adj_vector_label_max is not None:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX"] = str(args.non_adj_vector_label_max)
    if args.non_adj_vector_pair_max is not None:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_VECTOR_PAIR_MAX"] = str(args.non_adj_vector_pair_max)
    if args.non_adj_domain_semijoin:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN"] = "1"
    if args.non_adj_domain_semijoin_auto:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_AUTO"] = "1"
    if args.non_adj_domain_semijoin_pair_max is not None:
        os.environ["GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX"] = str(
            args.non_adj_domain_semijoin_pair_max
        )
    if args.edge_where_semijoin:
        os.environ["GRAPHISTRY_EDGE_WHERE_SEMIJOIN"] = "1"
    if args.edge_where_semijoin_auto:
        os.environ["GRAPHISTRY_EDGE_WHERE_SEMIJOIN_AUTO"] = "1"
    if args.edge_where_semijoin_pair_max is not None:
        os.environ["GRAPHISTRY_EDGE_WHERE_SEMIJOIN_PAIR_MAX"] = str(
            args.edge_where_semijoin_pair_max
        )
    setup_tracer()

    max_total_s = args.max_scenario_seconds if args.max_scenario_seconds and args.max_scenario_seconds > 0 else None
    max_call_s = args.max_call_seconds if args.max_call_seconds and args.max_call_seconds > 0 else None
    if max_call_s is None and max_total_s is not None:
        max_call_s = max_total_s

    opt_enabled = any(
        [
            bool(args.non_adj_mode),
            bool(args.non_adj_strategy),
            bool(args.non_adj_order),
            bool(args.non_adj_bounds),
            args.non_adj_value_card_max is not None,
            args.non_adj_vector_max_hops is not None,
            args.non_adj_vector_label_max is not None,
            args.non_adj_vector_pair_max is not None,
        ]
    )
    opt_call_s = None
    if opt_enabled and args.opt_max_call_ms and args.opt_max_call_ms > 0:
        opt_call_s = args.opt_max_call_ms / 1000.0

    where_call_s = max_call_s
    if opt_call_s is not None:
        where_call_s = opt_call_s if where_call_s is None else min(where_call_s, opt_call_s)

    dataset_filter = {d.strip() for d in args.datasets.split(",")} if args.datasets else {"all"}
    chain_filters = _parse_filters(args.chain_filter)
    where_filters = _parse_filters(args.where_filter)
    specs = build_specs(
        redteam_domain_categorical=args.redteam_domain_categorical,
        ndv_probes=args.ndv_probes,
        ndv_probe_buckets=args.ndv_probe_buckets,
        ndv_log=args.ndv_log,
    )
    if "all" not in dataset_filter:
        specs = [s for s in specs if s.name in dataset_filter]

    chain_results: List[ResultRow] = []
    where_results: List[ResultRow] = []
    engine_enum = _as_engine(args.engine)
    for dataset in specs:
        g = dataset.loader(engine_enum)
        chain_scenarios = dataset.scenarios
        where_scenarios = dataset.where_scenarios
        if chain_filters:
            chain_scenarios = [s for s in chain_scenarios if any(f in s.name for f in chain_filters)]
        if where_filters:
            where_scenarios = [s for s in where_scenarios if any(f in s.name for f in where_filters)]
        if not args.skip_chain:
            chain_results.extend(
                run_chain_scenarios(
                    g,
                    dataset.name,
                    chain_scenarios,
                    args.engine,
                    args.runs,
                    args.warmup,
                    max_total_s=max_total_s,
                    max_call_s=max_call_s,
                )
            )
        if not args.skip_where:
            where_results.extend(
                run_where_scenarios(
                    g,
                    dataset.name,
                    where_scenarios,
                    engine_enum,
                    args.runs,
                    args.warmup,
                    max_total_s=max_total_s,
                    max_call_s=where_call_s,
                )
            )

    if args.output:
        notes_extra = []
        if args.redteam_domain_categorical:
            notes_extra.append("Redteam nodes.domain cast to categorical.")
        if args.ndv_probes:
            notes_extra.append(f"NDV probes enabled (buckets={args.ndv_probe_buckets}).")
        if args.ndv_log:
            notes_extra.append("NDV logging enabled.")
        if args.skip_chain:
            notes_extra.append("Chain scenarios skipped.")
        if args.skip_where:
            notes_extra.append("WHERE scenarios skipped.")
        if chain_filters:
            notes_extra.append(f"Chain filter: {', '.join(chain_filters)}.")
        if where_filters:
            notes_extra.append(f"WHERE filter: {', '.join(where_filters)}.")
        if args.non_adj_mode:
            notes_extra.append(f"Non-adj mode: {args.non_adj_mode}.")
        if args.non_adj_value_card_max is not None:
            notes_extra.append(f"Non-adj value card max: {args.non_adj_value_card_max}.")
        if args.non_adj_order:
            notes_extra.append(f"Non-adj order: {args.non_adj_order}.")
        if args.non_adj_bounds:
            notes_extra.append("Non-adj bounds enabled.")
        if args.non_adj_domain_semijoin:
            notes_extra.append("Non-adj domain semijoin enabled.")
        if args.non_adj_domain_semijoin_auto:
            notes_extra.append("Non-adj domain semijoin auto enabled.")
        if args.non_adj_domain_semijoin_pair_max is not None:
            notes_extra.append(
                f"Non-adj domain semijoin pair max: {args.non_adj_domain_semijoin_pair_max}."
            )
        if args.edge_where_semijoin:
            notes_extra.append("Edge WHERE semijoin enabled.")
        if args.edge_where_semijoin_auto:
            notes_extra.append("Edge WHERE semijoin auto enabled.")
        if args.edge_where_semijoin_pair_max is not None:
            notes_extra.append(
                f"Edge WHERE semijoin pair max: {args.edge_where_semijoin_pair_max}."
            )
        if max_total_s is not None:
            notes_extra.append(f"Scenario timeout: {max_total_s:.1f}s total.")
        if max_call_s is not None:
            notes_extra.append(f"Per-call timeout: {max_call_s:.1f}s.")
        if opt_call_s is not None:
            notes_extra.append(f"Opt per-call timeout: {opt_call_s * 1000:.0f}ms.")
        write_markdown(chain_results, where_results, args.output, notes_extra=notes_extra)

    for title, rows in (
        ("Chain-only (GFQL)", chain_results),
        ("WHERE (df_executor)", where_results),
    ):
        lines = _table_lines(title, rows)
        if not lines:
            continue
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
