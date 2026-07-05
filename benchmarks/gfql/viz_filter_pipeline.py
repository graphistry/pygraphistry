#!/usr/bin/env python3
"""Regression benchmark: the streamgl-viz filter-pipeline panel shapes as SINGLE
cypher queries via ``g.gfql(query, engine=...)`` (viz-filter L3).

Scenarios mirror the panel states the viz server must answer interactively:
node filter + exclude + null-tolerant leaf + edge filter with graph-aware
ORDER BY + LIMIT (the 800K scenario), prune-isolated (both the GRAPH { } keep-self
form and the EXISTS { } row form), cross-column table search on nodes and edges
(``searchAny``, case-insensitive substring default), and one combined pipeline.

Fairness discipline (matches cypher_row_pipeline.py): per engine the node/edge
frames are converted ONCE up front to that engine's native frame type (pandas:
as-is; cudf: ``cudf.from_pandas``; polars / polars-gpu: ``pl.from_pandas``) and
bound via ``graphistry.nodes/edges`` — conversion time is excluded, exactly as a
real deployment keeps its graph resident in the engine's frame type. Engines run
back-to-back per scenario (implicit interleave on a shared host); for
regression-grade claims run several invocations and compare distributions.

Status semantics: an engine that raises NotImplementedError records ``nie``
(honest decline, not a crash); any other exception records ``err`` with the
message — the harness never dies mid-matrix.

The interactive reference (~sub-350ms panel apply @ 800K rows) is printed as a
comparison column only, never asserted.

Example (dgx via safe_run.sh; cudf/polars-gpu are opt-in so no GPU import
happens unless requested)::

    python benchmarks/gfql/viz_filter_pipeline.py --est --scale 1m
    python benchmarks/gfql/viz_filter_pipeline.py --scale 1m --runs 5 --warmup 2 \
        --engines pandas,polars,cudf,polars-gpu --output-json /tmp/viz-pipeline-1m.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VIZ_LIMIT = 800_000               # the streamgl-viz page-size ceiling scenario
INTERACTIVE_REF_S = 0.350         # sub-350ms @800K = interactive panel apply (comparison only)

SCALES: Dict[str, int] = {"100k": 100_000, "1m": 1_000_000, "10m": 10_000_000}
EDGE_FACTOR = 2.5
ISOLATED_FRAC = 0.03
SELF_LOOP_FRAC = 0.005
NULL_SCORE_FRAC = 0.05

KNOWN_ENGINES = ("pandas", "cudf", "polars", "polars-gpu")

# Zipf-weighted name vocabulary; 'ember' sits at a mid rank so the node search term
# hits a realistic ~2-4% of names (two tokens per name). no other token contains it (november swapped out for granite — wave-1 F1).
_TOKENS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "ember", "lima", "mike", "granite", "oscar",
    "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "onyx", "quartz", "topaz", "cobalt", "crimson",
    "indigo", "saffron", "umber", "viridian", "cerulean", "magenta", "ochre", "slate",
]
NODE_SEARCH_TERM = "Ember"        # case-insensitive substring default exercised on purpose
EDGE_SEARCH_TERM = "WIRE"         # hits etype 'wire-xfer' (~3% of edges)
_KINDS = ["account", "user", "device", "ip", "internal"]
_KIND_P = [0.30, 0.30, 0.15, 0.15, 0.10]
_ETYPES = ["ach", "card", "swift", "zelle", "check", "wire-xfer"]
_ETYPE_P = [0.30, 0.25, 0.20, 0.12, 0.10, 0.03]

# (name, cypher) — every scenario is ONE query string, run identically on every engine.
SCENARIOS: List[Tuple[str, str]] = [
    (
        # node filter + exclude (exclusion-dominates NOT) + (pred OR IS NULL) leaf
        # + edge filter, graph-aware ORDER BY + LIMIT 800K
        "panel_filters",
        "MATCH (a)-[e]->(b) "
        "WHERE a.kind <> 'internal' AND NOT (a.flag = true AND a.score < 0.1) "
        "AND (a.score > 0.25 OR a.score IS NULL) AND e.w > 2 "
        f"RETURN a.id AS id, a.score AS score ORDER BY score DESC LIMIT {VIZ_LIMIT}",
    ),
    (
        # full-graph keep-self prune: keeps every node with >=1 incident edge,
        # self-loops included (UI-parity zero-EDGES definition)
        "prune_keep_self",
        "GRAPH { MATCH (a)-[e]-(b) }",
    ),
    (
        # row-form prune via EXISTS pattern subquery, viz page ceiling
        "exists_prune",
        f"MATCH (n) WHERE EXISTS {{ (n)--() }} RETURN n.id AS id LIMIT {VIZ_LIMIT}",
    ),
    (
        # cross-column node table search (case-insensitive substring default)
        "search_nodes",
        f"MATCH (n) WHERE searchAny(n, '{NODE_SEARCH_TERM}') RETURN n.id AS id LIMIT {VIZ_LIMIT}",
    ),
    (
        # cross-column edge table search, different term
        "search_edges",
        f"MATCH (a)-[e]->(b) WHERE searchAny(e, '{EDGE_SEARCH_TERM}') RETURN e.w AS w LIMIT {VIZ_LIMIT}",
    ),
    (
        # filters + EXISTS prune + searchAny composed in ONE query
        "combined",
        "MATCH (n) WHERE n.kind <> 'internal' AND (n.score > 0.25 OR n.score IS NULL) "
        f"AND EXISTS {{ (n)--() }} AND searchAny(n, '{NODE_SEARCH_TERM}') "
        f"RETURN n.id AS id ORDER BY id LIMIT {VIZ_LIMIT}",
    ),
]


@dataclass
class CellResult:
    scenario: str
    engine: str
    status: str                       # ok | nie | err
    median_s: Optional[float] = None
    min_s: Optional[float] = None
    rows_out: Optional[int] = None    # table rows, or node count for graph results
    edges_out: Optional[int] = None   # only for graph-shaped results (S2)
    error: Optional[str] = None


def _zipf_p(k: int) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1)
    return w / w.sum()


def make_frames(n_nodes: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded synthetic viz workload: zipfian names (realistic search hit rates),
    ~5% null scores, power-lawish edge sources, ~0.5% self-loops, ~3% isolated nodes."""
    rng = np.random.default_rng(seed)
    n_edges = int(n_nodes * EDGE_FACTOR)

    tok_p = _zipf_p(len(_TOKENS))
    combos = np.array([f"{a}-{b}" for a in _TOKENS for b in _TOKENS], dtype=object)
    t1 = rng.choice(len(_TOKENS), size=n_nodes, p=tok_p)
    t2 = rng.choice(len(_TOKENS), size=n_nodes, p=tok_p)
    score = rng.random(n_nodes)
    score[rng.random(n_nodes) < NULL_SCORE_FRAC] = np.nan
    nodes = pd.DataFrame({
        "id": np.arange(n_nodes, dtype=np.int64),
        "name": combos[t1 * len(_TOKENS) + t2],
        "kind": rng.choice(_KINDS, size=n_nodes, p=_KIND_P),
        "score": score,
        "flag": rng.random(n_nodes) < 0.3,
    })

    # reserve a random ~3% of ids as guaranteed-isolated; endpoints draw from the rest
    perm = rng.permutation(n_nodes).astype(np.int64)
    connected = perm[int(n_nodes * ISOLATED_FRAC):]
    src = connected[(len(connected) * rng.random(n_edges) ** 3).astype(np.int64)]  # power-lawish hubs
    dst = connected[rng.integers(0, len(connected), size=n_edges)]
    loop = rng.random(n_edges) < SELF_LOOP_FRAC
    dst[loop] = src[loop]
    edges = pd.DataFrame({
        "src": src,
        "dst": dst,
        "etype": rng.choice(_ETYPES, size=n_edges, p=_ETYPE_P),
        "w": rng.integers(0, 10, size=n_edges, dtype=np.int64),
    })
    return nodes, edges


def bind_engine_graph(nodes: pd.DataFrame, edges: pd.DataFrame, engine: str):
    """One-time conversion to the engine's native frame type + graphistry binding.
    GPU/polars imports happen ONLY here, only for requested engines."""
    import graphistry
    if engine == "pandas":
        en, ee = nodes, edges
    elif engine == "cudf":
        import cudf
        en, ee = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in ("polars", "polars-gpu"):
        import polars as pl
        en, ee = pl.from_pandas(nodes), pl.from_pandas(edges)
    else:
        raise ValueError(f"unknown engine {engine!r}; known: {', '.join(KNOWN_ENGINES)}")
    return graphistry.nodes(en, "id").edges(ee, "src", "dst")


def _frame_len(df) -> Optional[int]:
    return None if df is None else int(len(df))


def run_cell(g, scenario: str, query: str, engine: str, runs: int, warmup: int) -> CellResult:
    """Warmup + timed runs for one (scenario, engine). NIE / errors are recorded,
    never raised — the matrix always completes."""
    def once():
        res = g.gfql(query, engine=engine)
        # touch BOTH result frames so lazy/async engines materialize inside the clock
        # (graph-shaped scenarios return edges too — wave-2 S1)
        _frame_len(res._nodes)
        _frame_len(res._edges)
        return res

    try:
        for _ in range(warmup):
            res = once()
        samples: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            res = once()
            samples.append(time.perf_counter() - t0)
    except NotImplementedError:
        return CellResult(scenario, engine, "nie")
    except Exception as exc:  # noqa: BLE001 - harness reports, never crashes the sweep
        return CellResult(scenario, engine, "err", error=f"{type(exc).__name__}: {exc}"[:300])

    is_graph = query.lstrip().startswith("GRAPH")
    return CellResult(
        scenario, engine, "ok",
        median_s=statistics.median(samples),
        min_s=min(samples),
        rows_out=_frame_len(res._nodes),
        edges_out=_frame_len(res._edges) if is_graph else None,
    )


def run_matrix(engines: List[str], scenarios: List[Tuple[str, str]],
               nodes: pd.DataFrame, edges: pd.DataFrame,
               runs: int, warmup: int) -> List[CellResult]:
    graphs = {eng: bind_engine_graph(nodes, edges, eng) for eng in engines}  # convert ONCE, untimed
    results: List[CellResult] = []
    for name, query in scenarios:                # scenario-major: engines back-to-back = interleave
        for eng in engines:
            print(f"  running {name} [{eng}] ...", file=sys.stderr, flush=True)
            cell = run_cell(graphs[eng], name, query, eng, runs, warmup)
            results.append(cell)
    return results


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v * 1000.0:.1f}" if v is not None else "-"


def _fmt_budget(median_s: Optional[float]) -> str:
    """Median as a multiple of the ~350ms interactive panel-apply reference @800K."""
    return f"{median_s / INTERACTIVE_REF_S:.2f}x" if median_s is not None else "-"


def to_table(results: List[CellResult]) -> str:
    lines = [
        "| scenario | engine | status | median_ms | min_ms | rows_out | x_350ms_ref |",
        "|----------|--------|--------|-----------|--------|----------|-------------|",
    ]
    for r in results:
        rows = "-" if r.rows_out is None else str(r.rows_out)
        if r.edges_out is not None:
            rows = f"{rows}n/{r.edges_out}e"
        note = r.error if r.status == "err" else _fmt_budget(r.median_s)
        lines.append(
            f"| {r.scenario} | {r.engine} | {r.status} | "
            f"{_fmt_ms(r.median_s)} | {_fmt_ms(r.min_s)} | {rows} | {note} |"
        )
    return "\n".join(lines)


def print_estimate(scale: str) -> None:
    """Row/memory budget for safe_run.sh planning (shared dgx box) — no generation, no run."""
    n = SCALES[scale]
    m = int(n * EDGE_FACTOR)
    node_gb = n * 130 / 1e9   # ~8B id + 8B score + 1B flag + object name/kind (~55-60B each)
    edge_gb = m * 85 / 1e9    # ~24B int64s + object etype
    print(f"scale={scale}: nodes={n:,} edges={m:,} (~{node_gb + edge_gb:.2f} GB pandas-resident, rough)")
    print("per extra engine add ~1x that resident (native-frame copy); "
          "timed set includes up to 800K-row results per run")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression benchmark: streamgl-viz filter-pipeline panel shapes as single GFQL cypher queries.")
    parser.add_argument("--scale", choices=sorted(SCALES), default="100k",
                        help="Node count (edges ~2.5x): 100k, 1m, 10m.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--engines", default="pandas,polars",
                        help="Comma list of engines; cudf / polars-gpu are opt-in (dgx). "
                             f"Known: {', '.join(KNOWN_ENGINES)}.")
    parser.add_argument("--scenarios", default="",
                        help="Optional comma list to run a subset (names: "
                             + ", ".join(name for name, _ in SCENARIOS) + ").")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default="", help="Optional path for machine-readable results.")
    parser.add_argument("--est", action="store_true",
                        help="Print estimated rows/memory for safe_run budgeting and exit.")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup >= 0")
    if args.output_json:
        # fail BEFORE a multi-minute sweep, not at the final write (wave-2 S4)
        with open(args.output_json, "a"):
            pass

    if args.est:
        print_estimate(args.scale)
        return

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in KNOWN_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engine(s) {unknown}; known: {', '.join(KNOWN_ENGINES)}")

    scenarios = SCENARIOS
    if args.scenarios:
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        by_name = dict(SCENARIOS)
        missing = [s for s in wanted if s not in by_name]
        if missing:
            raise SystemExit(f"unknown scenario(s) {missing}; known: {', '.join(by_name)}")
        scenarios = [(s, by_name[s]) for s in wanted]

    n_nodes = SCALES[args.scale]
    print(f"generating scale={args.scale} (nodes={n_nodes:,}, edges={int(n_nodes * EDGE_FACTOR):,}, "
          f"seed={args.seed}) ...", file=sys.stderr, flush=True)
    nodes, edges = make_frames(n_nodes, args.seed)

    results = run_matrix(engines, scenarios, nodes, edges, args.runs, args.warmup)

    print(to_table(results))
    print(f"\nx_350ms_ref: median vs the ~{INTERACTIVE_REF_S * 1000:.0f}ms interactive panel-apply "
          f"reference @{VIZ_LIMIT:,} rows (streamgl-viz budget; comparison only, not asserted).")

    if args.output_json:
        payload = {
            "meta": {
                "scale": args.scale, "n_nodes": n_nodes, "n_edges": int(len(edges)),
                "runs": args.runs, "warmup": args.warmup, "engines": engines,
                "seed": args.seed, "viz_limit": VIZ_LIMIT,
                "interactive_ref_s": INTERACTIVE_REF_S,
                "queries": {name: q for name, q in scenarios},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.output_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
