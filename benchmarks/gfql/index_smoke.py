#!/usr/bin/env python3
"""Differential parity smoke test for GFQL physical indexes.

Container-runnable MIRROR of graphistry/tests/compute/gfql/index/test_index.py
(the pytest suite is canonical); kept for quick in-container smoke without pytest.

Oracle: the index fast path MUST return the same subgraph (node-id set + edge
multiset) as the scan/join path, across engines / directions / hops / wavefront.
A fast wrong answer is not a win.

Run (dgx-spark container):
  python3 benchmarks/gfql/index_smoke.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import graphistry


def make_graph(n_nodes=2000, avg_deg=6, seed=0):
    rng = np.random.default_rng(seed)
    m = n_nodes * avg_deg
    src = rng.integers(0, n_nodes, size=m)
    dst = rng.integers(0, n_nodes, size=m)
    edf = pd.DataFrame({"src": src, "dst": dst, "w": rng.random(m)})
    ndf = pd.DataFrame({"id": np.arange(n_nodes), "label": rng.integers(0, 5, size=n_nodes)})
    return ndf, edf


def _to_pdf(df):
    if df is None:
        return None
    mod = type(df).__module__
    if "cudf" in mod:
        return df.to_pandas()
    if "polars" in mod:
        return df.to_pandas()
    return df


def result_signature(g, node_col, src, dst):
    n = _to_pdf(g._nodes)
    e = _to_pdf(g._edges)
    node_ids = sorted(n[node_col].tolist()) if n is not None else []
    edge_pairs = sorted(map(tuple, e[[src, dst]].itertuples(index=False, name=None))) if e is not None else []
    return node_ids, edge_pairs


def available_engines():
    engines = ["pandas"]
    try:
        import cudf  # noqa
        engines.append("cudf")
    except Exception:
        pass
    try:
        import polars  # noqa
        engines.append("polars")
        # polars-gpu only if cudf-polars present
        try:
            import cudf  # noqa
            engines.append("polars-gpu")
        except Exception:
            pass
    except Exception:
        pass
    return engines


SCENARIOS = [
    dict(hops=1, direction="forward", return_as_wave_front=False),
    dict(hops=1, direction="reverse", return_as_wave_front=False),
    dict(hops=1, direction="undirected", return_as_wave_front=False),
    dict(hops=2, direction="forward", return_as_wave_front=False),
    dict(hops=2, direction="undirected", return_as_wave_front=False),
    dict(hops=1, direction="forward", return_as_wave_front=True),
    dict(hops=2, direction="forward", return_as_wave_front=True),
    dict(hops=3, direction="forward", return_as_wave_front=False),
]


def main():
    ndf, edf = make_graph()
    g0 = graphistry.nodes(ndf, "id").edges(edf, "src", "dst")
    seeds = pd.DataFrame({"id": [0, 1, 2, 7, 42, 100]})

    engines = available_engines()
    print(f"engines: {engines}")
    total = 0
    fails = 0
    for engine in engines:
        gi = g0.gfql_index_edges("both", engine=engine)
        for sc in SCENARIOS:
            total += 1
            try:
                base = g0.hop(nodes=seeds, engine=engine, **sc)
                idx = gi.hop(nodes=seeds, engine=engine, **sc)
            except NotImplementedError as ex:
                print(f"  SKIP  {engine:10} {sc} :: NIE {ex}")
                total -= 1
                continue
            except Exception as ex:
                fails += 1
                print(f"  ERROR {engine:10} {sc} :: {type(ex).__name__}: {ex}")
                continue
            bn, be = result_signature(base, "id", "src", "dst")
            xn, xe = result_signature(idx, "id", "src", "dst")
            ok = (bn == xn) and (be == xe)
            if not ok:
                fails += 1
                print(f"  FAIL  {engine:10} {sc}")
                print(f"        nodes base={len(bn)} idx={len(xn)} eq={bn==xn}; edges base={len(be)} idx={len(xe)} eq={be==xe}")
                # show small diff
                sb, sx = set(bn), set(xn)
                print(f"        node only-base={sorted(sb-sx)[:8]} only-idx={sorted(sx-sb)[:8]}")
            else:
                print(f"  OK    {engine:10} {sc} :: nodes={len(bn)} edges={len(be)}")
    print(f"\n=== {total-fails}/{total} passed, {fails} failed ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
