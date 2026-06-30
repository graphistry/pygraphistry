"""Multi-hop polars-vs-pandas chain benchmark (validation for the native multi-hop port).

Confirms native multi-hop polars is on-par-or-better than pandas on non-trivial graphs.
Run on dgx (CPU lane is the relevant comparison; cudf/polars-gpu optional).
"""
import time
import numpy as np
import pandas as pd
import graphistry
from graphistry.compute.ast import n, e_forward, e_undirected


def _graph(nn, ne, seed=0):
    rng = np.random.default_rng(seed)
    nd = pd.DataFrame({"id": np.arange(nn), "k": rng.integers(0, 5, nn)})
    ed = pd.DataFrame({
        "s": rng.integers(0, nn, ne),
        "d": rng.integers(0, nn, ne),
        "w": rng.integers(0, 100, ne),
    })
    return graphistry.nodes(nd, "id").edges(ed, "s", "d")


def _time(g, ch, engine, reps=3):
    g.chain(ch, engine=engine)  # warm
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = g.chain(ch, engine=engine)
        dt = time.perf_counter() - t
        best = min(best, dt)
    ne = 0 if out._edges is None else len(out._edges)
    nn = 0 if out._nodes is None else len(out._nodes)
    return best * 1000, nn, ne


CHAINS = {
    "fwd-hops2":  [n({"id": [0]}), e_forward(hops=2), n()],
    "fwd-hops3":  [n({"id": [0]}), e_forward(hops=3), n()],
    "maxhops4":   [n({"id": [0]}), e_forward(max_hops=4), n()],
    "sandwiched": [n({"id": [0]}), e_forward(), n(), e_forward(hops=2), n()],
    "fwd-tofixed": [n({"id": [0]}), e_forward(to_fixed_point=True), n()],
    "und-hops2":  [n({"id": [0]}), e_undirected(hops=2), n()],
    "und-maxhops3": [n({"id": [0]}), e_undirected(max_hops=3), n()],
}

for (nn, ne) in [(10_000, 50_000), (100_000, 500_000), (500_000, 2_000_000)]:
    g = _graph(nn, ne)
    print(f"\n=== graph nn={nn:,} ne={ne:,} ===")
    for name, ch in CHAINS.items():
        engines = ["pandas", "polars"]
        try:
            import cudf  # noqa: F401
            engines.append("cudf")
        except Exception:
            pass
        row = []
        ref = None
        for eng in engines:
            try:
                ms, rnn, rne = _time(g, ch, eng)
                if eng == "pandas":
                    ref = ms
                spd = f"{ref/ms:.2f}x" if (ref and eng != "pandas") else "1.00x"
                row.append(f"{eng}={ms:.1f}ms({spd},n={rnn},e={rne})")
            except NotImplementedError:
                row.append(f"{eng}=NIE")
            except Exception as ex:
                row.append(f"{eng}=ERR:{type(ex).__name__}")
        print(f"  {name:12s} " + "  ".join(row))
