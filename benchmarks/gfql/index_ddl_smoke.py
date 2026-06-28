#!/usr/bin/env python3
"""Smoke test: Cypher DDL + JSON wire + index_policy auto/force + gfql_explain."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import graphistry
from graphistry.compute.gfql.index import (
    CreateIndex, DropIndex, ShowIndexes, index_op_from_json, parse_index_ddl,
)


def make_graph(n=3000, deg=6, seed=1):
    rng = np.random.default_rng(seed)
    m = n * deg
    edf = pd.DataFrame({"src": rng.integers(0, n, m), "dst": rng.integers(0, n, m)})
    ndf = pd.DataFrame({"id": np.arange(n)})
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst")


def check(name, cond):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    return 0 if cond else 1


def main():
    g = make_graph()
    seeds = pd.DataFrame({"id": [0, 1, 5, 9]})
    fails = 0

    # --- Cypher DDL recognizer ---
    fails += check("parse CREATE", isinstance(parse_index_ddl("CREATE GFQL INDEX FOR edge_out_adj"), CreateIndex))
    fails += check("parse CREATE named+col", parse_index_ddl("CREATE GFQL INDEX pk FOR node_id ON id").column == "id")
    fails += check("parse DROP FOR", isinstance(parse_index_ddl("DROP GFQL INDEX FOR edge_in_adj"), DropIndex))
    fails += check("parse SHOW", isinstance(parse_index_ddl("SHOW GFQL INDEXES"), ShowIndexes))
    fails += check("non-DDL -> None", parse_index_ddl("MATCH (a) RETURN a") is None)

    # --- Cypher DDL via gfql() drives the registry ---
    g2 = g.gfql("CREATE GFQL INDEX FOR edge_out_adj")
    g2 = g2.gfql("CREATE GFQL INDEX FOR edge_in_adj")
    g2 = g2.gfql("CREATE GFQL INDEX FOR node_id")
    si = g2.gfql("SHOW GFQL INDEXES")
    fails += check("SHOW returns 3 rows", hasattr(si, "shape") and si.shape[0] == 3)
    g2d = g2.gfql("DROP GFQL INDEX FOR edge_in_adj")
    fails += check("DROP removed one", g2d.show_indexes().shape[0] == 2)

    # --- JSON wire round-trip ---
    op = CreateIndex(kind="edge_out_adj")
    j = op.to_json()
    fails += check("wire round-trip", index_op_from_json(j) == op)
    g3 = g.gfql({"type": "CreateIndex", "kind": "edge_out_adj"})
    fails += check("wire CreateIndex via gfql", g3.show_indexes().shape[0] == 1)
    show_via_wire = g3.gfql({"type": "ShowIndexes"})
    fails += check("wire ShowIndexes", hasattr(show_via_wire, "shape") and show_via_wire.shape[0] == 1)

    # --- parity: cypher-DDL-built index hop == scan hop ---
    base = g.hop(nodes=seeds, hops=2, direction="forward")
    idxed = g2.hop(nodes=seeds, hops=2, direction="forward")
    bn = sorted(base._nodes["id"].tolist()); xn = sorted(idxed._nodes["id"].tolist())
    fails += check("DDL-built index parity (nodes)", bn == xn)

    # --- index_policy auto/force build-on-demand + explain ---
    gp = make_graph()  # fresh, no resident indexes
    exp_use = gp.gfql_explain([], index_policy="use") if False else None  # explain needs a real query
    # explain with a seeded 1-hop chain expressed in cypher
    q = "MATCH (a)-[e]->(b) RETURN b"  # not seeded -> not coverable; use python chain instead
    from graphistry.compute.ast import n, e_forward
    seeded_chain = [n({"id": 0}), e_forward(hops=1)]
    rep_off = gp.gfql_explain(seeded_chain, index_policy="off")
    fails += check("explain off -> scan", rep_off["used_index"] is False)
    rep_force = gp.gfql_explain(seeded_chain, index_policy="force")
    fails += check("explain force -> index", rep_force["used_index"] is True)

    # force on a fresh graph actually returns correct result
    gp_use = make_graph()
    r_scan = gp_use.gfql(seeded_chain)
    r_force = gp_use.gfql(seeded_chain, index_policy="force")
    fails += check("force parity", sorted(r_scan._nodes["id"].tolist()) == sorted(r_force._nodes["id"].tolist()))

    print(f"\n=== DDL/wire smoke: {'PASS' if fails == 0 else f'{fails} FAILED'} ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
