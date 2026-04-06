"""
Smoke test for #977: cudf SIGSEGV fix on RAPIDS 25.02.
Exercises Series.map(dict) paths that previously SIGSEGV'd via numba JIT.

Run inside RAPIDS container (--gpus all required):
  docker run --rm --gpus all -v /checkout:/work \
    --entrypoint /opt/conda/bin/python \
    graphistry/test-rapids-official:25.02-cuda12-gfql \
    /work/graphistry/tests/compute/gfql/cypher/_dgx_977_smoke.py
"""
import sys
sys.path.insert(0, "/work")
for key in list(sys.modules.keys()):
    if key == "graphistry" or key.startswith("graphistry."):
        del sys.modules[key]

import pandas as pd
import graphistry
from graphistry.Engine import EngineAbstract

failures = []


def check(label: str, result_val: object, expected: object) -> None:
    ok = result_val == expected
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  [{label}] got={result_val!r} expected={expected!r}")
    if not ok:
        failures.append(label)


def run_pandas() -> None:
    """Pandas path — regression guard, must not break."""
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "label__Person": [True, True, False, True]})
    edges = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["T", "T", "T"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    # label filter
    r = g.gfql("MATCH (p:Person) RETURN p.id AS pid ORDER BY pid")
    ids = sorted(r._nodes["pid"].tolist())
    check("pandas label filter", ids, ["a", "b", "d"])

    # hop with label_node_hops — exercises hop.py .map path
    r2 = g.hop(hops=2, label_node_hops="hop_num")
    check("pandas hop label_node_hops non-null", r2._nodes["hop_num"].notna().any(), True)


def run_cudf() -> None:
    """cudf path — exercises all safe_map_series sites."""
    try:
        import cudf
    except ImportError:
        print("  SKIP  [cudf] cudf not available")
        return

    nodes_pd = pd.DataFrame({"id": ["a", "b", "c", "d"], "label__Person": [True, True, False, True]})
    edges_pd = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["T", "T", "T"]})
    nodes = cudf.DataFrame(nodes_pd)
    edges = cudf.DataFrame(edges_pd)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    # label filter — exercises filter_by_dict cudf path
    r = g.gfql("MATCH (p:Person) RETURN p.id AS pid ORDER BY pid", engine=EngineAbstract.CUDF)
    ids = sorted(r._nodes["pid"].to_pandas().tolist())
    check("cudf label filter", ids, ["a", "b", "d"])

    # single hop — exercises df_executor safe_map_series
    r2 = g.gfql("MATCH (a)-[:T]->(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid", engine=EngineAbstract.CUDF)
    check("cudf single hop row count", len(r2._nodes), 3)

    # hop with label_node_hops — exercises hop.py safe_map_series
    r3 = g.hop(hops=2, label_node_hops="hop_num", engine=EngineAbstract.CUDF)
    check("cudf hop label_node_hops non-null", bool(r3._nodes["hop_num"].notna().any()), True)

    # pipeline .map path — ORDER BY exercises safe_map_series in pipeline.py
    r4 = g.gfql("MATCH (a)-[:T]->(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid", engine=EngineAbstract.CUDF)
    check("cudf ORDER BY no crash", len(r4._nodes) > 0, True)

    # Engine.py:399-400 — cudf series + pd.Series mapping (hop_map from set_index)
    # Exercises the elif isinstance(mapping, pd.Series) branch in safe_map_series
    from graphistry.Engine import safe_map_series
    s = cudf.Series(["a", "b", "c", "a"])
    mapping = pd.Series([0, 1, 2], index=["a", "b", "c"])
    result = safe_map_series(s, mapping)
    check("cudf safe_map_series + pd.Series mapping", result.to_pandas().tolist(), [0, 1, 2, 0])

    # hop.py:834-891 — output_max_hops filters via node_mask + fallback_map try/except
    nodes4_pd = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    edges4_pd = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "type": ["T", "T", "T"]})
    g4 = graphistry.nodes(cudf.DataFrame(nodes4_pd), "id").edges(cudf.DataFrame(edges4_pd), "s", "d")
    n_a = g4._nodes[g4._nodes["id"] == "a"]
    r5 = g4.hop(nodes=n_a, hops=3, direction="forward", label_node_hops="nh",
                label_seeds=True, output_max_hops=1, engine=EngineAbstract.CUDF)
    hop_vals = dict(zip(r5._nodes["id"].to_pandas(), r5._nodes["nh"].to_pandas()))
    check("cudf output_max_hops seed hop=0", int(hop_vals.get("a", -1)), 0)
    check("cudf output_max_hops hop=1 present", int(hop_vals.get("b", -1)), 1)
    check("cudf output_max_hops hop=2 filtered", "c" not in hop_vals, True)

    # hop.py:843-859 — label_seeds=True: seed rows added from node_hop_records
    r6 = g4.hop(nodes=n_a, hops=2, direction="forward", label_node_hops="nh",
                label_seeds=True, engine=EngineAbstract.CUDF)
    hop_vals6 = dict(zip(r6._nodes["id"].to_pandas(), r6._nodes["nh"].to_pandas()))
    check("cudf label_seeds seed hop=0", int(hop_vals6.get("a", -1)), 0)
    check("cudf label_seeds hop=1", int(hop_vals6.get("b", -1)), 1)

    # hop.py:987-1000 — label_seeds=False + undirected clears seed hop
    nodes3_pd = pd.DataFrame({"id": ["a", "b", "c"]})
    edges3_pd = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["T", "T"]})
    g3 = graphistry.nodes(cudf.DataFrame(nodes3_pd), "id").edges(cudf.DataFrame(edges3_pd), "s", "d")
    n_b = g3._nodes[g3._nodes["id"] == "b"]
    r7 = g3.hop(nodes=n_b, hops=1, direction="undirected", label_node_hops="nh",
                label_seeds=False, engine=EngineAbstract.CUDF)
    hop_vals7 = dict(zip(r7._nodes["id"].to_pandas(), r7._nodes["nh"].to_pandas()))
    import math
    b_hop = hop_vals7.get("b", float("nan"))
    check("cudf label_seeds=False undirected seed cleared",
          b_hop is None or (isinstance(b_hop, float) and math.isnan(b_hop)) or b_hop != b_hop,
          True)


run_pandas()
run_cudf()

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
else:
    print("\nAll #977 smoke tests passed.")
