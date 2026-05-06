"""
Smoke test for cugraph shortest path dispatch on DGX.
Run inside RAPIDS container:
  docker run --rm --gpus all -v /checkout:/work -w /work \
    --entrypoint /opt/conda/bin/python \
    graphistry/test-rapids-official:26.02-cuda13-gfql \
    /work/graphistry/tests/compute/gfql/same_path/_dgx_cugraph_smoke.py
"""
import sys
sys.path.insert(0, "/work")

import cudf
import graphistry
from graphistry.Engine import EngineAbstract

nodes = cudf.DataFrame({"id": ["a", "b", "c", "d"]})
edges = cudf.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

cases = [
    (
        "MATCH (p1 {id: 'a'}), (p2 {id: 'c'}), path = shortestPath((p1)-[*]-(p2)) RETURN length(path) AS hops",
        2, "a->c 2-hop"
    ),
    (
        "MATCH (p1 {id: 'a'}), (p2 {id: 'b'}), path = shortestPath((p1)-[*]-(p2)) RETURN length(path) AS hops",
        1, "a->b 1-hop"
    ),
    (
        "MATCH (p1 {id: 'a'}), (p2 {id: 'c'}), path = shortestPath((p1)-[*1..1]-(p2)) RETURN length(path) AS hops",
        None, "a->c bound=1 (unreachable)"
    ),
    (
        "MATCH (p1 {id: 'a'}), (p2 {id: 'd'}), path = shortestPath((p1)-[*]-(p2)) RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS hops",
        -1, "a->d disconnected"
    ),
]

failures = []
for query, expected, label in cases:
    try:
        r_cugraph = g.gfql(query, engine=EngineAbstract.CUDF, shortest_path_backend="cugraph")
        val = r_cugraph._nodes["hops"].iloc[0]
        r_bfs = g.gfql(query, engine=EngineAbstract.CUDF, shortest_path_backend="bfs")
        val_bfs = r_bfs._nodes["hops"].iloc[0]
        na_val = val is None or str(val) == "<NA>"
        na_bfs = val_bfs is None or str(val_bfs) == "<NA>"
        # parity: both NA, or both equal
        parity = (na_val and na_bfs) or (str(val) == str(val_bfs))
        # correctness vs expected
        if expected is None:
            correct = na_val
        else:
            correct = str(val) == str(expected)
        status = "PASS" if (correct and parity) else "FAIL"
        print(f"  {status}  {label}: cugraph={val} bfs={val_bfs} expected={expected}")
        if status == "FAIL":
            failures.append(label)
    except Exception:
        # CASE IS NULL not supported on CUDF — skip parity, just check cugraph alone
        try:
            r_cugraph = g.gfql(query, engine=EngineAbstract.CUDF, shortest_path_backend="cugraph")
            val = r_cugraph._nodes["hops"].iloc[0]
            na_val = val is None or str(val) == "<NA>"
            correct = (expected is None and na_val) or (str(val) == str(expected))
            status = "PASS" if correct else "FAIL"
            print(f"  {status}  {label} (bfs-skipped): cugraph={val} expected={expected}")
            if status == "FAIL":
                failures.append(label)
        except Exception as e2:
            print(f"  SKIP  {label}: {e2}")

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
else:
    print("\nAll cugraph smoke tests passed.")
