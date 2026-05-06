"""
Smoke test for #996: CASE x WHEN null on DGX.
Run inside RAPIDS container:
  docker run --rm --gpus all -v /checkout:/work -w /work \
    --entrypoint /opt/conda/bin/python \
    graphistry/test-rapids-official:26.02-cuda13-gfql \
    /work/graphistry/tests/compute/gfql/cypher/_dgx_996_smoke.py
"""
import sys
# Force /work ahead of the container-installed package at /opt/pygraphistry
sys.path.insert(0, "/work")
# Remove any already-cached graphistry modules so the insert takes effect
for key in list(sys.modules.keys()):
    if key == "graphistry" or key.startswith("graphistry."):
        del sys.modules[key]

import pandas as pd
import graphistry

nodes = pd.DataFrame({
    "id": ["m", "c", "c2", "p", "p2", "a"],
    "label__Message": [True, False, False, False, False, False],
    "label__Comment": [False, True, True, False, False, False],
    "label__Person":  [False, False, False, True, True, True],
})
edges = pd.DataFrame({
    "s":    ["c", "c",           "c2", "c2",          "m",           "a"],
    "d":    ["m", "p",           "m",  "p2",           "a",           "p"],
    "type": ["REPLY_OF", "HAS_CREATOR", "REPLY_OF", "HAS_CREATOR", "HAS_CREATOR", "KNOWS"],
})
g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

query = (
    "MATCH (m:Message)<-[:REPLY_OF]-(c:Comment)-[:HAS_CREATOR]->(p:Person) "
    "OPTIONAL MATCH (m)-[:HAS_CREATOR]->(a:Person)-[r:KNOWS]-(p) "
    "RETURN c.id AS commentId, p.id AS replyAuthorId, "
    "CASE r WHEN null THEN false ELSE true END AS knows "
    "ORDER BY commentId"
)

failures = []

def run_case(label, result_g):
    rows = result_g._nodes[["commentId", "replyAuthorId", "knows"]].to_dict(orient="records")
    expected = [
        {"commentId": "c",  "replyAuthorId": "p",  "knows": True},
        {"commentId": "c2", "replyAuthorId": "p2", "knows": False},
    ]
    for exp in expected:
        match = any(
            r.get("commentId") == exp["commentId"]
            and r.get("replyAuthorId") == exp["replyAuthorId"]
            and bool(r.get("knows")) == exp["knows"]
            for r in rows
        )
        if match:
            print(f"  PASS  [{label}] {exp}")
        else:
            print(f"  FAIL  [{label}] expected {exp}, got {rows}")
            failures.append(f"{label}: {exp}")

# pandas path
run_case("pandas", g.gfql(query))

# cudf path — exercises __cypher_case_eq__ null fix on GPU row pipeline
try:
    import cudf
    from graphistry.Engine import EngineAbstract
    nodes_cu = cudf.DataFrame(nodes)
    edges_cu = cudf.DataFrame(edges)
    g_cu = graphistry.nodes(nodes_cu, "id").edges(edges_cu, "s", "d")
    run_case("cudf", g_cu.gfql(query, engine=EngineAbstract.CUDF))
except Exception as e:
    print(f"  SKIP  [cudf] {e}")

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
else:
    print("\nAll #996 smoke tests passed.")
