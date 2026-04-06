"""
Step 3 bisect: find exact crash in hop(label_node_hops) on cudf 25.02.
Uses monkeypatching to add checkpoints inside hop.py operations.
"""
import sys
sys.path.insert(0, "/work")
for key in list(sys.modules.keys()):
    if key == "graphistry" or key.startswith("graphistry."):
        del sys.modules[key]

import pandas as pd
import cudf
import graphistry
from graphistry.Engine import EngineAbstract, Engine, resolve_engine
import graphistry.compute.hop as hop_module

sys.stdout.write("imports OK\n"); sys.stdout.flush()

nodes_pd = pd.DataFrame({"id": ["a", "b", "c", "d"]})
edges_pd = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"]})
nodes_cu = cudf.DataFrame(nodes_pd)
edges_cu = cudf.DataFrame(edges_pd)
g = graphistry.nodes(nodes_cu, "id").edges(edges_cu, "s", "d")
sys.stdout.write("g created\n"); sys.stdout.flush()

# Test: is the crash in the fast_path (BFS) or the hop-records hydration?
# First try with engine but NO fast_path by using allow_materialization
# Actually let's add our own step-by-step tracing

# Step 1: test that assign(col=int) on cudf works in the loop context
df_test = nodes_cu.copy()[[" id"[:3]]]  # id column only
sys.stdout.write("df_test created\n"); sys.stdout.flush()

# Simulate the inner loop new_node_ids.assign(**{node_hop_col: current_hop})
new_node_ids = cudf.DataFrame({"id": ["b", "c"]})
sys.stdout.write("new_node_ids created\n"); sys.stdout.flush()

labeled = new_node_ids.assign(**{"hop_num": 1})
sys.stdout.write(f"assign scalar in loop: {labeled['hop_num'].to_pandas().tolist()}\n"); sys.stdout.flush()

# Simulate concat of hop records
hop_rec1 = new_node_ids.assign(**{"hop_num": 1})
new_node_ids2 = cudf.DataFrame({"id": ["d"]})
hop_rec2 = new_node_ids2.assign(**{"hop_num": 2})
combined = cudf.concat([hop_rec1, hop_rec2], ignore_index=True, sort=False)
combined = combined.drop_duplicates(subset=["id"])
sys.stdout.write(f"concat hop_records: {combined.to_pandas().to_dict()}\n"); sys.stdout.flush()

# set_index + safe_map_series
from graphistry.Engine import safe_map_series
hop_map = combined.drop_duplicates(subset=["id"]).set_index("id")["hop_num"]
sys.stdout.write(f"hop_map created: {hop_map.to_pandas().to_dict()}\n"); sys.stdout.flush()

all_nodes = cudf.DataFrame({"id": ["a", "b", "c", "d"]})
mapped = safe_map_series(all_nodes["id"], hop_map)
sys.stdout.write(f"safe_map_series OK: {mapped.to_pandas().tolist()}\n"); sys.stdout.flush()

# combine_first (where notna)
hop_col = cudf.Series([None, 1, 2, None], dtype="Int64")
result = hop_col.where(hop_col.notna(), mapped)
sys.stdout.write(f"where(notna) OK: {result.to_pandas().tolist()}\n"); sys.stdout.flush()

# fillna(-1).eq(0) -- used for seed masking
hop_col2 = cudf.Series([0, 1, 2, None], dtype="Int64")
mask = hop_col2.fillna(-1).eq(0)
sys.stdout.write(f"fillna(-1).eq(0) OK: {mask.to_pandas().tolist()}\n"); sys.stdout.flush()

# to_numeric
r_numeric = cudf.to_numeric(result, errors="coerce")
sys.stdout.write(f"to_numeric OK: {r_numeric.to_pandas().tolist()}\n"); sys.stdout.flush()

# astype Int64 -- this is PANDAS nullable type, cudf may handle differently
try:
    r_int64 = r_numeric.astype("Int64")
    sys.stdout.write(f"astype Int64 OK dtype={r_int64.dtype}\n"); sys.stdout.flush()
except Exception as e:
    sys.stdout.write(f"astype Int64 RAISED (trying int64): {e}\n"); sys.stdout.flush()
    r_int64 = r_numeric.astype("int64")
    sys.stdout.write(f"astype int64 OK\n"); sys.stdout.flush()

# Test align_shared_column_dtypes which is called in hop.py line 927
from graphistry.compute.hop import align_shared_column_dtypes
sys.stdout.write("testing align_shared_column_dtypes...\n"); sys.stdout.flush()
rich_nodes = cudf.DataFrame({"id": ["a", "b", "c", "d"], "hop_num": cudf.Series([None, 1, 2, None], dtype="Int64")})
endpoints = cudf.DataFrame({"id": ["a", "b"]})
aligned = align_shared_column_dtypes(rich_nodes, endpoints)
sys.stdout.write(f"align_shared_column_dtypes OK\n"); sys.stdout.flush()

# Now try the actual hop
sys.stdout.write("about to call hop...\n"); sys.stdout.flush()
r = g.hop(hops=2, label_node_hops="hop_num", engine=EngineAbstract.CUDF)
sys.stdout.write(f"hop OK: {len(r._nodes)} nodes\n"); sys.stdout.flush()

sys.stdout.write("DONE\n"); sys.stdout.flush()
