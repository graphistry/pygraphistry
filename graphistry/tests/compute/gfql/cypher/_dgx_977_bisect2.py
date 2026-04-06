"""Minimal SIGSEGV bisect — find crash point before any cudf call."""
import sys
sys.stdout.write("A: python started\n")
sys.stdout.flush()

import os
sys.stdout.write("B: import os OK\n")
sys.stdout.flush()

import pandas as pd
sys.stdout.write("C: import pandas OK\n")
sys.stdout.flush()

sys.stdout.write("D: about to import cudf\n")
sys.stdout.flush()
import cudf
sys.stdout.write("E: import cudf OK\n")
sys.stdout.flush()

sys.stdout.write("F: cudf.Series(['a','b'])\n")
sys.stdout.flush()
s = cudf.Series(["a", "b"])
sys.stdout.write(f"G: Series OK len={len(s)}\n")
sys.stdout.flush()

sys.stdout.write("H: cudf.DataFrame({'id':['a','b']})\n")
sys.stdout.flush()
df = cudf.DataFrame({"id": ["a", "b"]})
sys.stdout.write(f"I: DataFrame OK len={len(df)}\n")
sys.stdout.flush()

sys.stdout.write("J: df.assign(hop_num=1)\n")
sys.stdout.flush()
df2 = df.assign(hop_num=1)
sys.stdout.write(f"K: assign scalar OK {df2['hop_num'].to_pandas().tolist()}\n")
sys.stdout.flush()

sys.stdout.write("L: isin\n")
sys.stdout.flush()
s2 = cudf.Series(["a"])
mask = s.isin(s2)
sys.stdout.write(f"M: isin OK {mask.to_pandas().tolist()}\n")
sys.stdout.flush()

sys.stdout.write("N: fillna\n")
sys.stdout.flush()
s3 = cudf.Series([1, None, 2], dtype="Int64")
r = s3.fillna(-1)
sys.stdout.write(f"O: fillna OK {r.to_pandas().tolist()}\n")
sys.stdout.flush()

sys.stdout.write("P: to_numeric\n")
sys.stdout.flush()
r2 = cudf.to_numeric(s3, errors="coerce")
sys.stdout.write(f"Q: to_numeric OK {r2.to_pandas().tolist()}\n")
sys.stdout.flush()

sys.stdout.write("R: astype Int64\n")
sys.stdout.flush()
hop_int = cudf.Series([1, 2, 3], dtype="int64")
try:
    r3 = hop_int.astype("Int64")
    sys.stdout.write(f"S: astype Int64 OK dtype={r3.dtype}\n")
except Exception as e:
    sys.stdout.write(f"S: astype Int64 RAISED: {e}\n")
sys.stdout.flush()

sys.stdout.write("T: groupby min\n")
sys.stdout.flush()
df3 = cudf.DataFrame({"node": ["a", "b", "a"], "hop": [1, 2, 3]})
gmin = df3.groupby("node")["hop"].min()
sys.stdout.write(f"U: groupby min OK\n")
sys.stdout.flush()

sys.stdout.write("V: import graphistry\n")
sys.stdout.flush()
sys.path.insert(0, "/work")
for key in list(sys.modules.keys()):
    if key == "graphistry" or key.startswith("graphistry."):
        del sys.modules[key]
import graphistry
sys.stdout.write("W: import graphistry OK\n")
sys.stdout.flush()

from graphistry.Engine import EngineAbstract, safe_map_series
sys.stdout.write("X: safe_map_series imported OK\n")
sys.stdout.flush()

# Now try actual hop
nodes_pd = pd.DataFrame({"id": ["a", "b", "c", "d"]})
edges_pd = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"]})
nodes_cu = cudf.DataFrame(nodes_pd)
edges_cu = cudf.DataFrame(edges_pd)
g = graphistry.nodes(nodes_cu, "id").edges(edges_cu, "s", "d")
sys.stdout.write("Y: graphistry graph created OK\n")
sys.stdout.flush()

r4 = g.hop(hops=1, engine=EngineAbstract.CUDF)
sys.stdout.write(f"Z1: hop without label_node_hops OK: {len(r4._nodes)} nodes\n")
sys.stdout.flush()

r5 = g.hop(hops=1, label_node_hops="hop_num", engine=EngineAbstract.CUDF)
sys.stdout.write(f"Z2: hop WITH label_node_hops OK: {len(r5._nodes)} nodes\n")
sys.stdout.flush()

sys.stdout.write("DONE: all bisect checks passed\n")
