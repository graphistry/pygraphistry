"""
Bisect script for #977: find which cudf operation SIGSEGVs on RAPIDS 25.02.
hop(label_node_hops=...) path still crashes after safe_map_series fix.

Run inside 25.02 container:
  docker run --rm --gpus all -v /checkout:/work \
    --entrypoint /opt/conda/bin/python \
    graphistry/test-rapids-official:25.02-cuda12-gfql \
    /work/graphistry/tests/compute/gfql/cypher/_dgx_977_bisect.py
"""
import sys
sys.path.insert(0, "/work")

import pandas as pd

print("1. import cudf")
import cudf
print("2. cudf imported OK")

# Simulate the node hop column data types produced by hop()
# node IDs are strings, hop numbers are integers
node_ids = cudf.Series(["a", "b", "c", "d"])
hop_vals = cudf.Series([1, 1, 2, 2], dtype='int64')

print("3. Basic cudf Series created OK")

# fillna(-1) -- used in hop.py line 976
print("4. Testing fillna(-1) on int64...")
hop_nullable = cudf.Series([1, None, 2, None], dtype='Int64')
result = hop_nullable.fillna(-1)
print(f"   fillna(-1) OK: {result.to_pandas().tolist()}")

# fillna(-1).eq(0)
print("5. Testing fillna(-1).eq(0)...")
result2 = hop_nullable.fillna(-1).eq(0)
print(f"   fillna(-1).eq(0) OK: {result2.to_pandas().tolist()}")

# astype('Int64') -- pandas extension type, cudf may not support
print("6. Testing astype('Int64') (nullable pandas extension type)...")
try:
    result3 = hop_vals.astype('Int64')
    print(f"   astype('Int64') OK: dtype={result3.dtype}")
except Exception as e:
    print(f"   astype('Int64') raised: {e}")
    print("   Trying astype('int64')...")
    result3 = hop_vals.astype('int64')
    print(f"   astype('int64') OK: dtype={result3.dtype}")

# cudf.to_numeric -- used in hop.py line 980
print("7. Testing cudf.to_numeric(errors='coerce')...")
hop_object = cudf.Series([1.0, None, 2.0, None])
result4 = cudf.to_numeric(hop_object, errors='coerce')
print(f"   cudf.to_numeric OK: {result4.to_pandas().tolist()}")

# assign scalar int column -- used at hop.py line 564, 579
print("8. Testing DataFrame.assign scalar int column...")
df = cudf.DataFrame({"id": ["a", "b", "c"]})
df2 = df.assign(hop_num=1)
print(f"   assign scalar OK: {df2['hop_num'].to_pandas().tolist()}")

# groupby min -- used at hop.py line 962
print("9. Testing groupby min...")
edges_df = cudf.DataFrame({"node": ["a", "b", "a", "c"], "hop": [1, 1, 2, 2]})
gmin = edges_df.groupby("node")["hop"].min()
print(f"   groupby min OK: {gmin.to_pandas().to_dict()}")

# where(notna, other) -- used at hop.py line 939-942
print("10. Testing Series.where(notna, other)...")
target = cudf.Series([1, None, 3, None], dtype='Int64')
fill = cudf.Series([10, 20, 30, 40], dtype='Int64')
result5 = target.where(target.notna(), fill)
print(f"    where(notna, fill) OK: {result5.to_pandas().tolist()}")

# .loc[mask, col] = safe_map_series result -- used at hop.py line 974
print("11. Testing .loc[mask, col] = mapped_series...")
nodes_df = cudf.DataFrame({"id": ["a", "b", "c", "d"], "hop_num": [1, None, 3, None]})
mask = nodes_df["hop_num"].isna()
mapping = cudf.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
# safe_map_series bridges through pandas
mapped_pd = nodes_df.loc[mask, "id"].to_pandas().map(mapping.to_pandas())
mapped_cu = cudf.Series(mapped_pd, index=nodes_df.loc[mask, "id"].index)
nodes_df.loc[mask, "hop_num"] = mapped_cu
print(f"    .loc[mask, col] = mapped OK: {nodes_df['hop_num'].to_pandas().tolist()}")

# isin with cudf Index -- used in chain.py / hop.py
print("12. Testing .isin on cudf Series...")
ids = cudf.Series(["a", "b"])
result6 = node_ids.isin(ids)
print(f"    isin OK: {result6.to_pandas().tolist()}")

# concat with sort=False -- heavily used
print("13. Testing concat...")
df_a = cudf.DataFrame({"id": ["a", "b"], "hop_num": [1, 2]})
df_b = cudf.DataFrame({"id": ["c", "d"], "hop_num": [3, 4]})
concat_result = cudf.concat([df_a, df_b], ignore_index=True, sort=False)
print(f"    concat OK: {len(concat_result)} rows")

# drop_duplicates
print("14. Testing drop_duplicates...")
dup_df = cudf.DataFrame({"id": ["a", "a", "b"], "hop_num": [1, 1, 2]})
deduped = dup_df.drop_duplicates(subset=["id"])
print(f"    drop_duplicates OK: {len(deduped)} rows")

# merge left
print("15. Testing merge left...")
left = cudf.DataFrame({"id": ["a", "b", "c"]})
right = cudf.DataFrame({"id": ["a", "b"], "hop_num": [1, 2]})
merged = left.merge(right, on="id", how="left")
print(f"    merge left OK: {merged['hop_num'].to_pandas().tolist()}")

print("\nAll bisect checks PASSED — SIGSEGV not reproduced with isolated ops.")
print("Check if hop() calls combine_first, mask(), or other ops not listed here.")
