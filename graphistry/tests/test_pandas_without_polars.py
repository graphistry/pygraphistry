"""The pandas engine must work with polars ABSENT — and never import it.

CI's minimal lanes lack the polars wheel, but dev boxes have it, so a future
module-level polars import would pass everywhere developers run tests and fail
only in CI. This subprocess harness hard-blocks the import and exercises every
release-touched surface on pandas.
"""
import subprocess
import sys

_HARNESS = r"""
import sys

class _BlockPolars:
    def find_spec(self, name, path=None, target=None):
        if name == "polars" or name.startswith("polars."):
            raise ImportError("polars blocked for test")
        return None

sys.meta_path.insert(0, _BlockPolars())

import pandas as pd
import graphistry

nodes = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["P"] * 4, "age": [25, 35, 45, 55]})
edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3], "rel": ["F"] * 3})
g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

out = g.gfql("MATCH (a {kind:'P'})-[]->(b) WHERE b.age >= 35 RETURN b.id AS x ORDER BY x")._nodes
assert out.to_dict("records") == [{"x": 1}, {"x": 2}, {"x": 3}], out
out = g.gfql("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})-[{rel:'F'}]->(c {kind:'P'}) RETURN count(*) AS n")._nodes
assert out.to_dict("records") == [{"n": 2}], out
assert sorted(g.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, direction="forward")._nodes["id"].tolist()) == [0, 1, 2]
gi = g.gfql_index_all().gfql_index_col_stats(node_type_column="kind", edge_type_column="rel")
assert gi is not None
out = g.gfql("MATCH (a) WITH a, a.age AS aa MATCH (a)-[]->(b) RETURN aa, b.id AS bid ORDER BY bid")._nodes
assert len(out) == 3, out
g.tree_layout()
from graphistry.compute.gfql_unified import gfql_clear_caches
gfql_clear_caches()
assert not any(m == "polars" for m in sys.modules), "polars was imported on a pandas-only path"
print("OK")
"""


def test_pandas_paths_work_and_never_import_polars() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _HARNESS],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    assert proc.stdout.strip().endswith("OK")
