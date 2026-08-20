"""`CALL graphistry.std.*` — the kernels reachable as GFQL queries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.algorithms import kernels as K
from graphistry.compute.algorithms._dfops import dense_renumber


@pytest.fixture(scope="module")
def g():
    return graphistry.edges(pd.DataFrame({"s": [10, 11, 10, 55], "d": [11, 12, 12, 66]}), "s", "d")


@pytest.mark.parametrize("query,col", [
    ("CALL graphistry.std.wcc.write()", "component"),
    ("CALL graphistry.std.cdlp.write({params: {iterations: 3}})", "cdlp"),
    ("CALL graphistry.std.mis.write()", "mis"),
    ("CALL graphistry.std.sssp.write({params: {source: 0}})", "distance"),
    ("CALL graphistry.std.pagerank.write()", "pagerank"),
])
def test_std_call_writes_expected_column(g, query, col):
    assert col in g.gfql(query)._nodes.columns


def test_out_col_override(g):
    assert "pr" in g.gfql("CALL graphistry.std.pagerank.write({out_col: 'pr'})")._nodes.columns


def test_unknown_std_procedure_is_rejected(g):
    from graphistry.compute.exceptions import GFQLValidationError

    with pytest.raises(GFQLValidationError):
        g.gfql("CALL graphistry.std.louvain.write()")


def test_call_result_matches_the_direct_kernel():
    """The query surface must not change the answer, and WCC labels must come
    back in the caller's id space -- the label IS a vertex id."""
    rng = np.random.default_rng(5)
    e = pd.DataFrame({"s": rng.integers(0, 500, 3000), "d": rng.integers(0, 500, 3000)})
    e = e[e["s"] != e["d"]]

    out = graphistry.edges(e, "s", "d").gfql("CALL graphistry.std.wcc.write()")
    got = out._nodes.sort_values("id").reset_index(drop=True)

    dense, ids, v_count = dense_renumber(e, "s", "d")
    ref = [int(ids.iloc[int(x)]) for x in K.wcc(dense, "s", "d", v_count)]

    assert list(got["component"]) == ref
    assert got["component"].min() == got["id"].min()
