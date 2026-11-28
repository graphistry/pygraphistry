"""A boolean value projected in a WITH is a value, never a node/relationship alias (#2368)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLCallExecutionError, GFQLValidationError


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


_NODES = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "flag": [True, False]})
_EDGES = pd.DataFrame({"s": [1], "d": [2], "ok": [True]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


_POLARS = {"polars", "polars-gpu"}


def _run(engine, query):
    out = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


def _expect_value_type_error(engine, query):
    """pandas and cuDF raise the row-expression type error; the Polars engines decline the `select` row op (NotImplementedError)."""
    with pytest.raises(NotImplementedError if engine in _POLARS else GFQLCallExecutionError):
        _run(engine, query)


_VALUE_MISUSE = [
    "MATCH (n) WITH n, n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n, n.flag AS t RETURN keys(t) AS v",
    "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(t) AS v",
]
_SHADOWING = [  # a projected value that reuses an alias name is a value, not the alias
    "MATCH (n) WITH n.flag AS n RETURN properties(n) AS v",
    "MATCH (n) WITH n.flag AS n RETURN keys(n) AS v",
    "MATCH (n) WITH n ORDER BY n.id LIMIT 2 WITH n.flag AS n RETURN properties(n) AS v",
    "MATCH (n) WITH DISTINCT n WITH n.flag AS n RETURN keys(n) AS v",
    "MATCH (n) WITH n.flag AS n, count(*) AS c RETURN properties(n) AS v",
    "MATCH ()-[r]->() WITH r.ok AS r RETURN properties(r) AS v",
]
_NO_MATCH = [  # a boolean from a source that has no MATCH alias at all
    "UNWIND [true, false] AS b RETURN properties(b) AS v",
    "UNWIND [true, false] AS b RETURN keys(b) AS v",
    "WITH true AS b RETURN properties(b) AS v",
    "WITH true AS b RETURN b.name AS v",
    "UNWIND [true] AS a UNWIND [false] AS b RETURN properties(a) AS v",
    "UNWIND [true] AS a UNWIND [false] AS b RETURN properties(b) AS v",
]
_DOTTED_READS = [
    "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v",
    "MATCH (n) WITH n.flag AS t RETURN t.name AS v",
    "MATCH (n) WITH n, n.id AS t RETURN t.name AS v",
    "MATCH (n) WITH n.flag AS t WHERE t.name = 'a' RETURN t AS v",
    "MATCH (n) WITH n.flag AS t RETURN t AS v ORDER BY t.name",
    "MATCH (n) WITH n.flag AS t RETURN t.name AS k, count(*) AS c",
    "MATCH (n) WITH n.flag AS t RETURN collect(t.name) AS v",
]
_TYPE_ERROR = "property access requires a graph element alias, entity value, or map"


@pytest.mark.parametrize("query", _VALUE_MISUSE + _SHADOWING + _NO_MATCH)
def test_a_projected_boolean_is_a_type_error_for_entity_functions(engine, query):
    _expect_value_type_error(engine, query)


@pytest.mark.parametrize("query", _DOTTED_READS)
def test_a_dotted_property_read_of_a_projected_scalar_is_a_type_error(engine, query):
    """Every engine, Polars lowering included, raises the row-expression type error (#2384)."""
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run(engine, query)


def test_a_dotted_property_read_of_a_projected_string_is_a_type_error_or_a_polars_decline(engine):
    """pandas' answer for a string depends on its values (entity or duration text), so Polars declines."""
    query = "MATCH (n) WITH n, n.name AS t RETURN t.name AS v"
    with pytest.raises(NotImplementedError if engine in _POLARS else GFQLCallExecutionError):
        _run(engine, query)


def _run_on(engine, flags, query):
    nodes = pd.DataFrame({"id": list(range(len(flags))), "name": ["x"] * len(flags), "flag": pd.array(flags, dtype="boolean")})
    if engine == "cudf":
        import cudf

        nodes = cudf.from_pandas(nodes)
    elif engine in _POLARS:
        import polars as pl

        nodes = pl.from_pandas(nodes)
    out = graphistry.nodes(nodes, "id").gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out


def test_a_dotted_property_read_of_a_projected_boolean_with_some_nulls_is_a_type_error(engine):
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run_on(engine, [True, None], "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v")


def test_a_dotted_property_read_of_an_all_null_projected_boolean_is_null(engine):
    out = _run_on(engine, [None, None], "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v")
    assert out["v"].isna().all() and len(out) == 2


def test_a_dotted_property_read_of_a_projected_boolean_over_no_rows_is_empty(engine):
    out = _run_on(engine, [True, False], "MATCH (n) WHERE n.id > 100 WITH n, n.flag AS t RETURN t.name AS v")
    assert len(out) == 0


def test_real_alias_temporal_comparison_keeps_its_engine_boundary(engine):
    query = "MATCH (n) WITH n, n.flag AS t RETURN n.ts = '2026-01-01' AS v"
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):
            _run_temporal(engine, [True, False], query)
        return
    out = _run_temporal(engine, [True, False], query)
    assert out["v"].tolist() == [True, True]


def test_cartesian_real_alias_read_beside_projected_value(engine):
    rows = _run(engine, "MATCH (a), (b) WITH a, b.flag AS t RETURN a.name AS v")
    assert sorted(row["v"] for row in rows) == ["a", "a", "b", "b"]


def test_real_aliases_beside_a_projected_boolean_keep_working(engine):
    node_query = "MATCH (n) WITH n, n.flag AS t RETURN properties(n) AS v, t AS b"
    rel_query = "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(r) AS v, t AS b"
    if engine in _POLARS:
        for query in (node_query, rel_query):
            with pytest.raises(NotImplementedError):
                _run(engine, query)
        return
    rows = _run(engine, node_query)
    assert sorted(row["b"] for row in rows) == [False, True]
    assert sorted(row["v"] for row in rows) == ["{id: 1, name: 'a', flag: true}", "{id: 2, name: 'b', flag: false}"]
    rel = _run(engine, rel_query)
    assert [row["b"] for row in rel] == [True]
    assert [row["v"] for row in rel] == ["{ok: true}"]


def test_a_map_variable_and_a_plain_alias_are_unchanged(engine):
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):
            _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")
        return
    assert [row["v"] for row in _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")] == ["{a: 1}", "{a: 1}"]
    assert len(_run(engine, "MATCH (n) RETURN properties(n) AS v")) == 2


@pytest.mark.parametrize("query", [
    "MATCH (n) WITH n.flag AS n RETURN n.name AS v",
    "MATCH (n) WITH n.flag AS n, n.name AS name RETURN n.name AS v",
    "MATCH (n) WITH n.flag AS n, n.name AS name RETURN n.n AS v",
    "MATCH (n) WITH n.id AS k, collect(n.flag) AS n RETURN n.name AS v",
])
def test_shadowed_alias_dotted_reads_are_type_errors(engine, query):
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run(engine, query)


@pytest.mark.parametrize("query", [
    "MATCH (n) WITH n RETURN n.name AS v",
])
def test_alias_carry_keeps_entity_reads(engine, query):
    assert sorted(row["v"] for row in _run(engine, query)) == ["a", "b"]


def test_alias_rename_keeps_the_current_pandas_validation_contract(engine):
    # The shared Cypher compiler currently rejects renamed aliases before engine dispatch.
    with pytest.raises(GFQLValidationError, match="Unknown Cypher alias 'm'"):
        _run(engine, "MATCH (n) WITH n AS m RETURN m.name AS v")


@pytest.mark.parametrize("query", [
    "UNWIND [true] AS t WITH t, 'x' AS name RETURN t.name AS v",
    "MATCH (a), (b) WITH a.flag AS t, a.name AS name RETURN t.name AS v",
])
def test_row_producers_without_alias_metadata_reject_value_reads(engine, query):
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run(engine, query)


def test_single_node_binding_rows_track_entity_aliases():
    pytest.importorskip("polars")
    import polars as pl

    from graphistry.compute.ast import n, rows, select
    from graphistry.compute.chain import serialize_binding_ops
    from graphistry.compute.gfql.lazy.engine.polars.pattern_apply import rows_binding_ops_polars

    g = graphistry.nodes(pl.from_pandas(_NODES), "id").edges(pl.from_pandas(_EDGES), "s", "d")
    out = rows_binding_ops_polars(g, serialize_binding_ops([n(name="bound")]))

    assert out is not None
    assert out._gfql_rows_node_aliases == {"bound"}
    assert out._gfql_rows_edge_aliases == set()

    ops = [
        rows(binding_ops=serialize_binding_ops([n(name="n")])),
        select(items=[("n", "n"), ("t", "n.flag"), ("name", "n.name")]),
        select(items=[("v", "t.name")]),
    ]
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        g.gfql(ops, engine="polars")

    positive = [
        rows(binding_ops=serialize_binding_ops([n(name="n")])),
        select(items=[("n", "n"), ("t", "n.flag"), ("v", "n.name")]),
    ]
    assert sorted(row["v"] for row in g.gfql(positive, engine="polars")._nodes.to_dicts()) == ["a", "b"]


def test_polars_projection_tracks_alias_carry_rename_and_shadowing():
    pytest.importorskip("polars")
    import polars as pl

    from graphistry.compute.gfql.identifiers import shadow_restore_column
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import _project_polars

    restore = shadow_restore_column("n")
    g = graphistry.nodes(pl.DataFrame({"id": [1], "n": [1], "value": [True], restore: ["saved"]}), "id")
    g._gfql_rows_node_aliases = {"n"}
    g._gfql_rows_edge_aliases = set()

    carried = _project_polars(g, ["n", (restore, restore)], extend=False)
    assert carried is not None
    assert carried._gfql_rows_node_aliases == {"n"}
    assert carried._gfql_rows_edge_aliases == set()

    renamed = _project_polars(g, [("m", "n")], extend=False)
    assert renamed is not None
    assert renamed._gfql_rows_node_aliases == {"m"}
    for expression in ("`n`", "(n)"):
        quoted_rename = _project_polars(g, [("m", expression)], extend=False)
        assert quoted_rename is not None
        assert quoted_rename._gfql_rows_node_aliases == {"m"}

    extended = _project_polars(g, [("n", "value")], extend=True)
    assert extended is not None
    assert extended._gfql_rows_node_aliases == set()


def test_polars_extend_projection_tracks_alias_rename_and_shadowing():
    pytest.importorskip("polars")
    import polars as pl

    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import select_extend_polars

    g = graphistry.nodes(pl.DataFrame({"id": [1], "n": [1], "m": [2], "value": [True]}), "id")
    g._gfql_rows_node_aliases = {"n"}
    g._gfql_rows_edge_aliases = set()

    renamed = select_extend_polars(g, [("m", "n")])
    assert renamed is not None
    assert renamed._gfql_rows_node_aliases == {"n", "m"}

    shadowed = select_extend_polars(g, [("n", "value")])
    assert shadowed is not None
    assert shadowed._gfql_rows_node_aliases == set()


def test_extend_alias_shadow_keeps_current_cross_engine_behavior(engine):
    # pandas currently resolves the retained qualified binding property after this extend.
    # Keep Polars in parity; a broader change to WITH scope belongs in a separate fix.
    rows = _run(engine, "MATCH (a), (b) WITH b, a.flag AS a RETURN a.name AS v")
    assert sorted(row["v"] for row in rows) == ["a", "a", "b", "b"]


def test_connected_binding_select_shadow_and_carry_boundaries(engine):
    shadowed = "MATCH (n)-[e]->(m) WITH n.flag AS n, n.name AS name RETURN n.name AS v"
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run(engine, shadowed)

    carried = _run(engine, "MATCH (n)-[e]->(m) WITH n RETURN n.name AS v")
    assert carried == [{"v": "a"}]


def test_bindings_select_rewrap_applies_projection_aliases(monkeypatch):
    pytest.importorskip("polars")
    import polars as pl

    from graphistry.compute.ast import e_forward, n, rows, select
    from graphistry.compute.gfql.lazy.engine.polars import chain as polars_chain
    from graphistry.compute.gfql.lazy.engine.polars.chain_specializations.bindings_select import (
        rewrap_projected_polars,
    )

    g = graphistry.nodes(pl.DataFrame({"id": [1], "name": ["source"]}), "id").edges(
        pl.DataFrame({"s": [1], "d": [1]}), "s", "d",
    )
    shadowed = rewrap_projected_polars(
        g, [n(name="n")], pl.DataFrame({"n": [True], "name": ["source"]}),
        [("n", "n.flag"), ("name", "n.name")],
    )
    assert shadowed._gfql_rows_node_aliases == set()
    assert shadowed._gfql_rows_edge_aliases == set()

    carried = rewrap_projected_polars(
        g, [n(name="n")], pl.DataFrame({"n": [1], "name": ["source"]}),
        [("n", "`n`"), ("name", "n.name")],
    )
    assert carried._gfql_rows_node_aliases == {"n"}

    ops = [
        n(name="n"), e_forward(name="e"), n(name="m"), rows(),
        select(items=[("n", "n.flag"), ("name", "n.name")]),
        select(items=[("v", "n.name")]),
    ]
    monkeypatch_frame = pl.DataFrame({"n": [True], "name": ["source"]})
    monkeypatch.setattr(
        polars_chain, "try_bindings_select_polars", lambda *_args: monkeypatch_frame,
    )
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        polars_chain.chain_polars(g, ops)


def test_projection_alias_tracking_tolerates_non_cypher_column_names():
    pytest.importorskip("polars")
    import polars as pl

    from graphistry.compute.ast import n
    from graphistry.compute.gfql.lazy.engine.polars.chain_specializations.bindings_select import (
        rewrap_projected_polars,
    )
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import _project_polars

    g = graphistry.nodes(pl.DataFrame({
        "id": [1], "n": [1], "first name": ["first"], "when": ["then"],
        "n.first name": ["qualified"],
    }), "id")
    g._gfql_rows_node_aliases = {"first name", "when", "n"}
    g._gfql_rows_edge_aliases = set()

    projected = _project_polars(
        g, ["first name", "when", ("qualified", "n.first name")], extend=False,
    )
    assert projected is not None
    assert projected._gfql_rows_node_aliases == {"first name", "when"}
    assert projected._gfql_rows_edge_aliases == set()

    rewrapped = rewrap_projected_polars(
        g, [n(name="n")], pl.DataFrame({"n": [1], "qualified": ["qualified"]}),
        [("n", "n"), ("qualified", "n.first name")],
    )
    assert rewrapped._gfql_rows_node_aliases == {"n"}
    assert rewrapped._gfql_rows_edge_aliases == set()


def _run_temporal(engine, flags, query):
    nodes = pd.DataFrame({
        "id": list(range(len(flags))),
        "flag": pd.array(flags, dtype="boolean"),
        "ts": pd.to_datetime(["2026-01-01"] * len(flags)),
    })
    if engine == "cudf":
        import cudf

        nodes = cudf.from_pandas(nodes)
    elif engine in _POLARS:
        import polars as pl

        nodes = pl.from_pandas(nodes)
    out = graphistry.nodes(nodes, "id").gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out


def test_temporal_compare_on_projected_non_null_value_is_type_error(engine):
    query = "MATCH (n) WITH n, n.flag AS t RETURN t.ts = '2026-01-01' AS v"
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run_temporal(engine, [True, False], query)


def test_temporal_compare_on_all_null_projected_value_returns_null(engine):
    query = "MATCH (n) WITH n, n.flag AS t RETURN t.ts = '2026-01-01' AS v"
    out = _run_temporal(engine, [None, None], query)
    assert len(out) == 2 and out["v"].isna().all()


def test_temporal_compare_on_empty_projected_value_returns_empty(engine):
    query = (
        "MATCH (n) WHERE n.id > 100 WITH n, n.flag AS t "
        "RETURN t.ts = '2026-01-01' AS v"
    )
    assert len(_run_temporal(engine, [True, False], query)) == 0
