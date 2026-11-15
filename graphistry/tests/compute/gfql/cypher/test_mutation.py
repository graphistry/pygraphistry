from __future__ import annotations

import importlib.util
import os
from typing import Any

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import cypher_to_gfql
from graphistry.compute.gfql.row.entity_props import MUTATION_NODE_ID, ID_PROPERTY_PRESENT


ENGINES = [
    pytest.param("pandas"),
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars", marks=pytest.mark.skipif(
        importlib.util.find_spec("polars") is None,
        reason="polars is not installed",
    )),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("label", ["None", "nan", "<NA>"])
def test_node_create_declines_filtered_label_names_without_effects(engine, label):
    source = _graph(engine)
    before_nodes, before_edges = _records(source._nodes), _records(source._edges)
    with pytest.raises(GFQLValidationError) as caught:
        source.gfql(f"CREATE (:`{label}`)", engine=engine)
    assert caught.value.code == ErrorCode.E108
    assert caught.value.context["field"] == "create"
    assert caught.value.context["value"] == [label]
    assert _records(source._nodes) == before_nodes and _records(source._edges) == before_edges


@pytest.mark.parametrize("engine", ENGINES)
def test_node_create_label_name_case_is_preserved(engine):
    result = _graph(engine).gfql("CREATE (:none)", engine=engine)
    assert sum(bool(row.get("label__none")) for row in _records(result._nodes)) == 1


def _records(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, pd.DataFrame):
        rows = frame.to_dict(orient="records")
    else:
        rows = frame.to_arrow().to_pylist()
    return [
        {key: None if value is None or not isinstance(value, (list, dict)) and pd.isna(value) else value for key, value in row.items()}
        for row in rows
    ]


def _assert_served(engine: str, frame: Any) -> None:
    expected = "pandas" if engine == "pandas" else engine
    assert type(frame).__module__.split(".")[0] == expected.split("-")[0]


def _graph(engine: str) -> Any:
    nodes = pd.DataFrame({"id": [1, 2], "label__Person": [True, False]})
    edges = pd.DataFrame({"src": [1], "dst": [2], "type": ["KNOWS"]})
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
    elif engine == "polars":
        polars = pytest.importorskip("polars")
        nodes = polars.from_pandas(nodes)
        edges = polars.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_adds_identity_labels_and_properties(engine: str) -> None:
    source = _graph(engine)

    result = source.gfql(
        "CREATE (a:Admin {name: 'Ada', age: 36}), (b:Admin {name: $name})",
        params={"name": "Grace"},
        engine=engine,
        language="cypher",
    )

    assert isinstance(result, type(source))
    assert result._edges is source._edges
    assert _records(result._nodes) == [
        {"id": 1, MUTATION_NODE_ID: 1, ID_PROPERTY_PRESENT: True, "label__Person": True, "label__Admin": False, "name": None, "age": None},
        {"id": 2, MUTATION_NODE_ID: 2, ID_PROPERTY_PRESENT: True, "label__Person": False, "label__Admin": False, "name": None, "age": None},
        {"id": None, MUTATION_NODE_ID: 3, ID_PROPERTY_PRESENT: False, "label__Person": False, "label__Admin": True, "name": "Ada", "age": 36},
        {"id": None, MUTATION_NODE_ID: 4, ID_PROPERTY_PRESENT: False, "label__Person": False, "label__Admin": True, "name": "Grace", "age": None},
    ]
    assert _records(source._nodes) == [
        {"id": 1, "label__Person": True},
        {"id": 2, "label__Person": False},
    ]
    assert result.gfql_graph is result
    assert source.gfql_graph is source
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_nodes_without_existing_nodes_preserves_inferred_graph(engine: str) -> None:
    edges = pd.DataFrame({"src": [7], "dst": [8], "type": ["R"]})
    source = graphistry.edges(edges, "src", "dst")

    result = source.gfql("CREATE (a:L {name: 'x'})", engine=engine, language="cypher")

    assert _records(result._nodes) == [
        {"id": 7, MUTATION_NODE_ID: 7, ID_PROPERTY_PRESENT: True, "label__L": False, "name": None},
        {"id": 8, MUTATION_NODE_ID: 8, ID_PROPERTY_PRESENT: True, "label__L": False, "name": None},
        {"id": None, MUTATION_NODE_ID: 9, ID_PROPERTY_PRESENT: False, "label__L": True, "name": "x"},
    ]
    assert _records(result._edges) == [{"src": 7, "dst": 8, "type": "R"}]
    assert source._nodes is None
    count = result.gfql("MATCH (a)-[r]->(b) RETURN count(*) AS count", engine=engine)
    assert _records(count._nodes) == [{"count": 1}]
    assert result.gfql_graph is result
    _assert_served(engine, result._nodes)


def test_create_preserves_string_existing_identity() -> None:
    source = graphistry.nodes(pd.DataFrame({"id": ["a", "b"]}), "id")
    result = source.gfql("CREATE ()", engine="pandas")
    assert result._nodes[result._node].tolist()[:2] == ["a", "b"]
    assert result._nodes[result._node].nunique() == 3
    assert _records(source._nodes) == [{"id": "a"}, {"id": "b"}]


def test_create_rejects_missing_existing_identity_column() -> None:
    source = graphistry.bind(node="id").nodes(pd.DataFrame({"name": ["Ada"]}))

    with pytest.raises(GFQLValidationError, match="missing identity column 'id'"):
        source.gfql("CREATE (a:L)", engine="pandas", language="cypher")


def test_create_return_has_rows_and_same_execution_graph() -> None:
    source = _graph("pandas")
    result = source.gfql("CREATE (a:L {name:'new'}) RETURN a.name AS name", engine="pandas")
    assert _records(result._nodes) == [{"name": "new"}]
    assert len(result.gfql_graph._nodes) == 3
    assert result.gfql_graph._nodes[result.gfql_graph._node].tolist() == [1, 2, 3]


def test_cypher_to_gfql_rejects_standalone_create() -> None:
    with pytest.raises(GFQLValidationError, match="Standalone Cypher CREATE"):
        cypher_to_gfql("CREATE (a:L)")


def test_gfql_graph_is_read_only() -> None:
    result = _graph("pandas").gfql(
        "CREATE (a:L)", engine="pandas", language="cypher"
    )

    with pytest.raises(AttributeError):
        result.gfql_graph = result


@pytest.mark.parametrize("engine", ENGINES)
def test_create_preserves_fractional_new_values_and_original_frame(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(1).alias("score"))
    else:
        nodes = nodes.assign(score=1)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    result = source.gfql("CREATE (a {score: 1.5})", engine=engine)
    assert [row["score"] for row in _records(result._nodes)] == [1, 1, 1.5]
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_sparse_created_integer_is_not_rounded_before_frame_construction(engine: str) -> None:
    wide = (1 << 60) + 1
    result = _graph(engine).gfql(
        "CREATE (a:Wide {serial: $wide}), (b:Other {name: 'other'})",
        params={"wide": wide}, engine=engine,
    )
    serials = [row["serial"] for row in _records(result._nodes) if row["serial"] is not None]
    assert serials == [wide]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_create_declines_boolean_numeric_column_union(engine: str) -> None:
    source = _graph(engine)
    nodes = source._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit(True).alias("flag"))
    else:
        nodes = nodes.assign(flag=True)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {flag: 2})", engine=engine)
    assert exc.value.context["field"] == "flag"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("dtype,maximum", [("int8", 127), ("int16", 32767), ("int64", (1 << 63) - 1)])
def test_create_checks_the_entire_allocated_id_range(engine: str, dtype: str, maximum: int) -> None:
    source = graphistry.nodes(_foundation_frame(engine, {"id": pd.Series([maximum - 1], dtype=dtype)}), "id")
    before = _records(source._nodes)
    before_dtypes = list(map(str, source._nodes.dtypes))
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a), (b)", engine=engine)
    assert exc.value.context["field"] == MUTATION_NODE_ID
    assert _records(source._nodes) == before
    assert list(map(str, source._nodes.dtypes)) == before_dtypes


@pytest.mark.parametrize("suffix", [
    "CALL graphistry.degree.write()",
    "UNWIND [1] AS x",
    "MATCH (n) RETURN n",
    "WITH 1 AS x RETURN x",
])
def test_create_rejects_the_entire_unsupported_mixed_program(suffix: str) -> None:
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLValidationError):
        source.gfql(f"CREATE (a:L) {suffix}")
    pd.testing.assert_frame_equal(source._nodes, before)


def test_strict_polars_gpu_creation_has_explicit_capability_decline() -> None:
    from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with pytest.raises(GFQLUnsupportedError) as exc:
        source.gfql("CREATE (a:L)", engine="polars-gpu")
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "engine"
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("transform", ["nodes", "edges", "bind"])
def test_ordinary_and_transformed_graph_accessors_return_their_own_graph(transform: str) -> None:
    source = _graph("pandas")
    assert source.gfql_graph is source
    created = source.gfql("CREATE (a:L)")
    if transform == "nodes":
        result = created.nodes(created._nodes.iloc[:1].copy())
    elif transform == "edges":
        result = created.edges(created._edges.iloc[:0].copy())
    else:
        result = created.bind()
    assert result is not created and result.gfql_graph is result
    assert created.gfql_graph is created and len(created._nodes) == 3


def test_create_preflight_is_validate_only_and_matches_execution(monkeypatch) -> None:
    import graphistry.compute.gfql_unified as unified
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    with monkeypatch.context() as context:
        def reject_execution(*args, **kwargs):
            pytest.fail("preflight dispatched a mutation executor")
        context.setattr(unified, "execute_create_nodes", reject_execution)
        assert source.gfql_validate("CREATE (a:L)")["compiled_kind"] == "mutation"
    pd.testing.assert_frame_equal(source._nodes, before)
    ordinary = source.gfql("CREATE (a:L)")
    checked = source.gfql("CREATE (a:L)", validate=True)
    pd.testing.assert_frame_equal(ordinary._nodes, checked._nodes)


@pytest.mark.parametrize("validate", [False, True])
def test_remote_mutation_declines_before_any_request(monkeypatch, validate: bool) -> None:
    from importlib import import_module
    remote_module = import_module("graphistry.compute.chain_remote")
    def reject_io(*args, **kwargs):
        pytest.fail("unsupported remote mutation reached HTTP")
    monkeypatch.setattr(remote_module.requests, "post", reject_io)
    with pytest.raises(GFQLValidationError) as exc:
        _graph("pandas").gfql_remote(
            "CREATE (a:L)", dataset_id="fixture", api_token="fixture-token",
            format="json", validate=validate,
        )
    assert exc.value.context["value"] == "mutation"


@pytest.mark.parametrize("engine", ENGINES)
def test_create_converts_pandas_input_to_the_explicit_native_engine(engine: str) -> None:
    source = _graph("pandas")
    before = source._nodes.copy(deep=True)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert len(result._nodes) == 3
    _assert_served(engine, result._nodes)
    _assert_served(engine, result._edges)
    pd.testing.assert_frame_equal(source._nodes, before)


@pytest.mark.parametrize("engine", ENGINES)
def test_wide_existing_integer_with_missing_new_property_remains_exact(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    result = source.gfql("CREATE (a:L)", engine=engine)
    assert [row["serial"] for row in _records(result._nodes)] == [wide, wide, None]


@pytest.mark.parametrize("engine", ENGINES)
def test_new_float_does_not_round_existing_wide_integer(engine: str) -> None:
    source = _graph(engine)
    wide = (1 << 60) + 1
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.lit(wide).alias("serial"))
    else:
        nodes = source._nodes.assign(serial=wide)
    source = source.nodes(nodes)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (a {serial: 1.5})", engine=engine)
    assert exc.value.context["field"] == "serial"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_new_integer_does_not_truncate_existing_fractional_values(engine: str) -> None:
    source = _graph(engine)
    if engine == "polars":
        import polars as pl
        nodes = source._nodes.with_columns(pl.Series("score", [1.5, -2.25]))
    else:
        nodes = source._nodes.assign(score=[1.5, -2.25])
    source = source.nodes(nodes)
    before = _records(source._nodes)
    result = source.gfql("CREATE (a {score: 2})", engine=engine)
    assert [row["score"] for row in _records(result._nodes)] == [1.5, -2.25, 2]
    assert _records(source._nodes) == before
    _assert_served(engine, result._nodes)


def _foundation_frame(engine,data):
    value=pd.DataFrame(data)
    if engine=="cudf":
        import cudf
        return cudf.from_pandas(value)
    if engine=="polars":
        pl = pytest.importorskip("polars")
        return pl.from_pandas(value)
    return value

@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("query,count,expected",[
    ("CREATE ()",1,None),
    ("CREATE (), ()",2,None),
    ("CREATE (:A:B), (:B:C)",2,None),
    ("CREATE (a {name:'foo'}) RETURN a.name AS p",1,[{"p":"foo"}]),
    ("CREATE (a {id:12,name:null}) RETURN a.id AS id,a.name AS name",1,[{"id":12,"name":None}]),
    ("CREATE (a {id:4611686018427387905}) RETURN a.id AS id",1,[{"id":4611686018427387905}]),
    ("CREATE (), () RETURN 1 AS x",2,[{"x":1}]),
    ("CREATE (a {name:'a'}),(b {name:'b'}) RETURN a.name AS a,b.name AS b",2,[{"a":"a","b":"b"}]),
])
def test_foundation(engine,query,count,expected):
    source=graphistry.nodes(_foundation_frame(engine,{"id":pd.Series([],dtype="int64")}),"id")
    before=_records(source._nodes)
    result=source.gfql(query,engine=engine)
    assert _records(source._nodes)==before
    graph=result.gfql_graph
    assert len(graph._nodes)==count
    assert graph._node!="id"
    assert type(graph._nodes).__module__.split(".")[0]==engine
    if expected is not None:
        assert _records(result._nodes)==expected
        assert result._gfql_rows_base_graph is None
        assert result._gfql_start_nodes is None
        assert graph.gfql_graph is graph
    else:
        assert result is graph

@pytest.mark.parametrize("engine",ENGINES)
def test_legacy_string_id(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"]}),"id")
    result=source.gfql("CREATE (a {id:'new'}),(b) RETURN a.id AS explicit,b.id AS missing",engine=engine)
    assert _records(result._nodes)==[{"explicit":"new","missing":None}]
    graph=result.gfql_graph
    assert graph._node!="id" and len(graph._nodes)==3
    assert _records(source._nodes)==[{"id":"old"}]
    again=graph.gfql("CREATE ()",engine=engine)
    assert len(again._nodes)==4 and again._node==graph._node
    assert list(again._nodes.columns)==list(graph._nodes.columns)

@pytest.mark.parametrize("engine", ENGINES[:2])
@pytest.mark.parametrize("old",["old",10])
def test_current_id_membership(engine,old):
    explicit="new" if isinstance(old,str) else 20
    source=graphistry.nodes(_foundation_frame(engine,{"id":[old]}),"id")
    graph=source.gfql("CREATE (a {id:$id}),(b)",params={"id":explicit},engine=engine)
    result=graph.gfql("MATCH (n) RETURN n.id AS id, keys(n) AS keys, properties(n) AS props",engine=engine)
    got=_records(result._nodes)
    assert [row["id"] for row in got]==[old,explicit,None]
    assert [list(row["keys"]) for row in got]==[[] if isinstance(old,str) else ["id"],["id"],[]]
    assert got[0]["props"]==("{}" if isinstance(old,str) else "{id: 10}")
    assert got[1]["props"]==("{id: 'new'}" if isinstance(old,str) else "{id: 20}")
    assert got[2]["props"]=="{}"

@pytest.mark.parametrize("engine",ENGINES)
def test_whole_entity_and_lifecycle(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"label__Old":[True]}),"id")
    result=source.gfql("CREATE (a:L {id:'new'}) RETURN a",engine=engine)
    assert set(result._nodes.columns)=={"a.id","a.label__Old","a.label__L"}
    assert _records(result._nodes)[0]["a.id"]=="new"
    graph=result.gfql_graph
    assert graph.gfql("MATCH (n) RETURN count(*) AS c",engine=engine)._nodes is not None
    count=_records(graph.gfql("MATCH (n) RETURN count(*) AS c",engine=engine)._nodes)
    assert count==[{"c":2}]
    # Result association survives cleanup; graph transformations describe themselves.
    assert result.gfql_graph is graph
    assert graph.nodes(graph._nodes).gfql_graph is not graph
    assert source.gfql_graph is source

@pytest.mark.parametrize("name",["__graphistry_node_identity__","__gfql_id_property_present__"])
def test_reserved_metadata_collision(name):
    source=graphistry.nodes(pd.DataFrame({"id":[1],name:[False]}),"id")
    before=source._nodes.copy(deep=True)
    from graphistry.compute.exceptions import GFQLValidationError
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE ()",engine="pandas")
    assert exc.value.context["field"] == "create"
    assert exc.value.context["value"] == [name]
    pd.testing.assert_frame_equal(source._nodes,before)

@pytest.mark.parametrize("engine",ENGINES)
def test_all_null_old_property_does_not_stringify_new_integer(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1],"p":pd.Series([None],dtype="string")}),"id")
    before=_records(source._nodes)
    result=source.gfql("CREATE (n {p:12}) RETURN n.p AS p",engine=engine)
    assert _records(result._nodes)==[{"p":12}]
    assert _records(source._nodes)==before


@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("expression",["__gfql_created_return__ AS leaked","*"])
def test_anonymous_scope_does_not_invent_a_public_binding(engine,expression):
    source=graphistry.nodes(_foundation_frame(engine,{"id":pd.Series([],dtype="int64")}),"id")
    with pytest.raises(GFQLValidationError):
        source.gfql("CREATE () RETURN "+expression,engine=engine)
    assert len(source._nodes)==0

@pytest.mark.parametrize("engine",ENGINES)
def test_replaced_ordinary_frame_is_readmitted(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    created=source.gfql("CREATE ()",engine=engine)
    replaced=created.nodes(_foundation_frame(engine,{"id":[10,20]}),"id")
    before=_records(replaced._nodes)
    result=replaced.gfql("CREATE ()",engine=engine)
    assert len(result._nodes)==3
    assert _records(replaced._nodes)==before
    assert result._node!="id"
    assert [row[result._node] for row in _records(result._nodes)][:2] == [10, 20]

@pytest.mark.parametrize("engine",ENGINES)
def test_rebinding_metadata_is_typed_not_silently_reused(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id").gfql("CREATE ()",engine=engine)
    rebound=source.bind(node="id")
    before=_records(rebound._nodes)
    with pytest.raises(GFQLValidationError):
        rebound.gfql("CREATE ()",engine=engine)
    assert _records(rebound._nodes)==before


def test_functional_create_documentation_example() -> None:
    from pathlib import Path
    import textwrap
    document = (Path(__file__).resolve().parents[5] / "docs/source/gfql/cypher.rst").read_text()
    section = document.split("Functional Node Creation\n", 1)[1]
    lines = section.split(".. code-block:: python\n", 1)[1].splitlines()
    code = []
    for line in lines:
        if line and not line.startswith("   "):
            break
        code.append(line)
    exec(textwrap.dedent("\n".join(code)), {})


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_alias_whole_entity_preserves_membership_without_helpers(engine):
    source = graphistry.nodes(_foundation_frame(engine, {"id": ["old"]}), "id")
    result = source.gfql("CREATE (a {id:'new'}), (b) RETURN a, b", engine=engine)
    assert set(result._nodes.columns) == {"a.id", "b.id"}
    assert _records(result._nodes) == [{"a.id": "new", "b.id": None}]
    assert len(result.gfql_graph._nodes) == 3
    assert _records(source._nodes) == [{"id": "old"}]


@pytest.mark.parametrize("engine",ENGINES)
@pytest.mark.parametrize("transform",["nodes","edges","bind"])
def test_unrelated_row_result_transforms_do_not_keep_saved_graph(engine,transform):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (n {name:'new'}) RETURN n.name AS name",engine=engine)
    saved=rows.gfql_graph
    if transform=="nodes":
        changed=rows.nodes(_foundation_frame(engine,{"id":[10]}),"id")
        count=changed.gfql("MATCH (n) RETURN count(*) AS count",engine=engine)
        assert _records(count._nodes)==[{"count":1}]
        assert count.gfql_graph is count
    elif transform=="edges":
        changed=rows.edges(_foundation_frame(engine,{"src":[],"dst":[]}),"src","dst")
    else:
        changed=rows.bind(node="name")
    assert changed.gfql_graph is changed
    assert rows.gfql_graph is saved and len(saved._nodes)==3
    assert source.gfql_graph is source and len(source._nodes)==2

@pytest.mark.parametrize("engine",ENGINES)
def test_cleanup_and_view_bind_preserve_same_execution_association(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (n {name:'new'}) RETURN n.name AS name",engine=engine)
    saved=rows.gfql_graph
    assert rows.bind().gfql_graph is saved
    from graphistry.compute.gfql.exec_context import attach_row_exec_context, clear_row_exec_context
    contextual=attach_row_exec_context(rows,rows_base_graph=source)
    clean=clear_row_exec_context(contextual)
    assert clean.gfql_graph is saved
    assert clean._gfql_rows_base_graph is None and clean._gfql_start_nodes is None


@pytest.mark.parametrize("value",[[1],{"a":1},float("nan")],ids=["list","map","nan"])
@pytest.mark.parametrize("mode",["validate_only","execute","execute_validate"])
def test_known_unsupported_property_parameters_decline_consistently(value,mode):
    source=graphistry.nodes(_foundation_frame("pandas",{"id":[1,2]}),"id")
    before=_records(source._nodes)
    query="CREATE (n {p:$value})"
    with pytest.raises(GFQLValidationError) as exc:
        if mode=="validate_only":
            source.gfql_validate(query,params={"value":value})
        else:
            source.gfql(query,params={"value":value},validate=mode=="execute_validate",engine="pandas")
    from graphistry.compute.exceptions import ErrorCode
    assert exc.value.code == ErrorCode.E108
    assert _records(source._nodes)==before

@pytest.mark.parametrize("engine",ENGINES)
def test_empty_return_retains_updated_graph_same_execution(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":[1,2]}),"id")
    rows=source.gfql("CREATE (a {name:'new'}) RETURN a.name AS name SKIP 1",engine=engine)
    assert _records(rows._nodes)==[]
    graph=rows.gfql_graph
    assert len(graph._nodes)==3 and graph.gfql_graph is graph
    assert len(graph.gfql("CREATE ()",engine=engine)._nodes)==4
    assert _records(source._nodes)==[{"id":1},{"id":2}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_old_legacy_type_with_new_canonical_label(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":["Old"]}),"id")
    with pytest.raises(GFQLValidationError,match="legacy type labels"):
        source.gfql("CREATE (:New)",engine=engine)
    assert _records(source._nodes)==[{"id":"old","type":"Old"}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_same_legacy_and_canonical_label(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":["Shared"]}),"id")
    with pytest.raises(GFQLValidationError,match="legacy type labels"):
        source.gfql("CREATE (:Shared)",engine=engine)
    assert _records(source._nodes)==[{"id":"old","type":"Shared"}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_declines_any_nonnull_legacy_type_with_new_canonical_label(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old","new"],"type":["Old",None]}),"id")
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (:New)",engine=engine)
    from graphistry.compute.exceptions import ErrorCode
    assert exc.value.code==ErrorCode.E108
    assert exc.value.context["field"]=="create"
    assert exc.value.context["value"]==["New"]
    assert _records(source._nodes)==[{"id":"old","type":"Old"},{"id":"new","type":None}]


@pytest.mark.parametrize("engine",ENGINES)
def test_create_admits_all_null_legacy_type_with_new_canonical_label(engine):
    source=graphistry.nodes(_foundation_frame(engine,{"id":["old"],"type":[None]}),"id")
    result=source.gfql("CREATE (:New)",engine=engine)
    assert len(result._nodes)==2
    assert [row["label__New"] for row in _records(result._nodes)]==[False,True]
    assert _records(source._nodes)==[{"id":"old","type":None}]


def _typed_frame(engine, data, dtypes):
    frame = pd.DataFrame({name: pd.Series(values, dtype=dtypes.get(name)) for name, values in data.items()})
    if engine == "cudf":
        import cudf
        return cudf.from_pandas(frame)
    if engine == "polars":
        return pytest.importorskip("polars").from_pandas(frame)
    return frame


# A16-F1: created ids must skip every edge endpoint, so a new node never attaches to a dangling edge.
@pytest.mark.parametrize("engine", ENGINES)
def test_create_ids_skip_edge_endpoints_so_a_new_node_has_no_dangling_edge(engine):
    nodes = _typed_frame(engine, {"id": [1, 2]}, {"id": "int64"})
    edges = _typed_frame(engine, {"s": [1], "d": [3]}, {"s": "int64", "d": "int64"})
    source = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    result = source.gfql("CREATE (:New)", engine=engine)
    assert sorted(row[result._node] for row in _records(result._nodes)) == [1, 2, 4]
    count = result.gfql("MATCH (n:New)--() RETURN count(*) AS c", engine=engine)
    assert _records(count._nodes) == [{"c": 0}]
    assert _records(source._nodes) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_ids_skip_edge_endpoints_on_an_empty_node_frame(engine):
    nodes = _typed_frame(engine, {"id": []}, {"id": "int64"})
    edges = _typed_frame(engine, {"s": [0], "d": [1]}, {"s": "int64", "d": "int64"})
    result = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql("CREATE (:New)", engine=engine)
    assert [row[result._node] for row in _records(result._nodes)] == [2]
    count = result.gfql("MATCH (n:New)--() RETURN count(*) AS c", engine=engine)
    assert _records(count._nodes) == [{"c": 0}]


# A16-F2: a legacy list-valued labels column is declined before effects, like the legacy type column.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", ["CREATE (:C)", "CREATE ()"])
def test_create_declines_a_legacy_list_valued_labels_column_before_effects(engine, query):
    frame = pd.DataFrame({"id": [1, 2], "labels": [["A"], ["B"]]})
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    elif engine == "polars":
        frame = pytest.importorskip("polars").from_pandas(frame)
    source = graphistry.nodes(frame, "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(query, engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "create" and exc.value.context["value"] == "legacy_labels"
    assert _records(source._nodes) == before


# B16-F1: the id contract is "never equal ANY existing node key or edge endpoint value, for every dtype".
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("edge_data,edge_dtypes", [
    ({"s": [1.0], "d": [3.0]}, {"s": "float64", "d": "float64"}),
    ({"s": [1, 2], "d": [3, None]}, {"s": "int64", "d": "float64"}),
    ({"s": [1], "d": ["3"]}, {"s": "int64", "d": "object"}),
])
def test_create_ids_skip_float_null_promoted_and_numeric_string_endpoints(engine, edge_data, edge_dtypes):
    nodes = _typed_frame(engine, {"id": [1, 2]}, {"id": "int64"})
    edges = _typed_frame(engine, edge_data, edge_dtypes)
    source = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    result = source.gfql("CREATE (:New)", engine=engine)
    assert sorted(row[result._node] for row in _records(result._nodes)) == [1, 2, 4]
    if edge_dtypes["d"] == "float64":
        count = result.gfql("MATCH (n:New)--() RETURN count(*) AS c", engine=engine)
        assert _records(count._nodes) == [{"c": 0}]


@pytest.mark.parametrize("engine", ENGINES)
def test_create_ids_skip_float_endpoints_on_an_empty_node_frame(engine):
    nodes = _typed_frame(engine, {"id": []}, {"id": "int64"})
    edges = _typed_frame(engine, {"s": [0.0], "d": [1.0]}, {"s": "float64", "d": "float64"})
    result = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql("CREATE (:New)", engine=engine)
    assert [row[result._node] for row in _records(result._nodes)] == [2]


# B16-F4: string ids skip string edge endpoints too.
@pytest.mark.parametrize("engine", ENGINES)
def test_create_string_ids_skip_string_edge_endpoints(engine):
    nodes = _typed_frame(engine, {"id": ["a"]}, {"id": "object"})
    edges = _typed_frame(engine, {"s": ["a"], "d": ["__gfql_created_0"]}, {"s": "object", "d": "object"})
    result = graphistry.nodes(nodes, "id").edges(edges, "s", "d").gfql("CREATE (:New)", engine=engine)
    assert sorted(row[result._node] for row in _records(result._nodes)) == ["__gfql_created_1", "a"]


# B16-F2: any non-null labels column is declined, also next to label__* flags (the shape the TCK harness builds).
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("data", [
    {"id": [1], "labels": [["A"]], "label__A": [True]},
    {"id": [1], "labels": ["A"]},
])
def test_create_declines_a_non_null_labels_column_with_or_without_flags(engine, data):
    source = graphistry.nodes(_typed_frame(engine, data, {}), "id")
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql("CREATE (:New)", engine=engine)
    assert exc.value.code == ErrorCode.E108
    assert exc.value.context["field"] == "create" and exc.value.context["value"] == "legacy_labels"
    assert _records(source._nodes) == before


@pytest.mark.parametrize("engine", ENGINES)
def test_create_accepts_an_all_null_labels_column(engine):
    source = graphistry.nodes(_typed_frame(engine, {"id": [1, 2], "labels": [None, None]}, {"labels": "float64"}), "id")
    result = source.gfql("CREATE (:New)", engine=engine)
    assert sorted(row[result._node] for row in _records(result._nodes)) == [1, 2, 3]


# B16-F3: a Polars LazyFrame is a structured lazy_frame decline (never an assert/TypeError), and AUTO may fall back.
@pytest.mark.skipif(importlib.util.find_spec("polars") is None, reason="polars is not installed")
@pytest.mark.parametrize("lazy", ["nodes", "edges", "both"])
def test_create_on_a_polars_lazy_frame_is_a_structured_decline(lazy):
    from graphistry.compute.exceptions import GFQLUnsupportedError
    polars = pytest.importorskip("polars")
    nodes, edges = polars.DataFrame({"id": [1, 2]}), polars.DataFrame({"s": [1], "d": [2]})
    source = graphistry.nodes(nodes.lazy() if lazy in {"nodes", "both"} else nodes, "id").edges(
        edges.lazy() if lazy in {"edges", "both"} else edges, "s", "d",
    )
    with pytest.raises(GFQLUnsupportedError) as exc:
        source.gfql("CREATE (:X)", engine="polars")
    assert exc.value.code == ErrorCode.E108 and exc.value.context["value"] == "lazy_frame"
    try:
        source.gfql("CREATE (:X)")
    except GFQLUnsupportedError as auto_exc:
        assert auto_exc.context["value"] == "lazy_frame"


# A16-F5: created ids keep an unsigned integer id dtype, so a second CREATE does not fail on a widened column.
@pytest.mark.parametrize("engine", ENGINES)
def test_create_keeps_an_unsigned_integer_id_dtype_across_repeated_creates(engine):
    source = graphistry.nodes(_typed_frame(engine, {"id": [1, 2]}, {"id": "uint64"}), "id")
    first = source.gfql("CREATE (:A)", engine=engine)
    second = first.gfql("CREATE (:B)", engine=engine)
    assert sorted(row[second._node] for row in _records(second._nodes)) == [1, 2, 3, 4]
    dtype = second._nodes.schema[second._node] if engine == "polars" else second._nodes[second._node].dtype
    assert "uint" in str(dtype).lower()


# A16-F6: reserved and duplicate property keys are declined before any effect.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,code,field", [
    ("CREATE (n {labels: 1})", ErrorCode.E108, "labels"),
    ("CREATE (n {type: 1})", ErrorCode.E108, "type"),
    ("CREATE (n {__x__: 1})", ErrorCode.E108, "__x__"),
    ("CREATE (n {label__X: 1})", ErrorCode.E108, "label__X"),
    ("CREATE (n {p: 1, p: 2})", ErrorCode.E105, "p"),
])
def test_create_declines_reserved_and_duplicate_keys_before_effects(engine, query, code, field):
    source = _graph(engine)
    before = _records(source._nodes)
    with pytest.raises(GFQLValidationError) as exc:
        source.gfql(query, engine=engine)
    assert exc.value.code == code
    assert exc.value.context["field"] == field
    assert _records(source._nodes) == before


# Only the internal CREATE identity column is hidden from flattened whole-entity RETURN fields; user keys starting
# with __ stay.
@pytest.mark.parametrize("engine", ENGINES)
def test_user_node_key_starting_with_double_underscore_stays_a_flattened_field(engine):
    graph = graphistry.nodes(_typed_frame(engine, {"__vid": [1, 2], "name": ["a", "b"]}, {}), "__vid")
    result = graph.gfql("MATCH (x) RETURN x", engine=engine)
    assert {"x.__vid", "x.name"} <= set(result._nodes.columns)


@pytest.mark.parametrize("engine", ENGINES)
def test_internal_create_identity_is_not_a_flattened_field(engine):
    result = _graph(engine).gfql("CREATE (n:L {name: 'z'}) RETURN n", engine=engine)
    assert f"n.{MUTATION_NODE_ID}" not in result._nodes.columns
    assert "n.name" in result._nodes.columns
