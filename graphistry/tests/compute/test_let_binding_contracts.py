"""Contracts a ``let()`` DAG binding must satisfy regardless of how the bindings dict is written.

Issue #1923.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.typing import DataFrameT
from graphistry.compute.ast import ASTLet, ASTRef, call, n
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLValidationError
from graphistry.compute.chain_let import detect_cycles, in_declaration_order

ENGINES: Tuple[str, ...] = ("pandas", "polars")

NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e"],
    "type": ["person", "person", "company", "person", "company"],
})
EDGES = pd.DataFrame({
    "s": ["a", "b", "a", "d", "c"],
    "d": ["b", "c", "c", "a", "e"],
})

#: Hand-computed from EDGES. Undirected incident-edge count per node.
ROOT_DEGREE: Dict[str, int] = {"a": 3, "b": 2, "c": 3, "d": 1, "e": 1}
#: Hand-computed from EDGES. Incoming-edge count per node.
ROOT_DEGREE_IN: Dict[str, int] = {"a": 1, "b": 1, "c": 2, "d": 0, "e": 1}
ROOT_NODE_COUNT = 5
ROOT_EDGE_COUNT = 5
PERSON_IDS = ["a", "b", "d"]


def _frame(engine: str, df: pd.DataFrame) -> DataFrameT:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return pl.from_pandas(df)
    return df


def _graph(engine: str) -> Plottable:
    return graphistry.nodes(_frame(engine, NODES), "id").edges(
        _frame(engine, EDGES), "s", "d"
    )


def _to_pandas(df: DataFrameT) -> pd.DataFrame:
    return df if "pandas" in type(df).__module__ else df.to_pandas()


def _node_column(g: Plottable, col: str) -> Dict[str, int]:
    nodes = _to_pandas(g._nodes)
    return dict(zip(nodes["id"], nodes[col]))


def _node_ids(g: Plottable) -> List[str]:
    return sorted(_to_pandas(g._nodes)["id"].tolist())


# --- Every binding kind filters from the DAG's root graph (#1923 F2)


@pytest.mark.parametrize("engine", ENGINES)
def test_call_get_degrees_binding_reads_root_graph_not_the_previous_binding(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"flt": n({"type": "person"}), "deg": call("get_degrees")}),
        output="deg",
        engine=engine,
    )
    assert _node_column(result, "degree") == ROOT_DEGREE


@pytest.mark.parametrize("engine", ENGINES)
def test_call_get_indegrees_binding_reads_root_graph_not_the_previous_binding(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"flt": n({"type": "person"}), "deg": call("get_indegrees")}),
        output="deg",
        engine=engine,
    )
    assert _node_column(result, "degree_in") == ROOT_DEGREE_IN


@pytest.mark.parametrize("engine", ENGINES)
def test_call_hop_binding_reads_root_graph_not_the_previous_binding(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"flt": n({"type": "person"}), "hopped": call("hop", {"hops": 1})}),
        output="hopped",
        engine=engine,
    )
    assert _node_ids(result) == sorted(NODES["id"].tolist())
    assert len(result._edges) == ROOT_EDGE_COUNT


@pytest.mark.parametrize("engine", ENGINES)
def test_call_count_table_binding_reads_root_graph_not_the_previous_binding(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"flt": n({"type": "person"}), "cnt": call("count_table")}),
        output="cnt",
        engine=engine,
    )
    assert _to_pandas(result._nodes)["count(*)"].tolist() == [ROOT_NODE_COUNT]


@pytest.mark.parametrize("engine", ENGINES)
def test_call_after_call_bindings_do_not_accumulate_each_others_columns(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"first": call("get_degrees"), "second": call("get_indegrees")}),
        output="second",
        engine=engine,
    )
    assert _node_column(result, "degree_in") == ROOT_DEGREE_IN
    assert "degree" not in _to_pandas(result._nodes).columns


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("output", ["deg", "flt"])
def test_reordering_independent_binding_keys_does_not_change_answers(engine: str, output: str) -> None:
    flt_first = ASTLet({"flt": n({"type": "person"}), "deg": call("get_degrees")})
    deg_first = ASTLet({"deg": call("get_degrees"), "flt": n({"type": "person"})})

    g = _graph(engine)
    assert _node_ids(g.gfql(flt_first, output=output, engine=engine)) == \
        _node_ids(g.gfql(deg_first, output=output, engine=engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_filter_binding_is_unaffected_by_a_sibling_call_binding(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"flt": n({"type": "person"}), "deg": call("get_degrees")}),
        output="flt",
        engine=engine,
    )
    assert _node_ids(result) == PERSON_IDS
    assert "degree" not in _to_pandas(result._nodes).columns


# --- Default output is the last declared binding, in every process (#1923 F1)

_SIBLING_ORDER_PROGRAM = """
import sys
import pandas as pd
import graphistry
from graphistry.compute.ast import ASTLet, ASTRef, n

engine = sys.argv[1]
nodes = pd.DataFrame({'id': ['a', 'b', 'c', 'd', 'e'], 'k': ['A', 'B', 'C', 'A', 'B']})
edges = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
if engine == 'polars':
    import polars as pl
    nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
g = graphistry.nodes(nodes, 'id').edges(edges, 's', 'd')
result = g.gfql(ASTLet({
    'root': n({}),
    'q1': ASTRef('root', [n({'k': 'A'})]),
    'q2': ASTRef('root', [n({'k': 'B'})]),
    'q3': ASTRef('root', [n({'k': 'C'})]),
}), engine=engine)
ids = result._nodes['id']
print(','.join(sorted(ids.to_list() if hasattr(ids, 'to_list') else ids.tolist())))
"""

#: ``q3`` is declared last and every sibling is ready at the same time, so ``q3`` is the answer.
_SIBLING_ORDER_ANSWER = "c"


def _importable_root() -> str:
    """Directory to put on the child's path so it imports THIS checkout of graphistry."""
    return os.path.dirname(os.path.dirname(os.path.abspath(graphistry.__file__)))


def _run_sibling_order(engine: str, hash_seed: str) -> str:
    root = _importable_root()
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=root)
    completed = subprocess.run(
        [sys.executable, "-c", _SIBLING_ORDER_PROGRAM, engine],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip()


@pytest.mark.parametrize("hash_seed", ["0", "1", "2", "3", "7", "13"])
def test_default_output_is_the_last_declared_binding_under_every_hash_seed(hash_seed: str) -> None:
    assert _run_sibling_order("pandas", hash_seed) == _SIBLING_ORDER_ANSWER


@pytest.mark.parametrize("hash_seed", ["0", "1", "7"])
def test_default_output_is_the_last_declared_binding_under_every_hash_seed_polars(hash_seed: str) -> None:
    pytest.importorskip("polars")
    assert _run_sibling_order("polars", hash_seed) == _SIBLING_ORDER_ANSWER


@pytest.mark.parametrize("engine", ENGINES)
def test_default_output_prefers_declaration_order_over_name_order(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({
            "root": n({}),
            "zebra": ASTRef("root", [n({"type": "company"})]),
            "alpha": ASTRef("root", [n({"type": "person"})]),
        }),
        engine=engine,
    )
    assert _node_ids(result) == PERSON_IDS


# --- The DAG root is not reachable through the user binding namespace (#1923 F3)


@pytest.mark.parametrize("engine", ENGINES)
def test_binding_named_original_graph_does_not_reroot_later_bindings(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({"__original_graph__": n({"type": "company"}), "x": n({})}),
        output="x",
        engine=engine,
    )
    assert _node_ids(result) == sorted(NODES["id"].tolist())


@pytest.mark.parametrize("engine", ENGINES)
def test_root_graph_is_not_addressable_as_an_output_binding(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(ASTLet({"x": n({})}), output="__original_graph__", engine=engine)
    assert exc_info.value.code == ErrorCode.E151


@pytest.mark.parametrize("engine", ENGINES)
def test_output_binding_error_lists_only_user_bindings(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(ASTLet({"x": n({})}), output="nope", engine=engine)
    assert "Available bindings: ['x']" in str(exc_info.value)


# --- Traversing a row-pipeline result (#1923 F4)


def test_ref_traversal_after_row_call_declines_typed_on_polars() -> None:
    pytest.importorskip("polars")
    dag = ASTLet({"lim": call("limit", {"value": 2}), "after": ASTRef("lim", [n({})])})
    with pytest.raises(NotImplementedError, match="unbound edge endpoints"):
        _graph("polars").gfql(dag, output="after", engine="polars")


def test_ref_traversal_after_row_call_runs_on_pandas() -> None:
    dag = ASTLet({"lim": call("limit", {"value": 2}), "after": ASTRef("lim", [n({})])})
    result = _graph("pandas").gfql(dag, output="after", engine="pandas")
    assert len(result._nodes) == 2


def test_polars_traversal_on_a_graph_without_bound_edges_declines_typed() -> None:
    pl = pytest.importorskip("polars")
    g = graphistry.nodes(pl.from_pandas(NODES), "id")
    with pytest.raises(NotImplementedError, match="unbound edge endpoints"):
        g.gfql(n({}), engine="polars")


# --- Scheduling and error typing (#1923 F5-F8)


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_let_reads_an_enclosing_binding_declared_after_it(engine: str) -> None:
    result = _graph(engine).gfql(
        ASTLet({
            "inner": ASTLet({"z": ASTRef("outer", [n({"type": "person"})])}),
            "outer": n({}),
        }),
        output="inner",
        engine=engine,
    )
    assert _node_ids(result) == PERSON_IDS


@pytest.mark.parametrize("engine", ENGINES)
def test_nested_let_cycle_through_an_enclosing_binding_is_a_coded_error(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(
            ASTLet({
                "inner": ASTLet({"bad": ASTRef("later", [])}),
                "later": ASTRef("inner", []),
            }),
            engine=engine,
        )
    assert exc_info.value.code == ErrorCode.E153


@pytest.mark.parametrize("engine", ENGINES)
def test_binding_schema_failure_keeps_its_gfql_error_type(engine: str) -> None:
    # strict= selects the level that still rejects an absent column; warn resolves it to null
    with pytest.raises(GFQLSchemaError) as exc_info:
        _graph(engine).gfql(
            ASTLet({"x": n({"nosuchcol": 1}), "y": n({})}), output="x", engine=engine,
            strict=True,
        )
    assert exc_info.value.code == ErrorCode.E301


@pytest.mark.parametrize("engine", ENGINES)
def test_single_binding_self_reference_reports_self_reference(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(ASTLet({"x": ASTRef("x", [])}), engine=engine)
    assert exc_info.value.code == ErrorCode.E153
    assert "Self-reference cycle detected: 'x' depends on itself" in str(exc_info.value)


@pytest.mark.parametrize("engine", ENGINES)
def test_single_binding_missing_reference_is_a_coded_error(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(ASTLet({"x": ASTRef("nope", [])}), engine=engine)
    assert exc_info.value.code == ErrorCode.E151


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_binding_cycle_is_a_coded_error(engine: str) -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        _graph(engine).gfql(
            ASTLet({"a": ASTRef("b", []), "b": ASTRef("c", []), "c": ASTRef("a", [])}),
            engine=engine,
        )
    assert exc_info.value.code == ErrorCode.E153


# --- the one ordering primitive the scheduler and the cycle report share ---------------------


def test_in_declaration_order_follows_the_declaration_index() -> None:
    order = {"z": 0, "a": 1, "m": 2}
    assert in_declaration_order({"a", "m", "z"}, order) == ["z", "a", "m"]


def test_enclosing_scope_names_trail_the_declared_ones_ordered_by_name() -> None:
    order = {"z": 0, "a": 1}
    assert in_declaration_order({"a", "z", "outer2", "outer1"}, order) == [
        "z", "a", "outer1", "outer2",
    ]


def test_without_a_declaration_order_names_fall_back_to_name_order() -> None:
    assert in_declaration_order({"m", "a", "z"}, {}) == ["a", "m", "z"]


def test_cycle_is_reported_in_declaration_order_not_name_order() -> None:
    dependencies = {"z": {"a"}, "a": {"z"}}
    assert detect_cycles(dependencies, {"z": 0, "a": 1}) == ["z", "a", "z"]
    assert detect_cycles(dependencies, {"a": 0, "z": 1}) == ["a", "z", "a"]


def test_cycle_report_is_stable_when_no_declaration_order_is_supplied() -> None:
    dependencies = {"z": {"a"}, "a": {"z"}}
    assert detect_cycles(dependencies) == detect_cycles(dependencies)
