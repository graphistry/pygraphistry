"""Execution must validate current ASTs, even after construction or prior execution."""

import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import ASTNode, e_forward, n
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


@pytest.fixture(scope="module", params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    from graphistry.tests.compute.gfql.routes.test_route_harness import _skip_unavailable
    _skip_unavailable(request.param)
    if request.param == "polars-gpu":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("GPU validation lane runs with TEST_CUDF=1")
        import polars as pl
        # An explicitly requested GPU run must fail if the backend is broken.
        pl.DataFrame({"a": [1]}).lazy().collect(engine=pl.GPUEngine(raise_on_fail=True))
    return request.param


def graph(engine):
    nodes = pd.DataFrame({"id": [0, 1], "value": [10, 20]})
    edges = pd.DataFrame({"s": [0], "d": [1]})
    if engine in ("polars", "polars-gpu"):
        import polars as pl
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        import cudf
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("entry", ["chain", "gfql"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_execution_revalidates_mutated_ast(engine, entry, wrapped):
    g = graph(engine)
    edge = e_forward(hops=1)
    ops = [n({"id": 0}), edge, n()]
    query = Chain(ops) if wrapped else ops
    execute = getattr(g, entry)
    assert len(execute(query, engine=engine)._edges) == 1
    # Cross the valid one-hop boundary using the same graph, AST and container.
    for invalid_hops in (0, -1, "1"):
        edge.hops = invalid_hops
        with pytest.raises(GFQLTypeError) as caught:
            execute(query, engine=engine)
        assert caught.value.code == ErrorCode.E103
        assert caught.value.context["field"] == "hops"
    edge.hops = 1
    assert len(execute(query, engine=engine)._edges) == 1


@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_deferred_chain_checks_structure_before_schema(engine, entry):
    g = graph(engine)
    # The first operation has a missing property, but structural validation of
    # the later edge must fail before schema lookup/execution starts.
    query = Chain([n({"missing": 3}), e_forward(hops=0)], validate=False)
    with pytest.raises(GFQLTypeError) as caught:
        getattr(g, entry)(query, engine=engine)
    assert caught.value.code == ErrorCode.E103
    assert caught.value.context["field"] == "hops"


@pytest.mark.parametrize("entry", ["chain", "gfql"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_mutated_operation_list_is_not_trusted(engine, entry, wrapped):
    g = graph(engine)
    ops = [n()]
    query = Chain(ops) if wrapped else ops
    assert len(getattr(g, entry)(query, engine=engine)._nodes) == 2
    ops.append("invalid operation")
    with pytest.raises(GFQLTypeError) as caught:
        getattr(g, entry)(query, engine=engine)
    assert caught.value.code == ErrorCode.E101
    assert caught.value.context["operation_index"] == 1


class RejectingNode(ASTNode):
    """A supported AST subclass can impose an additional validation rule."""

    def _validate_fields(self):
        super()._validate_fields()
        raise GFQLTypeError(ErrorCode.E103, "custom node rejected", field="custom")


@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_subclass_validator_cannot_be_bypassed(engine, entry):
    query = Chain([RejectingNode()], validate=False)
    with pytest.raises(GFQLTypeError, match="custom node rejected"):
        getattr(graph(engine), entry)(query, engine=engine)


def test_schema_opt_out_does_not_bypass_list_ast_validation(engine):
    # validate_schema=False is not permission to accept malformed list input.
    with pytest.raises(GFQLTypeError) as caught:
        graph(engine).chain([e_forward(hops=0)], engine=engine, validate_schema=False)
    assert caught.value.code == ErrorCode.E103


def test_schema_opt_out_accepts_valid_chain(engine):
    g = graph(engine)
    query = Chain([n({"id": 0})])
    assert len(g.chain(query, engine=engine, validate_schema=False)._nodes) == 1
