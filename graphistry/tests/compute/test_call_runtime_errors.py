from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.exceptions import ErrorCode, GFQLCallExecutionError, GFQLTypeError
from graphistry.compute.gfql.call.executor import execute_call


@pytest.fixture
def graph():
    edges = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "a"]})
    return graphistry.edges(edges, "src", "dst")


def test_handler_exception_is_call_execution_error(graph, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("engine kernel failed")

    monkeypatch.setattr(graph, "get_degrees", fail)

    with pytest.raises(GFQLCallExecutionError) as info:
        execute_call(graph, "get_degrees", {}, engine=Engine.PANDAS)

    assert info.value.code == ErrorCode.E306
    assert info.value.context["engine"] == "pandas"
    assert info.value.context["original_exception"] == "RuntimeError"
    assert isinstance(info.value.__cause__, RuntimeError)


def test_unknown_function_keeps_parameter_validation_error(graph):
    with pytest.raises(GFQLTypeError) as info:
        execute_call(graph, "not_safelisted", {}, engine=Engine.PANDAS)

    assert info.value.code == ErrorCode.E303
    assert not isinstance(info.value, GFQLCallExecutionError)
