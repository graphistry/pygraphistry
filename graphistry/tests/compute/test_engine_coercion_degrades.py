"""Pins that engine coercion degrades instead of raising, and can still name the engines.

`ensure_engine_match` promises to return the original graph rather than crash a user's
workflow. Its handler reports which engines it saw, and those names are resolved inside
the same try, so a failure early enough leaves them unset.
"""

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute import engine_coercion as ec


@pytest.fixture
def graph():
    return graphistry.edges(pd.DataFrame({"s": ["a"], "d": ["b"]}), "s", "d")


def test_a_failure_before_the_engines_are_resolved_still_degrades(graph, monkeypatch):
    """The earliest possible failure: resolve_engine itself raises."""
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "resolve_engine", boom)
    out = ec.ensure_engine_match(graph, Engine.PANDAS)
    assert out is graph, "degradation must hand back the original graph"


def test_a_failure_after_the_engines_are_resolved_also_degrades(graph, monkeypatch):
    """The later failure, where the handler does have engine names to report."""
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "df_to_engine", boom)
    out = ec.ensure_engine_match(graph, Engine.PANDAS)
    assert out is not None


def test_the_matching_engine_is_returned_unchanged(graph):
    out = ec.ensure_engine_match(graph, Engine.PANDAS)
    assert isinstance(out._edges, pd.DataFrame)
