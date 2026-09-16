"""`gfql` validates its ops once, and every other caller still re-validates.

The execution path builds a throwaway `Chain` purely to re-validate the ops, which is
what catches a caller that mutated a Chain after constructing it. When `gfql` built the
Chain itself in the same call, nothing can have changed in between, so that pass is
redundant — for every caller, including one that passed a plain list and never reused
anything. Skipping it there must not weaken the check anywhere else, so both sides are
pinned here.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import GFQLTypeError


@pytest.fixture
def g():
    return (graphistry
            .nodes(pd.DataFrame({"id": [1, 2, 3]}), "id")
            .edges(pd.DataFrame({"s": [1, 2], "d": [2, 3]}), "s", "d"))


def _validation_passes(monkeypatch):
    """Count Chain constructions that actually validate."""
    seen = {"n": 0}
    original = Chain.__init__

    def counting(self, chain, where=None, validate=True):
        seen["n"] += bool(validate)
        return original(self, chain, where, validate)

    monkeypatch.setattr(Chain, "__init__", counting)
    return seen


def test_gfql_validates_its_ops_once(g, monkeypatch):
    """A plain list through gfql must be validated once, not twice."""
    seen = _validation_passes(monkeypatch)
    g.gfql([n(), e_forward(), n()])
    assert seen["n"] == 1, f"ops were validated {seen['n']} times"


def test_chain_called_directly_still_validates(g, monkeypatch):
    """`chain` has no such guarantee about its caller, so it keeps re-validating."""
    seen = _validation_passes(monkeypatch)
    g.chain([n(), e_forward(), n()])
    assert seen["n"] >= 1


@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_a_chain_mutated_after_construction_is_still_caught(g, entry):
    """The case the re-validation exists for. Skipping it on the gfql path must not lose it.

    A Chain validates at construction; a caller can then mutate its ops and execute it. That
    is a different caller from the one gfql serves, and it must still raise.
    """
    ops = Chain([n(), e_forward(), n()])
    ops.chain[1].hops = -5
    with pytest.raises(GFQLTypeError) as error:
        getattr(g, entry)(ops)
    assert error.value.code == "invalid-hops-value"


def test_only_a_constructor_validated_chain_can_be_marked():
    """The mark means "this constructor validated these ops"; it cannot be forged onto one
    that was built with validation off."""
    assert Chain([n()])._gfql_validated_in_call is False, "the mark is never set by construction"
    assert Chain([n()]).gfql_validated()._gfql_validated_in_call is True
    assert Chain([n()], validate=False).gfql_validated()._gfql_validated_in_call is False


@pytest.mark.parametrize("ops,code", [
    ([n({"id": object()})], "type-mismatch"),
    (["not-an-op"], "invalid-chain-type"),
    ([n(), e_forward(hops=-1), n()], "invalid-hops-value"),
])
def test_malformed_queries_still_raise_the_same_code_through_gfql(g, ops, code):
    """The pass being skipped is redundant, not load-bearing: the errors are unchanged."""
    with pytest.raises(GFQLTypeError) as error:
        g.gfql(ops)
    assert error.value.code == code
