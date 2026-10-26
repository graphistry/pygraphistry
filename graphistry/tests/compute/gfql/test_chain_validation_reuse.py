"""Validation is not skipped for any caller who could have changed their ops.

`gfql` builds the Chain and runs it in the same call, so that one re-validation is
redundant. Every other caller may have mutated since, and must still raise.
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


MALFORMED = [
    ([n({"id": object()})], "type-mismatch"),
    (["not-an-op"], "invalid-chain-type"),
    ([n(), e_forward(hops=-1), n()], "invalid-hops-value"),
]


@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_a_chain_mutated_after_construction_still_raises(g, entry):
    """The reason the re-validation exists."""
    ops = Chain([n(), e_forward(), n()])
    ops.chain[1].hops = -5
    with pytest.raises(GFQLTypeError) as error:
        getattr(g, entry)(ops)
    assert error.value.code == "invalid-hops-value"


@pytest.mark.parametrize("entry", ["chain", "gfql"])
@pytest.mark.parametrize("ops,code", MALFORMED)
def test_malformed_ops_raise_the_same_code_through_both_entry_points(g, entry, ops, code):
    with pytest.raises(GFQLTypeError) as error:
        getattr(g, entry)(ops)
    assert error.value.code == code


def test_a_chain_reused_across_calls_is_revalidated_every_time(g):
    """The case a once-only memo would break."""
    ops = Chain([n(), e_forward(), n()])
    g.gfql(ops)                      # first run is fine
    ops.chain[1].hops = -5           # the caller mutates between runs
    with pytest.raises(GFQLTypeError) as error:
        g.gfql(ops)
    assert error.value.code == "invalid-hops-value"


def test_a_list_reused_across_calls_is_revalidated_every_time(g):
    ops = [n(), e_forward(), n()]
    g.gfql(ops)
    ops[1].hops = -5
    with pytest.raises(GFQLTypeError) as error:
        g.gfql(ops)
    assert error.value.code == "invalid-hops-value"


@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_well_formed_ops_return_the_same_result_through_both_entry_points(g, entry):
    ops = [n({"id": 1}), e_forward(), n()]
    out = getattr(g, entry)(ops)
    assert sorted(out._nodes["id"].tolist()) == [1, 2]
    assert sorted(out._edges["s"].tolist()) == [1]


def test_gfql_and_chain_agree_on_the_same_ops(g):
    ops = [n({"id": 1}), e_forward(), n()]
    assert g.gfql(list(ops))._nodes.equals(g.chain(list(ops))._nodes)


def test_gfql_does_not_validate_the_same_ops_twice(g, monkeypatch):
    """The only implementation assertion here: without it the rest passes with the change reverted."""
    seen = {"validating": 0}
    original = Chain.__init__

    def counting(self, chain, where=None, validate=True):
        seen["validating"] += bool(validate)
        return original(self, chain, where, validate)

    monkeypatch.setattr(Chain, "__init__", counting)
    g.gfql([n(), e_forward(), n()])
    assert seen["validating"] == 1, f"ops were validated {seen['validating']} times"


def test_the_marker_is_a_typed_check_not_an_attribute_probe():
    assert Chain.ops_were_validated_in_this_call([n()]) is False
    assert Chain.ops_were_validated_in_this_call(Chain([n()])) is False
    assert Chain.ops_were_validated_in_this_call(Chain([n()]).gfql_validated()) is True
    assert Chain.ops_were_validated_in_this_call(
        Chain([n()], validate=False).gfql_validated()
    ) is False, "an unvalidated Chain must never be markable"
