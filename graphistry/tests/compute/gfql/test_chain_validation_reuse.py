"""Validation is not skipped for any caller who could have changed their ops.

The execution path re-validates the ops on every run, which is what catches a caller that
built a `Chain`, mutated it, and then executed it. `gfql` builds the Chain itself and runs
it in the same call, so nothing can change in between and that pass is redundant there.

These tests are written on the BEHAVIOUR boundary -- what a caller observes -- not on how
the skip is implemented. The boundary is "could these ops have changed since they were
validated?", and both sides of it are covered: a caller who could (must still raise) and a
caller who could not (must still get the right answer). Whether validation ran once or
twice is an implementation detail and is asserted nowhere below, except in the one test
that exists to prove the optimization is live at all.
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


# --- the side that COULD have changed: validation must still fire ---

@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_a_chain_mutated_after_construction_still_raises(g, entry):
    """The reason the re-validation exists. Both entry points must still catch it."""
    ops = Chain([n(), e_forward(), n()])
    ops.chain[1].hops = -5
    with pytest.raises(GFQLTypeError) as error:
        getattr(g, entry)(ops)
    assert error.value.code == "invalid-hops-value"


@pytest.mark.parametrize("entry", ["chain", "gfql"])
@pytest.mark.parametrize("ops,code", MALFORMED)
def test_malformed_ops_raise_the_same_code_through_both_entry_points(g, entry, ops, code):
    """A caller passing bad ops gets the same diagnosis however they enter."""
    with pytest.raises(GFQLTypeError) as error:
        getattr(g, entry)(ops)
    assert error.value.code == code


def test_a_chain_reused_across_calls_is_revalidated_every_time(g):
    """Reuse is the case a once-only memo would break: validate, run, mutate, run again."""
    ops = Chain([n(), e_forward(), n()])
    g.gfql(ops)                      # first run is fine
    ops.chain[1].hops = -5           # the caller mutates between runs
    with pytest.raises(GFQLTypeError) as error:
        g.gfql(ops)
    assert error.value.code == "invalid-hops-value"


def test_a_list_reused_across_calls_is_revalidated_every_time(g):
    """Same, entering with a plain list, which is the common `gfql` shape."""
    ops = [n(), e_forward(), n()]
    g.gfql(ops)
    ops[1].hops = -5
    with pytest.raises(GFQLTypeError) as error:
        g.gfql(ops)
    assert error.value.code == "invalid-hops-value"


# --- the side that COULD NOT have changed: the answer is unchanged ---

@pytest.mark.parametrize("entry", ["chain", "gfql"])
def test_well_formed_ops_return_the_same_result_through_both_entry_points(g, entry):
    """Skipping a redundant pass must not change what comes back."""
    ops = [n({"id": 1}), e_forward(), n()]
    out = getattr(g, entry)(ops)
    assert sorted(out._nodes["id"].tolist()) == [1, 2]
    assert sorted(out._edges["s"].tolist()) == [1]


def test_gfql_and_chain_agree_on_the_same_ops(g):
    """The two entry points differ in whether the pass is skipped; not in the answer."""
    ops = [n({"id": 1}), e_forward(), n()]
    assert g.gfql(list(ops))._nodes.equals(g.chain(list(ops))._nodes)


# --- the one implementation pin, which exists to prove the optimization is live ---

def test_gfql_does_not_validate_the_same_ops_twice(g, monkeypatch):
    """Without this the suite above would pass with the optimization removed entirely.

    Deliberately an implementation assertion, and the only one here: it is the engagement
    pin, not a behaviour test.
    """
    seen = {"validating": 0}
    original = Chain.__init__

    def counting(self, chain, where=None, validate=True):
        seen["validating"] += bool(validate)
        return original(self, chain, where, validate)

    monkeypatch.setattr(Chain, "__init__", counting)
    g.gfql([n(), e_forward(), n()])
    assert seen["validating"] == 1, f"ops were validated {seen['validating']} times"


def test_the_marker_is_a_typed_check_not_an_attribute_probe():
    """`chain` takes a list or a Chain; only a Chain this call built can carry the mark."""
    assert Chain.ops_were_validated_in_this_call([n()]) is False
    assert Chain.ops_were_validated_in_this_call(Chain([n()])) is False
    assert Chain.ops_were_validated_in_this_call(Chain([n()]).gfql_validated()) is True
    assert Chain.ops_were_validated_in_this_call(
        Chain([n()], validate=False).gfql_validated()
    ) is False, "an unvalidated Chain must never be markable"
