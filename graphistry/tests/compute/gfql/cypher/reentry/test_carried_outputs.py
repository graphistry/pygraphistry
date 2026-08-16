"""Which optional-reentry outputs the unmatched-row null-fill can reproduce.

Unit pins for the source-column resolution behind
``carried_output_sources``: every branch that decides whether an unmatched
prefix row can be given its own identity, or whether the fill must decline.
"""
from __future__ import annotations

from typing import Optional

import pytest

from graphistry.compute.gfql.cypher.reentry.carried_outputs import (
    carried_output_source_column,
)


ALIASES = {"p"}
SCALARS = {"av"}


@pytest.mark.parametrize(
    "src,expected",
    [
        pytest.param("av", "av", id="carried_with_scalar_projected_as_is"),
        pytest.param("p.__cypher_reentry_av__", "av", id="carried_scalar_read_through_marker"),
        pytest.param("p.id", "p.id", id="bare_property_of_carried_alias"),
        pytest.param(
            "p.__cypher_reentry_gone__",
            "p.__cypher_reentry_gone__",
            id="marker_for_uncarried_scalar_stays_a_bare_property",
        ),
        pytest.param("p", None, id="whole_row_carry_is_not_reproducible"),
        pytest.param("p.id + 1", None, id="expression_over_carried_alias_is_not_reproducible"),
        pytest.param("q.id", None, id="property_of_another_alias_is_not_reproducible"),
    ],
)
def test_carried_output_source_column(src: str, expected: Optional[str]) -> None:
    assert (
        carried_output_source_column(src, alias_names=ALIASES, scalar_columns=SCALARS)
        == expected
    )


# ---------------------------------------------------------------------------
# carried_output_sources: the aggregating decision over a whole projection.
#
# The unit pins above cover ``carried_output_source_column`` one source string
# at a time. The branches BELOW -- which projection stage is read, whether a
# following group_by cancels it, and what happens when one output of many is
# irreproducible -- are only reachable through the aggregating function, and a
# mutation audit found every one of them survived both this file and the
# end-to-end #1896 pins. Two of them (the NOT_REPRODUCIBLE decline and its
# consumer in apply_optional_reentry_null_fill) have no Cypher spelling that
# reaches them today: they are defensive, and pinned here so they stay honest.
# ---------------------------------------------------------------------------

import dataclasses

from graphistry.compute.ast import ASTCall
from graphistry.compute.chain import Chain
from graphistry.compute.gfql.cypher.lowering import compile_cypher_query
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.reentry.carried_outputs import (
    CARRIED_OUTPUTS_NOT_REPRODUCIBLE,
    CarriedOutputSources,
    carried_output_sources,
)


#: A whole-row carry (alias ``a``) that ALSO carries the scalar column ``av``,
#: so both the alias branch and the scalar branch are in scope for one query.
_SCALAR_CARRY = ("MATCH (a:A) WITH a, a.v AS av OPTIONAL MATCH (a)-[:R]->(b) "
                 "RETURN av AS n, b.id AS bid")


def _with_ops(*ops):
    """The compiled scalar-carry query with its op chain swapped for ``ops``."""
    compiled = compile_cypher_query(parse_cypher(_SCALAR_CARRY))
    assert compiled.reentry_plan is not None, "shape no longer takes the reentry route"
    return dataclasses.replace(compiled, chain=Chain(list(ops)))


def _select(*items):
    return ASTCall("select", {"items": [list(i) for i in items]})


def test_reproducible_outputs_map_to_their_prefix_columns():
    out = carried_output_sources(_with_ops(_select(("n", "a.__cypher_reentry_av__"),
                                                   ("pid", "a.id"))))
    assert out.every_output_reproducible
    assert dict(out.columns) == {"n": "av", "pid": "a.id"}


def test_outputs_that_do_not_read_the_carried_alias_are_left_out():
    """`b` is bound by the OPTIONAL arm, not carried through the WITH, and `lit`
    names nothing at all. Neither can identify a prefix row, so neither may
    enter the anti-join key -- keying on `bid` would match a null-extended row
    against a matched one."""
    out = carried_output_sources(_with_ops(_select(("pid", "a.id"), ("bid", "b.id"),
                                                   ("k", "lit"))))
    assert out.every_output_reproducible
    assert dict(out.columns) == {"pid": "a.id"}


def test_a_bare_carried_scalar_column_is_its_own_source():
    """The scalar branch: a projection may read `av` directly rather than
    through the hidden `a.__cypher_reentry_av__` marker."""
    out = carried_output_sources(_with_ops(_select(("n", "av"))))
    assert dict(out.columns) == {"n": "av"}


def test_one_irreproducible_output_declines_the_whole_projection():
    """`a` alone is a whole-row read with no single prefix column behind it. A
    null-extended row could not be given its own value for `pp`, so the answer
    is the typed decline -- NOT 'skip that column and fill the rest', which
    would emit rows carrying another row's identity."""
    out = carried_output_sources(_with_ops(_select(("pid", "a.id"), ("pp", "a"))))
    assert out == CARRIED_OUTPUTS_NOT_REPRODUCIBLE
    assert not out.every_output_reproducible


def test_a_group_by_after_the_projection_cancels_the_column_mapping():
    """Grouping collapses rows, so the projected outputs no longer identify
    prefix rows one-for-one; the aggregate fill path owns that case instead."""
    out = carried_output_sources(_with_ops(_select(("pid", "a.id")),
                                           ASTCall("group_by", {"keys": ["pid"], "aggregations": []})))
    assert out.every_output_reproducible
    assert dict(out.columns) == {}


def test_a_projection_after_a_group_by_is_read_again():
    """The grouping flag is per-stage, not sticky: a projection that FOLLOWS a
    group_by is the one that shapes the result, so its columns count."""
    out = carried_output_sources(_with_ops(ASTCall("group_by", {"keys": ["pid"], "aggregations": []}),
                                           _select(("pid", "a.id"))))
    assert dict(out.columns) == {"pid": "a.id"}


def test_a_with_stage_counts_as_the_projection():
    """`with_` is a projection stage like `select`/`return_`; ignoring it would
    read an earlier stage's items, or none at all."""
    out = carried_output_sources(_with_ops(_select(("stale", "a.name")),
                                           ASTCall("with_", {"items": [["pid", "a.id"]]})))
    assert dict(out.columns) == {"pid": "a.id"}


def test_no_projection_stage_at_all_is_trivially_reproducible_not_a_decline():
    """Nothing reads the carried alias, so nothing has to be reproduced. This
    must NOT collapse into the decline: the row-count fill path still serves
    shapes with no lowered projection."""
    out = carried_output_sources(_with_ops(ASTCall("rows", {"table": "nodes"})))
    assert out.every_output_reproducible
    assert dict(out.columns) == {}


# ---------------------------------------------------------------------------
# The consumer: what the null-fill does with each CarriedOutputSources verdict.
# ---------------------------------------------------------------------------


def _fill(prefix_ids, result_rows, carried_outputs):
    import pandas as pd

    import graphistry
    from graphistry.compute.gfql.cypher.reentry.execution import (
        apply_optional_reentry_null_fill,
    )

    prefix = graphistry.nodes(pd.DataFrame({"p.id": list(prefix_ids)}), "p.id")
    result = graphistry.nodes(pd.DataFrame(result_rows), "bid") if result_rows else \
        graphistry.nodes(pd.DataFrame({"bid": pd.Series([], dtype=object)}), "bid")
    return apply_optional_reentry_null_fill(
        result, prefix_result=prefix, engine="pandas",
        empty_result_row={"bid": None}, carried_outputs=carried_outputs,
    )


def test_null_fill_declines_when_an_output_is_not_reproducible():
    """The CARRIED_OUTPUTS_NOT_REPRODUCIBLE verdict must reach a typed decline
    naming the projection as the reason -- never a silent short row set."""
    with pytest.raises(GFQLValidationError) as err:
        _fill(["a1", "a2"], [{"bid": "b1"}], CARRIED_OUTPUTS_NOT_REPRODUCIBLE)
    assert "the null-extension cannot reproduce" in str(err.value), str(err.value)


def test_null_fill_declines_ambiguously_when_nothing_identifies_the_rows():
    """A reproducible-but-empty mapping is a DIFFERENT failure from the one
    above: every output is fine, there is simply no carried column projected to
    anti-join on. The two must not share a message."""
    reproducible_but_empty = CarriedOutputSources(columns={}, every_output_reproducible=True)
    with pytest.raises(GFQLValidationError) as err:
        _fill(["a1", "a2"], [{"bid": "b1"}], reproducible_but_empty)
    msg = str(err.value)
    assert "no uniquely-identifying carried-alias columns" in msg, msg
    assert "the null-extension cannot reproduce" not in msg, msg


def test_null_fill_with_no_result_rows_null_extends_every_prefix_row():
    """Zero results needs no anti-join: each of the two carried rows gets its
    own null-extended row."""
    reproducible_but_empty = CarriedOutputSources(columns={}, every_output_reproducible=True)
    out = _fill(["a1", "a2"], [], reproducible_but_empty)
    assert len(out._nodes) == 2
    assert out._nodes["bid"].isna().all()


def test_only_the_last_grouping_stages_aggregate_fills_are_returned():
    """A superseded grouping stage must not leak its fills. `c1` belongs to a
    grouping the second group_by replaced; filling it would put an empty-group
    value in a column the result no longer has."""
    from graphistry.compute.gfql.cypher.reentry.carried_outputs import (
        optional_reentry_aggregate_fill_values,
    )

    compiled = _with_ops(
        ASTCall("with_", {"items": [["n", "a.__cypher_reentry_av__"]]}),
        ASTCall("group_by", {"keys": ["n"], "aggregations": [["c1", "count"]]}),
        ASTCall("group_by", {"keys": ["n"], "aggregations": [["c2", "count"]]}),
    )
    assert optional_reentry_aggregate_fill_values(compiled) == {"c2": 1}


# ---------------------------------------------------------------------------
# The identifier form these decisions are built on (graphistry/compute/gfql/
# identifiers.py). #1897 hoisted it out of three ad-hoc `re.fullmatch` copies;
# nothing pinned it directly, so a regex that admitted a digit-leading token
# survived every test in the repo.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    pytest.param("a", True, id="single_letter"),
    pytest.param("_priv", True, id="leading_underscore"),
    pytest.param("a1", True, id="trailing_digit"),
    pytest.param("__cypher_reentry_av__", True, id="hidden_carry_column"),
    pytest.param("1abc", False, id="leading_digit_is_not_an_identifier"),
    pytest.param("a.b", False, id="dotted_path"),
    pytest.param("a b", False, id="embedded_space"),
    pytest.param("a+1", False, id="operator"),
    pytest.param("count(a)", False, id="call"),
    pytest.param("", False, id="empty"),
])
def test_is_bare_identifier(text: str, expected: bool) -> None:
    from graphistry.compute.gfql.identifiers import is_bare_identifier

    assert is_bare_identifier(text) is expected


def test_identifier_tokens_finds_every_name_and_no_numbers():
    """Used to decide whether an expression READS a carried alias, so a literal
    must never contribute a token that could look like an alias."""
    from graphistry.compute.gfql.identifiers import identifier_tokens

    assert identifier_tokens("p.id + 1") == {"p", "id"}
    assert identifier_tokens("count(b.w)") == {"count", "b", "w"}
    assert identifier_tokens("42") == set()
