"""Direct tests for the ungrouped-aggregate identity-row synthesis (#1909).

``test_aggregate_identity_row_semantics.py`` pins the observable Cypher
semantics end-to-end; this file pins every admit/decline branch of the
synthesis itself, in the style of ``test_flatten_pure_carry_optional.py``:
the helpers are pure (compiled row steps -> dict), so the inputs are built
directly with the ``graphistry.compute.ast`` call constructors and the result
is asserted structurally, no engine execution involved.

The compiled shapes below mirror what the lowering actually emits, e.g.

    rows(table='nodes') | with_([('__cypher_group__', 1)])
      | group_by(['__cypher_group__'], [('c', 'count')]) | select([('c', 'c')])

so a decline pinned here is a decline the real compiler can hit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from graphistry.compute.ast import (
    ASTCall,
    count_table,
    distinct,
    drop_cols,
    group_by,
    limit,
    n,
    order_by,
    select,
    skip,
    where_rows,
    with_,
)
from graphistry.compute.gfql.cypher.aggregate_identity import (
    _identity_row_passthrough_projection,
    _identity_row_scalar,
    _identity_row_without_temps,
    _replay_identity_row,
    _ungrouped_aggregate_identity_seed,
    aggregate_identity_value,
    identity_row_after_paging,
    ungrouped_aggregate_identity_row,
)

GROUP_KEY = "__cypher_group__"


def _group_by(*aggregations: Any, keys: Any = (GROUP_KEY,), **kwargs: Any) -> ASTCall:
    """The constant-key group_by the lowering emits for an ungrouped aggregate."""
    return group_by(list(keys), list(aggregations), **kwargs)


# ===========================================================================
# 1. Identity values per aggregate function
# ===========================================================================


@pytest.mark.parametrize("func,expected", [
    ("count", 0),
    ("count_distinct", 0),
    ("sum", 0),
    ("collect", []),
    ("collect_distinct", []),
    ("min", None),
    ("max", None),
    ("avg", None),
    # anything the runtime lowers but this table does not name falls to null,
    # which is the openCypher default for a value-less aggregate
    ("first", None),
], ids=["count", "count_distinct", "sum", "collect", "collect_distinct",
        "min", "max", "avg", "unknown"])
def test_aggregate_identity_value_per_function(func, expected) -> None:
    assert aggregate_identity_value(func) == expected


def test_collect_identity_is_a_fresh_list() -> None:
    """The identity row is mutated downstream (projection/replay), so collect
    must not hand out a shared list."""
    first = aggregate_identity_value("collect")
    first.append(1)
    assert aggregate_identity_value("collect") == []


# ===========================================================================
# 2. Terminal SKIP/LIMIT paging of the synthesized row
# ===========================================================================


@pytest.mark.parametrize("row,skip_value,limit_value,expected", [
    (None, None, None, None),
    (None, 0, 5, None),
    ({"c": 0}, None, None, {"c": 0}),
    ({"c": 0}, 0, None, {"c": 0}),
    ({"c": 0}, 1, None, None),
    ({"c": 0}, 99, None, None),
    ({"c": 0}, None, 1, {"c": 0}),
    ({"c": 0}, None, 0, None),
    ({"c": 0}, None, -1, None),
    ({"c": 0}, 0, 5, {"c": 0}),
    ({"c": 0}, 1, 5, None),
], ids=["no_row", "no_row_paged", "unpaged", "skip_0", "skip_1", "skip_99",
        "limit_1", "limit_0", "limit_negative", "skip_0_limit_5", "skip_1_limit_5"])
def test_identity_row_after_paging(row, skip_value, limit_value, expected) -> None:
    assert identity_row_after_paging(
        row, skip_value=skip_value, limit_value=limit_value
    ) == expected


# ===========================================================================
# 3. Seeding: which compiled step is an ungrouped aggregate producer
# ===========================================================================


def test_seed_declines_non_call_step() -> None:
    assert _ungrouped_aggregate_identity_seed(n()) is None


def test_seed_declines_unrelated_call() -> None:
    assert _ungrouped_aggregate_identity_seed(select([("c", "c")])) is None


def test_seed_from_count_table_fast_path() -> None:
    """The count(*) fast path replaces group_by with count_table; its identity
    is the same zero."""
    assert _ungrouped_aggregate_identity_seed(count_table(alias="c")) == {"c": 0}


def test_seed_declines_count_table_without_string_alias() -> None:
    assert _ungrouped_aggregate_identity_seed(ASTCall("count_table", {})) is None
    assert _ungrouped_aggregate_identity_seed(
        ASTCall("count_table", {"alias": 7})
    ) is None


def test_seed_covers_every_aggregate_identity_in_one_group_by() -> None:
    seed = _ungrouped_aggregate_identity_seed(
        _group_by(("c", "count"), ("s", "sum"), ("col", "collect"),
                  ("mn", "min"), ("mx", "max"), ("av", "avg"))
    )
    assert seed == {GROUP_KEY: 1, "c": 0, "s": 0, "col": [],
                    "mn": None, "mx": None, "av": None}


def test_seed_keeps_extra_aggregation_tuple_fields() -> None:
    """Aggregations carry a source column as a third field; only the first two
    decide the identity."""
    assert _ungrouped_aggregate_identity_seed(
        _group_by(("s", "sum", "age"))
    ) == {GROUP_KEY: 1, "s": 0}


def test_seed_declines_group_by_with_key_prefixes() -> None:
    """key_prefixes adds grouping columns only known at runtime, so the step is
    not provably ungrouped."""
    assert _ungrouped_aggregate_identity_seed(
        _group_by(("c", "count"), key_prefixes=["tag."])
    ) is None


@pytest.mark.parametrize("keys", [
    ["city"],
    [GROUP_KEY, "city"],
    [],
    [1],
], ids=["real_grouping_key", "constant_plus_real_key", "no_keys", "non_string_key"])
def test_seed_declines_anything_but_the_lone_constant_key(keys) -> None:
    """A GROUPED aggregate yields one row per group, and an empty stream has no
    groups -- no identity row (pinned end-to-end as `grouped_stays_empty`)."""
    assert _ungrouped_aggregate_identity_seed(
        _group_by(("c", "count"), keys=keys)
    ) is None


@pytest.mark.parametrize("aggregations", [None, [], "count"],
                         ids=["missing", "empty", "not_a_sequence"])
def test_seed_declines_group_by_without_aggregations(aggregations) -> None:
    assert _ungrouped_aggregate_identity_seed(
        ASTCall("group_by", {"keys": [GROUP_KEY], "aggregations": aggregations})
    ) is None


@pytest.mark.parametrize("aggregation", [("c",), "count", ()],
                         ids=["short_tuple", "bare_string", "empty_tuple"])
def test_seed_declines_malformed_aggregation_entry(aggregation) -> None:
    assert _ungrouped_aggregate_identity_seed(
        ASTCall("group_by", {"keys": [GROUP_KEY], "aggregations": [aggregation]})
    ) is None


@pytest.mark.parametrize("aggregation", [(0, "count"), ("c", 0)],
                         ids=["non_string_output", "non_string_func"])
def test_seed_declines_non_string_aggregation_names(aggregation) -> None:
    assert _ungrouped_aggregate_identity_seed(
        ASTCall("group_by", {"keys": [GROUP_KEY], "aggregations": [aggregation]})
    ) is None


# ===========================================================================
# 4. Symbolic suffix: pass-through projection
# ===========================================================================


def test_projection_renames_and_drops_unprojected_columns() -> None:
    assert _identity_row_passthrough_projection(
        {GROUP_KEY: 1, "c": 0, "s": 0}, {"items": [("cnt", "c")]}
    ) == {"cnt": 0}


def test_projection_with_extend_keeps_the_prior_row() -> None:
    assert _identity_row_passthrough_projection(
        {GROUP_KEY: 1, "c": 0}, {"items": [("cnt", "c")], "extend": True}
    ) == {GROUP_KEY: 1, "c": 0, "cnt": 0}


def test_projection_declines_without_items() -> None:
    assert _identity_row_passthrough_projection({"c": 0}, {}) is None
    assert _identity_row_passthrough_projection({"c": 0}, {"items": "c"}) is None


@pytest.mark.parametrize("entry", [("c",), ("c1", "c", "extra"), "c"],
                         ids=["short", "long", "bare_string"])
def test_projection_declines_malformed_item(entry) -> None:
    assert _identity_row_passthrough_projection({"c": 0}, {"items": [entry]}) is None


@pytest.mark.parametrize("entry,label", [
    (("c1", "(c + 1)"), "expression"),
    (("c1", "nosuch"), "unknown_column"),
    (("c1", 1), "literal"),
    ((0, "c"), "non_string_output"),
])
def test_projection_declines_anything_but_a_column_carry(entry, label) -> None:
    """Only a rename of a column already in the row is symbolic; an expression
    (or a literal) has to be replayed instead."""
    assert _identity_row_passthrough_projection({"c": 0}, {"items": [entry]}) is None


# ===========================================================================
# 5. Replayed-row scalar normalization
# ===========================================================================


class _RaisingScalar:
    """A 0-d value whose .item() blows up -- the defensive arm of the unwrap."""

    ndim = 0

    def __init__(self, error: Exception) -> None:
        self._error = error

    def item(self) -> Any:
        raise self._error


@pytest.mark.parametrize("value,expected", [
    (None, None),
    (float("nan"), None),
    ("text", "text"),
    (b"bytes", b"bytes"),
    ([1, 2], [1, 2]),
    ((1, 2), (1, 2)),
    ({"a": 1}, {"a": 1}),
    (0, 0),
    (2.5, 2.5),
    (True, True),
], ids=["none", "nan", "str", "bytes", "list", "tuple", "dict", "int", "float", "bool"])
def test_identity_row_scalar_passes_through_plain_values(value, expected) -> None:
    assert _identity_row_scalar(value) == expected


def test_identity_row_scalar_unwraps_numpy_scalars() -> None:
    unwrapped = _identity_row_scalar(np.int64(3))
    assert unwrapped == 3 and type(unwrapped) is int
    assert _identity_row_scalar(np.float64(2.5)) == 2.5


def test_identity_row_scalar_unwraps_numpy_nan_to_none() -> None:
    """A replayed min/max lands as a numpy nan; the identity must be null."""
    assert _identity_row_scalar(np.float64("nan")) is None


def test_identity_row_scalar_leaves_ndarrays_alone() -> None:
    """collect() replays as an ndarray (ndim 1), which is a value, not a scalar."""
    array = np.array([1, 2])
    assert _identity_row_scalar(array) is array


@pytest.mark.parametrize("error", [ValueError("no"), AttributeError("no")],
                         ids=["value_error", "attribute_error"])
def test_identity_row_scalar_keeps_the_value_when_unwrapping_raises(error) -> None:
    value = _RaisingScalar(error)
    assert _identity_row_scalar(value) is value


# ===========================================================================
# 6. Replay fallback through the real row pipeline
# ===========================================================================

SEED = {GROUP_KEY: 1, "c": 0}


def test_replay_evaluates_a_post_aggregate_expression() -> None:
    """count -> 0, so `c + 1` is 1 (the identity row is a real row)."""
    assert _replay_identity_row(SEED, [select([("c1", "(c + 1)")])]) == {"c1": 1}


def test_replay_runs_the_whole_remaining_suffix() -> None:
    assert _replay_identity_row(
        SEED, [select([("c1", "(c + 1)")]), select([("c2", "(c1 * 2)")])]
    ) == {"c2": 2}


def test_replay_normalizes_scalars_and_stringifies_keys() -> None:
    replayed = _replay_identity_row(SEED, [drop_cols([GROUP_KEY])])
    assert replayed == {"c": 0}
    assert all(isinstance(key, str) for key in replayed)
    assert type(replayed["c"]) is int


@pytest.mark.parametrize("step", [
    where_rows(expr="(c = 0)"),
    ASTCall("unwind", {"expr": "[]", "as_": "x"}),
    ASTCall("hop", {}),
], ids=["filter", "unwind", "hop"])
def test_replay_declines_steps_outside_the_allowlist(step) -> None:
    """Whether the identity row survives a filter/unwind/join depends on data the
    compiler cannot see, so replay refuses rather than guessing."""
    assert _replay_identity_row(SEED, [step]) is None


def test_replay_declines_a_non_call_step() -> None:
    assert _replay_identity_row(SEED, [n()]) is None


@pytest.mark.parametrize("step", [
    select([("c1", "(nosuch + 1)")]),
    select([("c1", "(c +")]),
    ASTCall("limit", {"value": "three"}),
], ids=["unknown_column", "unparseable", "bad_limit_value"])
def test_replay_declines_when_the_pipeline_raises(step) -> None:
    """The replay is a compile-time probe: a raising step means "no identity
    row", never a failed query."""
    assert _replay_identity_row(SEED, [step]) is None


@pytest.mark.parametrize("step", [limit(0), skip(1), skip(99)],
                         ids=["limit_0", "skip_1", "skip_99"])
def test_replay_declines_when_the_pipeline_empties_the_row(step) -> None:
    assert _replay_identity_row(SEED, [step]) is None


# ===========================================================================
# 7. Temp-column stripping
# ===========================================================================


@pytest.mark.parametrize("temp", ["__cypher_group__", "__cypher_agg__0",
                                  "__cypher_postagg__0"])
def test_identity_row_without_temps_strips_compiler_temps(temp) -> None:
    assert _identity_row_without_temps({temp: 1, "c": 0}) == {"c": 0}


def test_identity_row_without_temps_on_an_all_temp_row_is_none() -> None:
    """An all-temp row is no row at all -- returning {} would emit a column-less
    result instead of the empty frame."""
    assert _identity_row_without_temps({GROUP_KEY: 1}) is None
    assert _identity_row_without_temps({}) is None
    assert _identity_row_without_temps(None) is None


# ===========================================================================
# 8. End to end over compiled row steps
# ===========================================================================

ROWS = ASTCall("rows", {"table": "nodes", "source": "m"})
SEED_STEPS = [ROWS, with_([(GROUP_KEY, 1)]), _group_by(("c", "count"))]


def test_declines_row_steps_without_an_aggregate() -> None:
    assert ungrouped_aggregate_identity_row([ROWS, select([("n", "m.name")])]) is None
    assert ungrouped_aggregate_identity_row([]) is None


def test_synthesizes_from_the_group_by_and_strips_the_group_temp() -> None:
    assert ungrouped_aggregate_identity_row(SEED_STEPS) == {"c": 0}


def test_synthesizes_from_the_count_table_fast_path() -> None:
    assert ungrouped_aggregate_identity_row([count_table(alias="c")]) == {"c": 0}


def test_the_last_aggregate_producer_wins() -> None:
    """A stage that aggregates an aggregate: the identity belongs to the FINAL
    ungrouped aggregate, not the first one."""
    assert ungrouped_aggregate_identity_row([
        ROWS,
        with_([(GROUP_KEY, 1)]), _group_by(("c", "count")),
        with_([(GROUP_KEY, 1)]), _group_by(("total", "sum"), ("names", "collect")),
    ]) == {"total": 0, "names": []}


def test_order_by_and_distinct_are_transparent() -> None:
    """Sorting or deduplicating one row leaves that one row."""
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [order_by([("c", "DESC")]), distinct(), select([("c", "c")])]
    ) == {"c": 0}


def test_trailing_projection_chain_is_applied_symbolically() -> None:
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [select([("c", "c")]), select([("cnt", "c")])]
    ) == {"cnt": 0}


@pytest.mark.parametrize("suffix,expected", [
    ([limit(1)], {"c": 0}),
    ([limit(0)], None),
    ([skip(0)], {"c": 0}),
    ([skip(1)], None),
    ([skip(0), limit(5)], {"c": 0}),
    ([skip(0), limit(0)], None),
    ([order_by([("c", "ASC")]), skip(2)], None),
], ids=["limit_1", "limit_0", "skip_0", "skip_1", "skip_0_limit_5",
        "skip_0_limit_0", "order_by_skip_2"])
def test_paging_suffix_pages_the_identity_row(suffix, expected) -> None:
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [select([("c", "c")])] + suffix
    ) == expected


@pytest.mark.parametrize("value", [True, False, "3", 3.0, None],
                         ids=["true", "false", "string", "float", "none"])
def test_declines_a_non_integer_paging_value(value) -> None:
    """bool is an int subclass; a LIMIT that is not a plain integer is not a
    paging decision this can make at compile time."""
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [ASTCall("limit", {"value": value})]
    ) is None
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [ASTCall("skip", {"value": value})]
    ) is None


def test_declines_a_non_call_suffix_step() -> None:
    assert ungrouped_aggregate_identity_row(SEED_STEPS + [n()]) is None


def test_declines_a_post_aggregate_filter() -> None:
    """#1909 residual: whether the identity row survives a post-aggregate WHERE
    depends on the real aggregate value, so the synthesis declines. Pinned as a
    strict xfail end-to-end in test_aggregate_identity_row_semantics.py."""
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [select([("c", "c")]), where_rows(expr="(c = 0)")]
    ) is None


def test_falls_back_to_replay_for_a_post_aggregate_expression() -> None:
    """The suffix stops being symbolic at `c + 1`, so the remaining steps are
    replayed through the real row pipeline on the one-row seed."""
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [with_([("c", "c")]), select([("c1", "(c + 1)")])]
    ) == {"c1": 1}


def test_replay_fallback_still_pages_and_strips_temps() -> None:
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [select([("c1", "(c + 1)")]), limit(0)]
    ) is None
    assert ungrouped_aggregate_identity_row(
        SEED_STEPS + [select([("c1", "(c + 1)")]), limit(1)]
    ) == {"c1": 1}
