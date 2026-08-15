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
