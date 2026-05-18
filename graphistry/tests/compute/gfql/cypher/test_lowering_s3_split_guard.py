from __future__ import annotations

import inspect

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher import lowering, projection_planning
from graphistry.compute.gfql.cypher.reentry import compiletime
from graphistry.compute.gfql.cypher.reentry import lowering_support


def test_issue_1514_projection_split_owned_by_projection_planning() -> None:
    assert projection_planning._split_qualified_name("person.name", line=1, column=1) == ("person", "name")
    assert projection_planning._split_qualified_name("person", line=1, column=1) == ("person", None)


def test_issue_1514_projection_split_rejects_nested_property_path() -> None:
    with pytest.raises(GFQLValidationError):
        projection_planning._split_qualified_name("person.address.city", line=1, column=1)


def test_issue_1514_projection_ref_owned_by_projection_planning() -> None:
    alias_obj = object()
    alias_targets = {"a": alias_obj}

    out = projection_planning._projection_ref_from_expr(
        "a.b",
        alias_targets=alias_targets,
        params={"p": 1},
        field="return.item",
        line=7,
        column=11,
    )

    assert out == ("a", "b")


@pytest.mark.parametrize(
    "name",
    [
        "_split_qualified_name",
        "_qualified_ref_from_node",
        "_projection_ref_from_expr",
        "_reject_duplicate_alias_row_refs",
        "_build_projection_plan",
        "_can_lower_multi_alias_projection_bindings",
        "_result_projection_plan",
        "_empty_optional_projection_row",
        "_optional_null_fill_plan",
        "_optional_projection_row_guard_plan",
        "_plan_with_visible_projected_columns",
        "_projection_output_names",
    ],
)
def test_issue_1514_projection_planning_delegates_retired_from_lowering(name: str) -> None:
    assert not hasattr(lowering, name)


def test_issue_1471_reentry_compiletime_has_no_lowering_symbol_table_shim() -> None:
    source = inspect.getsource(compiletime)

    assert "globals().update(vars(" not in source
    assert not hasattr(compiletime, "_lowering")
    assert compiletime.__name__ == "graphistry.compute.gfql.cypher.reentry.compiletime"
    assert compiletime.__package__ == "graphistry.compute.gfql.cypher.reentry"


def test_issue_1471_reentry_compiletime_keeps_lowering_entrypoints() -> None:
    assert callable(getattr(compiletime, "_compile_bounded_reentry_query", None))
    assert callable(getattr(lowering_support, "_drop_bare_alias_items_from_stage", None))
