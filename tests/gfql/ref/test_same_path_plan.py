from graphistry.compute.gfql.same_path_plan import plan_same_path
from graphistry.compute.gfql.same_path_types import col, compare


def test_plan_minmax_and_bitset():
    where = [
        compare(col("a", "balance"), ">", col("c", "credit")),
        compare(col("a", "owner"), "==", col("c", "owner")),
    ]
    plan = plan_same_path(where)
    assert plan.minmax_aliases == {"a": {"balance"}, "c": {"credit"}}
    assert any("owner" in key for key in plan.bitsets)


def test_plan_empty_when_no_where():
    plan = plan_same_path(None)
    assert plan.minmax_aliases == {}
    assert plan.bitsets == {}
