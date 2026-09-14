"""The semi-join key frame is deliberately NOT deduplicated.

``_semi`` skips the ``.unique()`` an inner join would need, on the argument that a
semi-join emits a left row iff at least one matching right row exists, so duplicate
keys cannot change the result. That argument previously lived only in a docstring;
these pin it, so the missing dedup is a verified property rather than a claim.
Its cost side is a benchmark, and belongs in pyg-bench.
"""
import pytest

pl = pytest.importorskip("polars")  # gfql-core lane has no polars; collection must not fail

from graphistry.compute.gfql.lazy.engine.polars.chain import _semi  # noqa: E402


def _left() -> pl.DataFrame:
    return pl.DataFrame({"src": [1, 2, 3, 4, 2], "payload": ["a", "b", "c", "d", "e"]})


@pytest.mark.parametrize("lazy", [False, True])
def test_duplicate_keys_do_not_change_the_result(lazy: bool) -> None:
    """The whole justification for skipping the dedup: N copies of a key select
    exactly the rows one copy selects -- never more, never duplicated."""
    left = _left()
    duped = pl.DataFrame({"id": [2, 2, 2, 3, 3]})
    once = pl.DataFrame({"id": [2, 3]})
    if lazy:
        got = _semi(left.lazy(), duped.lazy(), "src", "id").collect()
        want = _semi(left.lazy(), once.lazy(), "src", "id").collect()
    else:
        got = _semi(left, duped, "src", "id")
        want = _semi(left, once, "src", "id")
    assert got.to_dicts() == want.to_dicts()
    # and it selected real rows, so this cannot pass by both sides being empty
    assert got.to_dicts() == [{"src": 2, "payload": "b"},
                              {"src": 3, "payload": "c"},
                              {"src": 2, "payload": "e"}]


@pytest.mark.parametrize("lazy", [False, True])
def test_semi_join_never_multiplies_left_rows(lazy: bool) -> None:
    """The other half: unlike an inner join a semi-join cannot fan out, which is
    what makes the missing dedup safe rather than equivalent by luck."""
    left = _left()
    many = pl.DataFrame({"id": [2] * 50})
    got = (_semi(left.lazy(), many.lazy(), "src", "id").collect() if lazy
           else _semi(left, many, "src", "id"))
    assert got.height == 2  # the two src==2 rows, not 100
    inner = left.join(many.select("id"), left_on="src", right_on="id", how="inner")
    assert inner.height == 100  # contrast: an inner join DOES fan out, hence needs dedup


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_key_frame_selects_nothing_and_keeps_schema(lazy: bool) -> None:
    left = _left()
    empty = pl.DataFrame({"id": []}, schema={"id": pl.Int64})
    got = (_semi(left.lazy(), empty.lazy(), "src", "id").collect() if lazy
           else _semi(left, empty, "src", "id"))
    assert got.height == 0
    assert got.columns == left.columns


@pytest.mark.parametrize("lazy", [False, True])
def test_non_matching_keys_actually_drop_rows(lazy: bool) -> None:
    """NEGATIVE side, and it is load-bearing: without it a semi-join that returned
    the left frame UNCHANGED would still satisfy the duplicate-key test, since both
    of its arms would be equally wrong. This pins that the join filters at all."""
    left = _left()
    partial = pl.DataFrame({"id": [3, 99]})   # 99 matches nothing
    none_ = pl.DataFrame({"id": [7, 8, 9]})   # nothing matches
    def run(keys: pl.DataFrame) -> pl.DataFrame:
        return (_semi(left.lazy(), keys.lazy(), "src", "id").collect() if lazy
                else _semi(left, keys, "src", "id"))
    assert run(partial).to_dicts() == [{"src": 3, "payload": "c"}]
    assert run(none_).height == 0
    assert run(partial).height < left.height  # it is a filter, not a passthrough
