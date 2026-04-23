"""Tests for graphistry.compute.gfql.expr_split.split_top_level_and."""

from __future__ import annotations

from graphistry.compute.gfql.expr_split import split_top_level_and


def test_empty_returns_empty_tuple() -> None:
    assert split_top_level_and("") == ()
    assert split_top_level_and("   ") == ()


def test_single_term_returns_one_element_tuple() -> None:
    assert split_top_level_and("a = 1") == ("a = 1",)
    assert split_top_level_and("n:Admin") == ("n:Admin",)


def test_double_and_conjunction() -> None:
    assert split_top_level_and("a = 1 AND b = 2") == ("a = 1", "b = 2")


def test_triple_and_conjunction() -> None:
    assert split_top_level_and("a AND b AND c") == ("a", "b", "c")


def test_case_insensitive_and() -> None:
    assert split_top_level_and("a and b") == ("a", "b")
    assert split_top_level_and("a And b") == ("a", "b")
    assert split_top_level_and("a aNd b") == ("a", "b")


def test_word_boundary_not_broken_by_and_inside_word() -> None:
    # "ANDROID" contains "AND" but the trailing character is word-class.
    assert split_top_level_and("n:ANDROID") == ("n:ANDROID",)
    # "BRAND" likewise.
    assert split_top_level_and("n:BRAND = 1") == ("n:BRAND = 1",)


def test_and_inside_single_quoted_string_does_not_split() -> None:
    assert split_top_level_and("n.name = 'x AND y'") == ("n.name = 'x AND y'",)


def test_and_inside_double_quoted_string_does_not_split() -> None:
    assert split_top_level_and('n.name = "x AND y"') == ('n.name = "x AND y"',)


def test_and_inside_backtick_identifier_does_not_split() -> None:
    assert split_top_level_and("n.`weird AND name` = 1") == ("n.`weird AND name` = 1",)


def test_and_inside_parens_does_not_split() -> None:
    assert split_top_level_and("(a AND b) = c AND d = 1") == (
        "(a AND b) = c",
        "d = 1",
    )


def test_and_inside_square_brackets_does_not_split() -> None:
    assert split_top_level_and("a[0 AND 1] AND b") == ("a[0 AND 1]", "b")


def test_and_inside_curly_braces_does_not_split() -> None:
    assert split_top_level_and("{k: v AND w} AND z") == ("{k: v AND w}", "z")


def test_escape_sequence_in_string_does_not_break_quote_tracking() -> None:
    # Backslash-escaped quote inside a string — the string should remain
    # open across the escape, and the AND inside stays quoted.
    assert split_top_level_and(r"n.s = 'it\'s AND ok'") == (r"n.s = 'it\'s AND ok'",)


def test_trailing_and_returns_empty_tuple() -> None:
    # Trailing AND with no right-hand term is malformed; callers should
    # treat `()` as "do not split".
    assert split_top_level_and("a AND") == ()


def test_leading_and_returns_empty_tuple() -> None:
    assert split_top_level_and("AND a") == ()


def test_consecutive_ands_returns_empty_tuple() -> None:
    # An empty term between two ANDs is malformed.
    assert split_top_level_and("a AND AND b") == ()


def test_mixed_depth_preserves_split_at_top_level_only() -> None:
    assert split_top_level_and("(a AND b) AND (c AND d)") == (
        "(a AND b)",
        "(c AND d)",
    )


def test_and_adjacent_to_paren_at_depth_zero_splits() -> None:
    assert split_top_level_and("(a) AND (b)") == ("(a)", "(b)")
