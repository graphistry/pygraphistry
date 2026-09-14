"""Contract: python_remote accepts the code forms its own documentation prescribes.

``code`` may arrive as a callable or as a source string. Both are normalized to a
top-level ``def task`` source string before validation, so neither the function's name
nor the indentation of the literal it was written in can decide whether the call works.
"""

import pytest

from graphistry.compute.python_remote import normalize_task_code, validate_python_str


def task(g):
    return {"n": len(g._edges)}


def helper(g):
    return {"n": len(g._edges)}


def test_callable_named_task_normalizes_to_a_string() -> None:
    out = normalize_task_code(task)
    assert isinstance(out, str)
    assert validate_python_str(out) is True


def test_callable_named_other_than_task_normalizes_to_a_string() -> None:
    out = normalize_task_code(helper)
    assert isinstance(out, str)
    assert "def task(g)" in out
    assert validate_python_str(out) is True


def test_callable_named_task_and_renamed_callable_agree() -> None:
    assert normalize_task_code(task) == normalize_task_code(helper)


def test_nested_callable_source_is_dedented() -> None:
    def outer():
        def task(g):
            return {"n": len(g._edges)}
        return task

    out = normalize_task_code(outer())
    assert not out.startswith(" ")
    assert validate_python_str(out) is True


def test_indented_source_literal_is_accepted() -> None:
    code = """
        from typing import Any, Dict

        def task(g):
            return {'num_edges': len(g._edges)}
    """
    assert validate_python_str(normalize_task_code(code)) is True


def test_unindented_source_literal_is_unchanged() -> None:
    code = "def task(g):\n    return {'n': 1}\n"
    assert normalize_task_code(code) == code
    assert validate_python_str(normalize_task_code(code)) is True


def test_relative_indentation_inside_the_body_survives_dedent() -> None:
    code = """
        def task(g):
            if g is None:
                return {'n': 0}
            return {'n': 1}
    """
    out = normalize_task_code(code)
    assert "        return {'n': 0}" in out
    assert validate_python_str(out) is True


def test_missing_task_function_still_declines() -> None:
    with pytest.raises(ValueError, match="No top-level function 'task'"):
        validate_python_str(normalize_task_code("def other(g):\n    return {}\n"))


def test_task_with_wrong_arity_still_declines() -> None:
    with pytest.raises(ValueError, match="exactly one parameter"):
        validate_python_str(normalize_task_code("def task(g, extra):\n    return {}\n"))
