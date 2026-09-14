"""Arity + surfaced-exception pins for the lazy dependency probes."""

import builtins
from typing import Any, Callable, List, Tuple

import pytest

from graphistry.utils.lazy_import import (
    assert_imported,
    assert_imported_text,
    lazy_import_has_min_dependancy,
    lazy_sentence_transformers_import,
    lazy_umap_import,
)


class _Boom(RuntimeError):
    """A non-ModuleNotFoundError import failure (e.g. a broken/ABI-mismatched wheel)."""


def _raise_on(monkeypatch: pytest.MonkeyPatch, prefixes: Tuple[str, ...]) -> None:
    """Make ``import <prefix>...`` raise ``_Boom`` instead of ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if any(name == p or name.startswith(p + '.') for p in prefixes):
            raise _Boom(f'broken install: {name}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_min_dependancy_generic_failure_returns_two_tuple(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _raise_on(monkeypatch, ('scipy', 'sklearn'))
    out = lazy_import_has_min_dependancy()
    assert isinstance(out, tuple)
    # Every caller unpacks exactly two values.
    assert len(out) == 2, f'arity drift: {out!r}'
    ok, exn = out
    assert ok is False
    assert isinstance(exn, _Boom)


def test_assert_imported_surfaces_the_dependency_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _raise_on(monkeypatch, ('scipy', 'sklearn'))
    # Must be the underlying import failure, not ValueError from a bad unpack.
    with pytest.raises(_Boom):
        assert_imported()


def test_min_dependancy_module_not_found_returns_two_tuple(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'scipy' or name.startswith('scipy.'):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    out = lazy_import_has_min_dependancy()
    assert len(out) == 2
    assert out[0] is False
    assert isinstance(out[1], ModuleNotFoundError)


@pytest.mark.parametrize(
    'probe,prefixes',
    [
        (lazy_umap_import, ('umap',)),
        (lazy_sentence_transformers_import, ('sentence_transformers',)),
    ],
)
def test_three_tuple_probes_keep_their_arity_on_generic_failure(
    monkeypatch: pytest.MonkeyPatch,
    probe: Callable[[], Any],
    prefixes: Tuple[str, ...],
) -> None:
    _raise_on(monkeypatch, prefixes)
    out = probe()
    assert len(out) == 3, f'arity drift: {out!r}'
    assert out[0] is False
    assert isinstance(out[1], _Boom)
    assert out[2] is None


def test_assert_imported_text_surfaces_the_dependency_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _raise_on(monkeypatch, ('sentence_transformers',))
    with pytest.raises(_Boom):
        assert_imported_text()


def test_min_dependancy_success_returns_two_tuple() -> None:
    pytest.importorskip('scipy')
    pytest.importorskip('sklearn')
    out = lazy_import_has_min_dependancy()
    assert len(out) == 2
    ok, msg = out
    assert ok is True
    assert msg == 'ok'


def test_all_return_paths_of_min_dependancy_have_equal_arity() -> None:
    """Mutation guard: a new early-return with a different arity breaks callers."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(lazy_import_has_min_dependancy))
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    arities: List[int] = [
        len(node.value.elts)
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)
    ]
    assert arities, 'expected tuple returns'
    assert set(arities) == {2}, f'mixed return arity: {arities}'
