"""Pins that graphistry.outliers imports cleanly when matplotlib is absent.

The module guards its optional imports with try/except and binds each name to None on
failure. `import matplotlib.font_manager` binds `matplotlib`, and that name was missing
from the fallback list, so a module referencing `matplotlib.font_manager.FontProperties`
would raise NameError rather than degrade.
"""

import ast
import importlib
import os
import sys

import pytest


def _reimport_without(module_name: str):
    """Import graphistry.outliers with `module_name` unimportable."""
    blocked = {module_name: None}
    saved = {k: v for k, v in sys.modules.items()
             if k == module_name or k.startswith(module_name + ".")}
    saved["graphistry.outliers"] = sys.modules.get("graphistry.outliers")
    try:
        for k in list(saved):
            sys.modules.pop(k, None)
        sys.modules.update(blocked)
        return importlib.import_module("graphistry.outliers")
    finally:
        sys.modules.pop(module_name, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
        importlib.import_module("graphistry.outliers")


def test_outliers_imports_when_matplotlib_is_missing():
    mod = _reimport_without("matplotlib")
    assert mod.matplotlib is None
    assert mod.plt is None


def test_every_name_the_optional_import_binds_has_a_fallback():
    """The defect class, not just the one instance: no name may be bound on only one path."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outliers.py"
    )
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert tries, "outliers.py no longer guards its optional imports"

    for node in tries:
        handler_nodes = {id(n) for h in node.handlers for n in ast.walk(h)}
        bound_in_try = set()
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in handler_nodes:
                    for alias in n.names:
                        bound_in_try.add((alias.asname or alias.name).split(".")[0])
        bound_in_except = set()
        for handler in node.handlers:
            for n in ast.walk(handler):
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if isinstance(target, ast.Name):
                            bound_in_except.add(target.id)
        assert bound_in_try - bound_in_except == set(), (
            "names bound only when the import succeeds: %s"
            % sorted(bound_in_try - bound_in_except)
        )


def test_module_still_binds_the_real_matplotlib_when_present():
    pytest.importorskip("matplotlib")
    mod = importlib.import_module("graphistry.outliers")
    assert mod.matplotlib is not None
    assert mod.plt is not None
