"""Every process-lifetime GFQL cache must REGISTER itself in the cache registry.

The convention (see ``graphistry/compute/gfql/cache_registry.py`` and the GFQL
contributor guide): a cache setup site registers, adjacent to its definition,
either a clearable handle (``register_clearable`` / ``register_clearable_dict``
/ ``register_clearable_callable``) or a process-singleton exemption with a
written reason. ``gfql_clear_caches`` empties exactly the registered clearables.

THE LOCK here is completeness: a static AST sweep discovers every
``@lru_cache``/``@cache`` function, every module-level dict/set binding named
cache/memo, and every module-level mutable container that is WRITTEN at runtime
whatever its name, imports their modules, and fails when any discovered cache is
absent from the registry. Registration is enforced, not optional -- adding a memo
without registering it fails this file.

Why this exists: a clear that looked its target up BY NAME once turned into a
silent no-op (``parse_cypher`` vs the ``_parse_cypher_cached`` body holding the
memo) and published a wrong "cold-process" number for days. Registration hands
over the bound clear at definition time, so there is no later lookup to miss.

Two blind spots closed by #1913, both found with live (if benign) instances:

* SCOPE -- the sweep read only ``compute/gfql/**`` + ``gfql_unified.py``, while
  ``chain.py``, ``hop.py``, ``chain_fast_paths.py``, ``gfql_fast_paths.py`` and
  ``ComputeMixin.py`` are just as much the GFQL execution path.
* NAMING -- it matched only names containing cache/memo, so ``_OFFENGINE_BRIDGE_WARNED``
  (a warn-once ledger) and ``_COST_GATE_FRAC_OVERRIDES`` (planner tuning) were
  invisible process-global state. Neither could produce a wrong answer, which is
  exactly why a name-based lock would never have surfaced them.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Dict

import pytest

GFQL_TREE = Path(__file__).resolve().parents[3] / "compute" / "gfql"
_COMPUTE = Path(__file__).resolve().parents[3] / "compute"
GFQL_UNIFIED = _COMPUTE / "gfql_unified.py"
REPO_ROOT = Path(__file__).resolve().parents[4]

# The GFQL execution path is wider than the gfql/ package: these modules run on every
# query too, so a memo parked in one of them is exactly as process-global (#1913).
EXECUTION_PATH_MODULES = [
    GFQL_UNIFIED,
    _COMPUTE / "chain.py",
    _COMPUTE / "hop.py",
    _COMPUTE / "chain_fast_paths.py",
    _COMPUTE / "gfql_fast_paths.py",
    _COMPUTE / "ComputeMixin.py",
]

# Mutating method names + container constructors used by the runtime-write scan below.
_MUTATORS = frozenset({
    "add", "append", "clear", "discard", "extend", "insert", "pop", "popitem",
    "remove", "setdefault", "update", "__setitem__",
})
_CONTAINER_CTORS = frozenset({
    "dict", "set", "list", "OrderedDict", "WeakValueDictionary", "WeakKeyDictionary",
    "defaultdict", "deque", "Counter",
})


def _scanned_files() -> list:
    """Every file the lock reads: the GFQL package plus the wider execution path."""
    seen = sorted(GFQL_TREE.rglob("*.py"))
    for path in EXECUTION_PATH_MODULES:
        assert path.exists(), f"scan target moved or was renamed: {path}"
        if path not in seen:
            seen.append(path)
    return seen


def _lru_cached_functions() -> Dict[str, Path]:
    """Every ``@lru_cache``/``@cache``-decorated def in the GFQL tree, name -> file."""
    found: Dict[str, Path] = {}
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = (
                    target.attr if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name)
                    else ""
                )
                if name in ("lru_cache", "cache"):
                    found[node.name] = path
    return found


def _module_dict_caches() -> Dict[str, Path]:
    """Every MODULE-LEVEL dict/OrderedDict/set binding whose name says cache/memo.

    Hand-rolled memos are invisible to the decorator scan -- ``_SINGLE_ALIAS_CACHE``
    sat exactly in that blind spot until the 2026-08-01 audit. Name-based on
    purpose: a memo whose name hides what it is has worse problems than this
    file can catch.
    """
    found: Dict[str, Path] = {}
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for name in _module_level_containers(tree):
            lowered = name.lower()
            if "cache" in lowered or "memo" in lowered:
                found[name] = path
    return found


def _module_level_containers(tree: ast.Module) -> Dict[str, ast.AST]:
    """Module-level names bound to a mutable container literal/constructor."""
    out: Dict[str, ast.AST] = {}
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        is_container = (
            isinstance(value, (ast.Dict, ast.Set, ast.List))
            or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in _CONTAINER_CTORS
            )
        )
        if is_container:
            for target in targets:
                out[target.id] = value
    return out


def _runtime_written_names(tree: ast.Module) -> set:
    """Names this module MUTATES at runtime: ``x[k] = ``, ``del x[k]``, ``x.add(...)``
    and friends, or a ``global x`` rebind. Read-only lookup tables are left alone --
    the point is state that changes as queries run."""
    written: set = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.attr in _MUTATORS
        ):
            written.add(node.func.value.id)
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    written.add(target.value.id)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    written.add(target.value.id)
        elif isinstance(node, ast.Global):
            written.update(node.names)
    return written


def _mutated_process_state() -> Dict[str, Path]:
    """Every module-level mutable container that the module WRITES at runtime, whatever
    it is called.

    The name-based scan above only sees state that admits to being a cache.
    ``_OFFENGINE_BRIDGE_WARNED`` (a warn-once set) and ``_COST_GATE_FRAC_OVERRIDES``
    (planner tuning) are both process-lifetime and both were invisible to it (#1913).
    Neither can produce a wrong answer -- which is the point: a lock that only fires on
    correctness bugs is not a completeness lock. Each hit must end up registered as
    clearable or exempted with a written reason.
    """
    found: Dict[str, Path] = {}
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        containers = _module_level_containers(tree)
        if not containers:
            continue
        for name in sorted(set(containers) & _runtime_written_names(tree)):
            found[name] = path
    return found


def _import_module(path: Path) -> None:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    importlib.import_module(".".join(relative.parts))


def test_every_discovered_cache_is_registered() -> None:
    """THE LOCK. A new memo in the GFQL tree must register itself, at its own def site."""
    discovered = {**_lru_cached_functions(), **_module_dict_caches(), **_mutated_process_state()}
    assert discovered, "the AST scans found no caches at all -- the scan itself is broken"

    for path in sorted(set(discovered.values())):
        _import_module(path)

    from graphistry.compute.gfql.cache_registry import entries

    registered = entries()
    unregistered = sorted(set(discovered) - set(registered))
    assert not unregistered, (
        "these caches are not registered in graphistry/compute/gfql/cache_registry.py:\n"
        + "\n".join(f"  {n}  ({discovered[n]})" for n in unregistered)
        + "\n\nRegister each at its definition site: register_clearable(...) when it is "
        "keyed by caller input, register_process_singleton(..., reason) when it is a "
        "function of the code alone. An unaccounted cache makes results order-dependent "
        "and makes 'cold' measurements wrong."
    )

    stale = sorted(set(registered) - set(discovered))
    assert not stale, (
        f"registered but not discovered by the scans: {stale}. Either the cache moved out "
        "of the GFQL tree (move or drop its registration) or the scan needs a new pattern."
    )


def test_lock_scans_the_whole_gfql_execution_path() -> None:
    """#1913 SCOPE pin: narrowing the sweep back to compute/gfql/** must fail here."""
    scanned = set(_scanned_files())
    for path in EXECUTION_PATH_MODULES:
        assert path in scanned, f"{path.name} dropped out of the cache-coverage sweep"


def test_lock_sees_process_state_that_does_not_say_cache() -> None:
    """#1913 NAMING pin: the two live examples that a cache/memo name match cannot see.

    Both are benign today (warning order; plan choice) -- that is precisely why only a
    name-blind, mutation-based scan finds them. If either is renamed or retired, retarget
    this pin at whatever module-level mutable state remains; do not delete it.
    """
    discovered = _mutated_process_state()
    assert "_OFFENGINE_BRIDGE_WARNED" in discovered, discovered
    assert "_COST_GATE_FRAC_OVERRIDES" in discovered, discovered
    # ...and neither name would have been matched by the cache/memo scan
    named = _module_dict_caches()
    assert "_OFFENGINE_BRIDGE_WARNED" not in named and "_COST_GATE_FRAC_OVERRIDES" not in named


def test_clear_caches_empties_the_offengine_bridge_warn_ledger() -> None:
    """CLEARABLE verdict for the warn-once set: a ledger that cannot be emptied makes
    warning behavior depend on what ran earlier in the process."""
    from graphistry.compute.gfql.call import executor
    from graphistry.compute.gfql_unified import gfql_clear_caches

    executor._OFFENGINE_BRIDGE_WARNED.add("probe_fn")
    gfql_clear_caches()
    assert executor._OFFENGINE_BRIDGE_WARNED == set()


def test_clear_caches_preserves_cost_gate_overrides() -> None:
    """EXEMPT verdict, and it is load-bearing: set_cost_gate_frac is deliberate tuning,
    not a memo, so a call named "clear caches" must not silently revert it."""
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.cost import (
        cost_gate_frac, reset_cost_gate_frac, set_cost_gate_frac,
    )
    from graphistry.compute.gfql_unified import gfql_clear_caches

    try:
        set_cost_gate_frac(Engine.PANDAS, 0.123)
        gfql_clear_caches()
        assert cost_gate_frac(Engine.PANDAS) == 0.123
    finally:
        reset_cost_gate_frac(Engine.PANDAS)


def test_registry_fails_loud() -> None:
    """An empty registry, a duplicate handle, or a thin reason must raise, not shrug."""
    from graphistry.compute.gfql import cache_registry as reg

    original = dict(reg._REGISTRY)
    try:
        reg._REGISTRY.clear()
        with pytest.raises(RuntimeError):
            reg.clear_all()
    finally:
        reg._REGISTRY.update(original)

    from functools import lru_cache

    @lru_cache(maxsize=1)
    def thin_probe() -> None:
        return None

    with pytest.raises(ValueError, match="too thin"):
        reg.register_process_singleton(thin_probe, "because")

    def probe() -> None:
        pass

    reg._REGISTRY.pop("probe", None)
    reg.register_clearable_callable("probe", probe)
    try:
        with pytest.raises(ValueError, match="registered twice"):
            reg.register_clearable_callable("probe", lambda: None)
    finally:
        reg._REGISTRY.pop("probe", None)


def test_clear_caches_actually_empties_the_cypher_ast_memo() -> None:
    """THE REGRESSION PIN for the published-number bug."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    cypher_parser.parse_cypher("MATCH (n) RETURN n")
    assert cypher_parser._parse_cypher_cached.cache_info().currsize > 0

    gfql_clear_caches()
    assert cypher_parser._parse_cypher_cached.cache_info().currsize == 0, (
        "gfql_clear_caches() left the Cypher AST memo populated -- every 'cold-process' "
        "measurement taken through it is invalid"
    )


def test_clear_caches_actually_empties_the_row_expression_memo() -> None:
    pytest.importorskip("lark")
    from graphistry.compute.gfql import expr_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    expr_parser.parse_expr("age > 30")
    assert expr_parser._parse_expr_cached.cache_info().currsize > 0

    gfql_clear_caches()
    assert expr_parser._parse_expr_cached.cache_info().currsize == 0


def test_clear_caches_actually_empties_the_single_alias_lowering_memo() -> None:
    """THE 2026-08-01 AUDIT PIN: the dict-style memo the decorator scan never saw."""
    pytest.importorskip("lark")
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline
    from graphistry.compute.gfql_unified import gfql_clear_caches

    schema = pl.DataFrame({"age": [1]}).schema
    row_pipeline.lower_single_alias_predicate("age > 30", "n", schema)
    assert len(row_pipeline._SINGLE_ALIAS_CACHE) > 0, (
        "the probe call did not populate the memo; fix the probe before trusting this test"
    )

    gfql_clear_caches()
    assert len(row_pipeline._SINGLE_ALIAS_CACHE) == 0


def test_clear_caches_leaves_the_lalr_tables_built() -> None:
    """The exemption is load-bearing: rebuilding the LALR tables on every clear would put
    grammar construction inside any 'cold' number measured through this call."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    before = cypher_parser._parser_lalr()
    gfql_clear_caches()
    assert cypher_parser._parser_lalr() is before, (
        "the whole-query LALR parser was rebuilt by gfql_clear_caches()"
    )


def test_clear_survives_module_attribute_swap() -> None:
    """Registration binds the real object's clear at definition time, so replacing the
    module attribute later cannot silently disconnect the clear -- the opposite failure
    mode of the old name-lookup design."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    real = cypher_parser._parse_cypher_cached
    real("MATCH (n) RETURN n")
    assert real.cache_info().currsize > 0
    try:
        cypher_parser._parse_cypher_cached = object()  # type: ignore[assignment]
        gfql_clear_caches()
    finally:
        cypher_parser._parse_cypher_cached = real  # type: ignore[assignment]
    assert real.cache_info().currsize == 0


def test_single_alias_hit_path_survives_clear_during_read() -> None:
    """A clear (or eviction) landing between the memo's move_to_end and its read must
    degrade to a recompute, never raise KeyError to the caller. Deterministic interleave:
    an OrderedDict subclass whose move_to_end empties the cache mid-hit."""
    pytest.importorskip("lark")
    pl = pytest.importorskip("polars")
    from collections import OrderedDict

    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline

    class ClearsOnTouch(OrderedDict):
        def move_to_end(self, key, last=True):  # type: ignore[override]
            super().move_to_end(key, last)
            self.clear()  # simulate registry clear_all() winning the race

    schema = pl.DataFrame({"age": [1]}).schema
    oracle = row_pipeline._lower_single_alias_predicate_uncached("age > 30", "n", schema, False)
    original = row_pipeline._SINGLE_ALIAS_CACHE
    try:
        row_pipeline._SINGLE_ALIAS_CACHE = ClearsOnTouch()  # type: ignore[assignment]
        row_pipeline.lower_single_alias_predicate("age > 30", "n", schema)  # populate
        assert len(row_pipeline._SINGLE_ALIAS_CACHE) == 1
        # hit path: move_to_end fires the mid-read clear; must recompute, not KeyError
        second = row_pipeline.lower_single_alias_predicate("age > 30", "n", schema)
        assert str(second) == str(oracle), "race must recompute to the oracle answer"
    finally:
        row_pipeline._SINGLE_ALIAS_CACHE = original  # type: ignore[assignment]


def test_concurrent_queries_and_clears_never_corrupt() -> None:
    """Thread-stress: workers hammer every clearable cache's hot path while a clearer
    loops gfql_clear_caches(). Deterministic assertions only -- no exceptions escape,
    and every returned value matches its single-threaded oracle -- so the test cannot
    flake on timing; more threads simply exercise more interleavings."""
    pytest.importorskip("lark")
    pl = pytest.importorskip("polars")
    import threading

    from graphistry.compute.gfql import expr_parser
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql.lazy.engine.polars import row_pipeline
    from graphistry.compute.gfql_unified import gfql_clear_caches

    schema = pl.DataFrame({"age": [1], "name": [""]}).schema
    exprs = [f"age > {n}" for n in range(5)]
    oracles = {
        e: str(row_pipeline._lower_single_alias_predicate_uncached(e, "n", schema, False))
        for e in exprs
    }
    queries = [f"MATCH (n) WHERE n.age > {n} RETURN n" for n in range(5)]

    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker() -> None:
        try:
            barrier.wait()
            for i in range(150):
                e = exprs[i % len(exprs)]
                assert expr_parser.parse_expr(e) is not None
                lowered = row_pipeline.lower_single_alias_predicate(e, "n", schema)
                assert str(lowered) == oracles[e], f"corrupted value for {e!r}"
                cypher_parser.parse_cypher(queries[i % len(queries)])
        except BaseException as error:  # noqa: BLE001 - collect for the main thread
            errors.append(error)

    def clearer() -> None:
        try:
            barrier.wait()
            for _ in range(150):
                gfql_clear_caches()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    threads.append(threading.Thread(target=clearer))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "stress thread hung"
    assert errors == [], f"concurrency corruption: {errors[:3]}"
