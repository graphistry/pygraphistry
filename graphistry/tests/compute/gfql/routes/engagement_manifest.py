"""Record route calls per test and select only tests affected by a disabled route.

Loaded explicitly with ``pytest -p graphistry.tests.compute.gfql.routes.engagement_manifest``.
The normal replay records calls to every route switch target. A disabled route can
only change a test that called its target on the normal path; all-off uses the union.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pytest

from graphistry.tests.compute.gfql.routes.switch import ROUTES, _targets


def _digest(ids: List[str]) -> str:
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


_active: Optional[str] = None
_hits: Dict[str, Set[str]] = defaultdict(set)
_seen: Set[str] = set()
_collection: List[str] = []


def _install_wrappers() -> None:
    targets: Dict[Tuple[Any, str], Set[str]] = defaultdict(set)
    for route in ROUTES:
        for module, name in _targets((route,)):
            targets[(module, name)].add(route)
    for (module, name), routes in targets.items():
        original = getattr(module, name)
        if getattr(original, "_gfql_route_recording", False):
            continue

        def record(*args: Any, _original: Any = original, _routes: frozenset[str] = frozenset(routes), **kwargs: Any) -> Any:
            if _active is not None:
                for route in _routes:
                    _hits[route].add(_active)
            return _original(*args, **kwargs)

        record._gfql_route_recording = True  # type: ignore[attr-defined]
        record.__name__ = getattr(original, "__name__", name)
        record.__doc__ = getattr(original, "__doc__", None)
        setattr(module, name, record)


def pytest_sessionstart(session: pytest.Session) -> None:
    if os.environ.get("GFQL_ROUTE_RECORD_DIR"):
        _install_wrappers()


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    global _collection
    _collection = [item.nodeid for item in items]
    path = os.environ.get("GFQL_ROUTE_SELECT_MANIFEST")
    if not path:
        return
    manifest = json.loads(Path(path).read_text())
    if manifest["routes"] != list(ROUTES) or manifest["collection_sha256"] != _digest(_collection):
        pytest.exit("GFQL route manifest does not match this test collection")
    off = {route for route in os.environ.get("GFQL_ROUTES_OFF", "").split(",") if route}
    if not off or off - set(ROUTES):
        pytest.exit("GFQL route selection requires known disabled routes")
    selected = set().union(*(manifest["hits"][route] for route in off))
    if not selected:
        pytest.exit("GFQL route manifest selected no tests; refusing an empty replay")
    kept = [item for item in items if item.nodeid in selected]
    dropped = [item for item in items if item.nodeid not in selected]
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)


@pytest.hookimpl(hookwrapper=True)  # type: ignore[misc]
def pytest_runtest_protocol(item: pytest.Item, nextitem: Optional[pytest.Item]) -> Iterator[None]:
    global _active
    if not os.environ.get("GFQL_ROUTE_RECORD_DIR"):
        yield
        return
    _install_wrappers()
    _active = item.nodeid
    _seen.add(item.nodeid)
    yield
    _active = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    directory = os.environ.get("GFQL_ROUTE_RECORD_DIR")
    if not directory:
        return
    worker = getattr(session.config, "workerinput", None)
    if worker is None and getattr(session.config.option, "numprocesses", None):
        return
    name = worker["workerid"] if worker is not None else "standalone"
    Path(directory).mkdir(parents=True, exist_ok=True)
    payload = {
        "collection_sha256": _digest(_collection),
        "collected": len(_collection),
        "seen": sorted(_seen),
        "hits": {route: sorted(_hits[route]) for route in ROUTES},
        "exitstatus": exitstatus,
    }
    (Path(directory) / f"{name}.json").write_text(json.dumps(payload, sort_keys=True))
