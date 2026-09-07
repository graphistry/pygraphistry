"""Route forcing for test amplification.

``GFQL_ROUTES_OFF=<route>[,<route>...]`` makes the named hot paths decline for the whole
session, so every existing test — written against one specialization or against the
general path — is replayed through the other routes. Failures that appear only under a
mode are the cases where that route and the rest disagree.

Routes: native-fast (pandas/cuDF chain fast path), polars-seeded (polars seeded lane),
polars-plain (polars plain single-hop branches), index-hop (hop() index path),
indexed-kernel (indexed connected-bindings kernel), cypher-fast (the four Cypher lanes).

A test that asserts a route SERVES (trace, latency, served-by spy) is an engagement pin, not
a result pin: mark it ``@pytest.mark.route_engaged("<route>", ...)`` and it is skipped when
one of its routes is off, so the replay reports result divergences only
(``bin/test-routes-off.sh``).
"""
import os

import pytest


def _routes_off():
    raw = os.environ.get("GFQL_ROUTES_OFF", "")
    return {r.strip() for r in raw.split(",") if r.strip()}


def pytest_collection_modifyitems(config, items):
    off = _routes_off()
    if not off:
        return
    for item in items:
        for mark in item.iter_markers("route_engaged"):
            hit = off & set(mark.args)
            if hit:
                item.add_marker(pytest.mark.skip(reason="engagement pin for route(s) off: " + ",".join(sorted(hit))))


@pytest.fixture(autouse=True, scope="session")
def _gfql_routes_off():
    routes = _routes_off()
    if not routes:
        yield
        return
    from graphistry.tests.compute.gfql.routes.switch import routes_off
    with routes_off(routes):
        yield
