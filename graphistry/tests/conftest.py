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
    import graphistry.compute.chain as chain_mod
    import graphistry.compute.gfql_unified as unified
    import graphistry.compute.gfql.index as index_pkg
    import graphistry.compute.gfql.index.api as index_api
    import graphistry.compute.gfql.index.bindings as bindings
    import graphistry.compute.gfql.lazy.engine.polars.chain as pchain

    def none(*a, **k):
        return None

    patches = []

    def patch(mod, name, value):
        patches.append((mod, name, getattr(mod, name)))
        setattr(mod, name, value)

    if "native-fast" in routes:
        patch(chain_mod, "_try_chain_fast_path", none)
    if "polars-seeded" in routes:
        patch(pchain, "_try_seeded_chain_polars", none)
    if "polars-plain" in routes:
        patch(pchain, "polars_plain_single_hop_admits", none)
    if "index-hop" in routes:
        patch(index_pkg, "maybe_index_hop", none)
        patch(index_api, "maybe_index_hop", none)
    if "indexed-kernel" in routes:
        patch(bindings, "_try_indexed_connected_bindings_state", none)
    if "cypher-fast" in routes:
        for name in ("_execute_seeded_node_lookup_fast_path", "_execute_seeded_typed_hop_fast_path",
                     "_execute_single_hop_grouped_aggregate_fast_path", "_execute_two_hop_count_fast_path"):
            patch(unified, name, none)
    try:
        yield
    finally:
        for mod, name, value in reversed(patches):
            setattr(mod, name, value)
