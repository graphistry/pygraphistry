"""Route switch for test amplification: make named GFQL hot paths decline for a scope."""
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Tuple

ROUTES = ("native-fast", "polars-seeded", "polars-plain", "index-hop", "indexed-kernel", "cypher-fast")


def _none(*a, **k):
    return None


def _targets(routes: Iterable[str]) -> List[Tuple[object, str]]:
    import graphistry.compute.chain as chain_mod
    import graphistry.compute.gfql_unified as unified
    import graphistry.compute.gfql.index as index_pkg
    import graphistry.compute.gfql.index.api as index_api
    import graphistry.compute.gfql.index.bindings as bindings
    import graphistry.compute.gfql.lazy.engine.polars.chain as pchain
    routes = set(routes)
    unknown = routes - set(ROUTES)
    assert not unknown, f"unknown route(s) {sorted(unknown)}; known: {ROUTES}"
    out: List[Tuple[object, str]] = []
    if "native-fast" in routes:
        out.append((chain_mod, "_try_chain_fast_path"))
    if "polars-seeded" in routes:
        out.append((pchain, "_try_seeded_chain_polars"))
    if "polars-plain" in routes:
        out.append((pchain, "polars_plain_single_hop_admits"))
    if "index-hop" in routes:
        out += [(index_pkg, "maybe_index_hop"), (index_api, "maybe_index_hop")]
    if "indexed-kernel" in routes:
        out.append((bindings, "_try_indexed_connected_bindings_state"))
    if "cypher-fast" in routes:
        out += [(unified, name) for name in (
            "_execute_seeded_node_lookup_fast_path", "_execute_seeded_typed_hop_fast_path",
            "_execute_single_hop_grouped_aggregate_fast_path", "_execute_two_hop_count_fast_path")]
    return out


@contextmanager
def routes_off(routes: Iterable[str]) -> Iterator[None]:
    """Within the block the named routes decline, so the general path answers."""
    saved = []
    for mod, name in _targets(routes):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, _none)
    try:
        yield
    finally:
        for mod, name, value in reversed(saved):
            setattr(mod, name, value)
