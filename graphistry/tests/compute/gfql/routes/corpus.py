"""Shared shape corpus for the chain routes.

Every entry is a native op-list shape variant; a route test filters the corpus with the
route's own admission predicate (the function its dispatcher calls), so one corpus is reused
across hot paths and a shape is never hand-picked per lane. Tags name the defect classes an
entry exercises so coverage can be read per class.
"""
from typing import Callable, Dict, List, NamedTuple, Tuple

import pandas as pd

from graphistry.compute.ast import ASTObject, e_forward, e_reverse, e_undirected, n
from graphistry.compute.predicates.numeric import GT
from graphistry.tests.compute.gfql.routes.registry import Frames, register


class Entry(NamedTuple):
    name: str
    ops: Callable[[], List[ASTObject]]
    tags: Tuple[str, ...]


NODES = pd.DataFrame({"key": [1, 2, 3, 4, 5], "id": [10, 20, 30, 40, 50], "type": ["p", "p", "m", "m", "p"], "w": [1, 2, 3, 4, 5]})
EDGES = pd.DataFrame({"s": [1, 1, 2, 3, 3, 4], "d": [2, 3, 3, 1, 1, 5], "type": ["KNOWS", "KNOWS", "LIKES", "KNOWS", "KNOWS", "LIKES"], "eid": [0, 1, 2, 3, 4, 5], "w": [1, 2, 3, 4, 5, 6]})

CORPUS: List[Entry] = [
    Entry("single node, scalar filter", lambda: [n({"id": 30})], ("single-node",)),
    Entry("single node, named", lambda: [n({"id": 30}, name="a")], ("single-node", "alias")),
    Entry("single node, predicate filter", lambda: [n({"w": GT(2)})], ("single-node", "predicate")),
    Entry("single node, no filter", lambda: [n()], ("single-node",)),
    Entry("plain single hop, unseeded", lambda: [n(), e_forward(), n()], ("single-hop", "unseeded")),
    Entry("plain single hop, seeded", lambda: [n({"key": 1}), e_forward(), n()], ("single-hop", "seeded", "#2051")),
    Entry("plain single hop, seeded, reverse", lambda: [n({"key": 1}), e_reverse(), n()], ("single-hop", "seeded", "reverse")),
    Entry("plain single hop, seeded, destination filter", lambda: [n({"key": 1}), e_forward(), n({"id": 20})], ("single-hop", "seeded", "dest-filter", "#2051")),
    Entry("plain single hop, undirected, unconstrained", lambda: [n(), e_undirected(), n()], ("single-hop", "undirected")),
    Entry("plain single hop, undirected, seeded", lambda: [n({"key": 1}), e_undirected(), n()], ("single-hop", "undirected", "seeded")),
    Entry("typed single hop, seeded", lambda: [n({"key": 1}), e_forward({"type": "KNOWS"}), n()], ("single-hop", "seeded", "typed")),
    Entry("typed single hop, seeded, named", lambda: [n({"key": 1}, name="a"), e_forward({"type": "KNOWS"}, name="e"), n(name="b")], ("single-hop", "seeded", "typed", "alias")),
    Entry("typed single hop, seeded, named, undirected", lambda: [n({"key": 1}, name="a"), e_undirected({"type": "KNOWS"}, name="e"), n(name="b")], ("single-hop", "undirected", "alias")),
    Entry("single hop, node and edge alias share a name", lambda: [n({"key": 1}, name="a"), e_forward(name="a"), n()], ("single-hop", "alias", "shared-alias-name")),
    Entry("single hop, edge alias = filtered column", lambda: [n({"id": 30}, name="m"), e_forward({"type": "KNOWS"}, name="type"), n(name="p")], ("single-hop", "alias-collision", "#2039")),
    Entry("single hop, destination alias = its filtered column", lambda: [n({"id": 30}, name="m"), e_forward({"type": "KNOWS"}, name="e"), n({"type": "p"}, name="type")], ("single-hop", "alias-collision", "#2039")),
    Entry("single hop, source node match", lambda: [n(), e_forward(source_node_match={"type": "p"}), n()], ("single-hop", "endpoint-match")),
    Entry("single hop, prune to endpoints", lambda: [n({"key": 1}), e_forward(prune_to_endpoints=True), n()], ("single-hop", "prune")),
    Entry("hops=2, seeded", lambda: [n({"key": 1}), e_forward(hops=2), n()], ("multi-hop", "seeded")),
    Entry("hops=2, seeded, typed, named", lambda: [n({"key": 1}, name="a"), e_forward({"type": "KNOWS"}, hops=2, name="e"), n(name="b")], ("multi-hop", "typed", "alias", "#2049")),
    Entry("to_fixed_point, seeded", lambda: [n({"key": 1}), e_forward(to_fixed_point=True), n()], ("multi-hop", "fixed-point")),
    Entry("two single hops", lambda: [n({"key": 1}), e_forward(), n(), e_forward(), n()], ("two-steps",)),
]


def tagged(tag: str) -> List[Entry]:
    return [e for e in CORPUS if tag in e.tags]


def by_name() -> Dict[str, Entry]:
    return {e.name: e for e in CORPUS}


register("routes.corpus", [(e.name, e.ops, e.tags) for e in CORPUS], Frames(NODES, EDGES, "key", "s", "d", "eid"))
