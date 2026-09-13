"""Supplemental shape corpus for the chain route engagement matrix.

This matrix measures admission and engagement; it does not replace existing correctness
tests. ``bin/test-routes-off.sh`` broadcasts the existing suites through every route
switch and all routes disabled, retaining their independently written assertions.

Every entry is a native op-list shape variant; a route test filters the corpus with the
route's own admission predicate (the function its dispatcher calls), so one corpus is reused
across hot paths and a shape is never hand-picked per lane. Tags name the defect classes an
entry exercises so coverage can be read per class.
"""
from typing import Callable, Dict, List, NamedTuple, Tuple

import pandas as pd

from graphistry.compute.ast import ASTObject, e_forward, e_reverse, e_undirected, n, rows, select
from graphistry.compute.predicates.numeric import GT
from graphistry.tests.compute.gfql.routes.registry import Frames, register


KeyMap = Callable[[int], object]


class Entry(NamedTuple):
    name: str
    ops: Callable[[], List[ASTObject]]
    tags: Tuple[str, ...]
    shape: Callable[[KeyMap], List[ASTObject]]


def _entry(name: str, shape: Callable[[KeyMap], List[ASTObject]], tags: Tuple[str, ...]) -> Entry:
    """``ops()`` builds the shape over the base frames; ``shape(k)`` maps the node-key literals for a frame variant."""
    return Entry(name, lambda: shape(lambda v: v), tags, shape)


NODES = pd.DataFrame({"key": [1, 2, 3, 4, 5], "id": [10, 20, 30, 40, 50], "type": ["p", "p", "m", "m", "p"], "w": [1, 2, 3, 4, 5]})
EDGES = pd.DataFrame({"s": [1, 1, 2, 3, 3, 4], "d": [2, 3, 3, 1, 1, 5], "type": ["KNOWS", "KNOWS", "LIKES", "KNOWS", "KNOWS", "LIKES"], "eid": [0, 1, 2, 3, 4, 5], "w": [1, 2, 3, 4, 5, 6]})

CORPUS: List[Entry] = [
    _entry("point node rows", lambda k: [n({"key": k(1)}, name="a"), rows(source="a")], ("point-rows", "single-node")),
    _entry("point node projection", lambda k: [n({"key": k(1)}, name="a"), rows(source="a"), select(["key", ("value", "a.w")])], ("point-rows", "single-node", "projection")),
    _entry("point node coalesce", lambda k: [n({"key": k(1)}, name="a"), rows(source="a"), select([("key", "a.key"), ("value", "coalesce(a.w, a.key)")])], ("point-rows", "single-node", "projection")),
    _entry("point joined hop projection", lambda k: [n({"key": k(1)}, name="a"), e_forward({"type": "KNOWS"}), n(name="b"), rows(), select([("key", "b.key"), ("seed", "a.w"), ("tail", "b.w")])], ("point-rows", "single-hop", "projection")),
    _entry("point typed hop rows", lambda k: [n({"key": k(1)}, name="a"), e_forward({"type": "KNOWS"}), n(name="b"), rows(source="b")], ("point-rows", "single-hop")),
    _entry("point typed hop seed rows", lambda k: [n({"key": k(1)}, name="a"), e_forward({"type": "KNOWS"}), n(name="b"), rows(source="a")], ("point-rows", "single-hop")),
    _entry("point reverse hop projection", lambda k: [n({"key": k(1)}, name="a"), e_reverse({"type": "KNOWS"}), n(name="b"), rows(source="b"), select(["key", ("value", "b.w")])], ("point-rows", "single-hop", "reverse", "projection")),

    _entry("single node, scalar filter", lambda k: [n({"id": 30})], ("single-node",)),
    _entry("single node, named", lambda k: [n({"id": 30}, name="a")], ("single-node", "alias")),
    _entry("single node, predicate filter", lambda k: [n({"w": GT(2)})], ("single-node", "predicate")),
    _entry("single node, no filter", lambda k: [n()], ("single-node",)),
    _entry("plain single hop, unseeded", lambda k: [n(), e_forward(), n()], ("single-hop", "unseeded")),
    _entry("plain single hop, seeded", lambda k: [n({"key": k(1)}), e_forward(), n()], ("single-hop", "seeded", "#2051")),
    _entry("plain single hop, seeded, reverse", lambda k: [n({"key": k(1)}), e_reverse(), n()], ("single-hop", "seeded", "reverse")),
    _entry("plain single hop, seeded, destination filter", lambda k: [n({"key": k(1)}), e_forward(), n({"id": 20})], ("single-hop", "seeded", "dest-filter", "#2051")),
    _entry("plain single hop, undirected, unconstrained", lambda k: [n(), e_undirected(), n()], ("single-hop", "undirected")),
    _entry("plain single hop, undirected, seeded", lambda k: [n({"key": k(1)}), e_undirected(), n()], ("single-hop", "undirected", "seeded")),
    _entry("typed single hop, seeded", lambda k: [n({"key": k(1)}), e_forward({"type": "KNOWS"}), n()], ("single-hop", "seeded", "typed")),
    _entry("typed single hop, seeded, named", lambda k: [n({"key": k(1)}, name="a"), e_forward({"type": "KNOWS"}, name="e"), n(name="b")], ("single-hop", "seeded", "typed", "alias")),
    _entry("typed single hop, seeded, named, undirected", lambda k: [n({"key": k(1)}, name="a"), e_undirected({"type": "KNOWS"}, name="e"), n(name="b")], ("single-hop", "undirected", "alias")),
    _entry("single hop, node and edge alias share a name", lambda k: [n({"key": k(1)}, name="a"), e_forward(name="a"), n()], ("single-hop", "alias", "shared-alias-name")),
    _entry("single hop, edge alias = filtered column", lambda k: [n({"id": 30}, name="m"), e_forward({"type": "KNOWS"}, name="type"), n(name="p")], ("single-hop", "alias-collision", "#2039")),
    _entry("single hop, destination alias = its filtered column", lambda k: [n({"id": 30}, name="m"), e_forward({"type": "KNOWS"}, name="e"), n({"type": "p"}, name="type")], ("single-hop", "alias-collision", "#2039")),
    _entry("single hop, source node match", lambda k: [n(), e_forward(source_node_match={"type": "p"}), n()], ("single-hop", "endpoint-match")),
    _entry("single hop, prune to endpoints", lambda k: [n({"key": k(1)}), e_forward(prune_to_endpoints=True), n()], ("single-hop", "prune")),
    _entry("hops=2, seeded", lambda k: [n({"key": k(1)}), e_forward(hops=2), n()], ("multi-hop", "seeded")),
    _entry("hops=2, seeded, typed, named", lambda k: [n({"key": k(1)}, name="a"), e_forward({"type": "KNOWS"}, hops=2, name="e"), n(name="b")], ("multi-hop", "typed", "alias", "#2049")),
    _entry("to_fixed_point, seeded", lambda k: [n({"key": k(1)}), e_forward(to_fixed_point=True), n()], ("multi-hop", "fixed-point")),
    _entry("two single hops", lambda k: [n({"key": k(1)}), e_forward(), n(), e_forward(), n()], ("two-steps",)),
]


def tagged(tag: str) -> List[Entry]:
    return [e for e in CORPUS if tag in e.tags]


def by_name() -> Dict[str, Entry]:
    return {e.name: e for e in CORPUS}


register("routes.corpus", [(e.name, e.ops, e.tags) for e in CORPUS], Frames(NODES, EDGES, "key", "s", "d", "eid"))


def _frame_variants() -> Dict[str, Tuple[Frames, Tuple[str, ...], Callable[[int], object]]]:
    """The same shapes over frames that differ in what the routes must agree on: id dtype,
    null and duplicate ids, self-loops and cycles, an empty edge table, no edge-id binding."""
    str_nodes = NODES.assign(key=NODES["key"].map(lambda k: f"n{k}"))
    str_edges = EDGES.assign(s=EDGES["s"].map(lambda k: f"n{k}"), d=EDGES["d"].map(lambda k: f"n{k}"))
    null_nodes = pd.concat([NODES, pd.DataFrame({"key": [None], "id": [60], "type": ["p"], "w": [6]})], ignore_index=True).astype({"key": "Int64"})
    null_edges = pd.concat([EDGES, pd.DataFrame({"s": [1], "d": [None], "type": ["KNOWS"], "eid": [6], "w": [7]})], ignore_index=True).astype({"s": "Int64", "d": "Int64"})
    dup_nodes = pd.concat([NODES, NODES.iloc[[0]].assign(w=99)], ignore_index=True)
    loop_edges = pd.concat([EDGES, pd.DataFrame({"s": [1, 2], "d": [1, 1], "type": ["KNOWS", "KNOWS"], "eid": [6, 7], "w": [7, 8]})], ignore_index=True)
    return {
        "str-ids": (Frames(str_nodes, str_edges, "key", "s", "d", "eid"), ("dtype-str",), lambda v: f"n{v}"),
        "null-ids": (Frames(null_nodes, null_edges, "key", "s", "d", "eid"), ("null-ids",), lambda v: v),
        "dup-ids": (Frames(dup_nodes, EDGES, "key", "s", "d", "eid"), ("dup-ids",), lambda v: v),
        "self-loop-cycle": (Frames(NODES, loop_edges, "key", "s", "d", "eid"), ("self-loop", "cycle"), lambda v: v),
        "empty-edges": (Frames(NODES, EDGES.iloc[0:0], "key", "s", "d", "eid"), ("empty-edges",), lambda v: v),
        "no-edge-id": (Frames(NODES, EDGES.drop(columns=["eid"]), "key", "s", "d", None), ("no-edge-id",), lambda v: v),
    }


_VARIANT_ROW_TAGS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "dup-ids": {"single node, predicate filter": ("#2034",)},  # node lookup keeps each duplicate row; the general path collapses them
}

for _variant, (_frames, _tags, _key) in _frame_variants().items():
    register(f"routes.corpus.{_variant}", [(e.name, (lambda e=e, k=_key: e.shape(k)), e.tags) for e in CORPUS], _frames,
             tags=_tags + ("variant",), row_tags=_VARIANT_ROW_TAGS.get(_variant))
