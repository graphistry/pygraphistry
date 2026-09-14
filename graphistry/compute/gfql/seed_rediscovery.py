"""Seeds an undirected wavefront legitimately RE-ENCOUNTERS, computed in the frame engine.

``return_as_wave_front=True`` documents "exclude starting node(s) in return, returning only
encountered nodes". A seed counts as encountered only when a walk that never REUSES AN EDGE
arrives back at it -- coming back along the edge you left by is the trip home, not a
discovery. On the acyclic path ``a-b-c-d-e`` seeded at ``{a}`` the answer excludes ``a``
(returning needs edge ``ab`` twice); seeded at ``{a, b}`` it INCLUDES ``a``, which seed
``b`` reaches over ``ab`` on a one-edge walk that reuses nothing.

A seed ``s`` is re-encountered by an edge-disjoint walk of SOME length iff either

  (A) its component holds another seed -- the shortest path between two seeds is simple,
      hence reuses no edge; or
  (B) ``s`` lies on a cycle -- go around it.

(A) is answered by a multi-source label propagation from the seeds followed by a union
of the seed labels that meet across an edge; (B) by peeling nodes of degree <= 1 until
none remain (the multigraph 2-core) plus self-loops. Degree counts EDGE ROWS, not distinct
neighbours, so two PARALLEL edges between ``u`` and ``v`` leave both on a length-2 cycle.
Both rules run as joins and group-bys on the caller's own frames (pandas, cuDF, polars),
so nothing crosses to the host. NOTE the rule is LENGTH-BLIND: it answers "some length",
so a bounded hop whose window is shorter than the cycle back to the seed still keeps that
seed; both arms share that limit. A NULL endpoint is not an identity and is ignored.
Both rules iterate to a fixed point (BFS levels from the seeds; leaf-peeling rounds), so a
long pendant path costs one round per edge along it; scale behaviour is measured in
pyg-bench, not asserted here.
"""

from graphistry.compute.dataframe_utils import concat_frames, df_cons
from graphistry.compute.typing import DataFrameT

LABEL = "__seed_label__"
ROOT = "__seed_root__"
RANK = "__rank__"
DEG = "__deg__"


def _ids(frame: DataFrameT, col: str, out: str) -> DataFrameT:
    return frame[[col]].rename(columns={col: out}).dropna().drop_duplicates()


def _drop_ids(frame: DataFrameT, col: str, ids: DataFrameT, ids_col: str) -> DataFrameT:
    return frame[~frame[col].isin(ids[ids_col])]


def _labels_reached(edges: DataFrameT, src: str, dst: str, frontier: DataFrameT, id_col: str) -> DataFrameT:
    """(id, LABEL) rows for every neighbour of the frontier, over both edge directions."""
    out_step = edges.merge(frontier, left_on=src, right_on=id_col)[[dst, LABEL]].rename(columns={dst: id_col})
    in_step = edges.merge(frontier, left_on=dst, right_on=id_col)[[src, LABEL]].rename(columns={src: id_col})
    step = concat_frames([out_step, in_step])
    return out_step.iloc[:0] if step is None else step


def _seeds_sharing_a_component(edges: DataFrameT, src: str, dst: str, seeds: DataFrameT,
                               id_col: str) -> DataFrameT:
    """Rule A: seeds whose component holds another seed, as a one-column frame."""
    labeled = seeds.assign(**{LABEL: seeds[id_col]})
    frontier = labeled
    while len(frontier) > 0:
        step = _labels_reached(edges, src, dst, frontier, id_col)
        step = _drop_ids(step, id_col, labeled, id_col).drop_duplicates(subset=[id_col])
        if len(step) == 0:
            break
        labeled = concat_frames([labeled, step])
        frontier = step
    left = edges.merge(labeled, left_on=src, right_on=id_col)[[dst, LABEL]].rename(columns={LABEL: ROOT})
    pairs = left.merge(labeled, left_on=dst, right_on=id_col)[[ROOT, LABEL]]
    pairs = pairs[pairs[ROOT] != pairs[LABEL]].drop_duplicates()
    roots = _union_roots(seeds, id_col, pairs)
    counts = roots.groupby(ROOT).size().rename(DEG).reset_index()
    shared = roots.merge(counts, on=ROOT)
    return shared[shared[DEG] > 1][[id_col]]


def _union_roots(seeds: DataFrameT, id_col: str, pairs: DataFrameT) -> DataFrameT:
    """Root of every seed under the unions ``pairs`` (ROOT, LABEL): hook the larger root
    under the smaller, pointer-jump, until nothing moves. Ranks stand in for ids so the
    ordering is over integers whatever the id dtype."""
    rank = seeds[[id_col]].reset_index(drop=True).reset_index().rename(columns={"index": RANK})
    parent = rank.assign(**{ROOT: rank[RANK]})[[id_col, RANK, ROOT]]
    a_side = pairs.merge(rank, left_on=ROOT, right_on=id_col)[[LABEL, RANK]].rename(columns={RANK: "__a__"})
    ab = a_side.merge(rank, left_on=LABEL, right_on=id_col)[["__a__", RANK]].rename(columns={RANK: "__b__"})
    while len(ab) > 0:
        roots_of = parent[[RANK, ROOT]]
        ra = ab.merge(roots_of, left_on="__a__", right_on=RANK)[["__b__", ROOT]].rename(columns={ROOT: "__ra__"})
        rab = ra.merge(roots_of, left_on="__b__", right_on=RANK)[["__ra__", ROOT]].rename(columns={ROOT: "__rb__"})
        differ = rab[rab["__ra__"] != rab["__rb__"]]
        if len(differ) == 0:
            break
        hooks = df_cons(parent, {
            "__hi__": differ[["__ra__", "__rb__"]].max(axis=1),
            "__lo__": differ[["__ra__", "__rb__"]].min(axis=1),
        }).groupby("__hi__").min().reset_index()
        merged = parent.merge(hooks, left_on=ROOT, right_on="__hi__", how="left")
        rerooted = merged["__lo__"].fillna(merged[ROOT]).astype(parent[ROOT].dtype)
        parent = merged.assign(**{ROOT: rerooted})[[id_col, RANK, ROOT]]
        while True:
            jump_to = parent[[RANK, ROOT]].rename(columns={RANK: "__j__", ROOT: "__jr__"})
            jumped = parent.merge(jump_to, left_on=ROOT, right_on="__j__")
            if bool((jumped["__jr__"] == jumped[ROOT]).all()):
                break
            parent = jumped.assign(**{ROOT: jumped["__jr__"]})[[id_col, RANK, ROOT]]
    return parent[[id_col, ROOT]]


def _cycle_node_ids(edges: DataFrameT, src: str, dst: str, id_col: str) -> DataFrameT:
    """Rule B: nodes on a self-loop or in the multigraph 2-core, as a one-column frame."""
    loops = _ids(edges[edges[src] == edges[dst]], src, id_col)
    alive = edges[edges[src] != edges[dst]][[src, dst]]
    while len(alive) > 0:
        ends = concat_frames([_rename(alive, src, id_col), _rename(alive, dst, id_col)])
        assert ends is not None
        degree = ends.groupby(id_col).size().rename(DEG).reset_index()
        leaves = degree[degree[DEG] <= 1][[id_col]]
        if len(leaves) == 0:
            break
        alive = _drop_ids(_drop_ids(alive, src, leaves, id_col), dst, leaves, id_col)
    core = concat_frames([_ids(alive, src, id_col), _ids(alive, dst, id_col), loops])
    return loops if core is None else core.drop_duplicates()


def _rename(frame: DataFrameT, col: str, out: str) -> DataFrameT:
    return frame[[col]].rename(columns={col: out})


def rediscovered_seed_ids(edges: DataFrameT, src: str, dst: str, seeds: DataFrameT, id_col: str
                          ) -> DataFrameT:
    """The seeds (one column ``id_col``) an undirected wavefront over ``edges`` re-encounters."""
    seed_ids = _ids(seeds, id_col, id_col)
    edges = edges[[src, dst]].dropna()
    if len(seed_ids) == 0 or len(edges) == 0:
        return seed_ids.iloc[:0]
    shared = _seeds_sharing_a_component(edges, src, dst, seed_ids, id_col)
    on_cycle = _cycle_node_ids(edges, src, dst, id_col)
    kept = concat_frames([shared, on_cycle])
    if kept is None:
        return seed_ids.iloc[:0]
    return seed_ids[seed_ids[id_col].isin(kept[id_col])]
