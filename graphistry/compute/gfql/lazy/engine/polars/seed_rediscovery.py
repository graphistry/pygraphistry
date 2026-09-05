"""Polars twin of ``graphistry.compute.gfql.seed_rediscovery``: same rule, polars joins.

See that module for the rule (a seed survives an undirected wavefront iff another seed
shares its component, or it lies on a cycle: self-loops plus the multigraph 2-core).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

LABEL = "__seed_label__"
ROOT = "__seed_root__"
RANK = "__rank__"
DEG = "__deg__"


def _ids(frame: "pl.DataFrame", col: str, out: str) -> "pl.DataFrame":
    return frame.select(col).rename({col: out}).drop_nulls().unique()


def _seeds_sharing_a_component(edges: "pl.DataFrame", src: str, dst: str, seeds: "pl.DataFrame",
                               id_col: str) -> "pl.DataFrame":
    import polars as pl

    labeled = seeds.with_columns(pl.col(id_col).alias(LABEL))
    frontier = labeled
    while frontier.height > 0:
        out_step = edges.join(frontier, left_on=src, right_on=id_col).select(pl.col(dst).alias(id_col), LABEL)
        in_step = edges.join(frontier, left_on=dst, right_on=id_col).select(pl.col(src).alias(id_col), LABEL)
        step = pl.concat([out_step, in_step]).join(labeled, on=id_col, how="anti").unique(subset=[id_col])
        if step.height == 0:
            break
        labeled = pl.concat([labeled, step])
        frontier = step
    left = edges.join(labeled, left_on=src, right_on=id_col).select(dst, pl.col(LABEL).alias(ROOT))
    pairs = left.join(labeled, left_on=dst, right_on=id_col).select(ROOT, LABEL)
    pairs = pairs.filter(pl.col(ROOT) != pl.col(LABEL)).unique()
    roots = _union_roots(seeds, id_col, pairs)
    counts = roots.group_by(ROOT).len(name=DEG)
    return roots.join(counts, on=ROOT).filter(pl.col(DEG) > 1).select(id_col)


def _union_roots(seeds: "pl.DataFrame", id_col: str, pairs: "pl.DataFrame") -> "pl.DataFrame":
    import polars as pl

    rank = seeds.select(id_col).with_row_index(RANK)
    parent = rank.with_columns(pl.col(RANK).alias(ROOT))
    ab = (pairs.join(rank, left_on=ROOT, right_on=id_col).select(LABEL, pl.col(RANK).alias("__a__"))
          .join(rank, left_on=LABEL, right_on=id_col).select("__a__", pl.col(RANK).alias("__b__")))
    roots_of = parent.select(RANK, ROOT)
    while ab.height > 0:
        rab = (ab.join(roots_of, left_on="__a__", right_on=RANK).rename({ROOT: "__ra__"})
               .join(roots_of, left_on="__b__", right_on=RANK).rename({ROOT: "__rb__"}))
        differ = rab.filter(pl.col("__ra__") != pl.col("__rb__"))
        if differ.height == 0:
            break
        hooks = (differ.select(pl.max_horizontal("__ra__", "__rb__").alias("__hi__"),
                               pl.min_horizontal("__ra__", "__rb__").alias("__lo__"))
                 .group_by("__hi__").min())
        parent = (parent.join(hooks, left_on=ROOT, right_on="__hi__", how="left")
                  .with_columns(pl.coalesce("__lo__", ROOT).alias(ROOT)).select(id_col, RANK, ROOT))
        while True:
            jumped = parent.join(parent.select(pl.col(RANK).alias("__j__"), pl.col(ROOT).alias("__jr__")),
                                 left_on=ROOT, right_on="__j__")
            if jumped.select((pl.col("__jr__") == pl.col(ROOT)).all()).item():
                break
            parent = jumped.with_columns(pl.col("__jr__").alias(ROOT)).select(id_col, RANK, ROOT)
        roots_of = parent.select(RANK, ROOT)
    return parent.select(id_col, ROOT)


def _cycle_node_ids(edges: "pl.DataFrame", src: str, dst: str, id_col: str) -> "pl.DataFrame":
    import polars as pl

    loops = _ids(edges.filter(pl.col(src) == pl.col(dst)), src, id_col)
    alive = edges.filter(pl.col(src) != pl.col(dst)).select(src, dst)
    while alive.height > 0:
        ends = pl.concat([alive.select(pl.col(src).alias(id_col)), alive.select(pl.col(dst).alias(id_col))])
        leaves = ends.group_by(id_col).len(name=DEG).filter(pl.col(DEG) <= 1).select(id_col)
        if leaves.height == 0:
            break
        alive = (alive.join(leaves, left_on=src, right_on=id_col, how="anti")
                 .join(leaves, left_on=dst, right_on=id_col, how="anti"))
    return pl.concat([_ids(alive, src, id_col), _ids(alive, dst, id_col), loops]).unique()


def rediscovered_seed_ids(edges: "pl.DataFrame", src: str, dst: str, seeds: "pl.DataFrame", id_col: str
                          ) -> "pl.DataFrame":
    """The seeds (one column ``id_col``) an undirected wavefront over ``edges`` re-encounters."""
    import polars as pl

    seed_ids = _ids(seeds, id_col, id_col)
    edges = edges.select(src, dst).drop_nulls()
    edge_dtype = edges.schema[src]
    if seed_ids.schema[id_col] != edge_dtype and edge_dtype.is_numeric() and seed_ids.schema[id_col].is_numeric():
        seed_ids = seed_ids.with_columns(pl.col(id_col).cast(edge_dtype))  # a narrower id column
    if seed_ids.height == 0 or edges.height == 0:
        return seed_ids.clear()
    kept = pl.concat([
        _seeds_sharing_a_component(edges, src, dst, seed_ids, id_col),
        _cycle_node_ids(edges, src, dst, id_col),
    ], how="vertical_relaxed")
    return seed_ids.join(kept, on=id_col, how="semi")
