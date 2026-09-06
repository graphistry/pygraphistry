"""Shape registry for the route harness.

Each specialization's own test module registers the shape table it was written against
(``register(...)`` returns the table unchanged, so the module keeps using it), with the
frames those shapes address and the defect classes they exercise. The harness in
``test_route_harness.py`` then tries every registered shape against every route whose
admission predicate admits it, so one input is exercised by several hot paths, not one.
"""
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import pandas as pd
import pytest

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject

Build = Callable[[], List[ASTObject]]
Row = Union[Tuple[str, Build], Tuple[str, Build, Tuple[str, ...]]]


class Frames(NamedTuple):
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node: str
    src: str
    dst: str
    edge: Optional[str] = None


class Shape(NamedTuple):
    table: str
    label: str
    build: Build
    frames: Frames
    tags: Tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.table}/{self.label}"


REGISTRY: Dict[str, Shape] = {}


def register(table: str, rows: Sequence[Row], frames: Frames, tags: Iterable[str] = (),
             row_tags: Optional[Dict[str, Tuple[str, ...]]] = None) -> Sequence[Row]:
    """Register ``rows`` ((label, build[, tags]) ...) under ``table``; returns ``rows``."""
    base = tuple(tags)
    for row in rows:
        label, build = row[0], row[1]
        extra = tuple(row[2]) if len(row) > 2 else ()
        extra += (row_tags or {}).get(label, ())
        shape = Shape(table, label, build, frames, base + extra)
        REGISTRY.setdefault(shape.name, shape)
    return rows


def to_engine(df: pd.DataFrame, engine: str):
    if engine == "pandas":
        return df
    if engine == "cudf":
        return pytest.importorskip("cudf").from_pandas(df)
    if engine == "polars":
        return pytest.importorskip("polars").from_pandas(df)
    raise ValueError(engine)


def graph_for(shape: Shape, engine: str, indexed: bool = False) -> Plottable:
    import graphistry
    f = shape.frames
    g = graphistry.nodes(to_engine(f.nodes, engine), f.node).edges(to_engine(f.edges, engine), f.src, f.dst, f.edge)
    return g.gfql_index_all(engine=engine) if indexed else g
