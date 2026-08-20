"""Registry of `graphistry.std.*` procedures for the GFQL CALL surface.

`std` in the standard-library sense: these need no optional third-party
dependency and run on whatever engine the frames already use. That is the
distinction from `graphistry.cugraph.*` / `graphistry.igraph.*` / `graphistry.nx.*`,
which are named for the backend library doing the work and require it installed.

Deliberately NOT named after the external workload that motivated them. "Graphalytics"
is LDBC's external workload name; putting it in a shipped public API would imply an
endorsement and a conformance audit we do not have, and would read oddly for a
user who just wants label propagation. Conformance belongs in the docs and
tests, not the identifier.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Tuple

from graphistry.compute.typing import DataFrameT, SeriesT

from . import kernels as _k

# Algorithm -> output column and default options, following plugin naming.
STD_ALGS: Dict[str, Tuple[str, Dict[str, object]]] = {
    "wcc": ("component", {}),
    "pagerank": ("pagerank", {"iterations": 10, "damping": 0.85}),
    "cdlp": ("cdlp", {"iterations": 10}),
    "sssp": ("distance", {}),
    "mis": ("mis", {"seed": 0x5EED}),
}

STD_COMPUTE_ALGS: Tuple[str, ...] = tuple(STD_ALGS)

# Result dtypes, for the GFQL planner's schema effects.
STD_FLOAT64_ALGS = frozenset({"pagerank"})
STD_FLOAT32_ALGS = frozenset({"sssp"})
STD_LABEL_ALGS = frozenset({"wcc", "cdlp"})
STD_BOOL_ALGS = frozenset({"mis"})


_STD_RUNNERS: Dict[str, Callable[..., SeriesT]] = {
    "wcc": _k.wcc,
    "pagerank": _k.pagerank,
    "cdlp": _k.cdlp,
    "sssp": _k.sssp,
    "mis": _k.mis,
}


def _dispatch(alg: str) -> Callable[..., SeriesT]:
    return _STD_RUNNERS[alg]


def run(edges: DataFrameT, src: str, dst: str, v_count: int, alg: str, options: Mapping[str, object] | None = None) -> SeriesT:
    """Run a std kernel, merging caller options over the defaults."""
    if alg not in STD_ALGS:
        raise KeyError(f"unknown graphistry.std procedure {alg!r}; known: {list(STD_ALGS)}")
    _out_col, defaults = STD_ALGS[alg]
    opts = {**defaults, **dict(options or {})}
    fn = _dispatch(alg)

    if alg == "sssp":
        # Weight and source are SSSP parameters rather than graph-wide state.
        weight = opts.pop("weight", None)
        if weight is None:
            weight = "__std_w"
            edges = edges.assign(**{weight: _k.make_weights(edges, src, dst)}) if hasattr(edges, "assign") else edges
        source = opts.pop("source", 0)
        return fn(edges, src, dst, weight, v_count, source=source, **opts)

    return fn(edges, src, dst, v_count, **opts)


def output_column(alg: str) -> str:
    return STD_ALGS[alg][0]
