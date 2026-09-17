"""GFQL adapter for running a std kernel and binding its node result.

The kernels require dense int32 vertex ids 0..V-1 (that is what makes their
frontier expansion a positional gather rather than a hash join). Real graphs
have arbitrary ids, so this layer renumbers, runs, and maps back -- the same
service `compute_cugraph` performs for cuGraph's own renumbering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

from graphistry.compute.typing import DataFrameT, SeriesT

from ._dfops import arange, dense_renumber, df_cons, full, to_host_int
from .registry import STD_ALGS, output_column, run as _run_alg

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

# Algorithms whose output values are themselves vertex ids.
_LABEL_IS_VERTEX_ID = frozenset({"wcc", "cdlp"})


def _pagerank_vertex_values(
    value: object,
    ids: SeriesT,
    template: DataFrameT,
    value_column: str,
    name: str,
    fill_missing: bool,
) -> SeriesT:
    if isinstance(value, Mapping):
        frame = df_cons(
            template,
            {"vertex": list(value.keys()), value_column: list(value.values())},
        )
    elif isinstance(value, (list, tuple)) or hasattr(value, "columns"):
        frame = type(template)(value)
    else:
        raise ValueError(
            f"PageRank {name} must be a vertex-to-value map, record list, or dataframe"
        )
    required = {"vertex", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"PageRank {name} requires columns {sorted(required)}, got {list(frame.columns)}"
        )
    frame = frame[["vertex", value_column]].reset_index(drop=True)
    if to_host_int(frame["vertex"].duplicated().sum()) != 0:
        raise ValueError(f"PageRank {name} contains duplicate vertices")

    lookup = df_cons(
        template,
        {
            "vertex": ids.reset_index(drop=True),
            "__dense": arange(template, len(ids), "int32"),
        },
    )
    tagged = frame.merge(lookup, on="vertex", how="left")
    unknown = to_host_int(tagged["__dense"].isna().sum())
    if unknown:
        raise ValueError(
            f"PageRank {name} contains {unknown} unknown graph vertex value(s)"
        )
    aligned = lookup.merge(frame, on="vertex", how="left").sort_values("__dense")
    out = aligned[value_column].reset_index(drop=True).astype("float64")
    return out.fillna(0.0) if fill_missing else out


def compute_std(
    self: "Plottable",
    alg: str,
    out_col: Optional[str] = None,
    params: Optional[Mapping[str, object]] = None,
) -> "Plottable":
    """Run `alg` and return a new Plottable with the result bound to nodes."""
    if alg not in STD_ALGS:
        raise ValueError(
            f"unknown graphistry.std procedure {alg!r}; known: {sorted(STD_ALGS)}"
        )
    g = self.materialize_nodes()
    src, dst, node = g._source, g._destination, g._node
    if src is None or dst is None or node is None:
        raise ValueError(
            "graphistry.std requires source, destination, and node bindings"
        )
    options = dict(params or {})
    converged_col: Optional[str] = None
    if alg == "pagerank":
        converged_value = options.pop("converged_col", None)
        if converged_value is not None:
            if not isinstance(converged_value, str) or not converged_value:
                raise ValueError(
                    "PageRank converged_col must be a non-empty string"
                )
            if options.get("fail_on_nonconvergence", True) is not False:
                raise ValueError(
                    "PageRank converged_col requires fail_on_nonconvergence=False"
                )
            converged_col = converged_value
        if options.get("weight") is None and g._edge_weight is not None:
            options["weight"] = g._edge_weight

    edge_columns = [src, dst]
    if alg in {"sssp", "pagerank"} and options.get("weight") is not None:
        weight = options["weight"]
        if not isinstance(weight, str) or weight not in g._edges.columns:
            raise ValueError(
                f"{alg} weight must name an edge column, got {weight!r}"
            )
        edge_columns.append(weight)
    edges = g._edges[edge_columns]
    if alg == "mis":
        edges = edges[edges[src] != edges[dst]].reset_index(drop=True)

    dense, ids, v_count = dense_renumber(edges, src, dst, g._nodes[node])
    if alg == "sssp" and "source" in options:
        source = options["source"]
        try:
            dense_source = to_host_int(ids.searchsorted(source))
        except (TypeError, ValueError):
            dense_source = -1
        if (
            dense_source < 0
            or dense_source >= v_count
            or ids.iloc[dense_source] != source
        ):
            raise ValueError(f"SSSP source {source!r} is not a graph node")
        options["source"] = dense_source
    if alg == "pagerank":
        for name, value_column, fill_missing in (
            ("personalization", "values", True),
            ("nstart", "values", True),
            ("precomputed_vertex_out_weight", "sums", False),
        ):
            value = options.get(name)
            if value is not None:
                options[name] = _pagerank_vertex_values(
                    value, ids, dense, value_column, name, fill_missing
                )

    outcome = _run_alg(dense, src, dst, v_count, alg, options)
    converged: Optional[bool] = None
    if isinstance(outcome, tuple):
        result, converged = outcome
    else:
        result = outcome

    # WCC/CDLP labels are vertex IDs and must return to the caller ID space.
    if alg in _LABEL_IS_VERTEX_ID:
        result = ids.reset_index(drop=True).take(result.reset_index(drop=True))

    col = out_col or output_column(alg)
    if converged_col in {node, col}:
        raise ValueError(
            "PageRank converged_col must differ from node and result columns"
        )
    mapping_data = {node: ids, col: result.reset_index(drop=True)}
    if converged_col is not None:
        if converged is None:
            raise AssertionError("PageRank convergence status was not returned")
        mapping_data[converged_col] = full(dense, v_count, converged, "bool")
    mapping = type(g._nodes)(mapping_data)

    nodes = g._nodes
    replace_columns = [
        name for name in (col, converged_col) if name in nodes.columns
    ]
    if replace_columns:
        nodes = nodes.drop(columns=replace_columns)
    return g.nodes(nodes.merge(mapping, on=node, how="left"))
