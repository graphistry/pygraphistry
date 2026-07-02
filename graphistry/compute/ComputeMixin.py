import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal
from graphistry.Engine import Engine, EngineAbstract, EngineAbstractType, POLARS_ENGINES, resolve_engine, df_to_engine, df_concat, safe_merge
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.utils.json import JSONVal
from .ast import ASTLet, ASTObject
from .chain import Chain, chain as chain_base
from .chain_let import chain_let as chain_let_base
from .gfql_unified import gfql as gfql_base
from .gfql_validate import gfql_validate as gfql_validate_base
from .chain_remote import (
    chain_remote as chain_remote_base,
    chain_remote_shape as chain_remote_shape_base
)
from .python_remote import (
    python_remote_g as python_remote_g_base,
    python_remote_table as python_remote_table_base,
    python_remote_json as python_remote_json_base
)
from graphistry.models.compute.chain_remote import OutputTypeGraph, FormatType
from .collapse import collapse_by
from .hop import hop as hop_base
from .filter_by_dict import (
    filter_edges_by_dict as filter_edges_by_dict_base,
    filter_nodes_by_dict as filter_nodes_by_dict_base
)

logger = setup_logger(__name__)


def _safe_len(df: Any) -> int:
    """
    Safely get length of DataFrame, handling dask_cudf specially to avoid groupby aggregation issues.

    For dask_cudf DataFrames with lazy operations (concat + drop_duplicates), calling len() triggers
    a compute that can fail with "All requested aggregations are unsupported" error. This function
    uses an alternative method for dask_cudf.

    WORKAROUND: This is a workaround for dask_cudf limitations with groupby on empty DataFrames.
    TODO: Remove this function when dask_cudf properly supports len() on DataFrames with lazy operations,
    or when we materialize all dask_cudf DataFrames eagerly (which may have performance implications).
    Monitor: https://github.com/rapidsai/dask-cuda/issues and https://github.com/rapidsai/cudf/issues
    for fixes to groupby aggregation errors on empty DataFrames.
    """
    type_module = type(df).__module__
    if 'dask_cudf' in type_module:
        try:
            import dask_cudf
            if isinstance(df, dask_cudf.DataFrame):
                try:
                    partition_lengths = df.map_partitions(len, meta=pd.Series([], dtype='int64'))
                    total_length = partition_lengths.sum().compute()
                    return int(total_length)
                except Exception as e:
                    logger.warning("Could not compute length for dask_cudf DataFrame via map_partitions: %s", e)
                    return len(df.compute())
        except ImportError as e:
            logger.error("DataFrame type from dask_cudf module but import failed: %s", e)
            raise
        except AttributeError as e:
            logger.error("Imported dask_cudf but attribute error occurred: %s", e)
            raise

    return len(df)


def _coerce_input_formats(g: "Plottable", engine: Engine) -> "Plottable":
    """Coerce input-format types (Arrow, Spark, dask, Polars) to the target engine.

    Engine.PANDAS: polars/arrow/spark/dask → pandas; cuDF and dask_cudf preserved as GPU compute engines.
    Engine.CUDF:   polars/arrow/spark/dask/pandas → cuDF; dask_cudf preserved.

    The 'cudf' substring check catches both cudf.core.dataframe and dask_cudf.core —
    both are GPU compute engines that must not be silently downgraded to pandas.
    """
    def _is_already_correct(df: Any) -> bool:
        type_mod = str(type(df).__module__)
        if engine == Engine.PANDAS:
            return isinstance(df, pd.DataFrame) or 'cudf' in type_mod
        elif engine == Engine.CUDF:
            from graphistry.utils.lazy_import import lazy_cudf_import
            has_cudf, _, _ = lazy_cudf_import()
            if not has_cudf:
                return True  # cudf unavailable — skip coercion; downstream handles gracefully
            return 'cudf' in type_mod
        elif engine in POLARS_ENGINES:
            return 'polars' in type_mod
        return True

    if g._edges is not None and not _is_already_correct(g._edges):
        g = g.edges(df_to_engine(g._edges, engine), g._source, g._destination)
    if g._nodes is not None and not _is_already_correct(g._nodes):
        g = g.nodes(df_to_engine(g._nodes, engine), g._node)
    # A NATIVE-polars input is "already correct" and skips df_to_engine above, so it never
    # gets the NaN->null normalization the pandas->polars path does (pl.from_pandas nan_to_null).
    # Without it, engine='polars' on a frame carrying real NaN keeps rows a filter/aggregation
    # should drop (silent divergence from the pandas oracle, which treats NaN as missing).
    # _pl_nan_to_null is idempotent, so re-running it on a just-converted frame is a no-op.
    if engine in POLARS_ENGINES:
        from graphistry.Engine import _pl_nan_to_null, is_polars_df
        if g._edges is not None and is_polars_df(g._edges):
            g = g.edges(_pl_nan_to_null(g._edges), g._source, g._destination)
        if g._nodes is not None and is_polars_df(g._nodes):
            g = g.nodes(_pl_nan_to_null(g._nodes), g._node)
    return g


def _coerce_to_pandas(g: "Plottable") -> "Plottable":
    """Coerce input formats to pandas. Thin wrapper around _coerce_input_formats(g, PANDAS)."""
    return _coerce_input_formats(g, Engine.PANDAS)


def _degree_agg(edges: Any, key_col: str, out_name: str, node_id: str) -> Any:
    """Groupby edges on key_col, return small (node_id, out_name) frame. Caller handles empty edges."""
    return (
        edges[key_col].value_counts(sort=False)
        .reset_index(name=out_name)
        .rename(columns={key_col: node_id})
    )


class ComputeMixin(Plottable):
    
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
    
    def to_cudf(self) -> 'Plottable':
        """
        Convert to GPU mode by converting any defined nodes and edges to cudf dataframes

        When nodes or edges are already cudf dataframes, they are left as is

        :param g: Graphistry object
        :type g: Plottable

        :return: Graphistry object

        """

        import cudf

        g = self.bind()
        if g._edges is not None:
            if not isinstance(g._edges, cudf.DataFrame):
                if isinstance(g._edges, pd.DataFrame):
                    g = g.edges(cudf.from_pandas(g._edges))
                else:
                    raise ValueError('Expected edges to be pandas, got: {}'.format(type(g._edges)))
                
        if g._nodes is not None:
            if not isinstance(g._nodes, cudf.DataFrame):
                if isinstance(g._nodes, pd.DataFrame):
                    g = g.nodes(cudf.from_pandas(g._nodes))
                else:
                    raise ValueError('Expected nodes to be pandas, got: {}'.format(type(g._nodes)))
        
        return g
                
    def to_pandas(self) -> 'Plottable':
        """Convert nodes and edges to pandas DataFrames.

        Supports all input types: cuDF, Arrow, Polars, Spark, dask, and pandas (identity).
        """
        g = self.bind()
        if g._edges is not None and not isinstance(g._edges, pd.DataFrame):
            g = g.edges(df_to_engine(g._edges, Engine.PANDAS), g._source, g._destination)
        if g._nodes is not None and not isinstance(g._nodes, pd.DataFrame):
            g = g.nodes(df_to_engine(g._nodes, Engine.PANDAS), g._node)
        return g

    def materialize_nodes(
        self,
        reuse: bool = True,
        engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
    ) -> "Plottable":
        """
        Generate g._nodes based on g._edges

        Uses g._node for node id if exists, else 'id'

        Edges must be dataframe-like: cudf, pandas, ...

        When reuse=True and g._nodes is not None, use it

        **Example: Generate nodes**

            ::

                edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
                g = graphistry.edges(edges, 's', 'd')
                print(g._nodes)  # None
                g2 = g.materialize_nodes()
                print(g2._nodes)  # pd.DataFrame

        """

        if isinstance(engine, str):
            engine = EngineAbstract(engine)

        g: Plottable = self

        # Resolve target engine from ORIGINAL data BEFORE coercion, then coerce input formats
        # to that engine. This ensures GPU mode is preserved: polars/arrow/spark/dask are
        # converted to cuDF (not pandas) when engine='cudf'. The old pattern
        # (ensure_local_engine_match then _coerce_to_pandas) was wrong for cross-engine scenarios.
        engine_concrete = resolve_engine(engine, g)
        g = _coerce_input_formats(g, engine_concrete)

        if reuse:
            if g._nodes is not None:
                if g._node is None:
                    logger.warning(
                        "Must set node id binding, not just nodes; set via .bind() or .nodes()"
                    )
                else:
                    return g

        if g._edges is None:
            if reuse and g._nodes is not None and _safe_len(g._nodes) > 0:
                return g
            raise ValueError("Missing edges")
        if g._source is None or g._destination is None:
            raise ValueError(
                "Missing source/destination bindings; set via .bind() or .edges()"
            )
        node_id = g._node if g._node is not None else "id"
        if _safe_len(g._edges) == 0:
            if engine_concrete in POLARS_ENGINES:
                empty_nodes_df = g._edges.select(g._source).rename({g._source: node_id})
            else:
                empty_nodes_df = (
                    g._edges[[g._source]]
                    .rename(columns={g._source: node_id})
                    .reset_index(drop=True)
                )
            return g.nodes(empty_nodes_df, node_id)

        concat_fn = df_concat(engine_concrete)
        concat_df = concat_fn([g._edges[g._source], g._edges[g._destination]])
        if engine_concrete in POLARS_ENGINES:
            # polars Series has no row index and uses .unique() (== pandas drop_duplicates
            # keep-first with maintain_order) + .to_frame(); .drop_duplicates()/.reset_index
            # are pandas-only and raise on a polars Series (edges-only graph under engine='polars').
            nodes_df = concat_df.rename(node_id).unique(maintain_order=True).to_frame()
        else:
            nodes_df = concat_df.rename(node_id).drop_duplicates().to_frame().reset_index(drop=True)
        return g.nodes(nodes_df, node_id)

    def _single_direction_degree(self, key_col: str, col: str) -> "Plottable":
        """Shared body for get_indegrees / get_outdegrees: groupby one direction, merge into nodes."""
        engine_concrete = resolve_engine(EngineAbstract.AUTO, self)
        g = _coerce_input_formats(self, engine_concrete)
        g_nodes = g.materialize_nodes(engine=engine_concrete.value)
        node_id = g_nodes._node
        assert node_id is not None  # materialize_nodes raises otherwise

        if _safe_len(g._edges) == 0:
            if col in g_nodes._nodes.columns:
                return g.nodes(g_nodes._nodes.copy(), node_id)
            nodes_df = g_nodes._nodes.assign(**{col: 0}).astype({col: "int32"})
            return g.nodes(nodes_df, node_id)

        agg = _degree_agg(g._edges, key_col, col, node_id)
        nodes_subset = g_nodes._nodes[[c for c in g_nodes._nodes.columns if c != col]]
        nodes_df = safe_merge(nodes_subset, agg, on=node_id, how='left')
        nodes_df = nodes_df.assign(**{col: nodes_df[col].fillna(0).astype("int32")})
        return g.nodes(nodes_df, node_id)

    def get_indegrees(self, col: str = "degree_in"):
        """See get_degrees"""
        assert self._destination is not None, "Missing destination binding; set via .bind() or .edges()"
        return self._single_direction_degree(self._destination, col)

    def get_outdegrees(self, col: str = "degree_out"):
        """See get_degrees"""
        assert self._source is not None, "Missing source binding; set via .bind() or .edges()"
        return self._single_direction_degree(self._source, col)

    def get_degrees(
        self,
        col: str = "degree",
        degree_in: str = "degree_in",
        degree_out: str = "degree_out",
    ):
        """Decorate nodes table with degree info

        Edges must be dataframe-like: pandas, cudf, ...

        Parameters determine generated column names

        Warning: Self-cycles are currently double-counted. This may change.

        **Example: Generate degree columns**

            ::

                edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
                g = graphistry.edges(edges, 's', 'd')
                print(g._nodes)  # None
                g2 = g.get_degrees()
                print(g2._nodes)  # pd.DataFrame with 'id', 'degree', 'degree_in', 'degree_out'
        """
        engine_concrete = resolve_engine(EngineAbstract.AUTO, self)
        g = _coerce_input_formats(self, engine_concrete)
        g_nodes = g.materialize_nodes(engine=engine_concrete.value)
        node_id = g_nodes._node
        assert node_id is not None  # materialize_nodes raises otherwise
        assert g._source is not None and g._destination is not None  # likewise

        if _safe_len(g._edges) == 0:
            cols = (degree_in, degree_out, col)
            nodes_df = g_nodes._nodes.assign(**{c: 0 for c in cols}).astype({c: "int32" for c in cols})
            return g.nodes(nodes_df, node_id)

        in_df = _degree_agg(g._edges, g._destination, degree_in, node_id)
        out_df = _degree_agg(g._edges, g._source, degree_out, node_id)
        deg = safe_merge(in_df, out_df, on=node_id, how="outer")
        deg = deg.assign(**{
            degree_in: deg[degree_in].fillna(0).astype("int32"),
            degree_out: deg[degree_out].fillna(0).astype("int32"),
        })
        deg = deg.assign(**{col: (deg[degree_in] + deg[degree_out]).astype("int32")})

        keep = [c for c in g_nodes._nodes.columns if c not in (degree_in, degree_out, col)]
        nodes_df = safe_merge(g_nodes._nodes[keep], deg, on=node_id, how="left")
        nodes_df = nodes_df.assign(**{
            degree_in: nodes_df[degree_in].fillna(0).astype("int32"),
            degree_out: nodes_df[degree_out].fillna(0).astype("int32"),
            col: nodes_df[col].fillna(0).astype("int32"),
        })
        return g.nodes(nodes_df, node_id)

    def drop_nodes(self, nodes):
        """
        return g with any nodes/edges involving the node id series removed
        """

        g = self

        if len(nodes) == 0:
            return g

        g2 = g

        if g2._nodes is not None:
            node_hits = g2._nodes[g2._node].isin(nodes)
            if node_hits.any():
                g2 = g2.nodes(g2._nodes[~node_hits])

        src_hits = g2._edges[g2._source].isin(nodes)
        if src_hits.any():
            g2 = g2.edges(g2._edges[~src_hits])

        dst_hits = g2._edges[g2._destination].isin(nodes)
        if dst_hits.any():
            g2 = g2.edges(g2._edges[~dst_hits])

        return g2

    def keep_nodes(self, nodes):
        """
        Limit nodes and edges to those selected by parameter nodes
        For edges, both source and destination must be in nodes
        Nodes can be a list or series of node IDs, or a dictionary
        When a dictionary, each key corresponds to a node column, and nodes will be included when all match
        """
        g = self.materialize_nodes()

        if isinstance(nodes, dict):
            pass
        elif isinstance(nodes, np.ndarray) or isinstance(nodes, list):
            nodes = {g._node: nodes}
        else:
            if isinstance(nodes, pd.Series):
                nodes = {g._node: nodes.to_numpy()}
            else:
                import cudf
                if isinstance(nodes, cudf.Series):
                    nodes = {g._node: nodes.to_numpy()}
                else:
                    raise ValueError('Unexpected nodes type: {}'.format(type(nodes)))
        nodes = {
            k: v if isinstance(v, np.ndarray) or isinstance(v, list) else v.to_numpy()
            for k, v in nodes.items()
        }

        hits = g._nodes[list(nodes.keys())].isin(nodes)
        hits_s = hits[g._node]
        for c in hits.columns:
            if c != g._node:
                hits_s = hits_s & hits[c]
        new_nodes = g._nodes[hits_s]
        new_node_ids = new_nodes[g._node].to_numpy()
        new_edges_hits_df = (
            g._edges[[g._source, g._destination]]
            .isin({
                g._source: new_node_ids,
                g._destination: new_node_ids
            })
        )
        new_edges = g._edges[
            new_edges_hits_df[g._source] & new_edges_hits_df[g._destination]
        ]
        return g.nodes(new_nodes).edges(new_edges)

    def get_topological_levels(
        self,
        level_col: str = "level",
        allow_cycles: bool = True,
        warn_cycles: bool = True,
        remove_self_loops: bool = True,
    ) -> Plottable:
        """
        Label nodes on column level_col based on topological sort depth
        Supports pandas + cudf, using parallelism within each level computation
        Options:
        * allow_cycles: if False and detects a cycle, throw ValueException, else break cycle by picking a lowest-in-degree node
        * warn_cycles: if True and detects a cycle, proceed with a warning
        * remove_self_loops: preprocess by removing self-cycles. Avoids allow_cycles=False, warn_cycles=True messages.

        Example:

        edges_df = gpd.DataFrame({'s': ['a', 'b', 'c', 'd'],'d': ['b', 'c', 'e', 'e']})
        g = graphistry.edges(edges_df, 's', 'd')
        g2 = g.get_topological_levels()
        g2._nodes.info()  # pd.DataFrame with | 'id' , 'level' |

        """
        g2_base = self.materialize_nodes()

        g2 = g2_base
        if (g2._nodes is None) or (_safe_len(g2._nodes) == 0):
            return g2

        g2 = g2.edges(g2._edges.drop_duplicates([g2._source, g2._destination]))
        if remove_self_loops:
            non_self_loops = g2._edges[g2._source] != g2._edges[g2._destination]
            g2 = g2.edges(g2._edges[non_self_loops])

        nodes_with_levels: List[Any] = []
        while True:
            if _safe_len(g2._nodes) == 0:
                break
            g2 = g2.get_degrees()

            roots = g2._nodes[g2._nodes["degree_in"] == 0]
            if len(roots) == 0:
                if not allow_cycles:
                    raise ValueError(
                        "Cyclic graph in get_topological_levels(); remove cycles or set allow_cycles=True"
                    )
                max_degree = g2._nodes["degree"].max()
                roots = g2._nodes[g2._nodes["degree"] == max_degree][:1]
                if warn_cycles:
                    logger.warning(
                        "Cycle on computing level %s", len(nodes_with_levels)
                    )

            nodes_with_levels.append(
                (
                    roots[
                        [
                            c
                            for c in roots
                            if c not in ["degree_in", "degree_out", "degree"]
                        ]
                    ].assign(**{level_col: len(nodes_with_levels)})
                )
            )

            g2 = g2.drop_nodes(roots[g2._node])
        nodes_df0 = nodes_with_levels[0]
        if len(nodes_with_levels) > 1:
            engine = resolve_engine(EngineAbstract.AUTO, nodes_df0)
            concat_fn = df_concat(engine)
            nodes_df = concat_fn([nodes_df0] + nodes_with_levels[1:])
        else:
            nodes_df = nodes_df0

        if self._nodes is None:
            return self.nodes(nodes_df)
        else:
            levels_df = nodes_df[[g2_base._node, level_col]]
            out_df = safe_merge(g2_base._nodes, levels_df, on=g2_base._node, how='left')
            return self.nodes(out_df)

    def search_nodes(self, term, columns=None, case_sensitive=False, regex=False):
        """Keep nodes where ANY column matches ``term`` (viz-filter L2 inspector
        semantics: OR across columns; case-insensitive substring default; regex
        opt-in; string columns always, integer columns iff the term is a numeric
        literal — floats/dates via explicit ``columns=`` on pandas ONLY: cuDF
        declines them, its float/temporal stringification diverges from pandas).
        pandas/cuDF native; polars frames raise NotImplementedError (use the
        cypher ``search_any`` op).
        """
        from graphistry.compute.gfql.search_any import search_any_mask
        from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
        df = self._nodes
        if df is None:
            return self
        if "polars" in type(df).__module__:
            raise NotImplementedError(
                "search_nodes is not yet native on polars frames; use the cypher "
                "search_any op or engine='pandas'")
        mask = search_any_mask(
            df, term, case_sensitive=case_sensitive, regex=regex, columns=columns)
        if mask is None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "search_nodes columns= includes a column absent from the nodes table",
                field="columns", value=columns,
                suggestion="List only columns present on the nodes table.")
        return self.nodes(df[mask])

    def search_edges(self, term, columns=None, case_sensitive=False, regex=False):
        """Keep edges where ANY column matches ``term`` — see :meth:`search_nodes`."""
        from graphistry.compute.gfql.search_any import search_any_mask
        from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
        df = self._edges
        if df is None:
            return self
        if "polars" in type(df).__module__:
            raise NotImplementedError(
                "search_edges is not yet native on polars frames; use the cypher "
                "search_any op or engine='pandas'")
        mask = search_any_mask(
            df, term, case_sensitive=case_sensitive, regex=regex, columns=columns)
        if mask is None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "search_edges columns= includes a column absent from the edges table",
                field="columns", value=columns,
                suggestion="List only columns present on the edges table.")
        return self.edges(df[mask])

    def prune_self_edges(self):
        return self.edges(self._edges[ self._edges[self._source] != self._edges[self._destination] ])

    def collapse(
        self,
        node: Union[str, int],
        attribute: Union[str, int],
        column: Union[str, int],
        self_edges: bool = False,
        unwrap: bool = False,
        verbose: bool = False
    ):
        """
        Topology-aware collapse by given column attribute starting at `node`

        Traverses directed graph from start node `node` and collapses clusters of nodes that share
        the same property so that topology is preserved.

        :param node: start `node` to begin traversal
        :param attribute: the given `attribute` to collapse over within `column`
        :param column: the `column` of nodes DataFrame that contains `attribute` to collapse over
        :param self_edges: whether to include self edges in the collapsed graph
        :param unwrap: whether to unwrap the collapsed graph into a single node
        :param verbose: whether to print out collapse summary information

        :returns:A new Graphistry instance with nodes and edges DataFrame containing collapsed nodes and edges given by column attribute -- nodes and edges DataFrames contain six new columns `collapse_{node | edges}` and `final_{node | edges}`, while original (node, src, dst) columns are left untouched
        :rtype: Plottable
        """
        return collapse_by(
            self,
            start_node=node,
            parent=node,
            attribute=attribute,
            column=column,
            seen={},
            self_edges=self_edges,
            unwrap=unwrap,
            verbose=verbose
        )


    def hop(self, *args, **kwargs):
        return hop_base(self, *args, **kwargs)
    hop.__doc__ = hop_base.__doc__

    # ---- GFQL physical indexes (pay-as-you-go seeded-traversal acceleration) ----
    def create_index(self, kind, *, column=None, name=None, engine='auto'):
        """Build a GFQL physical index for O(degree) seeded traversal.

        :param kind: 'edge_out_adj' (forward hops), 'edge_in_adj' (reverse hops), or 'node_id' (node lookup)
        :param column: column to index (defaults to the binding for the kind, e.g. the edge source column)
        :param name: optional custom index name (defaults to 'kind:column')
        :param engine: 'auto' | 'pandas' | 'cudf' | 'polars' — array backend for the index
        :returns: new Plottable with the index resident (original is untouched)

        **Example**
            ::

                g2 = g.create_index('edge_out_adj')
                g2.gfql([n({'id': 0}), e_forward(hops=2)], index_policy='use')

        See :doc:`gfql/index_adjacency` for policies, cost gating, and benchmarks.
        """
        from graphistry.compute.gfql.index import create_index as _ci
        return _ci(self, kind, column=column, name=name, engine=engine)

    def drop_index(self, kind=None):
        """Drop one resident GFQL index (by kind) or all (kind=None). Idempotent; returns a new Plottable."""
        from graphistry.compute.gfql.index import drop_index as _di
        return _di(self, kind)

    def show_indexes(self):
        """Return a pandas DataFrame describing resident GFQL indexes (name, kind, column, valid). Empty if none; ``valid=False`` marks a stale index after a frame rebind."""
        from graphistry.compute.gfql.index import show_indexes as _si
        return _si(self)

    def gfql_index_edges(self, direction='both', engine='auto'):
        """Convenience: build the edge adjacency index(es) — 'forward', 'reverse', or 'both'. Returns a new Plottable."""
        from graphistry.compute.gfql.index import gfql_index_edges as _gie
        return _gie(self, direction, engine=engine)

    def gfql_index_all(self, engine='auto'):
        """Convenience: build all GFQL physical indexes (both edge adjacencies + node_id). Returns a new Plottable."""
        from graphistry.compute.gfql.index import gfql_index_all as _gia
        return _gia(self, engine=engine)

    def filter_nodes_by_dict(self, *args, **kwargs):
        return filter_nodes_by_dict_base(self, *args, **kwargs)
    filter_nodes_by_dict.__doc__ = filter_nodes_by_dict_base.__doc__

    def filter_edges_by_dict(self, *args, **kwargs):
        return filter_edges_by_dict_base(self, *args, **kwargs)
    filter_edges_by_dict.__doc__ = filter_edges_by_dict_base.__doc__

    def chain(self, *args, **kwargs):
        """
        .. deprecated:: 2.XX.X
           Use :meth:`gfql` instead for a unified API that supports both chains and DAGs.
        """
        import warnings
        warnings.warn(
            "chain() is deprecated. Use gfql() instead for a unified API.",
            DeprecationWarning,
            stacklevel=2
        )
        return chain_base(self, *args, **kwargs)
    chain.__doc__ = (chain.__doc__ or "") + "\n\n" + (chain_base.__doc__ or "")
    
    def gfql(self, *args, **kwargs):
        policy = kwargs.pop('index_policy', None)
        # Route GFQL index DDL (Python wire op, JSON dict, or Cypher string) to the
        # registry without touching the traversal executor.
        query = args[0] if args else kwargs.get('query')
        from graphistry.compute.gfql.index.wire import (
            is_index_op, is_index_op_json, index_op_from_json, apply_index_op,
        )
        from graphistry.compute.gfql.index.cypher_ddl import parse_index_ddl
        op = None
        if is_index_op(query):
            op = query
        elif is_index_op_json(query):
            op = index_op_from_json(query)
        elif isinstance(query, str):
            op = parse_index_ddl(query)
        if op is not None:
            return apply_index_op(self, op, engine=kwargs.get('engine', 'auto'))

        g = self
        if policy is not None:
            import copy as _copy
            g = _copy.copy(self)
            g._gfql_index_policy = policy
        return gfql_base(g, *args, **kwargs)
    gfql.__doc__ = (gfql_base.__doc__ or "") + """

        **GFQL physical indexes**

        :param index_policy: 'off' (never use indexes), 'use' (use resident,
            cost-gated; default planner behavior), 'auto' (build on demand), or
            'force' (always probe the index). Also accepts index DDL strings
            (``CREATE GFQL INDEX ...``) / wire ops as the query — routed to the
            index registry. See :meth:`create_index` and :doc:`gfql/index_adjacency`.
    """

    def gfql_explain(self, query, *, index_policy='use', engine='auto'):
        """Explain how the GFQL planner would run ``query``: per-hop index-vs-scan choice, cost-gate numbers, and resident-index validity. Read-only (no execution). Returns a report object; print it for a human-readable plan."""
        from graphistry.compute.gfql.index.explain import gfql_explain as _ge
        return _ge(self, query, index_policy=index_policy, engine=engine)

    def gfql_validate(self, *args, **kwargs):
        return gfql_validate_base(self, *args, **kwargs)
    gfql_validate.__doc__ = gfql_validate_base.__doc__

    def chain_remote(self, *args, **kwargs) -> Plottable:
        """
        .. deprecated:: 2.XX.X
           Use :meth:`gfql_remote` instead for a unified API that supports both chains and DAGs.
        """
        import warnings
        warnings.warn(
            "chain_remote() is deprecated. Use gfql_remote() instead for a unified API.",
            DeprecationWarning,
            stacklevel=2
        )
        return chain_remote_base(self, *args, **kwargs)
    chain_remote.__doc__ = (chain_remote.__doc__ or "") + "\n\n" + (chain_remote_base.__doc__ or "")

    def chain_remote_shape(self, *args, **kwargs) -> pd.DataFrame:
        """
        .. deprecated:: 2.XX.X
           Use :meth:`gfql_remote_shape` instead for a unified API that supports both chains and DAGs.
        """
        import warnings
        warnings.warn(
            "chain_remote_shape() is deprecated. Use gfql_remote_shape() instead for a unified API.",
            DeprecationWarning,
            stacklevel=2
        )
        return chain_remote_shape_base(self, *args, **kwargs)
    chain_remote_shape.__doc__ = (chain_remote_shape.__doc__ or "") + "\n\n" + (chain_remote_shape_base.__doc__ or "")

    def gfql_remote(
        self,
        chain: Union[Chain, List[ASTObject], ASTLet, Dict[str, JSONVal], str],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        output_type: OutputTypeGraph = "all",
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: EngineAbstractType = 'auto',
        validate: bool = True,
        persist: bool = False,
        params: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
    ) -> Plottable:
        """Run GFQL query remotely.

        This is the remote execution version of :meth:`gfql`. It supports chains,
        Let/DAG patterns, and Cypher strings.

        The query is compiled locally and sent to the server as wire-protocol
        JSON. A ``gfql_query`` field carries the full typed envelope (including
        WHERE clauses); ``gfql_operations`` carries a flat array for backward
        compatibility with older servers.

        :param chain: GFQL query — Chain, List[ASTObject], ASTLet, Dict, or
            Cypher string (compiled locally before sending).
        :param params: Optional parameter dict for Cypher string queries
            (e.g., ``params={"val": 10}`` for ``$val`` references).

        Example::

            # Chain (existing)
            g.gfql_remote([n(), e(), n()])

            # Cypher string with params
            g.gfql_remote(
                "MATCH (n) WHERE n.score > $cutoff RETURN n",
                params={"cutoff": 10},
            )

            # GRAPH constructor
            g.gfql_remote("GRAPH { MATCH (a)-[r]->(b) WHERE a.score > 5 }")

        See :meth:`chain_remote` for additional parameter documentation.
        """
        return chain_remote_base(
            self, chain, api_token, dataset_id, output_type, format,
            df_export_args, node_col_subset, edge_col_subset, engine, validate, persist,
            params=params, output=output,
        )
    
    def gfql_remote_shape(
        self,
        chain: Union[Chain, List[ASTObject], ASTLet, Dict[str, JSONVal], str],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: EngineAbstractType = 'auto',
        validate: bool = True,
        persist: bool = False
    ) -> pd.DataFrame:
        """Get shape metadata for remote GFQL query execution.

        This is the remote shape version of :meth:`gfql`. Returns metadata about the
        resulting graph without downloading the full data.

        See :meth:`chain_remote_shape` for detailed documentation (chain_remote_shape is deprecated).
        """
        return chain_remote_shape_base(
            self, chain, api_token, dataset_id, format, df_export_args,
            node_col_subset, edge_col_subset, engine, validate, persist
        )

    def python_remote_g(self, *args, **kwargs) -> Any:
        return python_remote_g_base(self, *args, **kwargs)
    python_remote_g.__doc__ = python_remote_g_base.__doc__

    def python_remote_table(self, *args, **kwargs) -> Any:
        return python_remote_table_base(self, *args, **kwargs)
    python_remote_table.__doc__ = python_remote_table_base.__doc__

    def python_remote_json(self, *args, **kwargs) -> Any:
        return python_remote_json_base(self, *args, **kwargs)
    python_remote_json.__doc__ = python_remote_json_base.__doc__
