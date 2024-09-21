import numpy as np, pandas as pd
from typing import Any, List, Union, TYPE_CHECKING
from inspect import getmodule

from graphistry.Engine import Engine, EngineAbstract
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .chain import chain as chain_base
from .collapse import collapse_by
from .hop import hop as hop_base
from .filter_by_dict import (
    filter_edges_by_dict as filter_edges_by_dict_base,
    filter_nodes_by_dict as filter_nodes_by_dict_base
)

logger = setup_logger(__name__)

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


class ComputeMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        pass

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

        g = self
        if g._edges is None:
            raise ValueError("Missing edges")
        if g._source is None or g._destination is None:
            raise ValueError(
                "Missing source/destination bindings; set via .bind() or .edges()"
            )
        if len(g._edges) == 0:
            return g
        # TODO use built-ins for igraph/nx/...

        if reuse:
            if g._nodes is not None and len(g._nodes) > 0:
                if g._node is None:
                    logger.warning(
                        "Must set node id binding, not just nodes; set via .bind() or .nodes()"
                    )
                    # raise ValueError('Must set node id binding, not just nodes; set via .bind() or .nodes()')
                else:
                    return g

        node_id = g._node if g._node is not None else "id"
        engine_concrete : Engine
        if engine == EngineAbstract.AUTO:
            if isinstance(g._edges, pd.DataFrame):
                engine_concrete = Engine.PANDAS
            else:

                def raiser(df: Any):
                    raise ValueError('Could not determine engine for edges, expected pandas or cudf dataframe, got: {}'.format(type(df)))

                try:
                    if 'cudf' in str(getmodule(g._edges)):
                        import cudf
                        if isinstance(g._edges, cudf.DataFrame):
                            engine_concrete = Engine.CUDF
                        else:
                            raiser(g._edges)
                    else:
                        raiser(g._edges)
                except ImportError as e:
                    raise e
                except Exception:
                    raiser(g._edges)

        else:
            engine_concrete = Engine(engine.value)

        if engine_concrete == Engine.PANDAS:
            concat_df = pd.concat([g._edges[g._source], g._edges[g._destination]])
        elif engine_concrete == Engine.CUDF:
            import cudf
            if isinstance(g._edges, cudf.DataFrame):
                edges_gdf = g._edges
            elif isinstance(g._edges, pd.DataFrame):
                edges_gdf = cudf.from_pandas(g._edges)
            else:
                raise ValueError('Unexpected edges type; convert edges to cudf.DataFrame')
            concat_df = cudf.concat([edges_gdf[g._source].rename(node_id), edges_gdf[g._destination].rename(node_id)])
        else:
            raise ValueError('Expected engine to be pandas or cudf, got: {}'.format(engine_concrete))
        nodes_df = concat_df.rename(node_id).drop_duplicates().to_frame().reset_index(drop=True)
        return g.nodes(nodes_df, node_id)

    def get_indegrees(self, col: str = "degree_in"):
        """See get_degrees"""
        g = self
        g_nodes = g.materialize_nodes()
        in_degree_df = (
            g._edges[[g._source, g._destination]]
            .groupby(g._destination)
            .agg({g._source: "count"})
            .reset_index()
            .rename(columns={g._source: col, g._destination: g_nodes._node})
        )
        nodes_df = g_nodes._nodes[
            [c for c in g_nodes._nodes.columns if c != col]
        ].merge(in_degree_df, how="left", on=g._node)
        nodes_df = nodes_df.assign(**{
            col: nodes_df[col].fillna(0).astype("int32")
        })
        return g.nodes(nodes_df, g_nodes._node)

    def get_outdegrees(self, col: str = "degree_out"):
        """See get_degrees"""
        g = self
        g2 = g.edges(
            g._edges.rename(
                columns={g._source: g._destination, g._destination: g._source}
            )
        ).get_indegrees(col)
        return g.nodes(g2._nodes, g2._node)

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
        g = self
        g2 = g.get_indegrees(degree_in).get_outdegrees(degree_out)
        g2._nodes[col] = g2._nodes["degree_in"] + g2._nodes["degree_out"]
        return g2

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

        #convert to Dict[Str, Union[Series, List-like]]
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
        #convert to Dict[Str, List-like]
        #print('nodes mid', nodes)
        nodes = {
            k: v if isinstance(v, np.ndarray) or isinstance(v, list) else v.to_numpy()
            for k, v in nodes.items()
        }

        #print('self nodes', g._nodes)
        #print('pre nodes', nodes)
        #print('keys', list(nodes.keys()))
        hits = g._nodes[list(nodes.keys())].isin(nodes)
        #print('hits', hits)
        hits_s = hits[g._node]
        for c in hits.columns:
            if c != g._node:
                hits_s = hits_s & hits[c]
        #print('hits_s', hits_s)
        new_nodes = g._nodes[hits_s]
        #print(new_nodes)
        new_node_ids = new_nodes[g._node].to_numpy()
        #print('new_node_ids', new_node_ids)
        #print('new node_ids', type(new_node_ids), len(g._nodes), '->', len(new_node_ids))
        new_edges_hits_df = (
            g._edges[[g._source, g._destination]]
            .isin({
                g._source: new_node_ids,
                g._destination: new_node_ids
            })
        )
        #print('new_edges_hits_df', new_edges_hits_df)
        new_edges = g._edges[
            new_edges_hits_df[g._source] & new_edges_hits_df[g._destination]
        ]
        #print('new_edges', new_edges)
        #print('new edges', len(g._edges), '->', len(new_edges))
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
        if (g2._nodes is None) or (len(g2._nodes) == 0):
            return g2

        g2 = g2.edges(g2._edges.drop_duplicates([g2._source, g2._destination]))
        if remove_self_loops:
            non_self_loops = g2._edges[g2._source] != g2._edges[g2._destination]
            g2 = g2.edges(g2._edges[non_self_loops])

        nodes_with_levels: List[Any] = []
        while True:
            if len(g2._nodes) == 0:
                break
            g2 = g2.get_degrees()

            roots = g2._nodes[g2._nodes["degree_in"] == 0]
            if len(roots) == 0:
                if not allow_cycles:
                    raise ValueError(
                        "Cyclic graph in get_topological_levels(); remove cycles or set allow_cycles=True"
                    )
                # tie break by picking biggest node
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
            nodes_df = pd.concat([nodes_df0] + nodes_with_levels[1:])
        else:
            nodes_df = nodes_df0

        if self._nodes is None:
            return self.nodes(nodes_df)
        else:
            # use orig cols, esp. in case collisions like degree
            out_df = g2_base._nodes.merge(
                nodes_df[[g2_base._node, level_col]], on=g2_base._node, how="left"
            )
            return self.nodes(out_df)

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
        # TODO FIXME CHECK SELF LOOPS?
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

    def filter_nodes_by_dict(self, *args, **kwargs):
        return filter_nodes_by_dict_base(self, *args, **kwargs)
    filter_nodes_by_dict.__doc__ = filter_nodes_by_dict_base.__doc__

    def filter_edges_by_dict(self, *args, **kwargs):
        return filter_edges_by_dict_base(self, *args, **kwargs)
    filter_edges_by_dict.__doc__ = filter_edges_by_dict_base.__doc__

    def chain(self, *args, **kwargs):
        return chain_base(self, *args, **kwargs)
    chain.__doc__ = chain_base.__doc__
