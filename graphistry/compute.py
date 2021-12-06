from typing import Any, Callable, Iterable, List, Optional, Set, Union, TYPE_CHECKING
import logging
import itertools, pandas
from .Plottable import Plottable
logger = logging.getLogger('compute')

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object

class ComputeMixin(MIXIN_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def materialize_nodes(self, reuse: bool = True):
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
        g = self
        if g._edges is None:
            raise ValueError('Missing edges')
        if g._source is None or g._destination is None:
            raise ValueError('Missing source/destination bindings; set via .bind() or .edges()')
        if len(g._edges) == 0:
            return g
        #TODO use built-ins for igraph/nx/...

        if reuse:
            if g._nodes is not None and len(g._nodes) > 0:
                if g._node is None:
                    logger.warning('Must set node id binding, not just nodes; set via .bind() or .nodes()')
                    #raise ValueError('Must set node id binding, not just nodes; set via .bind() or .nodes()')
                else:
                    return g

        node_id = g._node if g._node is not None else 'id'

        nodes_df = (g
                ._edges[g._source].append(g._edges[g._destination])
                .rename(node_id)
                .drop_duplicates()
                .to_frame())

        return g.nodes(nodes_df, node_id)


    def get_indegrees(self, col: str = 'degree_in'):
        """See get_degrees
        """
        g = self
        g_nodes = g.materialize_nodes()
        in_degree_df = (g
                    ._edges[[g._source, g._destination]]
                    .groupby(g._destination)
                    .agg({g._source: 'count'})
                    .reset_index()
                    .rename(columns={
                        g._source: col,
                        g._destination: g_nodes._node
                    }))
        nodes_df = (g_nodes._nodes
                    [[c for c in g_nodes._nodes.columns if c != col]]
                    .merge(in_degree_df, how='left', on=g._node))
        nodes_df[col].fillna(0, inplace=True)
        nodes_df[col] = nodes_df[col].astype('int32')
        return g.nodes(nodes_df, g_nodes._node)

    def get_outdegrees(self, col: str = 'degree_out'):
        """See get_degrees
        """
        g = self
        g2 = g.edges(
            g._edges.rename(
                columns={
                    g._source: g._destination,
                    g._destination: g._source
                })).get_indegrees(col)
        return g.nodes(g2._nodes, g2._node)


    def get_degrees(self, col: str = 'degree', degree_in: str = 'degree_in', degree_out: str = 'degree_out'):
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
        g2._nodes[col] = g2._nodes['degree_in'] + g2._nodes['degree_out']
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

    def replace_nodes_with_edges(self, node_selector):
        g = self

        node_selector = list(node_selector)
        if len(node_selector) == 0:
            return g

        g2 = g
        nodes_to_reduce = g._edges[g._edges[g._destination].isin(node_selector)][g._source]
        nodes_to_reduce = nodes_to_reduce.drop_duplicates()
        if len(nodes_to_reduce) <= 1:
            raise ValueError('Node reduction is not a meaningful operation in this context')

        edges_to_retain =  g._edges[~g._edges[g._destination].isin(node_selector)]
        g._edges = pandas.concat(
            [
                pandas.DataFrame(
                    itertools.combinations(nodes_to_reduce,2))\
                    .rename(columns={0: g._source, 1: g._destination})\
                    .assign(reduced_from=",".join(node_selector)),
                edges_to_retain
            ], ignore_index=True)
        return g

    def get_topological_levels(
        self,
        level_col: str = 'level',
        allow_cycles: bool = True,
        warn_cycles: bool = True,
        remove_self_loops: bool = True
    ) -> Plottable:
        """
        Label nodes on column level_col based on topological sort depth
        Supports pandas + cudf, using parallelism within each level computation
        Options:
        * allow_cycles: if False and detects a cycle, throw ValueException, else break cycle by picking a lowest-in-degree node
        * warn_cycles: if True and detects a cycle, proceed with a warning
        * remove_self_loops: preprocess by removing self-cycles. Avoids allow_cycles=False, warn_cycles=True messages.

        Example:

        edges_df = gpd.DataFrame({'s': ['a', 'b', 'c', 'd'], ['b', 'c', 'e', 'e']})
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

        nodes_with_levels : List[Any] = []
        while True:
            if len(g2._nodes) == 0:
                break
            g2 = g2.get_degrees()

            roots = g2._nodes[ g2._nodes['degree_in'] == 0 ]
            if len(roots) == 0:
                if not allow_cycles:
                    raise ValueError('Cyclic graph in get_topological_levels(); remove cycles or set allow_cycles=True')
                #tie break by picking biggest node
                max_degree = g2._nodes['degree'].max()
                roots = g2._nodes[ g2._nodes['degree'] == max_degree ][:1]
                if warn_cycles:
                    logger.warning('Cycle on computing level %s', len(nodes_with_levels))

            nodes_with_levels.append(
                (roots[[c for c in roots if c not in ['degree_in', 'degree_out', 'degree']]]
                    .assign(**{level_col: len(nodes_with_levels)})))

            g2 = g2.drop_nodes(roots[g2._node])
        nodes_df0 = nodes_with_levels[0]
        if len(nodes_with_levels) > 1:
            nodes_df = nodes_df0.append(nodes_with_levels[1:])
        else:
            nodes_df = nodes_df0

        if self._nodes is None:
            return self.nodes(nodes_df)
        else:
            #use orig cols, esp. in case collisions like degree
            out_df = g2_base._nodes.merge(nodes_df[[g2_base._node, level_col]], on=g2_base._node, how='left')
            return self.nodes(out_df)
