from typing import cast, List, Optional, Tuple, Union
import pandas as pd

from graphistry.Plottable import Plottable
from .ast import ASTObject, ASTNode, ASTEdge
from .filter_by_dict import filter_by_dict

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


###############################################################################


def combine_steps(g: Plottable, kind: str, steps: List[Tuple[ASTObject,Plottable]]) -> pd.DataFrame:
    """
    Collect nodes and edges, taking care to deduplicate and tag any names
    """

    id = getattr(g, '_node' if kind == 'nodes' else '_edge')
    df_fld = '_nodes' if kind == 'nodes' else '_edges'
    op_type = ASTNode if kind == 'nodes' else ASTEdge

    if id is None:
        raise ValueError(f'Cannot combine steps with empty id for kind {kind}')

    logger.debug('combine_steps ops pre: %s', [op for (op, _) in steps])
    if kind == 'edges':
        logger.debug('EDGES << recompute forwards given reduced set')
        steps = [
            (
                op,
                op(g=g.edges(g_step._edges), prev_node_wavefront=g_step._nodes)
            )
            for (op, g_step) in steps
        ]

    # df[[id]]
    out_df = pd.concat([
        getattr(g_step, df_fld)[[id]]
        for (_, g_step) in steps
    ]).drop_duplicates(subset=[id])

    # df[[id, op_name1, ...]]
    logger.debug('combine_steps ops: %s', [op for (op, _) in steps])
    for (op, g_step) in steps:
        if op._name is not None and isinstance(op, op_type):
            logger.debug('tagging kind [%s] name %s', op_type, op._name)
            out_df = out_df.merge(
                getattr(g_step, df_fld)[[id, op._name]],
                on=id,
                how='left'
            )
            out_df[op._name] = out_df[op._name].fillna(False).astype(bool)
    out_df = out_df.merge(getattr(g, df_fld), on=id, how='left')

    logger.debug('COMBINED[%s] >> %s', kind, out_df)

    return out_df


###############################################################################
#
#  Implementation: The algorithm performs three phases -
#
#     1. Forward wavefront (slowed)
#
#     Each step is processed, yielding the nodes it matches based on the nodes reached by the previous step
#
#     Full node/edge table merges are happening, so any pre-filtering would help
#
#     2. Reverse pruning pass  (fastish)
#
#     Some paths traversed during Step 1 are deadends that must be pruned
#    
#     To only pick nodes on full paths, we then run in a reverse pass on a graph subsetted to nodes along full/partial paths.
#
#     - Every node encountered on the reverse pass is guaranteed to be on a full path
#    
#     - Every 'good' node will be encountered
#    
#     - No 'bad' deadend nodes will be included
#
#     3. Forward output pass
#
#     This pass is likely fusable into Step 2: collect and label outputs
#
###############################################################################

def chain(self: Plottable, ops: List[ASTObject]) -> Plottable:
    """

    Experimental: Chain a list of operations

    Return subgraph of matches according to the list of node & edge matchers

    If any matchers are named, add a correspondingly named boolean-valued column to the output

    :param ops: List[ASTobject] Various node and edge matchers
    :type fg: dict

    :returns: Plotter
    :rtype: Plotter

    **Example: Find nodes of some type**
        ::

            from graphistry.ast import n

            people_nodes_df = g.chain([ n({"type": "person"}) ])._nodes
            
    **Example: Find 2-hop edge sequences with some attribute**
        ::

            from graphistry.ast import e_forward

            g_2_hops = g.chain([ e_forward({"interesting": True}, hops=2) ])
            g_2_hops.plot()

    **Example: Find any node 1-2 hops out from another node, and label each hop**

    ::

            from graphistry.ast import n, e_undirected

            g_2_hops = g.chain([ n({g._node: "a"}), e_undirected(name="hop1"), e_undirected(name="hop2") ])
            print('# first-hop edges:', len(g_2_hops._edges[ g_2_hops._edges.hop1 == True ]))

    **Example: Transaction nodes between two kinds of risky nodes**

    ::

            from graphistry.ast import n, e_forward, e_reverse

            g_risky = g.chain([
                n({"risk1": True}),
                e_forward(to_fixed=True),
                n({"type": "transaction"}, name="hit"),
                e_reverse(to_fixed=True),
                n({"risk2": True})
            ])
            print('# hits:', len(g_risky._nodes[ g_risky._nodes.hit ]))

    """

    if len(ops) == 0:
        return self

    logger.debug('orig chain >> %s', ops)

    if isinstance(ops[0], ASTEdge):
        logger.debug('adding initial node to ensure initial link has needed reversals')
        ops = cast(List[ASTObject], [ ASTNode() ]) + ops

    if isinstance(ops[-1], ASTEdge):
        logger.debug('adding final node to ensure final link has needed reversals')
        ops = ops + cast(List[ASTObject], [ ASTNode() ])

    logger.debug('final chain >> %s', ops)

    g = self.materialize_nodes()

    if g._edge is None:
        if 'index' in g._edges.columns:
            raise ValueError('Edges cannot have column "index", please remove or set as g._edge via bind() or edges()')
        added_edge_index = True
        indexed_edges_df = g._edges.reset_index()
        g = g.edges(indexed_edges_df, edge='index')
    else:
        added_edge_index = False
    

    logger.debug('============ FORWARDS ============')

    #forwards
    g_stack : List[Plottable] = []
    for op in ops:
        g_step = (
            op(
                g=g,
                prev_node_wavefront=(
                    None  # first uses full graph
                    if len(g_stack) == 0
                    else g_stack[-1]._nodes
                )))
        g_stack.append(g_step)

    encountered_nodes_df = pd.concat([
        g_step._nodes
        for g_step in g_stack
    ]).drop_duplicates(subset=[g._node])

    logger.debug('============ BACKWARDS ============')

    #backwards
    g_stack_reverse : List[Plottable] = [g_stack[-1]]
    for (op, g_step) in zip(reversed(ops), reversed(g_stack)):
        g_step_reverse = (
            (op.reverse())(

                # all encountered nodes + step's edges
                g=g_step.nodes(encountered_nodes_df),

                # check for hits against fully valid targets
                prev_node_wavefront=g_stack_reverse[-1]._nodes

            )
        )
        g_stack_reverse.append(g_step_reverse)

    logger.debug('============ COMBINE NODES ============')
    final_nodes_df = combine_steps(g, 'nodes', list(zip(reversed(ops), g_stack_reverse[1:])))

    logger.debug('============ COMBINE EDGES ============')
    final_edges_df = combine_steps(g, 'edges', list(zip(reversed(ops), g_stack_reverse[1:])))
    if added_edge_index:
        final_edges_df = final_edges_df.drop(columns=['index'])

    g_out = g.nodes(final_nodes_df).edges(final_edges_df)

    return g_out
