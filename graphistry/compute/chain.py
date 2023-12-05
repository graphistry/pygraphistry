from typing import cast, List, Tuple
import pandas as pd

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .ast import ASTObject, ASTNode, ASTEdge

logger = setup_logger(__name__)


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
                op,  # forward op
                op(
                    g=g.edges(g_step._edges),  # transition via any found edge
                    prev_node_wavefront=g_step._nodes,  # start from where backwards step says is reachable

                    #target_wave_front=steps[i+1][1]._nodes  # end at where next backwards step says is reachable
                    target_wave_front=None  # ^^^ optimization: valid transitions already limit to known-good ones
                )
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

    :param ops: List[ASTObject] Various node and edge matchers

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

    **Example: Filter by multiple node types at each step using is_in**

    ::

            from graphistry.ast import n, e_forward, e_reverse, is_in

            g_risky = g.chain([
                n({"type": is_in(["person", "company"])}),
                e_forward({"e_type": is_in(["owns", "reviews"])}, to_fixed=True),
                n({"type": is_in(["transaction", "account"])}, name="hit"),
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
    

    logger.debug('======================== FORWARDS ========================')

    # Forwards
    # This computes valid path *prefixes*, where each g nodes/edges is the path wavefront:
    #  g_step._nodes: The nodes reached in this step
    #  g_step._edges: The edges used to reach those nodes
    # At the paths are prefixes, wavefront nodes may invalid wrt subsequent steps (e.g., halt early)
    g_stack : List[Plottable] = []
    for op in ops:
        prev_step_nodes = (  # start from only prev step's wavefront node
            None  # first uses full graph
            if len(g_stack) == 0
            else g_stack[-1]._nodes
        )
        g_step = (
            op(
                g=g,  # transition via any original edge
                prev_node_wavefront=prev_step_nodes,
                target_wave_front=None  # implicit any
            )
        )
        g_stack.append(g_step)

    logger.debug('======================== BACKWARDS ========================')

    # Backwards
    # Compute reverse and thus complete paths. Dropped nodes/edges are thus the incomplete path prefixes.
    # Each g node/edge represents a valid wavefront entry for that step.
    g_stack_reverse : List[Plottable] = []
    for (op, g_step) in zip(reversed(ops), reversed(g_stack)):
        prev_loop_step = g_stack[-1] if len(g_stack_reverse) == 0 else g_stack_reverse[-1]
        if len(g_stack_reverse) == len(g_stack) - 1:
            prev_orig_step = None
        else:
            prev_orig_step = g_stack[-(len(g_stack_reverse) + 2)]
        assert prev_loop_step._nodes is not None
        g_step_reverse = (
            (op.reverse())(

                # Edges: edges used in step (subset matching prev_node_wavefront will be returned)
                # Nodes: nodes reached in step (subset matching prev_node_wavefront will be returned)
                g=g_step,

                # check for hits against fully valid targets
                # ast will replace g.node() with this as its starting points
                prev_node_wavefront=prev_loop_step._nodes,

                # only allow transitions to these nodes (vs prev_node_wavefront)
                target_wave_front=prev_orig_step._nodes if prev_orig_step is not None else None
            )
        )
        g_stack_reverse.append(g_step_reverse)

    logger.debug('============ COMBINE NODES ============')
    final_nodes_df = combine_steps(g, 'nodes', list(zip(ops, reversed(g_stack_reverse))))

    logger.debug('============ COMBINE EDGES ============')
    final_edges_df = combine_steps(g, 'edges', list(zip(ops, reversed(g_stack_reverse))))
    if added_edge_index:
        final_edges_df = final_edges_df.drop(columns=['index'])

    g_out = g.nodes(final_nodes_df).edges(final_edges_df)

    return g_out
