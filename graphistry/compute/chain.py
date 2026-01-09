import logging
import pandas as pd
from typing import Dict, Union, cast, List, Tuple, Optional, TYPE_CHECKING
from graphistry.Engine import Engine, EngineAbstract, df_concat, df_to_engine, resolve_engine

from graphistry.Plottable import Plottable
from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.Engine import safe_merge
from graphistry.util import setup_logger
from graphistry.utils.json import JSONVal
from .ast import ASTObject, ASTNode, ASTEdge, from_json as ASTObject_from_json
from .typing import DataFrameT
from .util import generate_safe_column_name
from graphistry.compute.validate.validate_schema import validate_chain_schema
from .gfql.policy import PolicyContext, PolicyException
from .gfql.policy.stats import extract_graph_stats

if TYPE_CHECKING:
    from graphistry.compute.exceptions import GFQLSchemaError, GFQLValidationError

logger = setup_logger(__name__)


def _filter_edges_by_endpoint(edges_df, nodes_df, node_id: str, edge_col: str):
    """Filter edges to those with edge_col values in nodes_df[node_id]."""
    if nodes_df is None or not node_id or not edge_col or edge_col not in edges_df.columns:
        return edges_df
    ids = nodes_df[[node_id]].drop_duplicates().rename(columns={node_id: edge_col})
    return edges_df.merge(ids, on=edge_col, how='inner')


###############################################################################


class Chain(ASTSerializable):

    def __init__(
        self,
        chain: List[ASTObject],
        validate: bool = True,
    ) -> None:
        self.chain = chain
        if validate:
            # Fail fast on invalid chains; matches documented automatic validation behavior
            self.validate(collect_all=False)

    def validate(self, collect_all: bool = False) -> Optional[List['GFQLValidationError']]:
        """Override to collect all chain validation errors."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
        
        if not collect_all:
            # Use parent's fail-fast implementation
            return super().validate(collect_all=False)
        
        # Collect all errors mode
        errors: List[GFQLValidationError] = []
        
        # Check if chain is a list
        if not isinstance(self.chain, list):
            errors.append(GFQLTypeError(
                ErrorCode.E101,
                f"Chain must be a list, but got {type(self.chain).__name__}. Wrap your operations in a list []."
            ))
            return errors  # Can't continue if not a list
        
        # Check each operation
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                errors.append(GFQLTypeError(
                    ErrorCode.E101,
                    f"Chain operation at index {i} is not a valid GFQL operation. Got {type(op).__name__} instead of an ASTObject.",
                    operation_index=i,
                    actual_type=type(op).__name__,
                    suggestion="Use n() for nodes, e() for edges, or other GFQL operations"
                ))
        
        # Validate child AST nodes
        for child in self._get_child_validators():
            child_errors = child.validate(collect_all=True)
            if child_errors:
                errors.extend(child_errors)
        
        return errors
    
    def _validate_fields(self) -> None:
        """Validate Chain fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.chain, list):
            raise GFQLTypeError(
                ErrorCode.E101,
                f"Chain must be a list, but got {type(self.chain).__name__}. Wrap your operations in a list []."
            )
        
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                raise GFQLTypeError(
                    ErrorCode.E101,
                    f"Chain operation at index {i} is not a valid GFQL operation. Got {type(op).__name__} instead of an ASTObject.",
                    operation_index=i,
                    actual_type=type(op).__name__,
                    suggestion="Use n() for nodes, e() for edges, or other GFQL operations"
                )
    
    def _get_child_validators(self) -> List[ASTSerializable]:
        """Return child AST nodes that need validation."""
        # Only return valid ASTObject instances
        return cast(List[ASTSerializable], [op for op in self.chain if isinstance(op, ASTObject)])

    @classmethod
    def from_json(cls, d: Dict[str, JSONVal], validate: bool = True) -> 'Chain':
        """
        Convert a JSON AST into a list of ASTObjects
        """
        from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
        
        if not isinstance(d, dict):
            raise GFQLSyntaxError(
                ErrorCode.E101,
                f"Chain JSON must be a dictionary, got {type(d).__name__}"
            )
        
        if 'chain' not in d:
            raise GFQLSyntaxError(
                ErrorCode.E105,
                "Chain JSON missing required 'chain' field"
            )
        
        if not isinstance(d['chain'], list):
            raise GFQLSyntaxError(
                ErrorCode.E101,
                f"Chain field must be a list, got {type(d['chain']).__name__}"
            )
        
        out = cls(
            [ASTObject_from_json(op, validate=validate) for op in d['chain']],
            validate=validate,
        )
        return out

    def to_json(self, validate=True) -> Dict[str, JSONVal]:
        """
        Convert a list of ASTObjects into a JSON AST
        """
        if validate:
            self.validate()
        return {
            'type': self.__class__.__name__,
            'chain': [op.to_json() for op in self.chain]
        }

    def validate_schema(self, g: Plottable, collect_all: bool = False) -> Optional[List['GFQLSchemaError']]:
        """Validate this chain against a graph's schema without executing.

        Args:
            g: Graph to validate against
            collect_all: If True, collect all errors. If False, raise on first.

        Returns:
            If collect_all=True: List of errors (empty if valid)  
            If collect_all=False: None if valid

        Raises:
            GFQLSchemaError: If collect_all=False and validation fails
        """
        return validate_chain_schema(g, self, collect_all)


###############################################################################


def combine_steps(
    g: Plottable,
    kind: str,
    steps: List[Tuple[ASTObject, Plottable]],
    engine: Engine,
    label_steps: Optional[List[Tuple[ASTObject, Plottable]]] = None
) -> DataFrameT:
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
        node_id = getattr(g, '_node')
        src_col = getattr(g, '_source')
        dst_col = getattr(g, '_destination')
        full_nodes = getattr(g, '_nodes', None)

        # Check if any edge op is multi-hop - if so, fall back to original re-run approach
        # Multi-hop edges span multiple nodes, so simple endpoint filtering doesn't work
        has_multihop = any(
            isinstance(op, ASTEdge) and not op.is_simple_single_hop()
            for op, _ in steps
        )

        if has_multihop:
            # Multi-hop: re-run forward ops (can't use simple endpoint filtering)
            logger.debug('EDGES << recompute forwards given reduced set (multihop)')
            new_steps = []
            for idx, (op, g_step) in enumerate(steps):
                prev_src = label_steps[idx - 1][1]._nodes if label_steps and idx > 0 else g_step._nodes
                prev_wf = (safe_merge(full_nodes, prev_src[[node_id]], on=node_id, how='inner', engine=engine)
                           if full_nodes is not None and node_id and prev_src is not None else prev_src)
                new_steps.append((op, op(g=g.edges(g_step._edges), prev_node_wavefront=prev_wf, target_wave_front=None, engine=engine)))
            steps = new_steps
        else:
            # Optimization: filter by valid endpoints instead of re-running op
            logger.debug('EDGES << filter by valid endpoints (optimized)')
            new_steps = []
            for idx, (op, g_step) in enumerate(steps):
                edges_df = g_step._edges
                if edges_df is None or len(edges_df) == 0:
                    new_steps.append((op, g_step))
                    continue

                prev_nodes = label_steps[idx - 1][1]._nodes if label_steps and idx > 0 else g._nodes
                next_nodes = label_steps[idx + 1][1]._nodes if label_steps and idx + 1 < len(label_steps) else None
                direction = getattr(op, 'direction', 'forward') if isinstance(op, ASTEdge) else 'forward'

                if direction == 'undirected' and prev_nodes is not None and next_nodes is not None and node_id:
                    prev_ids = prev_nodes[[node_id]].drop_duplicates()
                    next_ids = next_nodes[[node_id]].drop_duplicates()
                    # Either direction: (src in prev, dst in next) OR (dst in prev, src in next)
                    fwd = edges_df.merge(prev_ids.rename(columns={node_id: src_col}), on=src_col, how='inner') \
                                  .merge(next_ids.rename(columns={node_id: dst_col}), on=dst_col, how='inner')
                    rev = edges_df.merge(prev_ids.rename(columns={node_id: dst_col}), on=dst_col, how='inner') \
                                  .merge(next_ids.rename(columns={node_id: src_col}), on=src_col, how='inner')
                    edges_df = df_concat(engine)([fwd, rev]).drop_duplicates()
                else:
                    prev_col, next_col = (dst_col, src_col) if direction == 'reverse' else (src_col, dst_col)
                    edges_df = _filter_edges_by_endpoint(edges_df, prev_nodes, node_id, prev_col)
                    edges_df = _filter_edges_by_endpoint(edges_df, next_nodes, node_id, next_col)

                new_steps.append((op, g_step.edges(edges_df)))
            steps = new_steps

    logger.debug('-----------[ combine %s ---------------]', kind)

    # df[[id]] - with defensive checks for column existence
    if label_steps is None:
        label_steps = steps

    def apply_output_slice(op: ASTObject, op_label: ASTObject, df):
        if not isinstance(op_label, ASTEdge):
            return df
        out_min = getattr(op, 'output_min_hops', None) or getattr(op_label, 'output_min_hops', None)
        out_max = getattr(op, 'output_max_hops', None) or getattr(op_label, 'output_max_hops', None)
        if out_min is None and out_max is None:
            return df
        label_col = op_label.label_node_hops if kind == 'nodes' else op_label.label_edge_hops
        if label_col is None:
            hop_like = [c for c in df.columns if 'hop' in c]
            label_col = hop_like[0] if hop_like else None
        if not label_col or label_col not in df.columns:
            return df
        # Keep seeds (hop=0 or NA) and hops in range
        is_seed = (df[label_col] == 0) | df[label_col].isna()
        in_range = df[label_col].notna() & (df[label_col] > 0)
        if out_min is not None:
            in_range &= df[label_col] >= out_min
        if out_max is not None:
            in_range &= df[label_col] <= out_max
        return df[is_seed | in_range]

    dfs_to_concat = []
    extra_step_dfs = []
    base_cols = set(getattr(g, df_fld).columns)
    for idx, (op, g_step) in enumerate(steps):
        op_label = label_steps[idx][0] if idx < len(label_steps) else op
        step_df = apply_output_slice(op, op_label, getattr(g_step, df_fld))
        if id not in step_df.columns:
            step_id = getattr(g_step, '_node' if kind == 'nodes' else '_edge')
            raise ValueError(f"Column '{id}' not found in {kind} step DataFrame. "
                           f"Step has id='{step_id}', available columns: {list(step_df.columns)}. "
                           f"Operation: {op}")
        dfs_to_concat.append(step_df[[id]])

    for _, (_, g_step) in enumerate(label_steps):
        step_df = getattr(g_step, df_fld)
        if id not in step_df.columns:
            continue
        extra_cols = [c for c in step_df.columns if c != id and c not in base_cols and 'hop' in c]
        if extra_cols:
            extra_step_dfs.append(step_df[[id] + extra_cols])

    # Honor user's engine request by converting DataFrames to match requested engine
    # This ensures API contract: engine parameter guarantees output DataFrame type
    if len(dfs_to_concat) > 0:
        actual_engine = resolve_engine(EngineAbstract.AUTO, dfs_to_concat[0])
        if actual_engine != engine:
            logger.debug('Engine mismatch detected: param=%s, actual=%s. Converting data to match requested engine.', actual_engine, engine)
            dfs_to_concat = [df_to_engine(df, engine) for df in dfs_to_concat]

    concat = df_concat(engine)
    out_df = concat(dfs_to_concat).drop_duplicates(subset=[id])

    # Merge through any additional columns produced by steps (e.g., hop labels)
    label_cols = set()
    for step_df in extra_step_dfs:
        if len(step_df.columns) <= 1:  # only id column
            continue
        label_cols.update([c for c in step_df.columns if c != id])
        out_df = safe_merge(out_df, step_df, on=id, how='left', engine=engine)
        for col in step_df.columns:
            if col == id:
                continue
            col_x, col_y = f'{col}_x', f'{col}_y'
            if col_x in out_df.columns and col_y in out_df.columns:
                out_df[col] = out_df[col_x].fillna(out_df[col_y])
                out_df = out_df.drop(columns=[col_x, col_y])

    # Final post-filter: apply output slice to the combined result
    for idx, (op, _) in enumerate(steps):
        op_label = label_steps[idx][0] if idx < len(label_steps) else op
        if isinstance(op, ASTEdge):
            out_df = apply_output_slice(op, op_label, out_df)

    # If hop labels requested and seeds should be labeled, add hop 0 for seeds missing labels
    if kind == 'nodes' and label_cols:
        label_seeds_requested = any(isinstance(op, ASTEdge) and getattr(op, 'label_seeds', False) for op, _ in label_steps)
        if label_seeds_requested and label_steps:
            seed_df = getattr(label_steps[0][1], df_fld)
            if seed_df is not None and id in seed_df.columns:
                seed_ids = seed_df[[id]].drop_duplicates()
                # align engines defensively
                if resolve_engine(EngineAbstract.AUTO, seed_ids) != resolve_engine(EngineAbstract.AUTO, out_df):
                    seed_ids = df_to_engine(seed_ids, resolve_engine(EngineAbstract.AUTO, out_df))
                try:
                    seed_mask = out_df[id].isin(seed_ids[id])
                except Exception:
                    seed_mask = None
                if seed_mask is not None:
                    for col in label_cols:
                        if col in out_df.columns:
                            out_df.loc[seed_mask, col] = out_df.loc[seed_mask, col].fillna(0)
    if logger.isEnabledFor(logging.DEBUG):
        for (op, g_step) in steps:
            if kind == 'edges':
                logger.debug('adding edges to concat: %s', g_step._edges[[g_step._source, g_step._destination]])
            else:
                logger.debug('adding nodes to concat: %s', g_step._nodes[[g_step._node]])

    # df[[id, op_name1, ...]]
    logger.debug('combine_steps ops: %s', [op for (op, _) in steps])
    for idx, (op, g_step) in enumerate(steps):
        if op._name is not None and isinstance(op, op_type):
            logger.debug('tagging kind [%s] name %s', op_type, op._name)
            step_df = getattr(g_step, df_fld)[[id, op._name]]
            # Use safe_merge to handle engine type coercion automatically
            out_df = safe_merge(out_df, step_df, on=id, how='left', engine=engine)
            # Collapse any merge suffixes introduced by repeated tags
            x_name, y_name = f'{op._name}_x', f'{op._name}_y'
            if x_name in out_df.columns and y_name in out_df.columns:
                out_df[op._name] = out_df[x_name].fillna(out_df[y_name])
                out_df = out_df.drop(columns=[x_name, y_name])
            out_df[op._name] = out_df[op._name].fillna(False).astype('bool')

            # Restrict node aliases to endpoints that actually fed the next edge step
            if kind == 'nodes' and idx + 1 < len(steps):
                next_op, next_step = steps[idx + 1]
                if isinstance(next_op, ASTEdge):
                    allowed_ids = None
                    try:
                        if next_op.direction == 'forward':
                            allowed_ids = next_step._edges[next_step._source]
                        elif next_op.direction == 'reverse':
                            allowed_ids = next_step._edges[next_step._destination]
                        else:  # undirected
                            allowed_ids = pd.concat(
                                [
                                    next_step._edges[next_step._source],
                                    next_step._edges[next_step._destination],
                                ],
                                ignore_index=True,
                            )
                    except Exception:
                        allowed_ids = None

                    if allowed_ids is not None and id in out_df.columns:
                        out_df[op._name] = out_df[op._name] & out_df[id].isin(allowed_ids)

    # Final output_min/max_hops filter for nodes with hop=NA
    if kind == 'nodes':
        hop_cols = [c for c in out_df.columns if 'hop' in c.lower()]
        edge_ops = [op for op, _ in steps if isinstance(op, ASTEdge)]
        has_output_min = any(getattr(op, 'output_min_hops', None) is not None for op in edge_ops)
        has_output_max = any(getattr(op, 'output_max_hops', None) is not None for op in edge_ops)
        if (has_output_min or has_output_max) and hop_cols:
            hop_col = hop_cols[0]
            has_na = out_df[hop_col].isna()
            if has_output_min:
                # output_min_hops: drop hop=NA nodes (re-added via edge endpoint coverage)
                out_df = out_df[~has_na]
            elif has_na.any():
                # output_max_hops only: keep hop=NA nodes that have a True tag (seeds)
                tag_cols = [c for c in out_df.columns if c not in [id, 'id'] + hop_cols]
                has_tag = pd.Series(False, index=out_df.index)
                for col in tag_cols:
                    try:
                        vals = out_df[col].fillna(False)
                        if vals.dtype == 'bool' or vals.dtype == 'object':
                            has_tag |= vals.astype(bool)
                    except (TypeError, ValueError):
                        pass
                out_df = out_df[~has_na | has_tag]

    # Use safe_merge for final merge with automatic engine type coercion
    g_df = getattr(g, df_fld)
    out_df = safe_merge(out_df, g_df, on=id, how='left', engine=engine)

    logger.debug('COMBINED[%s] >>\n%s', kind, out_df)

    # Handle seed labeling toggles after slicing
    if kind == 'nodes' and label_cols:
        seeds_df = label_steps[0][1]._nodes if label_steps and label_steps[0][1]._nodes is not None else None
        seed_ids = seeds_df[[id]].drop_duplicates() if seeds_df is not None and id in seeds_df.columns else None
        label_seeds_true = any(isinstance(op, ASTEdge) and getattr(op, 'label_seeds', False) for op, _ in label_steps)
        if seed_ids is not None:
            if label_seeds_true:
                # Ensure seeds are present and labeled 0
                seeds_with_labels = seed_ids.copy()
                for col in label_cols:
                    if col in out_df.columns:
                        seeds_with_labels[col] = 0
                out_df = safe_merge(out_df, seeds_with_labels, on=id, how='outer', engine=engine)
            else:
                # Clear seed labels when label_seeds=False
                if id in out_df.columns:
                    mask = out_df[id].isin(seed_ids[id])
                    for col in label_cols:
                        if col in out_df.columns:
                            out_df.loc[mask, col] = pd.NA
        # Backfill missing hop labels from forward label steps
        hop_cols = [c for c in out_df.columns if 'hop' in c]
        if hop_cols:
            hop_maps = []
            for _, g_step in label_steps:
                step_df = getattr(g_step, df_fld)
                if id in step_df.columns:
                    for hc in hop_cols:
                        if hc in step_df.columns:
                            hop_maps.append(step_df[[id, hc]])
            hop_maps = [df for df in hop_maps if len(df) > 0]
            if hop_maps:
                hop_map_df = df_to_engine(df_concat(engine)(hop_maps), resolve_engine(EngineAbstract.AUTO, hop_maps[0]))
                for hc in hop_cols:
                    if hc in hop_map_df.columns:
                        hop_map = hop_map_df[[id, hc]].dropna(subset=[hc]).drop_duplicates(subset=[id]).set_index(id)[hc]
                        out_df[hc] = out_df[hc].combine_first(out_df[id].map(hop_map))

    # Collapse merge suffixes (_x/_y) into a single column
    cols = list(out_df.columns)
    for c in cols:
        if c.endswith('_x'):
            base = c[:-2]
            c_y = base + '_y'
            if c_y in out_df.columns:
                if len(out_df) > 0:
                    out_df[base] = out_df[c].where(out_df[c].notna(), out_df[c_y])
                out_df = out_df.drop(columns=[c, c_y])
        elif c.endswith('_y'):
            base = c[:-2]
            c_x = base + '_x'
            if c_x in out_df.columns:
                if len(out_df) > 0:
                    out_df[base] = out_df[c_x].where(out_df[c_x].notna(), out_df[c])
                out_df = out_df.drop(columns=[c, c_x])

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


def _get_boundary_calls(ops: List[ASTObject]) -> Tuple[List[ASTObject], List[ASTObject], List[ASTObject]]:
    """
    Split operations into boundary calls and middle segment.

    Detects call() operations at chain boundaries (start/end) vs interior positions.
    This enables convenient patterns like [call(), n(), e(), call()] while still
    rejecting interior mixing like [n(), call(), e()].

    Args:
        ops: List of chain operations (ASTCall, ASTNode, or ASTEdge)

    Returns:
        (prefix_calls, middle_ops, suffix_calls) where:
        - prefix_calls: call() operations at the start (may be empty)
        - middle_ops: n()/e() traversals or call()s in the middle (may be empty)
        - suffix_calls: call() operations at the end (may be empty)

    Examples:
        >>> _get_boundary_calls([call(), n(), e()])
        ([call()], [n(), e()], [])

        >>> _get_boundary_calls([n(), e(), call()])
        ([], [n(), e()], [call()])

        >>> _get_boundary_calls([call(), n(), e(), call()])
        ([call()], [n(), e()], [call()])

        >>> _get_boundary_calls([call(), call(), n()])
        ([call(), call()], [n()], [])

        >>> _get_boundary_calls([call(), call()])
        ([call(), call()], [], [])

    See: https://github.com/graphistry/pygraphistry/issues/792
    """
    from graphistry.compute.ast import ASTCall

    # Find first non-call operation
    first_traversal = next((i for i, op in enumerate(ops)
                           if not isinstance(op, ASTCall)), len(ops))

    # Find last non-call operation (search backwards)
    last_traversal = next((i for i, op in reversed(list(enumerate(ops)))
                          if not isinstance(op, ASTCall)), -1)

    # Extract segments
    prefix = ops[:first_traversal]  # All leading call() operations
    middle = ops[first_traversal:last_traversal + 1] if last_traversal >= 0 else []  # Middle segment
    suffix = ops[last_traversal + 1:] if last_traversal >= 0 else []  # All trailing call() operations

    return (prefix, middle, suffix)


def _handle_boundary_calls(
    self: Plottable,
    ops: List[ASTObject],
    engine: Union[EngineAbstract, str],
    validate_schema: bool,
    policy,
    context
) -> Optional[Plottable]:
    """
    Handle boundary call() patterns by splitting and executing sequentially.

    Detects patterns like [call(), n(), e(), call()] and executes as:
    prefix → middle → suffix via recursive chain() calls.

    Returns:
        Plottable if boundary pattern detected and executed, None otherwise

    Raises:
        GFQLValidationError: If interior mixing detected
    """
    from graphistry.compute.ast import ASTCall

    has_call = any(isinstance(op, ASTCall) for op in ops)
    has_traversal = any(isinstance(op, (ASTNode, ASTEdge)) for op in ops)

    # Only handle mixed chains (both call and traversal)
    if not (has_call and has_traversal):
        return None

    # Check if it's a boundary pattern or interior mixing
    prefix, middle, suffix = _get_boundary_calls(ops)

    # Validate middle segment doesn't have mixed operations
    if middle:
        has_call_in_middle = any(isinstance(op, ASTCall) for op in middle)
        has_traversal_in_middle = any(isinstance(op, (ASTNode, ASTEdge)) for op in middle)

        if has_call_in_middle and has_traversal_in_middle:
            from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
            raise GFQLValidationError(
                code=ErrorCode.E201,
                message="Cannot mix call() operations with n()/e() traversals in interior of chain",
                suggestion="call() operations are only allowed at chain boundaries (start/end). "
                          "For complex patterns, use either: "
                          "(1) let() composition: let({'filtered': [n(...), e(...)], 'enriched': call('get_degrees', g=ref('filtered'))}), or "
                          "(2) explicit cascading: g1 = g.chain([call(...)]); g2 = g1.chain([n(), e()]); g3 = g2.chain([call(...)]). "
                          "See issues #791, #792"
            )

    # Valid boundary pattern - execute segments sequentially
    logger.debug('Boundary call pattern detected: prefix=%s, middle=%s, suffix=%s',
                len(prefix), len(middle), len(suffix))

    g_temp = self

    if prefix:
        logger.debug('Executing boundary prefix calls: %s', prefix)
        g_temp = g_temp.chain(prefix, engine=engine, validate_schema=validate_schema, policy=policy, context=context)  # type: ignore[call-arg]

    if middle:
        logger.debug('Executing middle operations: %s', middle)
        g_temp = g_temp.chain(middle, engine=engine, validate_schema=validate_schema, policy=policy, context=context)  # type: ignore[call-arg]

    if suffix:
        logger.debug('Executing boundary suffix calls: %s', suffix)
        g_temp = g_temp.chain(suffix, engine=engine, validate_schema=validate_schema, policy=policy, context=context)  # type: ignore[call-arg]

    return g_temp


def chain(self: Plottable, ops: Union[List[ASTObject], Chain], engine: Union[EngineAbstract, str] = EngineAbstract.AUTO, validate_schema: bool = True, policy=None, context=None) -> Plottable:
    """
    Chain a list of ASTObject (node/edge) traversal operations

    Return subgraph of matches according to the list of node & edge matchers
    If any matchers are named, add a correspondingly named boolean-valued column to the output

    For direct calls, exposes convenience `List[ASTObject]`. Internal operational should prefer `Chain`.

    Use `engine='cudf'` to force automatic GPU acceleration mode

    :param ops: List[ASTObject] Various node and edge matchers
    :param validate_schema: Whether to validate the chain against the graph schema before executing
    :param policy: Optional policy dict for hooks
    :param context: Optional ExecutionContext for tracking execution state

    :returns: Plotter
    :rtype: Plotter
    """
    # Create context if not provided
    if context is None:
        from .execution_context import ExecutionContext
        context = ExecutionContext()

    # If policy provided, set it in thread-local for ASTCall operations
    if policy:
        from graphistry.compute.gfql.call_executor import _thread_local as call_thread_local
        old_policy = getattr(call_thread_local, 'policy', None)
        try:
            call_thread_local.policy = policy
            return _chain_impl(self, ops, engine, validate_schema, policy, context)
        finally:
            call_thread_local.policy = old_policy
    else:
        return _chain_impl(self, ops, engine, validate_schema, policy, context)


def _chain_impl(self: Plottable, ops: Union[List[ASTObject], Chain], engine: Union[EngineAbstract, str], validate_schema: bool, policy, context) -> Plottable:
    """
    Internal implementation of chain without policy wrapper indentation.

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
    
    **Example: Run with automatic GPU acceleration**

    ::

            import cudf
            import graphistry

            e_gdf = cudf.from_pandas(df)
            g1 = graphistry.edges(e_gdf, 's', 'd')
            g2 = g1.chain([ ... ])

    **Example: Run with automatic GPU acceleration, and force GPU mode**

    ::

            import cudf
            import graphistry

            e_gdf = cudf.from_pandas(df)
            g1 = graphistry.edges(e_gdf, 's', 'd')
            g2 = g1.chain([ ... ], engine='cudf')

    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    if isinstance(ops, Chain):
        ops = ops.chain

    # Recursive dispatch for schema-changing operations (UMAP, hypergraph, etc.)
    # These operations create entirely new graph structures, so we split the chain
    # and execute segments sequentially: before → schema_changer → rest
    from graphistry.compute.ast import ASTCall

    # Extensible list of schema-changing operations
    schema_changers = ['umap', 'hypergraph']

    # Find first schema-changer in ops
    schema_changer_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, ASTCall) and op.function in schema_changers:
            schema_changer_idx = i
            break

    if schema_changer_idx is not None:
        if len(ops) == 1:
            # Singleton schema-changer - execute directly without going through chain machinery
            from graphistry.compute.gfql.call_executor import execute_call
            from graphistry.compute.exceptions import GFQLTypeError, ErrorCode

            engine_concrete = resolve_engine(engine, self)
            schema_changer = ops[0]

            # Type narrowing: we know it's ASTCall from the isinstance check above
            if not isinstance(schema_changer, ASTCall):
                raise GFQLTypeError(
                    code=ErrorCode.E201,
                    message="Schema-changer operation must be ASTCall",
                    field="operation",
                    value=type(schema_changer).__name__,
                    suggestion="Use call('umap', {...}) or call('hypergraph', {...})"
                )

            # Validate schema if requested (even though ASTCall doesn't check columns, respect the flag)
            if validate_schema:
                validate_chain_schema(self, ops, collect_all=False)

            return execute_call(self, schema_changer.function, schema_changer.params, engine_concrete, policy=policy, context=context)
        else:
            # Multiple ops with schema-changer - split and recurse
            before = ops[:schema_changer_idx]
            schema_changer = ops[schema_changer_idx]
            rest = ops[schema_changer_idx + 1:]

            # Execute segments: before → schema_changer → rest
            # Recursion handles multiple schema-changers automatically
            g_temp = self.chain(before, engine=engine, validate_schema=validate_schema, policy=policy, context=context) if before else self  # type: ignore[call-arg]
            g_temp2 = g_temp.chain([schema_changer], engine=engine, validate_schema=validate_schema, policy=policy, context=context)  # type: ignore[call-arg]
            return g_temp2.chain(rest, engine=engine, validate_schema=validate_schema, policy=policy, context=context) if rest else g_temp2  # type: ignore[call-arg]

    if validate_schema:
        # Validate AST structure (including identifier validation) BEFORE schema validation
        # This ensures we catch reserved identifier errors before schema errors
        if isinstance(ops, Chain):
            ops.validate(collect_all=False)
        else:
            # Create temporary Chain for validation
            Chain(ops).validate(collect_all=False)

        validate_chain_schema(self, ops, collect_all=False)

    if len(ops) == 0:
        return self

    logger.debug('orig chain >> %s', ops)

    engine_concrete = resolve_engine(engine, self)
    logger.debug('chain engine: %s => %s', engine, engine_concrete)

    # Handle boundary call() patterns: [call(), ..., call()]
    # Allows call() at start/end for convenience, rejects interior mixing
    boundary_result = _handle_boundary_calls(self, ops, engine, validate_schema, policy, context)
    if boundary_result is not None:
        return boundary_result

    if isinstance(ops[0], ASTEdge):
        logger.debug('adding initial node to ensure initial link has needed reversals')
        ops = cast(List[ASTObject], [ ASTNode() ]) + ops

    if isinstance(ops[-1], ASTEdge):
        logger.debug('adding final node to ensure final link has needed reversals')
        ops = ops + cast(List[ASTObject], [ ASTNode() ])

    logger.debug('final chain >> %s', ops)

    # Store original edge binding from self before any transformations
    # This will be restored at the end if we add a temporary index column
    original_edge = self._edge

    # Initialize variables for finally block
    g_out = None
    error = None
    success = False

    try:
        g = self.materialize_nodes(engine=EngineAbstract(engine_concrete.value))

        # Handle node-only graphs (e.g., for hypergraph transformation)
        if g._edges is None:
            added_edge_index = False
        elif g._edge is None:
            # Generate a guaranteed unique internal column name to avoid conflicts with user data
            GFQL_EDGE_INDEX = generate_safe_column_name('edge_index', g._edges, prefix='__gfql_', suffix='__')

            added_edge_index = True
            # reset_index() adds the index as a column, creating 'index' if there's no name, or 'level_0', etc. if there is
            indexed_edges_df = g._edges.reset_index(drop=False)
            # Find the index column (first column not in original) with early exit
            original_cols = set(g._edges.columns)
            index_col_name = next(col for col in indexed_edges_df.columns if col not in original_cols)
            indexed_edges_df = indexed_edges_df.rename(columns={index_col_name: GFQL_EDGE_INDEX})
            g = g.edges(indexed_edges_df, edge=GFQL_EDGE_INDEX)
        else:
            added_edge_index = False

        # Prechain hook - fires BEFORE chain operations execute
        if policy and 'prechain' in policy:
            stats = extract_graph_stats(g)
            current_path = context.operation_path

            prechain_context: 'PolicyContext' = {
                'phase': 'prechain',
                'hook': 'prechain',
                'query': ops,
                'current_ast': ops,
                'query_type': 'chain',
                'plottable': g,
                'graph_stats': stats,
                'execution_depth': context.execution_depth,
                'operation_path': current_path,
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            try:
                policy['prechain'](prechain_context)
            except PolicyException:
                raise

        logger.debug('======================== FORWARDS ========================')

        # Forwards
        # This computes valid path *prefixes*, where each g nodes/edges is the path wavefront:
        #  g_step._nodes: The nodes reached in this step
        #  g_step._edges: The edges used to reach those nodes
        # At the paths are prefixes, wavefront nodes may invalid wrt subsequent steps (e.g., halt early)
        g_stack : List[Plottable] = []
        for i, op in enumerate(ops):
            # Determine graph to pass based on operation type
            # - ASTNode/ASTEdge: Use original graph `g` + wavefront tracking
            # - ASTCall: Use previous operation's result (for chaining filters/transforms)
            if isinstance(op, ASTCall):
                # For ASTCall operations (filter_edges_by_dict, etc.), pass previous result
                # This ensures chained filters apply sequentially: filter1(g) → filter2(result1) → ...
                current_g = g_stack[-1] if g_stack else g
                prev_step_nodes = None  # ASTCall doesn't use wavefronts
            else:
                # For ASTNode/ASTEdge operations, use original graph + wavefront
                # Wavefronts track which nodes are "active" at each step
                current_g = g
                prev_step_nodes = (
                    None  # first uses full graph
                    if len(g_stack) == 0
                    else g_stack[-1]._nodes
                )

            g_step = (
                op(
                    g=current_g,  # Pass appropriate graph for operation type
                    prev_node_wavefront=prev_step_nodes,
                    target_wave_front=None,  # implicit any
                    engine=engine_concrete
                )
            )
            g_stack.append(g_step)

        import logging
        if logger.isEnabledFor(logging.DEBUG):
            for (i, g_step) in enumerate(g_stack):
                logger.debug('~' * 10 + '\nstep %s', i)
                logger.debug('nodes: %s', g_step._nodes)
                logger.debug('edges: %s', g_step._edges)

        # Check if all operations are ASTCall (no traversals)
        # For pure ASTCall chains, skip backward pass and combine - just return the last result
        all_astcall = all(isinstance(op, ASTCall) for op in ops)

        if all_astcall:
            # For chains of only ASTCall operations (filters, transforms),
            # the forward pass result is final - no path validation needed
            g_out = g_stack[-1]
            if added_edge_index:
                # Drop the internal edge index column
                final_edges_df = g_out._edges.drop(columns=[g._edge])
                g_out = self.nodes(g_out._nodes).edges(final_edges_df, edge=original_edge)
            # Mark as successful
            success = True
        else:

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
                # Reattach node attributes for reverse wavefronts so downstream matches work
                prev_wavefront_nodes = prev_loop_step._nodes
                if g._node is not None and prev_wavefront_nodes is not None and g._nodes is not None:
                    prev_wavefront_nodes = safe_merge(
                        g._nodes,
                        prev_wavefront_nodes[[g._node]],
                        on=g._node,
                        how='inner',
                        engine=engine_concrete
                    )
                target_wave_front_nodes = prev_orig_step._nodes if prev_orig_step is not None else None
                if g._node is not None and target_wave_front_nodes is not None and g._nodes is not None:
                    target_wave_front_nodes = safe_merge(
                        g._nodes,
                        target_wave_front_nodes[[g._node]],
                        on=g._node,
                        how='inner',
                        engine=engine_concrete
                    )
                assert prev_loop_step._nodes is not None

                # Fast path: for simple single-hop edges, skip the full hop() call
                # and use vectorized merge filtering instead. This saves ~50% time on small graphs.
                use_fast_backward = (
                    isinstance(op, ASTEdge)
                    and op.is_simple_single_hop()
                    and g_step._edges is not None
                    and len(g_step._edges) > 0
                    and g._node is not None
                    and g._source is not None
                    and g._destination is not None
                )

                if use_fast_backward:
                    assert isinstance(op, ASTEdge)  # type narrowing for mypy
                    edges_df = g_step._edges
                    node_id, src_col, dst_col = g._node, g._source, g._destination
                    assert node_id is not None and src_col is not None and dst_col is not None
                    is_undirected = op.direction == 'undirected'
                    prev_set = set(prev_wavefront_nodes[node_id]) if prev_wavefront_nodes is not None else None
                    target_set = set(target_wave_front_nodes[node_id]) if target_wave_front_nodes is not None else None

                    # Filter edges by wavefronts
                    if is_undirected:
                        if prev_set and target_set:
                            mask = ((edges_df[src_col].isin(prev_set) & edges_df[dst_col].isin(target_set))
                                    | (edges_df[src_col].isin(target_set) & edges_df[dst_col].isin(prev_set)))
                            edges_df = edges_df[mask]
                        elif prev_set:
                            edges_df = edges_df[edges_df[src_col].isin(prev_set) | edges_df[dst_col].isin(prev_set)]
                        elif target_set:
                            edges_df = edges_df[edges_df[src_col].isin(target_set) | edges_df[dst_col].isin(target_set)]
                    else:
                        next_col, prev_col = (src_col, dst_col) if op.direction == 'reverse' else (dst_col, src_col)
                        edges_df = _filter_edges_by_endpoint(edges_df, prev_wavefront_nodes, node_id, next_col)
                        edges_df = _filter_edges_by_endpoint(edges_df, target_wave_front_nodes, node_id, prev_col)

                    # Get result nodes
                    if len(edges_df) > 0:
                        if is_undirected:
                            target_node_ids = df_concat(engine_concrete)([
                                edges_df[[src_col]].rename(columns={src_col: node_id}),
                                edges_df[[dst_col]].rename(columns={dst_col: node_id})
                            ]).drop_duplicates()
                        else:
                            target_col = dst_col if op.direction == 'reverse' else src_col
                            target_node_ids = edges_df[[target_col]].rename(columns={target_col: node_id}).drop_duplicates()
                        nodes_df = safe_merge(g._nodes, target_node_ids, on=node_id, how='inner', engine=engine_concrete) if g._nodes is not None else target_node_ids
                    else:
                        nodes_df = g._nodes.iloc[:0] if g._nodes is not None else None

                    g_step_reverse = g_step.nodes(nodes_df).edges(edges_df)
                else:
                    # Fall back to full hop() traversal for complex cases
                    g_step_reverse = op.reverse()(
                        g=g_step,
                        prev_node_wavefront=prev_wavefront_nodes,
                        target_wave_front=target_wave_front_nodes,
                        engine=engine_concrete
                    )
                g_stack_reverse.append(g_step_reverse)

            import logging
            if logger.isEnabledFor(logging.DEBUG):
                for (i, g_step) in enumerate(g_stack_reverse):
                    logger.debug('~' * 10 + '\nstep %s', i)
                    logger.debug('nodes: %s', g_step._nodes)
                    logger.debug('edges: %s', g_step._edges)

            logger.debug('============ COMBINE NODES ============')
            final_nodes_df = combine_steps(
                g,
                'nodes',
                list(zip(ops, reversed(g_stack_reverse))),
                engine_concrete,
                label_steps=list(zip(ops, g_stack))
            )

            logger.debug('============ COMBINE EDGES ============')
            final_edges_df = combine_steps(
                g,
                'edges',
                list(zip(ops, reversed(g_stack_reverse))),
                engine_concrete,
                label_steps=list(zip(ops, g_stack))
            )
            if added_edge_index:
                # Drop the internal edge index column (stored in g._edge after we added it)
                final_edges_df = final_edges_df.drop(columns=[g._edge])
                # Fix: Restore original edge binding instead of using modified 'index' binding
                g_out = self.nodes(final_nodes_df).edges(final_edges_df, edge=original_edge)
            else:
                g_out = g.nodes(final_nodes_df).edges(final_edges_df)

            # Ensure node set covers edge endpoints after any output slicing
            if g_out._edges is not None and len(g_out._edges) > 0:
                concat_fn = df_concat(engine_concrete)
                endpoints = concat_fn(
                    [
                        g_out._edges[[g_out._source]].rename(columns={g_out._source: g_out._node}),
                        g_out._edges[[g_out._destination]].rename(columns={g_out._destination: g_out._node}),
                    ],
                    ignore_index=True,
                    sort=False,
                ).drop_duplicates(subset=[g_out._node])
                if resolve_engine(EngineAbstract.AUTO, endpoints) != resolve_engine(EngineAbstract.AUTO, g_out._nodes):
                    endpoints = df_to_engine(endpoints, resolve_engine(EngineAbstract.AUTO, g_out._nodes))
                g_out = g_out.nodes(
                    concat_fn([g_out._nodes, endpoints], ignore_index=True, sort=False).drop_duplicates(subset=[g_out._node])
                )

            # Mark as successful
            success = True

    except Exception as e:
        # Capture error for postload hook
        error = e
        # Don't re-raise yet - let finally block run first

    finally:
        # Postchain hook - fires AFTER chain operations complete (even on error)
        postchain_policy_error = None
        if policy and 'postchain' in policy:

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, g_out is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, g_out) if success else self
            stats = extract_graph_stats(graph_for_stats)
            current_path = context.operation_path

            postchain_context: 'PolicyContext' = {
                'phase': 'postchain',
                'hook': 'postchain',
                'query': ops,
                'current_ast': ops,
                'query_type': 'chain',
                'plottable': graph_for_stats,
                'graph_stats': stats,
                'success': success,
                'execution_depth': context.execution_depth,
                'operation_path': current_path,
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            # Add error information if execution failed
            if error is not None:
                postchain_context['error'] = str(error)  # type: ignore
                postchain_context['error_type'] = type(error).__name__  # type: ignore

            try:
                policy['postchain'](postchain_context)
            except PolicyException as e:
                # Capture policy error instead of raising immediately
                postchain_policy_error = e

        # Postload policy phase - ALWAYS fires (even on error)
        policy_error = None
        if policy and 'postload' in policy:

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, g_out is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, g_out) if success else self
            stats = extract_graph_stats(graph_for_stats)

            policy_context: 'PolicyContext' = {
                'phase': 'postload',
                'hook': 'postload',
                'query': ops,
                'current_ast': ops,  # For chain, current == ops
                'query_type': 'chain',
                'plottable': graph_for_stats,  # RESULT or INPUT
                'graph_stats': stats,
                'success': success,  # True if successful, False if error
                'execution_depth': context.execution_depth,  # Add execution depth
                '_policy_depth': getattr(ops, '_policy_depth', 0) if hasattr(ops, '_policy_depth') else 0
            }

            # Add error information if execution failed
            if error is not None:
                policy_context['error'] = str(error)  # type: ignore
                policy_context['error_type'] = type(error).__name__  # type: ignore

            try:
                # Policy can only accept (None) or deny (exception)
                policy['postload'](policy_context)

            except PolicyException as e:
                # Enrich exception with context if not already set
                if e.query_type is None:
                    e.query_type = 'chain'
                if e.data_size is None:
                    e.data_size = stats
                # Capture policy error instead of raising immediately
                policy_error = e

    # After finally block, decide which error to raise
    # Priority: postchain PolicyException > postload PolicyException > operation error
    if postchain_policy_error is not None:
        # postchain policy error takes highest priority
        if error is not None:
            raise postchain_policy_error from error
        else:
            raise postchain_policy_error
    elif policy_error is not None:
        # postload policy error is second priority
        if error is not None:
            raise policy_error from error
        else:
            raise policy_error
    elif error is not None:
        raise error

    # Cast: At this point, all error paths have been handled, so g_out is guaranteed to be a Plottable
    return cast(Plottable, g_out)
