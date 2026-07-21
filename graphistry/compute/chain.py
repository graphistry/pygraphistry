import logging
from typing import Any, Dict, Union, cast, List, Tuple, Sequence, Optional, TYPE_CHECKING
from graphistry.Engine import Engine, EngineAbstract, POLARS_ENGINES, align_shared_column_dtypes, df_concat, df_to_engine, resolve_engine, safe_map_series, safe_row_concat, s_na
from graphistry.compute.dataframe_utils import dbg_df

from graphistry.Plottable import Plottable
from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.Engine import safe_merge
from graphistry.util import setup_logger
from graphistry.utils.json import JSONVal
from .ast import ASTObject, ASTNode, ASTEdge, Direction, from_json as ASTObject_from_json, serialize_binding_ops
from .typing import DataFrameT, SeriesT
from .util import generate_safe_column_name
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.gfql.same_path_types import (
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
    where_to_json,
)
from .gfql.policy import PolicyContext, PolicyException
from .gfql.policy.stats import extract_graph_stats
from graphistry.otel import otel_traced, otel_detail_enabled

if TYPE_CHECKING:
    from graphistry.compute.exceptions import GFQLSchemaError, GFQLValidationError

logger = setup_logger(__name__)


def _filter_edges_by_endpoint(
    edges_df: DataFrameT, nodes_df: Optional[DataFrameT], node_id: str, edge_col: str
) -> DataFrameT:
    if nodes_df is None or not node_id or not edge_col or edge_col not in edges_df.columns:
        return edges_df
    # isin() is set-membership, so the dropped .unique() is redundant (byte-identical).
    return edges_df[edges_df[edge_col].isin(nodes_df[node_id])]


from .chain_lean_combine import (
    _lean_combine_enabled, _is_unique_ids, _lean_engine_ok,
    _lean_intersect_full, _lean_prefilter_right,
)


class Chain(ASTSerializable):

    def __init__(
        self,
        chain: List[ASTObject],
        where: Optional[Sequence[WhereComparison]] = None,
        validate: bool = True,
    ) -> None:
        self.chain = chain
        self.where = normalize_where_entries(where or [])
        if validate:
            self.validate(collect_all=False)

    def validate(self, collect_all: bool = False) -> Optional[List['GFQLValidationError']]:
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
        
        if not collect_all:
            return super().validate(collect_all=False)
        
        errors: List[GFQLValidationError] = []
        
        if not isinstance(self.chain, list):
            errors.append(GFQLTypeError(
                ErrorCode.E101,
                f"Chain must be a list, but got {type(self.chain).__name__}. Wrap your operations in a list []."
            ))
            return errors
        
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                errors.append(GFQLTypeError(
                    ErrorCode.E101,
                    f"Chain operation at index {i} is not a valid GFQL operation. Got {type(op).__name__} instead of an ASTObject.",
                    operation_index=i,
                    actual_type=type(op).__name__,
                    suggestion="Use n() for nodes, e() for edges, or other GFQL operations"
                ))
        
        for child in self._get_child_validators():
            child_errors = child.validate(collect_all=True)
            if child_errors:
                errors.extend(child_errors)
        
        return errors
    
    def _validate_fields(self) -> None:
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
        return [op for op in self.chain if isinstance(op, ASTObject)]

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
        
        where = parse_where_json(d.get('where'))
        out = cls(
            [ASTObject_from_json(op, validate=validate) for op in d['chain']],
            where=where,
            validate=validate,
        )
        return out

    def to_json(self, validate=True) -> Dict[str, JSONVal]:
        """
        Convert a list of ASTObjects into a JSON AST
        """
        if validate:
            self.validate()
        data: Dict[str, JSONVal] = {
            'type': self.__class__.__name__,
            'chain': [op.to_json() for op in self.chain]
        }
        if self.where:
            data['where'] = where_to_json(self.where)
        return data

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

        has_multihop = any(
            isinstance(op, ASTEdge) and not op.is_simple_single_hop()
            for op, _ in steps
        )

        if has_multihop:
            logger.debug('EDGES << recompute forwards given reduced set (multihop)')
            new_steps = []
            for idx, (op, g_step) in enumerate(steps):
                prev_src = label_steps[idx - 1][1]._nodes if label_steps and idx > 0 else g_step._nodes
                prev_wf = (safe_merge(full_nodes, prev_src[[node_id]], on=node_id, how='inner', engine=engine)
                           if full_nodes is not None and node_id and prev_src is not None else prev_src)
                new_steps.append((op, op.execute(g=g.edges(g_step._edges), prev_node_wavefront=prev_wf, target_wave_front=None, engine=engine)))
            steps = new_steps
        else:
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
                    # isin() dedups internally -> the .unique() pass is redundant
                    prev_ids = prev_nodes[node_id]
                    next_ids = next_nodes[node_id]
                    fwd_mask = edges_df[src_col].isin(prev_ids) & edges_df[dst_col].isin(next_ids)
                    rev_mask = edges_df[dst_col].isin(prev_ids) & edges_df[src_col].isin(next_ids)
                    edges_df = edges_df[fwd_mask | rev_mask]
                else:
                    prev_col, next_col = (dst_col, src_col) if direction == 'reverse' else (src_col, dst_col)
                    edges_df = _filter_edges_by_endpoint(edges_df, prev_nodes, node_id, prev_col)
                    edges_df = _filter_edges_by_endpoint(edges_df, next_nodes, node_id, next_col)

                new_steps.append((op, g_step.edges(edges_df)))
            steps = new_steps

    logger.debug('-----------[ combine %s ---------------]', kind)

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

    if len(dfs_to_concat) > 0:
        actual_engine = resolve_engine(EngineAbstract.AUTO, dfs_to_concat[0])
        if actual_engine != engine:
            logger.debug('Engine mismatch detected: param=%s, actual=%s. Converting data to match requested engine.', actual_engine, engine)
            dfs_to_concat = [df_to_engine(df, engine) for df in dfs_to_concat]

    concat = df_concat(engine)
    out_df = concat(dfs_to_concat).drop_duplicates(subset=[id])

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

    for idx, (op, _) in enumerate(steps):
        op_label = label_steps[idx][0] if idx < len(label_steps) else op
        if isinstance(op, ASTEdge):
            out_df = apply_output_slice(op, op_label, out_df)

    if kind == 'nodes' and label_cols:
        label_seeds_requested = any(isinstance(op, ASTEdge) and getattr(op, 'label_seeds', False) for op, _ in label_steps)
        if label_seeds_requested and label_steps:
            seed_df = getattr(label_steps[0][1], df_fld)
            if seed_df is not None and id in seed_df.columns:
                seed_ids = seed_df[[id]].drop_duplicates()
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
                logger.debug('adding edges to concat: %s', dbg_df(g_step._edges))
            else:
                logger.debug('adding nodes to concat: %s', dbg_df(g_step._nodes))

    logger.debug('combine_steps ops: %s', [op for (op, _) in steps])

    # Reject duplicate alias names — silent overwrite would lose data
    seen_names: Dict[str, int] = {}
    for idx, (op, _) in enumerate(steps):
        if op._name is not None and isinstance(op, op_type):
            if op._name in seen_names:
                from graphistry.compute.exceptions import GFQLValidationError, ErrorCode
                raise GFQLValidationError(
                    code=ErrorCode.E201,
                    message=(
                        f"Duplicate alias name '{op._name}' in chain "
                        f"(steps {seen_names[op._name]} and {idx})"
                    ),
                    suggestion="Use distinct alias names for each step in the chain",
                )
            seen_names[op._name] = idx

    for idx, (op, g_step) in enumerate(steps):
        if op._name is not None and isinstance(op, op_type):
            logger.debug('tagging kind [%s] name %s', op_type, op._name)
            step_df = getattr(g_step, df_fld)[[id, op._name]]
            out_df = safe_merge(out_df, step_df, on=id, how='left', engine=engine)
            x_name, y_name = f'{op._name}_x', f'{op._name}_y'
            if x_name in out_df.columns and y_name in out_df.columns:
                out_df[op._name] = out_df[x_name].where(out_df[x_name].notna(), out_df[y_name])
                out_df = out_df.drop(columns=[x_name, y_name])
            label_col = out_df[op._name]
            if engine == Engine.PANDAS:
                label_col = label_col.astype('boolean').fillna(False).astype('bool')
            else:
                label_col = label_col.fillna(False).astype('bool')
            out_df[op._name] = label_col

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
                            allowed_ids = df_concat(engine)(
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

    if kind == 'nodes':
        hop_cols = [c for c in out_df.columns if 'hop' in c.lower()]
        edge_ops = [op for op, _ in steps if isinstance(op, ASTEdge)]
        has_output_min = any(getattr(op, 'output_min_hops', None) is not None for op in edge_ops)
        has_output_max = any(getattr(op, 'output_max_hops', None) is not None for op in edge_ops)
        if (has_output_min or has_output_max) and hop_cols:
            hop_col = hop_cols[0]
            has_na = out_df[hop_col].isna()
            if has_output_min:
                out_df = out_df[~has_na]
            elif has_na.any():
                tag_cols = [c for c in out_df.columns if c not in [id, 'id'] + hop_cols]
                has_tag = out_df[id].isin([])  # engine-agnostic all-False boolean Series
                for col in tag_cols:
                    try:
                        vals = out_df[col].fillna(False)
                        if vals.dtype == 'bool' or vals.dtype == 'object':
                            has_tag |= vals.astype(bool)
                    except (TypeError, ValueError):
                        pass
                out_df = out_df[~has_na | has_tag]

    g_df = getattr(g, df_fld)
    # slice 5 (#1755): a seeded result attaches the full node/edge frame via a
    # how='left' merge whose big side (g_df) is scanned in full even for a 1-row
    # out_df. Pre-shrink g_df to the ids actually present (unmatched rows are
    # dropped by how='left' regardless) so the join runs small-vs-small.
    g_df_join = _lean_prefilter_right(out_df, g_df, id, engine) if id in out_df.columns else g_df
    out_df = safe_merge(out_df, g_df_join, on=id, how='left', engine=engine)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('COMBINED[%s] >> %s', kind, dbg_df(out_df))

    if kind == 'nodes' and label_cols:
        seeds_df = label_steps[0][1]._nodes if label_steps and label_steps[0][1]._nodes is not None else None
        seed_ids = seeds_df[[id]].drop_duplicates() if seeds_df is not None and id in seeds_df.columns else None
        label_seeds_true = any(isinstance(op, ASTEdge) and getattr(op, 'label_seeds', False) for op, _ in label_steps)
        if seed_ids is not None:
            if label_seeds_true:
                seeds_with_labels = seed_ids.copy()
                for col in label_cols:
                    if col in out_df.columns:
                        seeds_with_labels[col] = 0
                out_df = safe_merge(out_df, seeds_with_labels, on=id, how='outer', engine=engine)
            else:
                if id in out_df.columns:
                    mask = out_df[id].isin(seed_ids[id])
                    for col in label_cols:
                        if col in out_df.columns:
                            out_df[col] = out_df[col].where(~mask, s_na(engine))
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
                hop_map_df = df_concat(engine)(hop_maps)
                for hc in hop_cols:
                    if hc in hop_map_df.columns:
                        hop_map = hop_map_df[[id, hc]].dropna(subset=[hc]).drop_duplicates(subset=[id]).set_index(id)[hc]
                        mapped_vals = safe_map_series(out_df[id], hop_map)
                        out_df[hc] = out_df[hc].where(out_df[hc].notna(), mapped_vals)

        if hop_cols:
            for idx, (op, _g_step) in enumerate(steps):
                if op._name is None or not isinstance(op, ASTNode) or op._name not in out_df.columns or idx == 0:
                    continue
                prev_op, _ = steps[idx - 1]
                if not isinstance(prev_op, ASTEdge):
                    continue
                prev_step_nodes = label_steps[idx - 1][1]._nodes if idx - 1 < len(label_steps) else None
                prev_hop_cols = (
                    [c for c in prev_step_nodes.columns if 'hop' in c.lower()]
                    if prev_step_nodes is not None
                    else []
                )
                hop_col = None
                if prev_hop_cols:
                    preferred = prev_hop_cols[0]
                    if preferred in out_df.columns:
                        hop_col = preferred
                    elif f'{preferred}_x' in out_df.columns:
                        hop_col = f'{preferred}_x'
                    elif f'{preferred}_y' in out_df.columns:
                        hop_col = f'{preferred}_y'
                min_hop = (
                    prev_op.output_min_hops
                    if prev_op.output_min_hops is not None
                    else (
                        prev_op.min_hops
                        if prev_op.min_hops is not None
                        else (prev_op.hops if prev_op.hops is not None else 1)
                    )
                )
                max_hop = (
                    prev_op.output_max_hops
                    if prev_op.output_max_hops is not None
                    else (
                        prev_op.max_hops
                        if prev_op.max_hops is not None
                        else prev_op.hops
                    )
                )
                if prev_op.to_fixed_point:
                    max_hop = None
                label_mask = out_df[op._name].fillna(False).astype(bool)
                if hop_col is not None and min_hop > 1:
                    label_mask = label_mask & out_df[hop_col].notna() & (out_df[hop_col] >= min_hop)
                if hop_col is not None and max_hop is not None:
                    label_mask = label_mask & out_df[hop_col].notna() & (out_df[hop_col] <= max_hop)
                out_df[op._name] = label_mask

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


def _get_boundary_calls(ops: List[ASTObject]) -> Tuple[List[ASTObject], List[ASTObject], List[ASTObject]]:
    """Split ops into call-prefix, traversal middle, and call-suffix segments.

    This helper is intentionally structural only: validation for illegal
    call/traversal mixing in the middle segment is handled by
    ``_handle_boundary_calls()``.
    """
    from graphistry.compute.ast import ASTCall

    first_traversal = next((i for i, op in enumerate(ops)
                           if not isinstance(op, ASTCall)), len(ops))

    last_traversal = next((i for i, op in reversed(list(enumerate(ops)))
                          if not isinstance(op, ASTCall)), -1)

    prefix = ops[:first_traversal]
    middle = ops[first_traversal:last_traversal + 1] if last_traversal >= 0 else []
    suffix = ops[last_traversal + 1:] if last_traversal >= 0 else []

    return (prefix, middle, suffix)


def _handle_boundary_calls(
    self: Plottable,
    ops: List[ASTObject],
    engine: Union[EngineAbstract, str],
    validate_schema: bool,
    policy,
    context,
    start_nodes: Optional[DataFrameT]
) -> Optional[Plottable]:
    """Handle boundary ``call()`` operations around traversal chains.

    Allowed pattern:
    - ``[call..., traversal..., call...]`` where traversals are ``n()/e()``.

    Rejected pattern:
    - Any interior segment that mixes ``call()`` with traversal ops.

    Returns:
    - ``None`` when there is no boundary-call pattern to handle.
    - A materialized ``Plottable`` when boundary-call execution was applied.
    """
    from graphistry.compute.ast import ASTCall, rows as rows_fn

    has_call = any(isinstance(op, ASTCall) for op in ops)
    has_traversal = any(isinstance(op, (ASTNode, ASTEdge)) for op in ops)

    if not (has_call and has_traversal):
        return None

    prefix, middle, suffix = _get_boundary_calls(ops)

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
                          "(1) let() composition: let({'filtered': [n(...), e(...)], 'enriched': ref('filtered', [call('get_degrees', {'col': 'degree'})])}, output='enriched'), or "
                          "(2) explicit cascading: g1 = g.chain([call(...)]); g2 = g1.chain([n(), e()]); g3 = g2.chain([call(...)]). "
                          "See issues #791, #792"
            )

    logger.debug('Boundary call pattern detected: prefix=%s, middle=%s, suffix=%s',
                len(prefix), len(middle), len(suffix))

    g_temp = self
    suffix_base_graph = g_temp

    if prefix:
        logger.debug('Executing boundary prefix calls: %s', prefix)
        g_temp = _chain_impl(
            g_temp,
            prefix,
            engine,
            validate_schema,
            policy,
            context,
            start_nodes
        )
        suffix_base_graph = g_temp

    if middle:
        logger.debug('Executing middle operations: %s', middle)
        g_temp = _chain_impl(
            g_temp,
            middle,
            engine,
            validate_schema,
            policy,
            context,
            start_nodes
        )

    if suffix:
        logger.debug('Executing boundary suffix calls: %s', suffix)
        if start_nodes is not None:
            setattr(g_temp, "_gfql_start_nodes", start_nodes)
        setattr(g_temp, "_gfql_rows_base_graph", suffix_base_graph)
        setattr(g_temp, "_gfql_shortest_path_backend", getattr(g_temp, "_gfql_shortest_path_backend", "auto"))
        if (
            middle
            and any(getattr(op, "_name", None) is not None for op in middle)
            and isinstance(suffix[0], ASTCall)
            and suffix[0].function == "rows"
            and suffix[0].params.get("binding_ops") is None
            and suffix[0].params.get("source") is None
            and suffix[0].params.get("alias_endpoints") is None
            and all(isinstance(op, (ASTNode, ASTEdge)) for op in middle)
        ):
            suffix = [rows_fn(binding_ops=serialize_binding_ops(middle))] + list(suffix[1:])
        g_temp = _chain_impl(
            g_temp,
            suffix,
            engine,
            validate_schema,
            policy,
            context,
            start_nodes
        )

    return g_temp


def _chain_otel_attrs(
    self: Plottable,
    ops: Union[List[ASTObject], "Chain"],
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    validate_schema: bool = True,
    policy=None,
    context=None,
    start_nodes: Optional[DataFrameT] = None,
) -> Dict[str, Any]:
    chain_len = len(ops.chain) if isinstance(ops, Chain) else len(ops)
    attrs: Dict[str, Any] = {"gfql.chain_len": chain_len}
    if isinstance(ops, Chain):
        attrs["gfql.has_where"] = bool(ops.where)
    if otel_detail_enabled():
        attrs["gfql.engine"] = str(engine)
        attrs["gfql.validate_schema"] = validate_schema
        attrs["gfql.has_policy"] = policy is not None
        attrs["gfql.has_start_nodes"] = start_nodes is not None
    return attrs


def _seeded_scalar_filters(fd: Optional[Dict[str, Any]], df: DataFrameT) -> Optional[Dict[str, Any]]:
    """Resolve a filter dict to plain scalar column==value pairs, or None to bail
    to the general path. Mirrors filter_by_dict.resolve_filter_column exactly for
    the shapes it accepts: the cypher ``label__X: True`` form maps to ``type``
    equality ONLY when no list-valued ``labels`` column exists (labels-containment
    is not scalar equality) and the frame is not edge-shaped — same precedence as
    the live resolver. Anything else (predicates, non-scalar values, absent
    columns) bails, so the full path keeps its exact semantics incl. E301."""
    from graphistry.compute.filter_by_dict import _looks_like_edge_dataframe
    if not fd:
        return {}
    cols = set(df.columns)
    out: Dict[str, Any] = {}
    for k, v in fd.items():
        if not isinstance(v, (int, float, str, bool)):
            return None  # predicate / non-scalar -> bail to the general path
        if k in cols:
            out[k] = v
        elif (isinstance(k, str) and k.startswith("label__") and v is True
              and "labels" not in cols and "type" in cols
              and not _looks_like_edge_dataframe(df)):
            out["type"] = k[len("label__"):]
        else:
            return None  # labels-list / unknown column -> bail
    return out


def _seeded_typed_hop_pandas_cudf(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Plottable]:
    """#1755 lever-3: engine-generic (pandas + cuDF) fast path for a scalar-filtered
    seeded typed 1-hop. Value-identical to the general seeded branch for the covered
    shape (all node/edge filters are plain scalars, directed) — same rows, columns,
    and dtypes; row order and RangeIndex may differ — collapsing it into a
    few DataFrame filters so a seeded lookup lands sub-ms. Uses only the shared
    pandas/cuDF DataFrame API (no numpy array drops) so the same body runs on both
    engines. Returns None to fall back for anything it does not cover (predicates,
    undirected, missing columns) — the caller then runs the general branch."""
    if direction == "undirected":
        return None

    nodes_df, edges_df = g._nodes, g._edges
    if nodes_df is None or edges_df is None:
        return None
    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)

    # from-side seed FIRST: reduce edges to the seed's out-edges before the
    # edge_match compare, so the type filter runs on the tiny frontier rather than
    # all edges — this is what makes a seeded lookup sub-ms. The id filter goes
    # first (int, unique -> ~1 row in one pass) so any remaining object filters
    # (label__X->type) run on that tiny survivor frame, not the whole node table.
    if n0f:
        seed_nodes = nodes_df
        for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
            seed_nodes = seed_nodes[seed_nodes[k] == v]
        edges = edges_df[edges_df[from_col].isin(seed_nodes[node].dropna())]
    else:
        edges = edges_df
    if ef:  # typed edge (edge_match) — now on the reduced frontier
        for k, v in ef.items():
            edges = edges[edges[k] == v]

    # Gather candidate endpoint nodes (both endpoints of surviving edges), then run
    # the dest filter, dangling-edge drop and final-node selection on the small
    # candidate/edge frames. Selecting from nodes_df keeps only real nodes, so the
    # endpoint-in-nodes check subsumes the old NaN-endpoint guard. Membership sets
    # are dropna()'d: pandas .isin matches NaN<->NaN, but the general branch's BFS
    # joins never join on null keys, so a null id/endpoint must not link.
    cand = nodes_df[
        nodes_df[node].isin(edges[src].dropna()) | nodes_df[node].isin(edges[dst].dropna())
    ].drop_duplicates(subset=[node])
    if n2f:  # destination-node filter (to-side)
        n2_cand = cand
        for k, v in n2f.items():
            n2_cand = n2_cand[n2_cand[k] == v]
        n2_ok = n2_cand[node]
    else:
        n2_ok = cand[node]
    to_vals = edges[to_col]
    keep = edges[src].isin(cand[node].dropna()) & edges[dst].isin(cand[node].dropna()) & to_vals.isin(n2_ok.dropna())
    edges = edges[keep]
    cand = cand[cand[node].isin(edges[src]) | cand[node].isin(edges[dst])]
    return g.nodes(cand).edges(edges)


def _seeded_typed_return_dst_pandas_cudf(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    """#1755 cypher RETURN-alias fast path: like _seeded_typed_hop_pandas_cudf but
    returns ONLY the destination (RETURN-alias) node rows + surviving edges — no
    seed-node gather, no Plottable round-trip — so the seeded cypher projection
    lands sub-ms. Engine-generic (pandas + cuDF): only the shared DataFrame API,
    no numpy array drops. Returns ``(dst_node_rows, edges)`` or None to fall back."""
    if direction == "undirected":
        return None
    nodes_df, edges_df = g._nodes, g._edges
    if nodes_df is None or edges_df is None:
        return None
    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None or not n0f:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)
    # id-first seed reduction: filter by the id column first (int/unique -> ~1 row)
    # so any remaining object filters (label__X->type) run on the tiny survivor
    # frame, never materializing an object column over the whole node table.
    # Membership sets are dropna()'d: pandas .isin matches NaN<->NaN, but the full
    # pipeline's joins never join on null keys, so a null id/endpoint must not link.
    seed_nodes = nodes_df
    for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
        seed_nodes = seed_nodes[seed_nodes[k] == v]
    edges = edges_df[edges_df[from_col].isin(seed_nodes[node].dropna())]
    if ef:
        for k, v in ef.items():
            edges = edges[edges[k] == v]
    # destination nodes = real nodes that are edge to-endpoints, then the dest
    # filter, dangling-edge drop and dedup on the small dst/edge frames.
    dstn = nodes_df[nodes_df[node].isin(edges[to_col].dropna())]
    if n2f:
        for k, v in n2f.items():
            dstn = dstn[dstn[k] == v]
    edges = edges[edges[to_col].isin(dstn[node].dropna())]
    dstn = dstn[dstn[node].isin(edges[to_col].dropna())].drop_duplicates(subset=[node])
    return dstn, edges


def _seeded_typed_return_dst_polars(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    """#1755 polars analog of _seeded_typed_return_dst_pandas_cudf: same seed-first
    reduction (seed out-edges -> typed-edge filter -> destination nodes) expressed
    with polars filters, so a seeded cypher RETURN on polars/polars-gpu also lands
    sub-ms. Returns ``(dst_node_rows, edges)`` (polars frames) or None to fall back
    to the full lazy pipeline. Byte-identical node set to the full path for the
    covered shape (scalar filters, directed, single hop)."""
    import polars as pl
    if direction == "undirected":
        return None
    nodes_df, edges_df = g._nodes, g._edges
    if nodes_df is None or edges_df is None:
        return None
    node_cols, edge_cols = set(nodes_df.columns), set(edges_df.columns)

    def _sc(fd: Any, cols: set) -> Optional[Dict[str, Any]]:
        if not fd:
            return {}
        out: Dict[str, Any] = {}
        for k, v in fd.items():
            if not isinstance(v, (int, float, str, bool)):
                return None
            if k in cols:
                out[k] = v
            elif isinstance(k, str) and k.startswith("label__") and v is True and "type" in cols:
                out["type"] = k[len("label__"):]
            else:
                return None
        return out

    n0f = _sc(n0.filter_dict, node_cols)
    n2f = _sc(n2.filter_dict, node_cols)
    ef = _sc(e1.edge_match, edge_cols)
    if n0f is None or n2f is None or ef is None or not n0f:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)

    # from-side seed: reduce the node frame to the seed rows, take their ids.
    seed_nodes = nodes_df
    for k, v in n0f.items():
        seed_nodes = seed_nodes.filter(pl.col(k) == v)
    from_ids = seed_nodes.get_column(node)
    if from_ids.len() == 0:
        return nodes_df.clear(), edges_df.clear()
    edges = edges_df.filter(pl.col(from_col).is_in(from_ids))
    for k, v in ef.items():  # typed edge on the reduced frontier
        edges = edges.filter(pl.col(k) == v)
    dst_ids = edges.get_column(to_col).drop_nulls().unique()
    dstn = nodes_df.filter(pl.col(node).is_in(dst_ids))
    for k, v in n2f.items():  # destination-node filter
        dstn = dstn.filter(pl.col(k) == v)
    # drop dangling edges + dedup destination nodes (mirror the pandas tail)
    keep_ids = dstn.get_column(node)
    edges = edges.filter(pl.col(to_col).is_in(keep_ids))
    dstn = dstn.filter(pl.col(node).is_in(edges.get_column(to_col))).unique(subset=[node], maintain_order=True)
    return dstn, edges


def _try_chain_fast_path(
    g_in: Plottable,
    ops: List[ASTObject],
    engine_concrete: Engine,
    start_nodes: Optional[DataFrameT] = None,
) -> Optional[Plottable]:
    """Degenerate-shape fast path (pandas/cuDF): node-only ``MATCH (n)`` or a plain
    single-hop ``MATCH (a)-[e]->(b)`` skip the forward/backward/combine BFS machinery.
    Returns the result Plottable, or ``None`` to fall through to the full path.

    Same node/edge sets + VALUES as the full machinery (trackA_golden + hop/chain
    suites); the 1-hop additionally preserves int node dtypes (the full path upcasts
    int→float via merge). Gated to unnamed/unqueried nodes + a plain single-hop edge;
    filtered-undirected and seeded chains fall through. polars/dask/spark also fall
    through (own fast path / lazy semantics)."""
    from graphistry.compute.filter_by_dict import filter_by_dict

    if engine_concrete not in (Engine.PANDAS, Engine.CUDF):
        return None
    if start_nodes is not None:
        return None  # seeded chains use the full path (fast path has no seed)
    engine_abs = EngineAbstract(engine_concrete.value)

    def _materialize_fast_path_graph() -> Plottable:
        from graphistry.compute.ComputeMixin import _coerce_input_formats  # lazy — avoids circular import
        g = g_in.materialize_nodes(engine=EngineAbstract(engine_concrete.value))
        return _coerce_input_formats(g, engine_concrete)

    if len(ops) == 1:
        n0 = ops[0]
        if not (isinstance(n0, ASTNode) and n0._name is None and n0.query is None):
            return None
        g = _materialize_fast_path_graph()
        if g._nodes is None:
            return None
        nodes = filter_by_dict(g._nodes, n0.filter_dict, engine_abs) if n0.filter_dict else g._nodes
        edges = g._edges.iloc[0:0] if g._edges is not None else None
        return g.nodes(nodes).edges(edges) if edges is not None else g.nodes(nodes)

    if len(ops) != 3:
        return None
    n0, e1, n2 = ops
    if not (isinstance(n0, ASTNode) and n0._name is None and n0.query is None):
        return None
    if not (isinstance(n2, ASTNode) and n2._name is None and n2.query is None):
        return None
    if not (isinstance(e1, ASTEdge) and e1.is_simple_single_hop()
            and e1.source_node_match is None
            and e1.destination_node_match is None and e1._name is None
            and e1.source_node_query is None and e1.destination_node_query is None
            and e1.edge_query is None and not e1.include_zero_hop_seed
            and not e1.prune_to_endpoints):  # prune keeps only the arrival side -> full path
        return None
    # #1755 lever-3: a typed edge (edge_match, e.g. -[:HAS_CREATOR]->) is a plain
    # equality/predicate filter on the edge frame — apply it in the fast-path body
    # below rather than falling through to the full two-pass machinery. source/dest
    # node match + edge_query (richer predicates) still bail above.
    direction = e1.direction
    unconstrained = not n0.filter_dict and not n2.filter_dict
    if not unconstrained and direction == "undirected":
        return None  # filtered-undirected (OR of both directions) -> full path
    g = _materialize_fast_path_graph()
    if g._nodes is None or g._edges is None:
        return None
    src, dst, node = g._source, g._destination, g._node
    if src is None or dst is None or node is None:
        return None  # no edge/node bindings -> can't fast-path; full path handles it
    concat = df_concat(engine_concrete)
    if unconstrained:
        # No node filter to reduce by: validate BOTH endpoints against the full
        # node table (the full path drops dangling edges via its joins). dropna so
        # a NaN node id can't validate a NaN endpoint — .isin treats NaN as
        # matchable but the BFS joins never match NaN<->NaN.
        node_ids = g._nodes[node].dropna()
        edges = g._edges[g._edges[src].isin(node_ids) & g._edges[dst].isin(node_ids)]
        if e1.edge_match:
            # typed edge (e.g. -[:HAS_CREATOR]->) — same edge-frame filter the full
            # hop applies, so the result set is identical.
            edges = filter_by_dict(edges, e1.edge_match, engine_abs)
    else:
        # #1755 lever-3 seed-first: a seeded 1-hop must be O(result), not O(E).
        # Reduce edges by the selective node filter(s) BEFORE the typed-edge scan
        # and endpoint validation, so the expensive object/isin passes run on the
        # tiny frontier, not all edges. The from-side ids come from the node table
        # (so that endpoint is validated); the node gather below validates the to
        # side and drops any edge dangling off the node table.
        # pandas + cuDF: a scalar-filtered seeded typed hop collapses to a few
        # DataFrame filters (sub-ms); falls back to the general branch below for
        # predicates / undirected / missing columns (and non-pandas/cuDF engines).
        if engine_concrete in (Engine.PANDAS, Engine.CUDF):
            _fast_res = _seeded_typed_hop_pandas_cudf(g, n0, n2, e1, src, dst, node, direction)
            if _fast_res is not None:
                return _fast_res
        from_col, to_col = (src, dst) if direction == "forward" else (dst, src)
        edges = g._edges
        if n0.filter_dict:
            from_ids = filter_by_dict(g._nodes, n0.filter_dict, engine_abs)[node]
            edges = edges[edges[from_col].isin(from_ids)]
        if e1.edge_match:
            edges = filter_by_dict(edges, e1.edge_match, engine_abs)
        if n2.filter_dict:
            # Apply the destination filter to the SMALL set of gathered dst nodes,
            # not the full node table — an O(N) object/type scan on all nodes is
            # exactly the tax we're removing. Gather the frontier's dst nodes
            # (small isin key), filter those, then drop edges to the losers.
            to_present = edges[to_col].dropna().unique()
            to_nodes = filter_by_dict(
                g._nodes[g._nodes[node].isin(to_present)], n2.filter_dict, engine_abs)
            edges = edges[edges[to_col].isin(to_nodes[node])]
        # Validate endpoints + build result nodes on the reduced edge set (small
        # isin key -> small hashtable; no O(E)-values scan). Engine-agnostic
        # (pandas + cuDF): gather candidate endpoint nodes, drop edges dangling off
        # the node table, then keep only nodes still referenced by a surviving edge.
        ep = concat([
            edges[[src]].rename(columns={src: node}),
            edges[[dst]].rename(columns={dst: node}),
        ]).drop_duplicates()
        cand = g._nodes[g._nodes[node].isin(ep[node])].drop_duplicates(subset=[node])
        valid = cand[node].dropna()
        edges = edges[edges[src].isin(valid) & edges[dst].isin(valid)]
        final = concat([
            edges[[src]].rename(columns={src: node}),
            edges[[dst]].rename(columns={dst: node}),
        ]).drop_duplicates()
        nodes = cand[cand[node].isin(final[node])]
        return g.nodes(nodes).edges(edges)
    endpoints = concat([
        edges[[src]].rename(columns={src: node}),
        edges[[dst]].rename(columns={dst: node}),
    ]).drop_duplicates()
    nodes = g._nodes[g._nodes[node].isin(endpoints[node])]
    # match the full path's merge, which collapses duplicate node-id rows
    nodes = nodes.drop_duplicates(subset=[node])
    return g.nodes(nodes).edges(edges)


@otel_traced("gfql.chain", attrs_fn=_chain_otel_attrs)
def chain(
    self: Plottable,
    ops: Union[List[ASTObject], Chain],
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    validate_schema: bool = True,
    policy=None,
    context=None,
    start_nodes: Optional[DataFrameT] = None
) -> Plottable:
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
    :param start_nodes: Optional node wavefront for the first traversal step

    :returns: Plotter
    :rtype: Plotter
    """
    if context is None:
        from .execution_context import ExecutionContext
        context = ExecutionContext()

    # Resolve engine from original data BEFORE coercion so GPU mode is preserved end-to-end.
    # _coerce_input_formats then converts input formats (polars, arrow, spark, dask) to that engine.
    if isinstance(engine, str):
        engine = EngineAbstract(engine)
    from graphistry.compute.ComputeMixin import _coerce_input_formats  # lazy — avoids circular import
    engine_concrete_early = resolve_engine(engine, self)
    if engine_concrete_early in (Engine.POLARS, Engine.POLARS_GPU):
        # Clean dependency errors BEFORE _coerce_input_formats (which imports polars to
        # coerce frames) so the user sees actionable install guidance, not a raw ImportError
        # deep in coercion / the lazy engine. Both engines need polars; polars-gpu also needs
        # the RAPIDS cudf_polars stack (checked here so it's consistent regardless of whether a
        # given query reaches a GPU collect, and reads as an install issue rather than the
        # genuine not-GPU-capable signal from lazy._engine_for).
        try:
            import polars  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"GFQL engine={engine_concrete_early.value!r} requires the 'polars' package; "
                "install it with `pip install polars` (or use engine='pandas')."
            ) from e
        if engine_concrete_early == Engine.POLARS_GPU:
            import importlib.util
            if importlib.util.find_spec("cudf_polars") is None:
                raise ImportError(
                    "GFQL engine='polars-gpu' requires the RAPIDS cudf_polars stack (NVIDIA GPU). "
                    "Install RAPIDS/cudf_polars, or use engine='polars' for native CPU execution."
                )
    self = _coerce_input_formats(self, engine_concrete_early)

    if engine_concrete_early in POLARS_ENGINES:
        # Native polars chain lives in a dedicated dispatched module so the
        # production pandas/cuDF orchestration below stays untouched (see
        # no-silent-fallback policy). Correctness gated by differential parity.
        # POLARS_GPU = the same lazy engine with the GPU execution target.
        # (Dependency guards for polars / cudf_polars are above, pre-coercion.)
        if validate_schema:
            Chain(ops if not isinstance(ops, Chain) else ops.chain).validate(collect_all=False)
        from graphistry.compute.gfql.lazy.engine.polars.chain import chain_polars
        from graphistry.compute.gfql.lazy import target_mode, ExecutionTarget
        # NO pandas fallback here (no-silent-fallback policy): chain_polars raises
        # NotImplementedError for deferred features (var-length/multi-hop edges,
        # undirected multi-edge); that honest signal propagates to the caller.
        _tgt = ExecutionTarget.GPU if engine_concrete_early == Engine.POLARS_GPU else ExecutionTarget.CPU
        with target_mode(_tgt):
            return chain_polars(self, ops, start_nodes=start_nodes)

    if policy:
        from graphistry.compute.gfql.call.executor import _thread_local as call_thread_local
        old_policy = getattr(call_thread_local, 'policy', None)
        try:
            call_thread_local.policy = policy
            return _chain_impl(self, ops, engine, validate_schema, policy, context, start_nodes)
        finally:
            call_thread_local.policy = old_policy
    else:
        return _chain_impl(self, ops, engine, validate_schema, policy, context, start_nodes)


def _chain_impl(
    self: Plottable,
    ops: Union[List[ASTObject], Chain],
    engine: Union[EngineAbstract, str],
    validate_schema: bool,
    policy,
    context,
    start_nodes: Optional[DataFrameT]
) -> Plottable:
    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    if isinstance(ops, Chain):
        ops = ops.chain

    if validate_schema:
        Chain(ops).validate(collect_all=False)

    from graphistry.compute.ast import ASTCall

    schema_changers = ['umap', 'hypergraph']

    schema_changer_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, ASTCall) and op.function in schema_changers:
            schema_changer_idx = i
            break

    if schema_changer_idx is not None:
        if len(ops) == 1:
            from graphistry.compute.gfql.call.executor import execute_call
            from graphistry.compute.exceptions import GFQLTypeError, ErrorCode

            engine_concrete = resolve_engine(engine, self)
            schema_changer = ops[0]

            if not isinstance(schema_changer, ASTCall):
                raise GFQLTypeError(
                    code=ErrorCode.E201,
                    message="Schema-changer operation must be ASTCall",
                    field="operation",
                    value=type(schema_changer).__name__,
                    suggestion="Use call('umap', {...}) or call('hypergraph', {...})"
                )

            if validate_schema:
                validate_chain_schema(self, ops, collect_all=False)

            return execute_call(self, schema_changer.function, schema_changer.params, engine_concrete, policy=policy, context=context)
        before = ops[:schema_changer_idx]
        schema_changer = ops[schema_changer_idx]
        rest = ops[schema_changer_idx + 1:]

        g_temp = _chain_impl(self, before, engine, validate_schema, policy, context, start_nodes=None) if before else self
        g_temp2 = _chain_impl(g_temp, [schema_changer], engine, validate_schema, policy, context, start_nodes=None)
        return _chain_impl(g_temp2, rest, engine, validate_schema, policy, context, start_nodes=None) if rest else g_temp2

    if len(ops) == 0:
        return self

    logger.debug('orig chain >> %s', ops)

    engine_concrete = resolve_engine(engine, self)
    logger.debug('chain engine: %s => %s', engine, engine_concrete)

    boundary_result = _handle_boundary_calls(self, ops, engine, validate_schema, policy, context, start_nodes)
    if boundary_result is not None:
        return boundary_result

    if validate_schema:
        validate_chain_schema(self, ops, collect_all=False)

    # The fast path skips the policy hook dispatch (prechain/postchain below), so only
    # take it when no policy is attached — policy-bearing queries must keep hook firing.
    if not policy:
        _fast = _try_chain_fast_path(self, ops, engine_concrete, start_nodes)
        if _fast is not None:
            return _fast

    if isinstance(ops[0], ASTEdge):
        logger.debug('adding initial node to ensure initial link has needed reversals')
        ops = cast(List[ASTObject], [ ASTNode() ]) + ops

    if isinstance(ops[-1], ASTEdge):
        logger.debug('adding final node to ensure final link has needed reversals')
        ops = ops + cast(List[ASTObject], [ ASTNode() ])

    logger.debug('final chain >> %s', ops)

    original_edge = self._edge

    g_out = None
    error = None
    success = False

    try:
        g = self.materialize_nodes(engine=EngineAbstract(engine_concrete.value))

        if g._edges is None:
            added_edge_index = False
        elif g._edge is None:
            GFQL_EDGE_INDEX = generate_safe_column_name('edge_index', g._edges, prefix='__gfql_', suffix='__')

            added_edge_index = True
            # Attach the synthetic per-edge id WITHOUT copying edge data (#1670):
            # the previous reset_index(drop=False) + rename deep-copied AND
            # block-consolidated the whole edge frame (~70ms @2M edges) on every
            # chain call — even node-only queries. A shallow copy + assigning the
            # index as a column yields the identical id values (the frame's index)
            # with no O(E) data copy. The column is internal-only — dropped on
            # every exit path (see added_edge_index consumers below) — so only
            # uniqueness matters.
            indexed_edges_df = g._edges.copy(deep=False)
            indexed_edges_df[GFQL_EDGE_INDEX] = indexed_edges_df.index
            g = g.edges(indexed_edges_df, edge=GFQL_EDGE_INDEX)
            # The shallow copy above only ADDS the synthetic id column; the indexed
            # src/dst columns are preserved by value. Re-point any resident #1658
            # adjacency index at the new edge frame so the seeded fast path still
            # engages through gfql()/Cypher chains (else the identity guard misses and
            # every chain hop falls back to the O(E) scan).
            from graphistry.compute.gfql.index import get_registry, set_registry
            _reg = get_registry(g)
            if not _reg.is_empty():
                g = set_registry(g, _reg.rebind_edges(indexed_edges_df))
        else:
            added_edge_index = False

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

        # Phase 1: Forward pass to build step-local wavefronts.
        logger.debug('======================== FORWARDS ========================')
        g_stack : List[Plottable] = []
        for i, op in enumerate(ops):
            if isinstance(op, ASTCall):
                current_g = g_stack[-1] if g_stack else g
                prev_step_nodes = None  # ASTCall doesn't use wavefronts
            else:
                current_g = g
                prev_step_nodes = (
                    start_nodes
                    if len(g_stack) == 0
                    else g_stack[-1]._nodes
                )

            g_step = (
                op.execute(
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
                logger.debug('nodes: %s', dbg_df(g_step._nodes))
                logger.debug('edges: %s', dbg_df(g_step._edges))

        all_astcall = all(isinstance(op, ASTCall) for op in ops)

        if all_astcall:
            g_out = g_stack[-1]
            if added_edge_index:
                final_edges_df = g_out._edges.drop(columns=[g._edge])
                g_out = self.nodes(g_out._nodes).edges(final_edges_df, edge=original_edge)
            success = True
        else:
            # Phase 2: Backward pass to propagate downstream constraints.
            g_stack_reverse : List[Plottable] = []
            for (op, g_step) in zip(reversed(ops), reversed(g_stack)):
                prev_loop_step = g_stack[-1] if len(g_stack_reverse) == 0 else g_stack_reverse[-1]
                if len(g_stack_reverse) == len(g_stack) - 1:
                    prev_orig_step = None
                else:
                    prev_orig_step = g_stack[-(len(g_stack_reverse) + 2)]
                prev_wavefront_nodes = prev_loop_step._nodes
                if g._node is not None and prev_wavefront_nodes is not None and g._nodes is not None:
                    # slice 5 (#1755): re-attach full node columns to the reverse
                    # wavefront. The inner merge scans all of g._nodes; when the
                    # wavefront is tiny (seeded chain), the byte-identical result
                    # is an isin membership filter (see _lean_intersect_full).
                    _lean = _lean_intersect_full(g._nodes, prev_wavefront_nodes[[g._node]], g._node, engine_concrete)
                    prev_wavefront_nodes = _lean if _lean is not None else safe_merge(
                        g._nodes,
                        prev_wavefront_nodes[[g._node]],
                        on=g._node,
                        how='inner',
                        engine=engine_concrete
                    )
                target_wave_front_nodes = prev_orig_step._nodes if prev_orig_step is not None else None
                if g._node is not None and target_wave_front_nodes is not None and g._nodes is not None:
                    _lean = _lean_intersect_full(g._nodes, target_wave_front_nodes[[g._node]], g._node, engine_concrete)
                    target_wave_front_nodes = _lean if _lean is not None else safe_merge(
                        g._nodes,
                        target_wave_front_nodes[[g._node]],
                        on=g._node,
                        how='inner',
                        engine=engine_concrete
                    )
                assert prev_loop_step._nodes is not None

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
                    prev_ids = prev_wavefront_nodes[node_id] if prev_wavefront_nodes is not None else None
                    target_ids = target_wave_front_nodes[node_id] if target_wave_front_nodes is not None else None

                    if is_undirected:
                        if prev_ids is not None and target_ids is not None:
                            mask = ((edges_df[src_col].isin(prev_ids) & edges_df[dst_col].isin(target_ids))
                                    | (edges_df[src_col].isin(target_ids) & edges_df[dst_col].isin(prev_ids)))
                            edges_df = edges_df[mask]
                        elif prev_ids is not None:
                            edges_df = edges_df[edges_df[src_col].isin(prev_ids) | edges_df[dst_col].isin(prev_ids)]
                        elif target_ids is not None:
                            edges_df = edges_df[edges_df[src_col].isin(target_ids) | edges_df[dst_col].isin(target_ids)]
                    else:
                        next_col, prev_col = (src_col, dst_col) if op.direction == 'reverse' else (dst_col, src_col)
                        edges_df = _filter_edges_by_endpoint(edges_df, prev_wavefront_nodes, node_id, next_col)
                        edges_df = _filter_edges_by_endpoint(edges_df, target_wave_front_nodes, node_id, prev_col)

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
                    g_step_reverse = op.reverse().execute(
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
                    logger.debug('nodes: %s', dbg_df(g_step._nodes))
                    logger.debug('edges: %s', dbg_df(g_step._edges))

            # Phase 3: Materialize final node/edge outputs from pruned steps.
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
                final_edges_df = final_edges_df.drop(columns=[g._edge])
                # `self` restores the original edge binding, but carry the materialized
                # `g._node` explicitly: an edges-only `self._node is None` would drop the
                # node binding, making the reconciliation concat synthesize a corrupt
                # `None`-named column (and a void-block concat crash on newer pandas).
                g_out = self.nodes(final_nodes_df, g._node).edges(final_edges_df, edge=original_edge)
            else:
                g_out = g.nodes(final_nodes_df).edges(final_edges_df)

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
                endpoints = align_shared_column_dtypes(g_out._nodes, endpoints)
                g_out = g_out.nodes(safe_row_concat([g_out._nodes, endpoints], ignore_index=True, sort=False).drop_duplicates(subset=[g_out._node]))

            success = True

    except Exception as e:
        error = e

    finally:
        postchain_policy_error = None
        if policy and 'postchain' in policy:

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

            if error is not None:
                postchain_context['error'] = str(error)  # type: ignore
                postchain_context['error_type'] = type(error).__name__  # type: ignore

            try:
                policy['postchain'](postchain_context)
            except PolicyException as e:
                postchain_policy_error = e

        policy_error = None
        if policy and 'postload' in policy:

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

            if error is not None:
                policy_context['error'] = str(error)  # type: ignore
                policy_context['error_type'] = type(error).__name__  # type: ignore

            try:
                policy['postload'](policy_context)

            except PolicyException as e:
                if e.query_type is None:
                    e.query_type = 'chain'
                if e.data_size is None:
                    e.data_size = stats
                policy_error = e

    if postchain_policy_error is not None:
        if error is not None:
            raise postchain_policy_error from error
        else:
            raise postchain_policy_error
    elif policy_error is not None:
        if error is not None:
            raise policy_error from error
        else:
            raise policy_error
    elif error is not None:
        raise error

    return cast(Plottable, g_out)
