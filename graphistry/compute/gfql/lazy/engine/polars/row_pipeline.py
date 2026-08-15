"""Native polars lowering for the cypher row pipeline (Phase 2, vectorized).

NO-CHEATING contract: no pandas bridge — ``lower_expr`` returns ``None`` for anything not
provably pandas-equivalent, and ``chain._run_calls_polars`` raises NotImplementedError (NIE)
pointing at ``engine='pandas'``. Differential parity vs pandas is the correctness gate.

Lowered: property access / bare columns / literals; arithmetic/comparison/boolean BinaryOp,
UnaryOp, IsNullOp, CaseWhen (ternary); function whitelist (coalesce/abs/sqrt/sign + dtype-gated
size/substring/toInteger/toFloat/toBoolean/toString); homogeneous list literals and
``x IN [literals]``. Ops wired native: select/with_/return_ projection, order_by, where_rows,
group_by, unwind. Everything else (mixed/nested/empty list, map, subscript, other functions,
temporal arithmetic) → NIE.
"""
from __future__ import annotations

import operator
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.expr_parser import ExprNode, FunctionCall
    from graphistry.compute.ast import ASTObject

    # Within ONE call the path bag and its per-alias frames are the same polars
    # flavour — the generic builder works in LazyFrames, the indexed one in eager
    # frames. A constrained TypeVar says that; a plain union does not, and then
    # `state.join(lookup)` cannot type-check.
    PolarsFrameT = TypeVar("PolarsFrameT", "pl.DataFrame", "pl.LazyFrame")

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.cache_registry import register_clearable_dict
from graphistry.utils.json import JSONVal
# Engine-neutral wire-format payload types (ASTCall.params). Shapes are safelist-validated
# (gfql/call/validation.py) before reaching these helpers, so the runtime isinstance/len
# checks below are defense-in-depth, not the contract.
from graphistry.compute.gfql.agg_types import (
    GFQL_NUMERIC_ONLY_AGGREGATIONS,
    numeric_agg_all_null_value,
    polars_non_numeric_agg_dtype,
    raise_non_numeric_aggregation,
)
from graphistry.compute.gfql.call.support import AggSpec, OrderKey, SelectItem
from .dtypes import is_float as _dtype_is_float, is_int as _dtype_is_int, is_numeric as _dtype_is_numeric, is_stringlike as _dtype_is_stringlike
# Same-package sibling holding the var-length specializations. Safe at module scope:
# `varlen_rows` has no runtime module-level imports of its own (polars and the pandas
# mixin are both function-local there), so it cannot cycle back through this module.
from .varlen_rows import (
    _directed_varlen_reachable_polars,
    _directed_fixed_point_binding_rows_polars,
)


# Active row-table schema (col -> dtype), set around lowering so lower_expr can infer FLOAT
# operands for the NaN guard. Lowering contextvars live in the per-engine `lowering_context`
# registry (aliased here to keep call sites terse); the whole-entity identity sentinel is the
# shared cypher-lowering constant, NOT a local literal.
from .lowering_context import (
    SCHEMA as _SCHEMA,
    NODE_ID as _NODE_ID,
    COLUMNS_NAN_FREE as _COLUMNS_NAN_FREE,
)
from graphistry.compute.gfql.same_path_types import NODE_IDENTITY_COLUMN as _NODE_ID_TOKEN
from graphistry.compute.gfql.identifiers import (
    TRAIL_EDGE_IDENT_COL,
    WALK_CURRENT_COL,
    WALK_FROM_COL,
    WALK_TO_COL,
    trail_column_name,
)

# Ops needing the NaN guard: polars treats NaN as the LARGEST value (>/>=/== TRUE), but
# IEEE/Python/pandas/Cypher compare NaN as FALSE (!= TRUE; Neo4j TCK agrees). Float operands
# get masked to the IEEE answer; ``is_nan()`` is float-only, hence the dtype inference.
_NAN_GUARD_OPS = frozenset({"<", ">", "<=", ">=", "=", "==", "<>", "!="})
_NAN_NE_OPS = frozenset({"<>", "!="})
_ORDER_OPS = frozenset({"<", ">", "<=", ">="})
# ops whose numeric-vs-string operands make polars raise (compare AND arithmetic)
_NUMSTR_OPS = _NAN_GUARD_OPS | frozenset({"+", "-", "*", "/", "%"})


def _parser():
    from graphistry.compute.gfql.row.pipeline import _gfql_expr_runtime_parser_bundle
    bundle = _gfql_expr_runtime_parser_bundle()
    if bundle is None:
        return None
    parse_expr, _validate, _mod = bundle
    return parse_expr


# Cypher binary op → polars expr via operator.* (pl.Expr implements the Python arithmetic/
# rich-comparison protocol). Null-propagating semantics match pandas here (parity-verified);
# anything subtler returns None upstream.
_BINOP_FNS: Dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "%": operator.mod,  # non-numeric fallback only: numeric % is conformed to TRUNCATED semantics in lower_expr (#1900)
    "=": operator.eq, "==": operator.eq, "<>": operator.ne, "!=": operator.ne,
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}


def _apply_binop(op: str, left: pl.Expr, right: pl.Expr) -> Optional[pl.Expr]:
    fn = _BINOP_FNS.get(op)
    if fn is not None:
        return fn(left, right)
    o = op.upper()
    if o in ("AND", "OR"):
        # Kleene 3VL: polars Boolean &/| already match (true|null=true, false&null=false,
        # null&null=null). Cast to Boolean so a bare null lit doesn't raise
        # `bitand not supported for dtype null`; no-op on a real Boolean column.
        import polars as pl
        lb, rb = left.cast(pl.Boolean), right.cast(pl.Boolean)
        return lb & rb if o == "AND" else lb | rb
    return None


def _resolve_property(alias: str, prop: str, columns: Sequence[str]) -> Optional[str]:
    """Resolve ``alias.prop`` to a row-table column (None if ambiguous/absent). Prefer the
    multi-entity prefixed form (``n.val``) over single-entity bare ``val`` + ``alias`` marker
    column, avoiding cross-entity collisions."""
    prefixed = f"{alias}.{prop}"
    if prefixed in columns:
        return prefixed
    if prop == _NODE_ID_TOKEN and alias in columns:
        # Whole-entity identity key (#1650 lowering groups by `alias.__gfql_node_id__`).
        # pandas' bindings table carries it as a join-residue column; the polars table
        # deliberately doesn't — its value IS the bare alias id column.
        return alias
    if prop in columns and alias in columns:
        return prop
    return None


def _lower_function(node: FunctionCall, columns: Sequence[str]) -> Optional[pl.Expr]:
    """Lower a whitelisted scalar cypher function to polars, or None to defer. Only
    parity-verified mappings admitted; anything else returns None (caller NIEs, never guesses)."""
    import polars as pl  # function-local: polars is an optional dependency
    name = node.name.lower()
    if name == "__cypher_case_eq__" and len(node.args) == 2:
        # Simple-CASE equality marker (`CASE x WHEN v`). openCypher simple CASE
        # uses '=': null NEVER matches (conformed #1900 -- `CASE x WHEN null`
        # matches no row and falls to ELSE, mirroring the pandas evaluator);
        # the general form carries pandas' bool/numeric cross-dtype rules --
        # decline it rather than diverge.
        from graphistry.compute.gfql.expr_parser import Literal as _Lit
        a_node, b_node = node.args
        if (isinstance(b_node, _Lit) and b_node.value is None) or (
            isinstance(a_node, _Lit) and a_node.value is None
        ):
            return pl.lit(False)
        return None
    args: List[pl.Expr] = []
    for arg in node.args:
        lowered = lower_expr(arg, columns)
        if lowered is None:
            return None
        args.append(lowered)
    if name == "coalesce" and args:
        # cypher coalesce = first non-null; pl.coalesce has identical semantics.
        return pl.coalesce(args)
    if name == "abs" and len(args) == 1:
        return args[0].abs()
    # neo4j/openCypher numeric fns (parity-verified vs the pandas engine).
    if name == "sqrt" and len(args) == 1:
        # sqrt of a negative -> NaN on both pandas and polars; Float64 cast so sqrt(int)
        # returns float like neo4j/pandas; parity-verified.
        return args[0].cast(pl.Float64).sqrt()
    if name == "sign" and len(args) == 1:
        # polars .sign() == np.sign (-1/0/1; null/NaN preserved); neo4j sign() returns an
        # Integer, so cast to match the pandas engine (which yields int). Parity-verified.
        return args[0].sign().cast(pl.Int64)
    if name in {"floor", "ceil", "ceiling"} and len(args) == 1:
        # Float64 cast like sqrt: neo4j floor/ceil return Float, and the pandas engine
        # astype(float)s — bare polars .floor() on an Int64 column stays Int64.
        x = args[0].cast(pl.Float64)
        return x.ceil() if name in {"ceil", "ceiling"} else x.floor()
    if name == "round" and len(args) in {1, 2}:
        from graphistry.compute.gfql.expr_parser import Literal
        ndigits = 0
        if len(args) == 2:
            arg1 = node.args[1]
            # isinstance narrowing (a bare .value probe also matched non-Literal nodes)
            if not isinstance(arg1, Literal) or not isinstance(arg1.value, int) \
                    or isinstance(arg1.value, bool):
                return None  # non-literal precision -> defer (honest NIE)
            ndigits = arg1.value
        if ndigits < 0:
            return None  # neo4j raises on negative precision; decline (honest NIE)
        # neo4j tie-breaking (matches the pandas engine): precision 0 -> ties toward
        # +inf; precision > 0 -> ties away from zero (HALF_UP). polars' .round default
        # (half-to-even) would be a wrong answer vs the spec. p=0 uses a floor+frac
        # kernel (NOT floor(x+0.5): the +0.5 rounds when x is 1 ulp below a tie —
        # round(0.49999999999999994) must be 0.0). p>0 uses the native mode= (bit-exact;
        # a manual scale/divide formula picks up 1-ulp noise from polars' reassociating
        # optimizer). Requires polars >= 1.29 for the mode kwarg (see setup.py extra;
        # the kwarg shipped in py-1.29.0, pola-rs/polars#22248 — NOT 1.5). The trailing
        # + 0.0 normalizes -0.0 like the pandas kernel's scale/divide does (polars'
        # native mode keeps -0.0: round(-0.04, 1) was 0.0 vs -0.0, dgx-repro'd).
        x = args[0].cast(pl.Float64)
        if ndigits > 308:
            # Identity, mirroring the pandas kernel's p>308 guard: polars' own
            # identity only starts at p>=326 (its [300,325] split-multiplier window
            # quantizes tiny values where pandas returns identity), and p >= 2**32
            # is a raw PyO3 OverflowError (decimals is u32) — #1677 wave-2.
            return x + 0.0
        if ndigits == 0:
            fl = x.floor()
            return fl + ((x - fl) >= 0.5).cast(pl.Float64)  # ties toward +inf
        return x.round(ndigits, mode="half_away_from_zero") + 0.0
    if name in {"tolower", "toupper", "lower", "upper"} and len(args) == 1:
        # toLower/toUpper + GQL-conformance aliases lower/upper (as neo4j accepts both).
        # String-only like neo4j (type error there); a non-string column must decline —
        # pandas declines too, and bare .str here raised a non-NIE SchemaError on
        # polars-gpu (dgx-repro'd).
        if _expr_output_dtype(args[0]) != pl.String:
            return None
        to_lower = name in {"tolower", "lower"}
        return args[0].str.to_lowercase() if to_lower else args[0].str.to_uppercase()
    if name == "size" and len(args) == 1:
        # size(x): #chars (String) or #elements (List) — different polars ops, so gate by output
        # dtype. str.len_chars == pandas str.len (code points); list.len parity; null/empty
        # preserved — parity-verified. Numeric/Categorical/unknown decline (NIE): pandas size()
        # over a non-sequence Series returns the ROW COUNT (quirk we refuse to replicate), and
        # Categorical .str raises in polars only.
        dt = _expr_output_dtype(args[0])
        if dt == pl.String:
            return args[0].str.len_chars()
        if isinstance(dt, pl.List):
            return args[0].list.len()
        return None
    if name == "substring" and len(args) in (2, 3):
        from graphistry.compute.gfql.expr_parser import Literal
        # substring(s, start[, length]), 0-based: pandas slices s[start:start+length]; polars
        # str.slice(offset, length). Equal ONLY for non-negative int start/length (negative start
        # + length diverges: pandas s[-2:1]=='' vs polars slice(-2,3) keeps chars — silent wrong
        # answer). Admit int literals >= 0 (negatives parse as UnaryOp, so the Literal gate also
        # declines them) over a String column (polars raises otherwise; pandas declines).
        start_node = node.args[1]
        length_node = node.args[2] if len(node.args) == 3 else None
        if not (isinstance(start_node, Literal) and isinstance(start_node.value, int)
                and not isinstance(start_node.value, bool) and start_node.value >= 0):
            return None
        length_val = None
        if length_node is not None:
            if not (isinstance(length_node, Literal) and isinstance(length_node.value, int)
                    and not isinstance(length_node.value, bool) and length_node.value >= 0):
                return None
            length_val = length_node.value
        if _expr_output_dtype(args[0]) != pl.String:
            return None
        # offset>=0, length>=0 (or None=to-end) → identical chars on pandas/polars.
        return args[0].str.slice(start_node.value, length_val)
    if name == "tointeger" and len(args) == 1:
        # toInteger oracle: pandas inner.astype(float).fillna(0).astype("int64") + isna() null_mask
        # restored, so NaN IS null (NaN/null -> null); finite floats truncate toward zero
        # (== polars float->int cast). Admit only output dtypes polars provably reproduces:
        dt = _expr_output_dtype(args[0])
        if _dtype_is_int(dt) or dt == pl.Boolean:
            # Int/Bool: identity widening (True/False -> 1/0); no NaN possible, nulls preserved.
            return args[0].cast(pl.Int64)
        if _dtype_is_float(dt):
            # Float: NaN AND null -> null on pandas; finite truncates. Mask NaN/null EXPLICITLY
            # (don't trust polars' NaN->int cast internals); strict=False truncates the rest
            # (masked rows never fail the cast).
            return pl.when(args[0].is_nan() | args[0].is_null()).then(
                pl.lit(None, dtype=pl.Int64)
            ).otherwise(args[0].cast(pl.Int64, strict=False))
        # String: pandas astype(float) RAISES on non-numeric (not null-on-failure); polars
        # strict=False would silently null -> divergence. Decline (NIE).
        return None
    if name == "tofloat" and len(args) == 1:
        # toFloat oracle: pandas inner.astype(float) + isna() mask via .where(~mask, pd.NA).
        # CRUCIALLY no .fillna(0)/int step (contrast toInteger): float64 has no null sentinel, so
        # a masked NaN re-materializes as NaN — NaN is PRESERVED, not nulled. A plain cast
        # preserves both NaN and null, so no explicit NaN mask. Admit provable dtypes only:
        dt = _expr_output_dtype(args[0])
        if _dtype_is_int(dt) or dt == pl.Boolean or _dtype_is_float(dt):
            # Int/UInt/Bool/Float -> Float64: exact IEEE widening (True/False -> 1.0/0.0;
            # nulls + NaN preserved) == pandas inner.astype(float).
            return args[0].cast(pl.Float64)
        # String: pandas astype(float) RAISES on non-numeric (data-dependent); polars
        # strict=False would silently null -> divergence. Decline (NIE).
        return None
    if name == "toboolean" and len(args) == 1:
        # toBoolean oracle: pandas parses fixed tokens ("true"/"t"/"1"/"yes" vs "false"/"f"/"0"/
        # "no") over astype(str), ERRORING otherwise; numerics data-dependent (only exact 0/1;
        # "2"/"1.0" error). Only statically-provable case: Boolean identity (nulls preserved).
        # Strings (polars cast won't parse "yes"/"t"/...) and numerics decline (NIE).
        if _expr_output_dtype(args[0]) == pl.Boolean:
            return args[0].cast(pl.Boolean)
        return None
    if name == "tostring" and len(args) == 1:
        # toString oracle: pandas astype(str) + "True"/"False" -> "true"/"false" rewrite. Admit
        # dtypes whose text polars reproduces EXACTLY: Boolean (lowercase), Int (decimal digits),
        # String (identity). Decline Float (repr diverges: pandas str(1e20)='1e+20' vs polars
        # formatting) and temporal/Categorical/other.
        dt = _expr_output_dtype(args[0])
        if dt == pl.Boolean or _dtype_is_int(dt) or dt == pl.String:
            return args[0].cast(pl.String)
        return None
    return None


_ISO_DURATION_RE = re.compile(r"^-?P(?=[0-9T])")

# ISO-8601 date/datetime/time-with-seconds-or-timezone — what cypher date()/time()/datetime()
# lower to; polars string </> compares these LEXICOGRAPHICALLY (wrong across timezones/precision).
# Bare times require seconds or a timezone so ordinary '10:00' strings don't match.
_ISO_TEMPORAL_RE = re.compile(
    r"""^(
        \d{4}-\d{2}-\d{2}([T\ ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?
      | \d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})
      | \d{2}:\d{2}:\d{2}(\.\d+)?
    )$""",
    re.VERBOSE,
)


def _is_int_literal(node: ExprNode) -> bool:
    """True iff integer Literal (not bool). Gates the literal/literal int-division decline (NIE):
    cypher 5/2 == 2 (truncating) vs polars 2.5; column / int is Float on both, so it matches."""
    from graphistry.compute.gfql.expr_parser import Literal
    return isinstance(node, Literal) and isinstance(node.value, int) and not isinstance(node.value, bool)


def _nonzero_int_literal(node: ExprNode) -> bool:
    """True iff the node is a NONZERO integer literal (unary +/- admitted).

    Gates the native int `/` and `%` lowering (#1900): openCypher mandates an
    ERROR for an integer zero divisor, but polars `// 0` silently yields null,
    so only a provably nonzero literal divisor may run natively -- anything
    else declines to the pandas lane's typed error."""
    from graphistry.compute.gfql.expr_parser import Literal, UnaryOp as _UnaryOp
    if isinstance(node, _UnaryOp) and node.op in ("+", "-"):
        node = node.operand
    return (
        isinstance(node, Literal)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value != 0
    )


def _is_iso_duration_literal(node: ExprNode) -> bool:
    """True iff string Literal is an ISO-8601 duration (``PT6M``, ``P1Y``, …) — what cypher
    ``duration({...})`` lowers to. ``^-?P(?=[0-9T])`` avoids misfiring on strings like 'Prefix'."""
    from graphistry.compute.gfql.expr_parser import Literal
    return (
        isinstance(node, Literal)
        and isinstance(node.value, str)
        and _ISO_DURATION_RE.match(node.value) is not None
    )


def _is_iso_temporal_literal(node: ExprNode) -> bool:
    """True iff string Literal is ISO date/datetime/time (cypher date()/time()/datetime() output).
    Gates the temporal-comparison decline (NIE) — polars would compare lexicographically (wrong)."""
    from graphistry.compute.gfql.expr_parser import Literal
    return (
        isinstance(node, Literal)
        and isinstance(node.value, str)
        and _ISO_TEMPORAL_RE.match(node.value) is not None
    )


def _is_temporal_column_ref(node: ExprNode, columns: Sequence[str]) -> bool:
    """True iff ``node`` references a column with TEMPORAL schema dtype (Datetime/Date/Time).
    Temporal column vs ISO temporal STRING literal makes polars raise -> decline; a String
    column holding ISO text compares lexicographically (correct) and must NOT be declined."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import Identifier, PropertyAccessExpr
    name: Optional[str] = None
    if isinstance(node, PropertyAccessExpr) and isinstance(node.value, Identifier):
        name = _resolve_property(node.value.name, node.property, columns)
    elif isinstance(node, Identifier) and node.name in columns:
        name = node.name
    if name is None:
        return False
    dt = _SCHEMA.get().get(name)
    return dt is not None and (isinstance(dt, pl.Datetime) or dt == pl.Date or dt == pl.Time)


def _expr_output_dtype(expr: pl.Expr) -> Optional[pl.DataType]:
    """Output dtype of a lowered expr under the active schema (None if unresolvable). Schema-only
    (empty LazyFrame, no data); catches what AST inference misses — int/int → Float (NaN-capable),
    function results (abs/coalesce), Categorical/Enum. Drives the NaN + cross-type guards."""
    import polars as pl
    try:
        return pl.LazyFrame(schema=_SCHEMA.get()).select(expr.alias("__gfql_dt__")).collect_schema()["__gfql_dt__"]
    except Exception:
        return None


def _is_cross_type(ldt: Optional[pl.DataType], rdt: Optional[pl.DataType]) -> bool:
    """Numeric vs string-like operand: polars raises (compare + arithmetic; incl. all-null
    columns, which from_pandas types as String) where pandas/cypher return a value/null, so
    decline natively. None dtype = unknown → not flagged."""
    if ldt is None or rdt is None:
        return False
    return (_dtype_is_numeric(ldt) and _dtype_is_stringlike(rdt)) or (_dtype_is_stringlike(ldt) and _dtype_is_numeric(rdt))


def _operand_is_nan_free_column(node: Optional[ExprNode], columns: Sequence[str]) -> bool:
    """True when ``node`` is a DIRECT reference to a column of an ingest-cleaned frame.

    Two conjuncts, both required:

    1. ``COLUMNS_NAN_FREE`` is set — the caller has declared this frame's float columns NaN-free
       because it came through gfql ingest (``nan_clean._pl_nan_to_null``, applied to ``_nodes``
       and ``_edges`` in ``_coerce_input_formats``). Default False, so a caller that does not opt
       in never reaches the second conjunct.
    2. the operand is a bare column read — an ``Identifier``/``PropertyAccessExpr`` that
       ``lower_expr`` resolves to ``pl.col(...)`` and nothing else. Its values ARE the ingested
       column's values, so "the column carries no NaN" transfers to "this operand yields no NaN".

    Anything COMPUTED is excluded on purpose, including function calls and arithmetic: ``n.a/n.b``
    is NaN at 0.0/0.0 and ``sqrt(n.x)`` is NaN at x<0 even on a perfectly clean column, so
    in-query float math manufactures NaN that ingest cannot have removed and MUST stay masked.
    """
    if node is None or not _COLUMNS_NAN_FREE.get():
        return False
    from graphistry.compute.gfql.expr_parser import Identifier, PropertyAccessExpr
    if isinstance(node, PropertyAccessExpr):
        return (
            isinstance(node.value, Identifier)
            and _resolve_property(node.value.name, node.property, columns) is not None
        )
    return isinstance(node, Identifier) and node.name in columns


def _nan_guard(
    result: pl.Expr, op: str, left: pl.Expr, right: pl.Expr,
    ldt: Optional[pl.DataType], rdt: Optional[pl.DataType],
    *, left_nan_free: bool = False, right_nan_free: bool = False,
) -> pl.Expr:
    """Mask a comparison so NaN compares IEEE/pandas/Cypher-style (false; ``!=`` true), not
    polars-style (NaN = largest). ``is_nan()`` applied only to float-OUTPUT operands; no-op
    for int/string/bool comparisons.

    ``left_nan_free``/``right_nan_free`` drop that operand's ``is_nan()`` term because the operand
    provably cannot be NaN — see ``_operand_is_nan_free_column``. They default False (mask ON), so
    the mask is only ever skipped by a caller that opted in explicitly, and the mask that remains
    is exactly the mask this function would have built for the operands that can still be NaN.
    """
    nan_terms = []
    if _dtype_is_float(ldt) and not left_nan_free:
        nan_terms.append(left.is_nan())
    if _dtype_is_float(rdt) and not right_nan_free:
        nan_terms.append(right.is_nan())
    if not nan_terms:
        return result
    any_nan = nan_terms[0]
    for term in nan_terms[1:]:
        any_nan = any_nan | term
    return (result | any_nan) if op in _NAN_NE_OPS else (result & ~any_nan)


def _dtype_category(dt: Optional[pl.DataType]) -> Optional[str]:
    """Coarse dtype category for list/IN parity gating: int/float/str/bool (None if unknown/other,
    e.g. List/Struct/Null/temporal). Only same-category elements coerce to a list/``is_in``
    supertype preserving VALUE + repr vs pandas — drives the homogeneity requirement."""
    import polars as pl
    if dt is None:
        return None
    if dt == pl.Boolean:
        return "bool"
    if _dtype_is_int(dt):
        return "int"
    if _dtype_is_float(dt):
        return "float"
    if _dtype_is_stringlike(dt):
        return "str"
    return None


def _value_category(v: Any) -> Optional[str]:
    """Python-literal mirror of ``_dtype_category`` (bool checked BEFORE int — bool subclasses int)."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    return None


def _lower_list_literal(items: Sequence[ExprNode], columns: Sequence[str]) -> Optional[pl.Expr]:
    """Lower ``[e0, e1, ...]`` to a per-row list via ``pl.concat_list``, or None to defer.

    concat_list preserves written element ORDER, matching the pandas oracle (cudf is known to
    REORDER list elements — an orthogonal cudf bug not inherited; conformance scoped
    pandas-vs-polars). SAFE subset: non-empty, all elements lower, all ONE dtype category —
    same-category coercion preserves value + repr (Int32->Int64 widening; all-float rounds
    equal). Mixed category (supertype coercion drifts value/repr or raises), nested/temporal,
    null/unknown-dtype element, or EMPTY list (no inferable dtype) -> decline (NIE)."""
    import polars as pl
    if not items:
        return None
    lowered: List[pl.Expr] = []
    cats = set()
    for item in items:
        expr = lower_expr(item, columns)
        if expr is None:
            return None
        cat = _dtype_category(_expr_output_dtype(expr))
        if cat is None:
            return None
        cats.add(cat)
        lowered.append(expr)
    if len(cats) != 1:
        return None
    return pl.concat_list(lowered)


def _lower_in(left: pl.Expr, items: Sequence[ExprNode], columns: Sequence[str]) -> Optional[pl.Expr]:
    """Lower ``x IN [literals]`` to a 3-valued membership test, or None to defer.

    SAFE subset: non-empty, non-null literals, single category matching the lhs dtype category.
    Cypher IN is 3-valued (NULL lhs -> NULL, not False): mask explicitly, independent of the
    polars version's ``is_in`` null handling; with no null elements the null lhs is the only
    unknown source, so the masked result is parity-equal to pandas. Null element, cross-type
    list (``is_in`` would raise), or non-literal element -> decline (NIE)."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import Literal
    if not items or not all(isinstance(it, Literal) and it.value is not None for it in items):
        return None
    literals: List[Literal] = [it for it in items if isinstance(it, Literal)]
    cats = {_value_category(it.value) for it in literals}
    if len(cats) != 1 or None in cats:
        return None
    if _dtype_category(_expr_output_dtype(left)) != next(iter(cats)):
        return None
    values = [it.value for it in literals]
    return pl.when(left.is_null()).then(pl.lit(None, dtype=pl.Boolean)).otherwise(left.is_in(values))


def lower_expr(node: ExprNode, columns: Sequence[str]) -> Optional[pl.Expr]:
    """Lower a parsed cypher ExprNode to a polars expression, or None to defer."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import (
        Identifier, Literal, BinaryOp, UnaryOp, IsNullOp, PropertyAccessExpr, FunctionCall, CaseWhen,
        ListLiteral,
    )

    if isinstance(node, Literal):
        return pl.lit(node.value)
    if isinstance(node, CaseWhen):
        cond = lower_expr(node.condition, columns)
        wt = lower_expr(node.when_true, columns)
        wf = lower_expr(node.when_false, columns)
        if cond is None or wt is None or wf is None:
            return None
        # cast cond to Boolean: a Null-dtype/3-valued WHEN takes the ELSE branch (Cypher,
        # matching pandas); no-op on a real Boolean.
        return pl.when(cond.cast(pl.Boolean)).then(wt).otherwise(wf)
    if isinstance(node, FunctionCall):
        return _lower_function(node, columns)
    if isinstance(node, ListLiteral):
        return _lower_list_literal(node.items, columns)
    if isinstance(node, Identifier):
        if node.name in columns:
            return pl.col(node.name)
        # Bare whole-entity identity sentinel -> the graph node-id column (pandas
        # _gfql_resolve_token bare form). Only when the id column is actually present;
        # otherwise decline (None -> NIE) rather than invent a column.
        if node.name == _NODE_ID_TOKEN:
            node_id = _NODE_ID.get()
            if node_id is not None and node_id in columns:
                return pl.col(node_id)
        return None
    if isinstance(node, PropertyAccessExpr):
        if isinstance(node.value, Identifier):
            src = _resolve_property(node.value.name, node.property, columns)
            if src is not None:
                return pl.col(src)
        return None
    if isinstance(node, BinaryOp):
        if node.op == "in" and isinstance(node.right, ListLiteral):
            # x IN [literals] on the row-expression surface (distinct from the WHERE/IsIn
            # predicate path); 3-valued, parity-checked. Non-literal/non-list RHS falls
            # through to the generic handler (-> None -> NIE).
            left = lower_expr(node.left, columns)
            if left is None:
                return None
            return _lower_in(left, node.right.items, columns)
        # decline (NIE): temporal arithmetic — duration({...}) lowers to an ISO duration STRING
        # ('PT6M'), so +/- would become string concatenation (silent wrong answer); pandas handles it.
        if node.op in ("+", "-") and (_is_iso_duration_literal(node.left) or _is_iso_duration_literal(node.right)):
            return None
        # decline (NIE): ORDERING two ISO temporal constructor-string literals = lexicographic
        # (wrong across timezones). Only literal-vs-literal ordering declines: =/<> are
        # lexicographically correct, and literal-vs-real-string-column must NOT decline.
        if node.op in _ORDER_OPS and _is_iso_temporal_literal(node.left) and _is_iso_temporal_literal(node.right):
            return None
        # decline (NIE): TEMPORAL column vs ISO constructor-string literal (n.ts > date('2020-01-15'))
        # — the constructor lowers to a STRING literal, so Datetime/Date vs string makes polars raise
        # InvalidOperationError; pandas compares temporally. A String column holding ISO text is NOT
        # temporal here and still computes lexicographically. (The chain p.gt(date(...)) predicate
        # carries a typed value + schema dtype and IS lowered natively in predicates.py.)
        if node.op in _NAN_GUARD_OPS and (
            (_is_iso_temporal_literal(node.left) and _is_temporal_column_ref(node.right, columns))
            or (_is_iso_temporal_literal(node.right) and _is_temporal_column_ref(node.left, columns))
        ):
            return None
        left = lower_expr(node.left, columns)
        right = lower_expr(node.right, columns)
        if left is None or right is None:
            return None
        ldt = rdt = None
        if node.op in _NUMSTR_OPS:
            # decline (NIE): numeric-vs-string-like makes polars raise (compare AND arithmetic;
            # incl. AllOf-nested, all-null→String, Categorical). Output dtypes catch
            # int/int→Float division + function results AST inference missed.
            ldt, rdt = _expr_output_dtype(left), _expr_output_dtype(right)
            if _is_cross_type(ldt, rdt):
                return None
            # decline (NIE): Boolean modulo — pandas raises GFQLTypeError on n.flag % 2 while
            # polars computes it (bool→int). Verified bool +,-,*,/ are IDENTICAL on both
            # engines; only % diverges.
            if node.op == "%" and (ldt == pl.Boolean or rdt == pl.Boolean):
                return None
            # openCypher: ordering a BOOLEAN against a NUMBER is incomparable -> null
            # (rows drop in WHERE); boolean-vs-boolean ordering stays served (#1900).
            if node.op in _ORDER_OPS and ldt is not None and rdt is not None and (
                (ldt == pl.Boolean and _dtype_is_numeric(rdt) and rdt != pl.Boolean)
                or (rdt == pl.Boolean and _dtype_is_numeric(ldt) and ldt != pl.Boolean)
            ):
                return pl.lit(None, dtype=pl.Boolean)
            # openCypher numeric tower (#1900): `%` is TRUNCATED (sign of the
            # dividend, -7 % 3 = -1), int `/` truncates toward zero, and an
            # integer zero divisor is an ERROR -- polars `// 0` yields null, so
            # int `/` and int `%` run natively only with a provably nonzero
            # literal divisor; other divisors decline to pandas' typed error.
            if (
                node.op in ("/", "%")
                and ldt is not None and rdt is not None
                and _dtype_is_numeric(ldt) and _dtype_is_numeric(rdt)
                and ldt != pl.Boolean and rdt != pl.Boolean
            ):
                both_int = _dtype_is_int(ldt) and _dtype_is_int(rdt)
                if node.op == "/" and both_int:
                    if not _nonzero_int_literal(node.right):
                        return None
                    return (left.abs() // right.abs()) * left.sign() * right.sign()
                if node.op == "%":
                    if both_int:
                        if not _nonzero_int_literal(node.right):
                            return None
                        quotient = (left.abs() // right.abs()) * left.sign() * right.sign()
                        return left - quotient * right
                    true_q = left / right
                    quotient = pl.when(true_q >= 0).then(true_q.floor()).otherwise(true_q.ceil())
                    return left - quotient * right
        result = _apply_binop(node.op, left, right)
        if result is not None and node.op in _NAN_GUARD_OPS:
            result = _nan_guard(
                result, node.op, left, right, ldt, rdt,
                left_nan_free=_operand_is_nan_free_column(node.left, columns),
                right_nan_free=_operand_is_nan_free_column(node.right, columns),
            )
        return result
    if isinstance(node, UnaryOp):
        operand = lower_expr(node.operand, columns)
        if operand is None:
            return None
        if node.op == "-":
            return -operand
        if node.op.upper() == "NOT":
            # Cast to Boolean so NOT null (Null-dtype lit) yields null (Cypher 3VL: NOT null =
            # null) instead of raising `dtype Null not supported in 'not' operation`; no-op
            # on a real Boolean column.
            return ~operand.cast(pl.Boolean)
        return None
    if isinstance(node, IsNullOp):
        value = lower_expr(node.value, columns)
        if value is None:
            return None
        return value.is_not_null() if node.negated else value.is_null()
    return None


def lower_expr_str(expr: str, columns: Sequence[str]) -> Optional[pl.Expr]:
    """Parse + lower an expression string; None if unparseable or not lowerable."""
    import polars as pl
    if expr in columns:
        return pl.col(expr)
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    return lower_expr(node, columns)


def _bare_column_ast(node: ExprNode, alias: str) -> Optional[ExprNode]:
    """Rewrite ``alias.prop`` property access to the BARE column ``prop``; None to decline.

    Used by callers that hold ONE alias's own frame (bare column names) rather than the
    joined row table (``alias.col`` names). ``alias.col -> col`` is a bijection over that
    frame's columns, so lowering the rewritten tree against the bare schema builds the SAME
    polars expression the row table would build against the prefixed schema.

    Declines (returns None):
    - a property access on any OTHER alias -- that alias's columns are not in this frame, and
      the row-table route cannot resolve them either (``_resolve_property`` -> None -> NIE);
    - a BARE ``Identifier`` -- no bare name is a column of the prefixed row table, so the
      row-table route declines it and accepting it here would invent a resolution. The one
      exception is the whole-entity identity sentinel ``__gfql_node_id__``, which the
      row-table route DOES resolve through ``_NODE_ID``; this frame publishes no identity
      column, so it is declined here too. That decline costs only speed -- the caller falls
      back and the row-table route answers it -- and never changes an answer;
    - any node type outside the set ``lower_expr`` itself handles (map/subscript/slice/
      quantifier/comprehension/wildcard), which ``lower_expr`` declines anyway.
    """
    from graphistry.compute.gfql.expr_parser import (
        BinaryOp, CaseWhen, FunctionCall, Identifier, IsNullOp, ListLiteral, Literal,
        PropertyAccessExpr, UnaryOp,
    )
    if isinstance(node, PropertyAccessExpr):
        if isinstance(node.value, Identifier) and node.value.name == alias:
            return Identifier(name=node.property)
        return None
    if isinstance(node, Literal):
        return node
    if isinstance(node, BinaryOp):
        left = _bare_column_ast(node.left, alias)
        right = _bare_column_ast(node.right, alias)
        if left is None or right is None:
            return None
        return BinaryOp(op=node.op, left=left, right=right)
    if isinstance(node, UnaryOp):
        operand = _bare_column_ast(node.operand, alias)
        return None if operand is None else UnaryOp(op=node.op, operand=operand)
    if isinstance(node, IsNullOp):
        value = _bare_column_ast(node.value, alias)
        return None if value is None else IsNullOp(value=value, negated=node.negated)
    if isinstance(node, CaseWhen):
        condition = _bare_column_ast(node.condition, alias)
        when_true = _bare_column_ast(node.when_true, alias)
        when_false = _bare_column_ast(node.when_false, alias)
        if condition is None or when_true is None or when_false is None:
            return None
        return CaseWhen(condition=condition, when_true=when_true, when_false=when_false)
    if isinstance(node, ListLiteral):
        items: List[ExprNode] = []
        for item in node.items:
            rewritten_item = _bare_column_ast(item, alias)
            if rewritten_item is None:
                return None
            items.append(rewritten_item)
        return ListLiteral(items=tuple(items))
    if isinstance(node, FunctionCall):
        args: List[ExprNode] = []
        for arg in node.args:
            rewritten_arg = _bare_column_ast(arg, alias)
            if rewritten_arg is None:
                return None
            args.append(rewritten_arg)
        return FunctionCall(name=node.name, args=tuple(args), distinct=node.distinct)
    return None


# Memo for `lower_single_alias_predicate`. Its callers re-lower a handful of predicate STRINGS,
# fixed per query, once per execution -- and the lowering is NOT cheap (parse, then a
# schema-width LazyFrame probe per operand in `_expr_output_dtype`). Bounded LRU so a
# long-lived process running unboundedly many distinct queries cannot grow it without limit.
#: (expr, alias, columns_nan_free, ((col, dtype-repr), ...)) -- see `_single_alias_cache_key`.
_SingleAliasKey = Tuple[str, str, bool, Tuple[Tuple[str, str], ...]]
_SINGLE_ALIAS_CACHE: "OrderedDict[_SingleAliasKey, Optional[pl.Expr]]" = OrderedDict()
_SINGLE_ALIAS_CACHE_MAX = 512


register_clearable_dict("_SINGLE_ALIAS_CACHE", _SINGLE_ALIAS_CACHE)


def _single_alias_cache_key(
    expr: str, alias: str, schema: Mapping[str, "pl.DataType"], columns_nan_free: bool
) -> _SingleAliasKey:
    """Every input the lowered expression depends on, in one hashable key.

    A missing input serves a stale expression, i.e. a silent wrong answer, so
    completeness is pinned in ``test_single_alias_cache_key.py`` -- including the
    non-obvious one: the key holds ``str(dtype)``, not the dtype OBJECT, because
    polars equates a dtype class with its parameterized instances
    (``Datetime == Datetime('ns')``) and they would collide as dict keys.

    Parser AVAILABILITY is deliberately NOT keyed: it is a property of the process,
    not of the arguments, so the caller probes it BEFORE consulting the memo -- which
    also means no cached entry can outlive a parser that has gone away.
    """
    return (expr, alias, columns_nan_free, tuple((k, str(v)) for k, v in schema.items()))


def lower_single_alias_predicate(
    expr: str, alias: str, schema: Mapping[str, "pl.DataType"], *, columns_nan_free: bool = False
) -> Optional[pl.Expr]:
    """Lower a single-alias predicate STRING against a BARE-column frame schema; None to defer.

    The parity seam for callers that filter one alias's own frame directly instead of routing
    the predicate through ``where_rows_polars`` on a prefixed row table. Both routes run the
    SAME parser and the SAME ``lower_expr`` under the SAME ``_SCHEMA`` dtypes, and
    ``where_rows_polars`` does nothing to the lowered expression but ``table.filter(...)`` --
    so ``frame.filter(lower_single_alias_predicate(...))`` is value-identical to the where_rows
    route by construction, including every decline (a decline here is a decline there, i.e. the
    caller's fallback reaches the row op's designed NotImplementedError rather than guessing).

    ``_NODE_ID`` is pinned to None: this frame has no row-table identity column, so the bare
    ``__gfql_node_id__`` sentinel must resolve to nothing, exactly as it does on the prefixed
    row table where the sentinel is not a column either.

    ``columns_nan_free`` (default False = fully guarded) declares this frame's float COLUMNS
    already NaN-free, which lets a comparison against a bare column skip the IEEE NaN mask; see
    ``lowering_context.COLUMNS_NAN_FREE`` and ``_operand_is_nan_free_column``. Only a caller
    filtering a gfql-INGESTED frame may set it; computed float operands stay masked either way.

    MEMOIZED on ``_single_alias_cache_key`` (see there for why that key is complete). The
    returned ``pl.Expr`` is safe to hand out repeatedly: a polars expression is an immutable
    plan fragment naming columns symbolically -- ``frame.filter(e)`` builds a new plan and
    neither mutates ``e`` nor binds it to that frame -- so one expression applies to any frame
    whose schema matches the key, which is the only frame the key can be reached with.
    """
    if _parser() is None:
        return None  # not keyable (process state, not an argument) -> never memoized
    key = _single_alias_cache_key(expr, alias, schema, columns_nan_free)
    if key in _SINGLE_ALIAS_CACHE:
        try:
            _SINGLE_ALIAS_CACHE.move_to_end(key)
            # The read stays INSIDE the try: a concurrent eviction or a registry
            # clear_all() landing between move_to_end and this lookup must degrade
            # to a recompute, never surface KeyError to the caller.
            return _SINGLE_ALIAS_CACHE[key]
        except KeyError:  # concurrent eviction/clear; recompute-safe
            pass
    lowered = _lower_single_alias_predicate_uncached(expr, alias, schema, columns_nan_free)
    _SINGLE_ALIAS_CACHE[key] = lowered
    # Eviction races can only DROP an entry (a redundant recompute), never fabricate a hit.
    while len(_SINGLE_ALIAS_CACHE) > _SINGLE_ALIAS_CACHE_MAX:
        try:
            _SINGLE_ALIAS_CACHE.popitem(last=False)
        except KeyError:  # pragma: no cover - concurrent eviction
            break
    return lowered


def _lower_single_alias_predicate_uncached(
    expr: str, alias: str, schema: Mapping[str, "pl.DataType"], columns_nan_free: bool
) -> Optional[pl.Expr]:
    """``lower_single_alias_predicate`` without the memo (also the unit-test seam for it)."""
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    bare = _bare_column_ast(node, alias)
    if bare is None:
        return None
    schema_token = _SCHEMA.set(dict(schema))
    node_id_token = _NODE_ID.set(None)
    nan_free_token = _COLUMNS_NAN_FREE.set(columns_nan_free)
    try:
        return lower_expr(bare, list(schema))
    finally:
        _SCHEMA.reset(schema_token)
        _NODE_ID.reset(node_id_token)
        _COLUMNS_NAN_FREE.reset(nan_free_token)


def lower_select_items(items: Sequence[SelectItem], columns: Sequence[str]) -> Optional[List["pl.Expr"]]:
    """Lower projection items [(alias, expr) | 'col'] to polars exprs, or None."""
    out: List["pl.Expr"] = []
    for item in items:
        alias: str
        expr: JSONVal
        if isinstance(item, str):
            alias, expr = item, item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            alias, expr = str(item[0]), item[1]
        else:
            return None
        if not isinstance(expr, str):
            # Non-string value = constant literal (e.g. synthetic __cypher_group__=1 for
            # keyless aggregation).
            import polars as pl
            out.append(pl.lit(expr).alias(alias))
            continue
        lowered = lower_expr_str(expr, columns)
        if lowered is None:
            return None
        out.append(lowered.alias(alias))
    return out


def lower_order_by_keys(keys: Sequence[OrderKey], columns: Sequence[str]) -> Optional[Tuple[List["pl.Expr"], List[bool]]]:
    """Lower order_by [(expr, direction)] to (polars exprs, descending flags)."""
    exprs: List["pl.Expr"] = []
    descending: List[bool] = []
    for key in keys:
        if not isinstance(key, (list, tuple)) or len(key) != 2:
            return None
        expr, direction = key
        if not isinstance(expr, str) or not isinstance(direction, str):
            return None
        lowered = lower_expr_str(expr, columns)
        if lowered is None:
            return None
        exprs.append(lowered)
        descending.append(direction.lower() == "desc")
    return exprs, descending


def _active_table(g: Plottable) -> Any:
    if g._nodes is not None:
        return g._nodes
    return g._edges


def _rewrap(g: Plottable, table_df: Any) -> Plottable:
    """Set the new active row table (mirrors frame_ops.row_table for polars)."""
    from graphistry.compute.gfql.row import frame_ops
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter
    return frame_ops.row_table(_RowPipelineAdapter(g), table_df)


def _finish_binding_rows_polars(
    g: Plottable,
    ops: "Sequence[ASTObject]",
    state: "PolarsFrameT",
    alias_frames: Dict[str, "PolarsFrameT"],
    node_id: str,
    attach_prop_aliases: Optional[Sequence[str]],
    *,
    decline_on_schema_error: bool,
) -> Optional[Plottable]:
    """Canonical property attachment/materialization for generic or indexed state.

    ``decline_on_schema_error`` reflects WHOSE state this is. The generic builder
    joins frames polars will not unify the way pandas implicitly does (int vs float
    join keys), so a ``SchemaError`` there is an honest decline. The indexed state
    comes from a helper that already checked those dtypes, so a ``SchemaError``
    there is a BUG and must surface rather than become a silent slow-path fallback.
    """
    import polars as pl
    from graphistry.compute.ast import ASTEdge, ASTNode
    from graphistry.compute.gfql.lazy import collect as _lazy_collect

    def names(frame: "PolarsFrameT") -> List[str]:
        return (
            frame.collect_schema().names()
            if isinstance(frame, pl.LazyFrame)
            else list(frame.columns)
        )

    try:
        attach_set = (
            None if attach_prop_aliases is None else set(attach_prop_aliases)
        )
        node_aliases: List[str] = [
            op._name
            for op in ops[::2]
            if isinstance(op, (ASTNode, ASTEdge)) and isinstance(op._name, str)
        ]
        for alias in node_aliases:
            if attach_set is not None and alias not in attach_set:
                continue
            lookup_src = alias_frames[alias]
            lookup = lookup_src.select(
                [
                    pl.col(node_id),
                    pl.col(node_id).alias(f"{alias}.{node_id}"),
                ]
                + [
                    pl.col(col).alias(f"{alias}.{col}")
                    for col in names(lookup_src)
                    if col != node_id
                ]
            )
            if (set(names(lookup)) - {node_id}) & set(names(state)):
                return None
            state = state.join(
                lookup, left_on=alias, right_on=node_id, how="left",
            )
        state = state.drop(WALK_CURRENT_COL)
        out_df = (
            _lazy_collect(state)
            if isinstance(state, pl.LazyFrame)
            else state
        )
    except pl.exceptions.SchemaError:
        if not decline_on_schema_error:
            raise
        return None

    out = _rewrap(g, out_df)
    edge_aliases = {
        alias
        for op in ops[1::2]
        for alias in [op._name]
        if isinstance(alias, str)
    }
    out._gfql_rows_edge_aliases = edge_aliases
    return out


_LowerT = TypeVar("_LowerT")


def _lower_with_schema(table: "pl.DataFrame", fn: Callable[[], _LowerT],
                       node_id: Optional[str] = None) -> _LowerT:
    """Run a lowering callable with the table schema published to ``_SCHEMA`` (float-operand
    inference for the NaN guard) and the graph node-id column published to ``_NODE_ID`` (bare
    ``__gfql_node_id__`` identity-sentinel resolution)."""
    schema_token = _SCHEMA.set(dict(table.schema))
    node_id_token = _NODE_ID.set(node_id)
    try:
        return fn()
    finally:
        _SCHEMA.reset(schema_token)
        _NODE_ID.reset(node_id_token)


def _project_preserving_height(table: Any, exprs: List[Any]) -> Any:
    """Project ``exprs`` while preserving the frame's row cardinality.

    Cypher ``WITH``/``RETURN`` projection is a map, not a reduce. Polars
    ``DataFrame.select`` collapses to one row when every projected expression is
    scalar, so broadcast all-scalar projections through ``with_columns`` first.
    """
    if exprs and all(len(e.meta.root_names()) == 0 for e in exprs):
        names = [e.meta.output_name() for e in exprs]
        return table.with_columns(exprs).select(names)
    return table.select(exprs)


def _project_polars(g: Plottable, items: Sequence[SelectItem], extend: bool) -> Optional[Plottable]:
    """Shared body of ``select_polars`` / ``with_columns_polars``; None if any item isn't
    lowerable (honest NIE, no pandas bridge)."""
    table = _active_table(g)
    exprs = _lower_with_schema(table, lambda: lower_select_items(items, list(table.columns)), node_id=g._node)
    if exprs is None:
        return None
    out = table.with_columns(exprs) if extend else _project_preserving_height(table, exprs)
    if _select_emits_temporal_constructor_text(out):
        # decline (NIE): projected String column holds temporal-constructor text (date({...})
        # etc.) that pandas normalizes to ISO, not yet native — don't leak the raw text.
        # Only String columns are scanned, so numeric/bool projections pay nothing.
        return None
    return _rewrap(g, out)


def _select_emits_temporal_constructor_text(out: Any) -> bool:
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.projection import _has_temporal_constructor_text
    for name, dtype in out.schema.items():
        if dtype == pl.String and _has_temporal_constructor_text(out, name):
            return True
    return False


def select_polars(g: Plottable, items: Sequence[SelectItem]) -> Optional[Plottable]:
    """Native polars projection (replaces the row table)."""
    return _project_polars(g, items, extend=False)


def with_columns_polars(g: Plottable, items: Sequence[SelectItem]) -> Optional[Plottable]:
    """Native polars WITH extend=True: add/overwrite columns, keep the rest. Mirrors pandas
    ``with_(extend=True)`` (``table_df.assign``): ``with_columns`` matches — an existing alias
    REPLACES in place (position kept), a new alias APPENDS at the end in item order."""
    return _project_polars(g, items, extend=True)


def where_rows_polars(
    g: Plottable,
    filter_dict: Optional[dict] = None,
    expr: Optional[str] = None,
) -> Optional[Plottable]:
    """Native polars row-table WHERE; None if the predicate isn't lowerable.

    Cypher 3-valued WHERE keeps only TRUE rows (NULL and FALSE dropped) — polars ``filter``
    plus Kleene ``|``/``&`` match pandas/cypher NULL handling with no special-casing.
    filter_dict entries are scalar-equality conjuncts; PREDICATE values (``gt(1)`` etc. —
    the legacy path lowers them via ``filter_by_dict``) are not natively lowered here yet
    and defer (NIE) rather than reach ``pl.lit`` and leak a raw polars ``TypeError``
    (observed via the AUTO cudf->polars route, where the leak also broke the guard's
    decline-and-fall-back contract). Native predicate lowering exists for the traversal
    lane (``predicates.filter_expr_by_dict_polars``); wiring it here is a future upgrade,
    kept out of this decline-shaped fix.
    """
    import polars as pl
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    table = _active_table(g)
    columns = list(table.columns)
    preds: List[Any] = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in columns or isinstance(val, dict):
                return None  # missing column / nested-struct value -> defer (NIE)
            if isinstance(val, ASTPredicate):
                return None  # predicate object (gt/lt/contains/...) -> defer (NIE)
            if isinstance(val, (list, tuple, set)):
                # IN: `is_in` on a null cell -> null -> filter drops it, i.e. openCypher 3VL
                # (`null IN [...]` = null -> excluded), matching the filter_by_dict membership
                # fix. (Equality below also drops nulls: `null == v` -> null -> dropped.)
                preds.append(pl.col(col).is_in(list(val)))
            else:
                try:
                    preds.append(pl.col(col) == val)
                except TypeError:
                    # any other value polars can't lower to a literal -> defer (NIE),
                    # never a raw third-party error
                    return None
    if expr is not None:
        if not isinstance(expr, str):
            return None
        lowered = _lower_with_schema(table, lambda: lower_expr_str(expr, columns), node_id=g._node)
        if lowered is None:
            return None
        preds.append(lowered)
    if not preds:
        return g  # empty WHERE -> identity
    combined = preds[0]
    for pred in preds[1:]:
        combined = combined & pred
    return _rewrap(g, table.filter(combined))


def _order_keys_hold_list_like_values(table: "Union[pl.DataFrame, pl.LazyFrame]", exprs: "List[pl.Expr]") -> bool:
    """Decline sniff for ``order_by_polars``: would a native polars sort diverge from
    the legacy pipeline's Cypher list-orderability semantics?

    The triggers -- nested/object dtypes, and strings carrying list-syntax text -- and
    their parity consequences are pinned in ``test_engine_polars_row_pipeline.py`` and
    ``test_row_pipeline_ops.py``.

    Deliberately BROADER than legacy engagement; do not tighten it to match. Over-
    declining re-serves those corners on the legacy path with IDENTICAL values, trading
    a native sort for a fallback; under-declining returns a silently wrong ORDER. Errors
    while resolving the schema also decline, for the same reason.
    """
    import polars as pl
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.ordering import _GFQL_LIST_TEXT_RE
    aliased = [e.alias(f"__gfql_order_sniff_{i}__") for i, e in enumerate(exprs)]
    lazy = table.lazy() if isinstance(table, pl.DataFrame) else table
    try:
        keyed = lazy.select(aliased)
        schema = dict(keyed.collect_schema())
    except Exception:
        return True
    if any(isinstance(dt, (pl.List, pl.Array, pl.Struct, pl.Object)) for dt in schema.values()):
        return True
    text_cols = [name for name, dt in schema.items() if _dtype_is_stringlike(dt)]
    if not text_cols:
        return False
    try:
        flags = keyed.select([
            pl.col(name).cast(pl.String).str.strip_chars()  # hygiene-ok: explicit-cast -- pl.Expr.cast is a runtime dtype conversion (Categorical/Enum -> String for the str scan), not typing.cast
            .str.contains(_GFQL_LIST_TEXT_RE.pattern).any()
            for name in text_cols
        ])
        row = _lazy_collect(flags).row(0)
    except Exception:
        return True
    return any(bool(v) for v in row if v is not None)


def order_by_polars(g: Plottable, keys: Sequence[OrderKey]) -> Optional[Plottable]:
    """Native polars sort; None if any key isn't lowerable OR holds list-like values whose
    legacy ordering semantics a plain sort can't reproduce (see the sniff helper)."""
    table = _active_table(g)
    lowered = _lower_with_schema(table, lambda: lower_order_by_keys(keys, list(table.columns)), node_id=g._node)
    if lowered is None:
        return None
    exprs, descending = lowered
    if _order_keys_hold_list_like_values(table, exprs):
        return None  # legacy list-orderability semantics -> defer (NIE)
    # openCypher orders NULL as the LARGEST value: ASC -> nulls last, DESC -> nulls FIRST.
    # (Previously hardcoded nulls_last=True, which mis-ordered DESC keys and silently returned
    # the wrong `... DESC LIMIT k` top-k over a column containing NULLs.) `descending` is one
    # bool per key (see lower_order_by_keys), so `nulls_last` mirrors it per key.
    nulls_last = [not d for d in descending]
    return _rewrap(g, table.sort(exprs, descending=descending, nulls_last=nulls_last))


# Native aggs: count/sum/avg/min/max/count_distinct/collect/collect_distinct; stdev/percentile
# etc. return None → caller declines (NIE).
def _agg_expr(func: str, expr: Optional[str], columns: Sequence[str], alias: str,
              schema: Optional[Mapping[str, "pl.DataType"]] = None,
              is_all_null: Optional[Callable[[str], bool]] = None) -> Optional[pl.Expr]:
    import polars as pl
    func = func.lower()
    if func == "count" and (expr is None or expr == "*"):
        return pl.len().alias(alias)
    if not isinstance(expr, str) or expr not in columns:
        return None
    col = pl.col(expr)
    dtype = schema.get(expr) if schema is not None else None
    if dtype is not None and _dtype_is_stringlike(dtype) and dtype != pl.String:
        # Categorical/Enum is a STRING column to cypher (its categories are the values). Twin of
        # the pandas row pipeline's decategorize, and load-bearing on older polars: 1.35.2 (the
        # RAPIDS 26.02 image) PANICS in the rust core on a grouped min/max over a Categorical
        # (`categorical.rs: not implemented`), which escapes as a pyo3 PanicException -- not even
        # a polars exception, so nothing on the python side can wrap it. Casting to String makes
        # the aggregate well-defined on every polars version AND matches what pandas returns.
        col = col.cast(pl.String)  # hygiene-ok: explicit-cast -- pl.Expr.cast is a runtime dtype conversion, not typing.cast
    # pandas aggs skip NaN (skipna); polars skips only NULL and treats NaN as a value (NaN == NaN
    # is True, so self-inequality can't detect it). For FLOAT columns convert in-query NaN -> null
    # first so every agg matches the oracle (pandas sum([nan, 1]) == 1 vs raw polars == nan).
    # fill_nan is float-only, hence the dtype gate. Stored NaN is nulled at ingestion; this covers
    # NaN created mid-query (e.g. 0.0/0.0).
    if schema is not None and _dtype_is_float(schema.get(expr)):
        col = col.fill_nan(None)
    if func in GFQL_NUMERIC_ONLY_AGGREGATIONS and schema is not None:
        # DTYPE FIRST, data second -- deliberately, for cost. Cypher restricts sum()/avg() to
        # INTEGER|FLOAT|DURATION (see gfql/agg_types.py for the sources), and that verdict is a
        # schema lookup; only a column the SCHEMA already rejects is worth an O(n) null scan. A
        # numeric column -- every served aggregate -- therefore pays nothing here.
        if dtype == pl.Null:
            # all-null by construction: `sum`/`mean` are unsupported on `null` dtype in polars,
            # while cypher says 0 / null.
            return pl.lit(numeric_agg_all_null_value(func)).alias(alias)
        dtype_label = polars_non_numeric_agg_dtype(dtype)
        if dtype_label is not None:
            # An ALL-NULL column carries no type evidence, so it is never a type error: cypher
            # answers `sum(null)` with 0 and `avg(null)` with null whatever the declared type,
            # and pandas already did (an all-None pandas object column arrives here typed
            # `String`). Both would otherwise raise -- `sum`/`mean` are unsupported on `str`.
            if is_all_null is not None and is_all_null(expr):
                return pl.lit(numeric_agg_all_null_value(func)).alias(alias)
            # Raise, don't return None: None is an NIE-decline that falls back to the pandas
            # kernel, which would then ANSWER the same wrong-typed query.
            raise_non_numeric_aggregation(func, expr, dtype_label, alias)
    if func == "count":
        return col.count().alias(alias)
    if func == "sum":
        return col.sum().alias(alias)
    if func in ("avg", "mean"):
        return col.mean().alias(alias)
    if func == "min":
        return col.min().alias(alias)
    if func == "max":
        return col.max().alias(alias)
    if func == "count_distinct":
        # count(DISTINCT x) drops nulls (pandas nunique(dropna=True)); polars n_unique() counts
        # null, so drop_nulls first.
        return col.drop_nulls().n_unique().alias(alias)
    if func == "collect":
        # collect(x) drops nulls, keeps within-group row order (pandas row/pipeline.py:4552-4582:
        # ~isna() then agg(list)). Inside group_by(maintain_order=True).agg a multi-valued expr
        # yields a List column, so drop_nulls() alone reproduces it; all-null/empty group -> []
        # never [null], matching the oracle's []-coercion (4597-4614). NO .implode() — that
        # would double-wrap to List(List).
        return col.drop_nulls().alias(alias)
    if func == "collect_distinct":
        # collect(DISTINCT x): drop nulls + keep-first dedup in first-occurrence order (pandas
        # drop_duplicates(keep="first") + agg(list)); unique(maintain_order=True) matches;
        # empty/all-null group -> [].
        return col.drop_nulls().unique(maintain_order=True).alias(alias)
    return None


def group_by_polars(
    g: Plottable,
    keys: Sequence[str],
    aggregations: Sequence[AggSpec],
    key_prefixes: Optional[Sequence[str]] = None,
) -> Optional[Plottable]:
    """Native polars group-by; None if a key/agg isn't lowerable. Matches pandas dropna=False
    (null keys kept) + non-null agg semantics; output order is first-occurrence (maintain_order),
    though the parity gate compares order-insensitively. ``key_prefixes`` mirrors the pandas
    whole-entity expansion: every ``<prefix>*`` column of the row table joins the key set (the
    entity's columns are functionally dependent on its identity key, so this only carries them
    through — group sizes are unchanged)."""
    table = _active_table(g)
    cols = list(table.columns)
    key_cols = [str(k) for k in keys]
    if key_prefixes:
        seen = set(key_cols)
        for prefix in key_prefixes:
            for col in cols:
                if isinstance(col, str) and col.startswith(prefix) and col not in seen:
                    key_cols.append(col)
                    seen.add(col)
    if not key_cols or not all(isinstance(k, str) and k in cols for k in key_cols):
        return None
    aggs: List["pl.Expr"] = []
    for agg in aggregations:
        if not isinstance(agg, (list, tuple)) or len(agg) not in (2, 3):
            return None
        # cast: the AggSpec tuple variants make agg[2] an out-of-range index to mypy;
        # the len guard above already proved the shape.
        spec = cast("Sequence[Optional[str]]", agg)
        alias = str(spec[0])
        func = str(spec[1])
        expr = spec[2] if len(spec) == 3 else None
        # Passed as a CALLABLE, not a precomputed flag: the null scan is O(n) and is only ever
        # consulted for a column the dtype check has already rejected, so a normal numeric
        # aggregate never runs it.
        def _is_all_null(col_name: str) -> bool:
            return table.height > 0 and table[col_name].null_count() == table.height

        lowered = _agg_expr(func, expr, cols, alias, table.schema, _is_all_null)
        if lowered is None:
            return None
        aggs.append(lowered)
    out = table.group_by(key_cols, maintain_order=True).agg(aggs)
    return _rewrap(g, out)


def unwind_polars(g: Plottable, expr: str, as_: str = "value") -> Optional[Plottable]:
    """Native UNWIND for two shapes; None → caller NIEs.

    1. Literal scalar list (``UNWIND [1, 2] AS x``): cross-join each row with the values
       (cypher per-row expansion; empty list → 0 rows).
    2. Carried list column (``WITH collect(x) AS xs UNWIND xs AS y``, i.e. a ``collect()``
       output or any List-dtype binding): explode the list column. Mirrors the pandas oracle
       (``RowPipelineMixin.unwind`` list-column branch) exactly — an empty-list or null cell
       contributes 0 rows; nulls WITHIN a list survive as real elements; the source column is
       retained and the exploded values are appended as ``as_``.

    Everything else — nested-list literals, scalar/non-list columns (whose single-element-list
    Cypher coercion is not yet ported), function/arithmetic results — still declines (NIE)
    rather than risk diverging from pandas."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import Identifier, ListLiteral, Literal

    if not isinstance(expr, str):
        return None
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    table = _active_table(g)
    if as_ in table.columns:
        return None
    if isinstance(node, ListLiteral) and all(isinstance(it, Literal) for it in node.items):
        values = [it.value for it in node.items if isinstance(it, Literal)]
        rhs = pl.DataFrame({as_: values})
        return _rewrap(g, table.join(rhs, how="cross"))
    if isinstance(node, Identifier) and node.name in table.columns:
        col = node.name
        if not isinstance(table.schema[col], pl.List):
            # Non-list column: Cypher UNWIND coerces a scalar to a 1-element list (null → 0
            # rows). Those semantics aren't ported here yet, so decline rather than diverge.
            return None
        # pandas oracle: empty/null list cells drop out (0 rows); nulls within a list survive.
        # Copy the source into ``as_`` first (keeping the source column, like pandas), filter
        # out empty/null cells (``list.len()`` is null for a null cell → excluded by the
        # predicate), then explode. Pre-filtering empties also makes the explode independent of
        # the polars ``empty_as_null`` default (stable across polars versions).
        out = table.with_columns(pl.col(col).alias(as_))
        out = out.filter(pl.col(as_).list.len() > 0)
        out = out.explode(as_)
        return _rewrap(g, out)
    return None


def select_extend_polars(g: Plottable, items: Sequence[SelectItem]) -> Optional[Plottable]:
    """Native polars ``with_(items, extend=True)``: add/overwrite projected columns
    while keeping the existing row table (pandas ``assign`` semantics). Emitted by
    the bindings-path aggregate lowering (pre-aggregation group keys / agg args),
    so it is required for binding-row queries (#1709). None → NIE."""
    table = _active_table(g)
    exprs = _lower_with_schema(table, lambda: lower_select_items(items, list(table.columns)), node_id=g._node)
    if exprs is None:
        return None
    out = table.with_columns(exprs)
    if _select_emits_temporal_constructor_text(out):
        return None
    return _rewrap(g, out)


def _cartesian_node_bindings_polars(
    g: Plottable,
    ops: "Sequence[ASTObject]",
    node_id: Optional[str],
) -> Optional[Plottable]:
    """Native polars cross-product for disconnected MATCH aliases (#1273).

    Mirrors the pandas ``_gfql_cartesian_node_bindings_row_table`` oracle: each
    node alias is independently filtered, projected into the ``_gfql_node_alias_lookup_frame``
    schema (``alias``, ``alias.node_id``, ``alias.<col>``, plus the leaked named-op
    FLAG column ``alias.alias = True``), then cross-joined in op order (left-major,
    matching pandas' ``merge`` order so no ORDER BY is needed for parity). The bare
    ``node_id`` residue column that pandas carries (``id_x``/``id_y`` merge suffixes)
    is intentionally dropped: no lowered query references it, and dropping avoids
    polars cross-join column collisions on 3+ aliases.

    Returns None to DECLINE (caller raises the honest NIE) outside the supported
    subset: node ``query=`` params, ``alias == node_id`` (pandas' flag column
    overwrites the id column — no sane shared semantics), and seeded re-entry
    (already gated by the caller). NO-CHEATING: never bridges to pandas.
    """
    import polars as pl
    from graphistry.compute.ast import ASTNode
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from .predicates import filter_by_dict_polars

    nodes = g._nodes
    if nodes is None or node_id is None:  # pragma: no cover - defensive: bindings run post-materialize
        return None
    node_id = str(node_id)
    if node_id not in nodes.columns:  # pragma: no cover - defensive: node_id is the bound id column
        return None

    aliases = [op._name for op in ops]
    # Decline outside pandas' RELIABLE zone (empirically derived, keeps parity):
    #  - anonymous node op: the pandas cartesian raises a spurious schema error on
    #    an EMPTY result when a bare `()` is present (it drops the id column) rather
    #    than returning empty — declining avoids that divergence.
    #  - >3 named aliases: the pandas builder's per-alias bare-id merge residue
    #    collides on the 4th frame ("Passing 'suffixes' which cause duplicate
    #    columns"). Both engines must not diverge, so decline to the honest NIE.
    if any(not isinstance(a, str) for a in aliases):
        return None
    named = [a for a in aliases if isinstance(a, str)]
    if len(named) > 3:
        return None

    nodes_lf = nodes.lazy()
    # filter_by_dict_polars is frame-polymorphic (LazyFrame in -> LazyFrame out) but
    # annotated ``pl.DataFrame``; keep the accumulator loose, as this module does.
    per_alias: List[Any] = []
    for op in ops:
        if not isinstance(op, ASTNode) or op.query is not None:  # pragma: no cover - node_cartesian only routes bare ASTNode ops
            return None
        alias = op._name
        if not isinstance(alias, str):  # pragma: no cover - non-str aliases already declined above
            return None
        if alias == node_id:
            # pandas' named-op flag column overwrites the id column here — neither
            # engine has sane semantics; decline (mirrors the single-entity
            # ``rows_binding_ops_polars`` corner).
            return None
        try:
            matched = filter_by_dict_polars(nodes_lf, op.filter_dict)
        except NotImplementedError:  # pragma: no cover - propagate exotic-predicate NIE unchanged
            raise
        except Exception:  # pragma: no cover - defensive: unexpected filter failure declines
            return None
        cols = matched.collect_schema().names()
        # prop_cols excludes node_id and any real column named == alias: the pandas
        # node execute() leaks a boolean FLAG into a column named ``alias``
        # (shadowing a same-named real property), which the lookup frame surfaces
        # as ``alias.alias = True``. Reproduce that exactly.
        prop_cols = [c for c in cols if c != node_id and c != alias]
        exprs = [
            pl.col(node_id).alias(alias),
            pl.col(node_id).alias(f"{alias}.{node_id}"),
            pl.lit(True).alias(f"{alias}.{alias}"),
        ]
        exprs.extend(pl.col(c).alias(f"{alias}.{c}") for c in prop_cols)
        per_alias.append(matched.select(exprs))

    if not per_alias:  # pragma: no cover - defensive: ops is non-empty so per_alias is too
        return None
    state = per_alias[0]
    for frame in per_alias[1:]:
        # Left-major cross join → same row order as the pandas constant-key merge.
        state = state.join(frame, how="cross")
    try:
        out_df = _lazy_collect(state)
    except pl.exceptions.SchemaError:  # pragma: no cover - defensive: cross-join schema clash declines
        return None
    return _rewrap(g, out_df)


def binding_rows_polars(
    g: Plottable,
    binding_ops: Sequence[Dict[str, JSONVal]],
    attach_prop_aliases: Optional[Sequence[str]] = None,
) -> Optional[Plottable]:
    """Native polars bindings-row table for connected alias patterns (#1709).

    Materializes one row per matched path for an alternating ``n/e/n/...`` pattern
    (the ``rows(binding_ops=...)`` op emitted by Cypher multi-alias lowering), with
    the same meaningful schema as the pandas engine: bare ``alias`` id columns,
    ``edge_alias.col`` edge-payload columns, and ``alias.{col}`` node-property
    columns per node alias. (The pandas frame additionally carries join-residue
    columns — raw ``node_id``, ``a__a_join__``, leaked ``__gfql_edge_index__`` —
    that no lowered query references; those are intentionally not replicated.)

    Covers fixed-length hops, bounded variable-length (directed ``-[*i..k]->`` and
    undirected ``-[*1..k]-``), unbounded DIRECTED fixed point (``-[*]->`` /
    ``-[*0..]->``), and the node-only cartesian mode.

    Returns None to DECLINE (caller raises the honest NIE) for anything outside
    that subset: undirected variable-length outside ``min_hops == 1`` (including
    undirected unbounded), aliased variable-length relationships, unbounded
    segments without ``to_fixed_point``, shortestPath scalar bindings, node
    ``query=`` / edge query or endpoint-match params, hop labels, HAS_-label
    destination disambiguation on duplicate-node-id graphs (unique-id graphs run
    native — pandas would not narrow there either), duplicate-id re-entry seeds,
    and the legacy ``alias_endpoints`` variant. NO-CHEATING: never bridges to
    pandas. Parity gate: differential tests vs the pandas oracle.
    """
    import polars as pl
    from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject, from_json as ast_from_json
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
    from graphistry.compute.gfql.same_path.edge_semantics import EdgeSemantics
    from.predicates import filter_by_dict_polars

    def _names(lf: pl.LazyFrame) -> List[str]:
        # LazyFrame column names WITHOUT collecting data (schema-only resolve).
        return lf.collect_schema().names()

    # Build from the PRE-CHAIN base graph, exactly like the pandas oracle
    # (`_gfql_binding_rows`: `base_nodes = base_graph._nodes`, every alias step
    # `op.execute(g=base_graph,...)`) and like the indexed builder below, which is
    # already handed `base_graph`. Rebuilding from the chain OUTPUT instead would
    # silently under-report: the traversal prunes to nodes/edges IT considers
    # matched, and that is not the same set the bindings builder matches — e.g. a
    # zero-hop var-length segment (`-[*0..k]->`, `-[*0..]->`) binds a seed with no
    # outgoing edge, which the traversal drops entirely. Fuzz-verified: with the
    # chain output as the source, `-[*0..2]->` lost those rows against pandas.
    base_graph = g._gfql_rows_base_graph
    if base_graph is None:
        base_graph = g
    nodes = base_graph._nodes
    edges = base_graph._edges
    node_id = base_graph._node
    src = base_graph._source
    dst = base_graph._destination
    if (nodes is None or node_id is None) and edges is not None:
        # Edges-only graph: materialize nodes (the node-set path did this
        # implicitly through the hop executor); keep the frame on polars.
        try:
            base_graph = base_graph.materialize_nodes()
        except Exception:
            return None
        nodes = base_graph._nodes
        node_id = base_graph._node
        from graphistry.Engine import Engine as _MatEngine, df_to_engine as _mat_to_engine, is_polars_df as _mat_is_polars
        if nodes is not None and not _mat_is_polars(nodes):
            nodes = _mat_to_engine(nodes, _MatEngine.POLARS)
    if nodes is None or edges is None or node_id is None or src is None or dst is None:
        return None
    seed_ids_lf: Optional[Any] = None  # LazyFrame; Any avoids the union-typed seed_nodes.join mismatch
    start_nodes = g._gfql_start_nodes
    if start_nodes is not None:
        # Bounded WITH->MATCH re-entry: the carried WITH rows seed the first
        # alias. Constrain the first node alias to the carried ids via a semi-join —
        # the native twin of the pandas wavefront seed. Support only UNIQUE carried
        # ids: then the semi-join contributes each seed node exactly once, matching the
        # pandas seed row-for-row; duplicate carried ids could change path multiplicity,
        # so decline (return None -> honest NIE), never risk a silent-wrong count.
        from graphistry.Engine import Engine as _Engine, df_to_engine as _df_to_engine, is_polars_df as _is_polars
        sn = start_nodes.collect() if isinstance(start_nodes, pl.LazyFrame) else start_nodes
        if not _is_polars(sn):
            sn = _df_to_engine(sn, _Engine.POLARS)
        if node_id not in sn.columns:
            return None
        seed_ids = sn.select(pl.col(node_id)).drop_nulls()
        # eager by construction (a LazyFrame start_nodes is collected above), but the polars
        # guard narrows to the eager-or-lazy union and cannot express "eager only"
        if seed_ids.height != seed_ids.unique().height:  # type: ignore[union-attr]
            return None
        seed_ids_lf = seed_ids.lazy()

    ops: List[ASTObject] = [ast_from_json(op_json, validate=False) for op_json in binding_ops]
    # Shared validation (engine-agnostic): raises the canonical GFQLValidationError
    # for malformed op sequences / duplicate aliases — same error as pandas.
    RowPipelineMixin._gfql_validate_binding_ops(ops)
    if RowPipelineMixin._gfql_binding_ops_mode(ops) == "node_cartesian":
        if seed_ids_lf is not None:
            # A WITH->MATCH seed constrains the FIRST alias, but the cartesian builder
            # below does not thread it; running it would silently ignore the seed
            # (wrong cross-product). Decline honestly (the alternating-path seed is
            # applied at seed_nodes; node-cartesian re-entry stays pandas-only).
            return None
        # MATCH (a), (b), ... disconnected node aliases: native cross-product.
        return _cartesian_node_bindings_polars(g, ops, node_id)
    if RowPipelineMixin._gfql_is_shortest_path_scalar_binding_ops(ops):
        return None  # shortestPath scalar contract: BFS/native backends, pandas-only

    from graphistry.compute.gfql.index import bindings as indexed_bindings
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
    from graphistry.Engine import Engine
    engine_concrete = (
        Engine.POLARS_GPU
        if active_target() == ExecutionTarget.GPU
        else Engine.POLARS
    )
    # The chain boundary may already have decided this exact plan (served or
    # declined) before the canonical traversal; reuse that decision instead of
    # recomputing it — and re-recording a duplicate trace step.
    from graphistry.compute.gfql.index.handoff import read_handoff

    handoff = read_handoff(g)
    plan = list(binding_ops)
    indexed_state = (
        handoff.state
        if handoff is not None and handoff.serves(plan, engine_concrete)
        else None
    )
    if indexed_state is None and not (handoff is not None and handoff.declined(plan)):
        indexed_state = indexed_bindings.try_indexed_connected_bindings_state(
            base_graph,
            ops,
            engine=engine_concrete,
            start_nodes=start_nodes,
        )
    if indexed_state is not None:
        return _finish_binding_rows_polars(
            g,
            ops,
            # engine-polymorphic by declaration; on the polars branch the helper
            # builds polars frames, so narrow once here rather than widening the
            # finisher's contract back to Any
            cast("pl.DataFrame", indexed_state.state),
            cast(Dict[str, "pl.DataFrame"], indexed_state.alias_frames),
            str(node_id),
            attach_prop_aliases,
            decline_on_schema_error=False,  # our own state: a schema clash is a bug
        )

    for idx, op in enumerate(ops):
        if idx % 2 == 0:
            if not isinstance(op, ASTNode) or op.query is not None:
                return None
        else:
            if not isinstance(op, ASTEdge):
                return None
            sem = EdgeSemantics.from_edge(op)
            if sem.is_multihop:
                # Served: bounded fwd/rev/undirected windows and the unbounded DIRECTED fixed point.
                if isinstance(op._name, str):
                    return None
                _resolved_max = op.max_hops if op.max_hops is not None else op.hops
                if bool(op.to_fixed_point) and _resolved_max is not None:
                    #  - `to_fixed_point` COMBINED WITH an explicit bound. Master declined
                    #    this outright (it declined on `bool(op.to_fixed_point)` alone), and
                    #    serving it is silently wrong for min_hops >= 3 — the same
                    #    reconstruction gap as the unbounded case: `MATCH (a)-[*3..5]->(b)`
                    #    with the flag set gives pandas 0 and polars 30 on a 7-node acyclic
                    #    graph. Cypher never emits this combination (the parser sets
                    #    to_fixed_point False for `*k` / `*i..k` and only leaves max_hops
                    #    None for `*` / `*k..`), so it is reachable through the AST /
                    #    `rows(binding_ops=...)` wire surface only — but "hard to reach" is
                    #    not "correct", and declining it merely restores what master did.
                    return None
                if _resolved_max is None:
                    if not bool(op.to_fixed_point) or op.direction == "undirected":
                        return None
                    # Unbounded serves min_hops 0 and 1 only; >= 2 is the divergence above.
                    _resolved_min_unbounded = op.min_hops if op.min_hops is not None else (
                        op.hops if op.hops is not None else 1
                    )
                    if _resolved_min_unbounded > 1:
                        return None
                if op.direction == "undirected":
                    _resolved_min = op.min_hops if op.min_hops is not None else (
                        op.hops if op.hops is not None else 1
                    )
                    if _resolved_min != 1:
                        return None
            # Residual decline: pandas' `max_reached_hop` is a dedup-by-node eccentricity, not a
            # longest-trail length, so a DIRECTED min_hops window under-reports on the pandas lane.
            if op.min_hops is not None or op.max_hops is not None:
                _vl_min = op.min_hops if op.min_hops is not None else (
                    op.hops if op.hops is not None else 1
                )
                _prev_op = ops[idx - 1] if idx >= 1 else None
                _seeded_start = start_nodes is not None or (
                    isinstance(_prev_op, ASTNode) and bool(_prev_op.filter_dict)
                )
                if op.direction != "undirected" and (
                    _vl_min >= 3 or (_vl_min >= 2 and _seeded_start)
                ):
                    return None
            if op.direction not in ("forward", "reverse", "undirected"):
                return None
            if any(
                value is not None
                for value in (
                    op.edge_query, op.source_node_match, op.destination_node_match,
                    op.source_node_query, op.destination_node_query,
                    op.label_node_hops, op.label_edge_hops,
                    op.output_min_hops, op.output_max_hops,
                )
            ):
                return None
            if bool(op.label_seeds) or bool(op.include_zero_hop_seed):
                return None

    node_id = str(node_id)
    src = str(src)
    dst = str(dst)

    try:
        # Build the WHOLE binding table as ONE deferred pl.LazyFrame and collect
        # ONCE on the active target: under engine='polars-gpu' the
        # entire join chain + property attach runs on cudf_polars in a single GPU
        # collect (~4-5× vs CPU on the join phase — de-risk probe 2026-07-06);
        # under 'polars' it collects on CPU (parity-identical). NO-CHEATING: a
        # GPU-incapable plan node makes `collect` raise NotImplementedError (honest
        # NIE → use engine='pandas'/'polars'), never a silent CPU fallback.
        nodes_lf = nodes.lazy()
        edges_lf = edges.lazy()
        # int-vs-float endpoint dtype mismatch (e.g. a null endpoint promoted the
        # column) SchemaErrors the join chain; align endpoints to the node-id
        # dtype when lossless, mirroring the chain traversal's join-key fix.
        # Output dtypes are untouched (alias columns come from the node frames).
        _node_dtype = nodes_lf.collect_schema().get(node_id)
        _edge_schema = edges_lf.collect_schema()
        _endpoint_casts = []
        for _endpoint in {src, dst}:
            _e_dtype = _edge_schema.get(_endpoint)
            if _e_dtype is None or _node_dtype is None or _e_dtype == _node_dtype:
                continue
            if _dtype_is_int(_e_dtype) and _dtype_is_float(_node_dtype):
                _endpoint_casts.append(pl.col(_endpoint).cast(_node_dtype))
            elif _dtype_is_float(_e_dtype) and _dtype_is_int(_node_dtype):
                _nonintegral = bool(
                    edges_lf.select(
                        (
                            pl.col(_endpoint).is_not_null()
                            & (pl.col(_endpoint) != pl.col(_endpoint).round(0))
                        ).any()
                    ).collect().item()
                )
                if not _nonintegral:
                    _endpoint_casts.append(pl.col(_endpoint).cast(_node_dtype))
        if _endpoint_casts:
            edges_lf = edges_lf.with_columns(_endpoint_casts)
        # openCypher trail semantics: stable per-edge identity for the
        # at-most-once-per-path relationship constraint (pandas twin:
        # _gfql_connected_bindings_state's __gfql_edge_ident__).
        edges_lf = edges_lf.with_row_index(TRAIL_EDGE_IDENT_COL)
        trail_cols_pl: List[str] = []
        first_op = ops[0]
        if not isinstance(first_op, ASTNode):
            return None
        seed_nodes = filter_by_dict_polars(nodes_lf, first_op.filter_dict)
        if seed_ids_lf is not None:
            # WITH->MATCH re-entry seed: constrain the first alias to the carried ids.
            seed_nodes = seed_nodes.join(seed_ids_lf, on=node_id, how="semi")
        # The whole generic builder works in LazyFrames (`nodes_lf` / `edges_lf` above);
        # `filter_by_dict_polars` is frame-polymorphic at runtime but declares the eager
        # type, so pin the path bag lazy here instead of leaving every downstream lazy
        # op to fight an eager inference.
        state: pl.LazyFrame = seed_nodes.select(pl.col(node_id).alias(WALK_CURRENT_COL))  # type: ignore[assignment]
        alias_frames: Dict[str, pl.LazyFrame] = {}
        node_aliases: List[str] = []
        first_alias = first_op._name
        if isinstance(first_alias, str):
            state = state.with_columns(pl.col(WALK_CURRENT_COL).alias(first_alias))
            alias_frames[first_alias] = seed_nodes
            node_aliases.append(first_alias)

        for edge_idx in range(1, len(ops), 2):
            edge_op = ops[edge_idx]
            if not isinstance(edge_op, ASTEdge):
                return None
            sem = EdgeSemantics.from_edge(edge_op)
            edges_f = filter_by_dict_polars(edges_lf, edge_op.edge_match)
            edge_alias = edge_op._name
            if isinstance(edge_alias, str):
                payload_renames = {
                    col: f"{edge_alias}.{col}"
                    for col in _names(edges_f)
                    if col not in (src, dst, TRAIL_EDGE_IDENT_COL)
                }
            else:
                # Unaliased edge payload is unaddressable downstream; carrying it
                # unprefixed (as pandas does) only risks column collisions.
                edges_f = edges_f.select([src, dst, TRAIL_EDGE_IDENT_COL])
                payload_renames = {}
            if sem.is_undirected:
                fwd = edges_f.rename({src: WALK_FROM_COL, dst: WALK_TO_COL})
                rev = edges_f.rename({dst: WALK_FROM_COL, src: WALK_TO_COL})
                oriented = pl.concat([fwd, rev.select(_names(fwd))], how="vertical")
                # A self-loop's two undirected orientations are the SAME binding:
                # dedupe the flip twin.
                oriented = oriented.unique(
                    subset=[WALK_FROM_COL, WALK_TO_COL, TRAIL_EDGE_IDENT_COL],
                    keep="first",
                    maintain_order=True,
                )
            else:
                join_col, result_col = (dst, src) if edge_op.direction == "reverse" else (src, dst)
                oriented = edges_f.rename({join_col: WALK_FROM_COL, result_col: WALK_TO_COL})
            if payload_renames:
                oriented = oriented.rename(payload_renames)

            next_op = ops[edge_idx + 1]
            if not isinstance(next_op, ASTNode):
                return None
            next_nodes = filter_by_dict_polars(nodes_lf, next_op.filter_dict)
            next_node_ids = next_nodes.select(node_id).unique()
            if not sem.is_multihop:
                # Filter endpoint candidates before joining from the current state.
                # For graph-bench q5/q6/q7, pushed Interest/City predicates make
                # this turn an all-edges scan into a small-domain edge semi-join.
                oriented = oriented.join(
                    next_node_ids,
                    left_on=WALK_TO_COL,
                    right_on=node_id,
                    how="semi",
                )

            # Column collision between edge payload and accumulated state → decline
            # (pandas resolves via merge suffixes; unreferenced-by-queries either way).
            overlap = (set(_names(oriented)) - {WALK_FROM_COL}) & set(_names(state))
            if overlap:
                return None
            if sem.is_multihop:
                # Same defaults as the pandas builder: a bare hops=k means exactly k.
                min_hops_value = edge_op.min_hops if edge_op.min_hops is not None else (
                    edge_op.hops if edge_op.hops is not None else 1
                )
                max_hops_value = edge_op.max_hops if edge_op.max_hops is not None else edge_op.hops
                min_hops = int(min_hops_value)
                state_cols = _names(state)
                if max_hops_value is None:
                    # UNBOUNDED directed fixed point (`-[*0..]->`, LDBC IS6). Gated
                    # above to to_fixed_point=True and a directed edge. Termination is
                    # data-dependent, so unlike the bounded branch this one cannot stay
                    # fully lazy — it collects one frontier per hop.
                    state, _fp_trail_cols = _directed_fixed_point_binding_rows_polars(
                        state,
                        oriented.select([WALK_FROM_COL, WALK_TO_COL]),
                        state_cols,
                        min_hops=min_hops,
                    )
                elif sem.is_undirected:
                    # One row per orientation (a self-loop just one); a relationship binds once per
                    # path, so a same-edge backtrack dies while a PARALLEL-edge return trip is legal.
                    max_hops = int(max_hops_value)
                    normal = edges_f.filter(pl.col(src) != pl.col(dst))
                    loops = edges_f.filter(pl.col(src) == pl.col(dst))
                    ident = pl.col(TRAIL_EDGE_IDENT_COL)
                    fwd = normal.select([pl.col(src).alias(WALK_FROM_COL), pl.col(dst).alias(WALK_TO_COL), ident])
                    rev = normal.select([pl.col(dst).alias(WALK_FROM_COL), pl.col(src).alias(WALK_TO_COL), ident])
                    loop = loops.select([pl.col(src).alias(WALK_FROM_COL), pl.col(dst).alias(WALK_TO_COL), ident])
                    pairs = pl.concat([fwd, rev, loop], how="vertical")
                    reachable = [state.select(state_cols)] if min_hops == 0 else []
                    current = state
                    _und_trail_cols: List[str] = []
                    for _hop in range(1, max_hops + 1):
                        joined = current.join(
                            pairs, left_on=WALK_CURRENT_COL, right_on=WALK_FROM_COL, how="inner"
                        )
                        for _used in trail_cols_pl + _und_trail_cols:
                            joined = joined.filter(
                                (pl.col(TRAIL_EDGE_IDENT_COL) != pl.col(_used)) | pl.col(_used).is_null()
                            )
                        _hop_trail = trail_column_name(len(trail_cols_pl) + len(_und_trail_cols))
                        joined = joined.rename({TRAIL_EDGE_IDENT_COL: _hop_trail})
                        _und_trail_cols.append(_hop_trail)
                        joined = joined.drop(WALK_CURRENT_COL).rename({WALK_TO_COL: WALK_CURRENT_COL})
                        current = joined.select(state_cols + _und_trail_cols)
                        if _hop >= min_hops:
                            reachable.append(current)
                    state = pl.concat(reachable, how="diagonal") if reachable else state.limit(0)
                    trail_cols_pl = trail_cols_pl + _und_trail_cols
                else:
                    # Bounded directed var-length (`-[*1..k]->`), trail-tracked.
                    state, _seg_trail_cols = _directed_varlen_reachable_polars(
                        state,
                        oriented.select([WALK_FROM_COL, WALK_TO_COL, TRAIL_EDGE_IDENT_COL]),
                        state_cols,
                        min_hops=min_hops,
                        max_hops=int(max_hops_value),
                        trail_cols_in=trail_cols_pl,
                    )
                    trail_cols_pl = trail_cols_pl + _seg_trail_cols
            else:
                state = (
                    state.join(oriented, left_on=WALK_CURRENT_COL, right_on=WALK_FROM_COL, how="inner")
.drop(WALK_CURRENT_COL)
.rename({WALK_TO_COL: WALK_CURRENT_COL})
                )
                for _used in trail_cols_pl:
                    state = state.filter(
                        (pl.col(TRAIL_EDGE_IDENT_COL) != pl.col(_used)) | pl.col(_used).is_null()
                    )
                _new_trail = trail_column_name(len(trail_cols_pl))
                state = state.rename({TRAIL_EDGE_IDENT_COL: _new_trail})
                trail_cols_pl = trail_cols_pl + [_new_trail]

            state = state.join(
                next_node_ids,
                left_on=WALK_CURRENT_COL,
                right_on=node_id,
                how="semi",
            )
            # HAS_<Label> destination disambiguation (pandas'
            # _gfql_disambiguate_has_edge_destination_nodes): on DUPLICATE-id graphs
            # pandas narrows the unlabeled next op to the edge's HAS_<Label> rows
            # taken from the ORIGINAL node table, which still carries the colliding
            # label rows. Reproducing that narrowing natively would be silently
            # row-order-dependent, so: unique-id graphs need no narrowing (pandas'
            # duplicated() probe is False) → native is parity-exact; duplicate-id
            # graphs DECLINE (honest NIE). ``nodes`` above IS the pre-chain node
            # table; when there is no pre-chain graph to probe we cannot prove
            # uniqueness of what pandas would have seen, so decline.
            dis_label_col = RowPipelineMixin._gfql_has_edge_destination_label_col(edge_op, nodes.columns)
            if (
                dis_label_col is not None
                and not sem.is_multihop
                and edge_op.direction == "forward"
                and not RowPipelineMixin._gfql_node_filter_has_label(next_op.filter_dict)
            ):
                if g._gfql_rows_base_graph is None:
                    return None
                _base_dup = bool(
                    nodes.lazy()
.select(pl.col(node_id).is_duplicated().any())
.collect()
.item()
                )
                if _base_dup:
                    return None
            next_alias = next_op._name
            if isinstance(next_alias, str):
                state = state.with_columns(pl.col(WALK_CURRENT_COL).alias(next_alias))
                alias_frames[next_alias] = next_nodes
                node_aliases.append(next_alias)

        if trail_cols_pl:
            _present = [c for c in trail_cols_pl if c in _names(state)]
            if _present:
                state = state.drop(_present)
        # The finisher's frame type is a constrained TypeVar so `state.join(lookup)`
        # type-checks. The GENERIC builder above mixes eager and lazy frames across
        # its (pre-existing) branches, so inference cannot pick one here; the
        # indexed caller binds cleanly. Narrow just this call rather than widening
        # the finisher back to `Any`.
        return _finish_binding_rows_polars(  # type: ignore[misc]
            g, ops, state, alias_frames, node_id, attach_prop_aliases,
            decline_on_schema_error=True,  # pandas-vs-polars join-key dtype divergence
        )
    except pl.exceptions.SchemaError:
        return None


def can_select_native(items: Sequence[SelectItem], columns: Sequence[str]) -> bool:
    return lower_select_items(items, columns) is not None


def can_order_by_native(keys: Sequence[OrderKey], columns: Sequence[str]) -> bool:
    return lower_order_by_keys(keys, columns) is not None
