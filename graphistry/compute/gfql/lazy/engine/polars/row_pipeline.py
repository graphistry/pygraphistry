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

import contextvars
import operator
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.expr_parser import ExprNode, FunctionCall

from graphistry.Plottable import Plottable
from .dtypes import is_float as _dtype_is_float, is_int as _dtype_is_int, is_numeric as _dtype_is_numeric, is_stringlike as _dtype_is_stringlike


# Active row-table schema (col -> dtype), set around lowering so lower_expr can infer FLOAT
# operands for the NaN guard. Free to populate — schema is already on the table, no scan.
_SCHEMA: "contextvars.ContextVar[dict]" = contextvars.ContextVar("gfql_polars_schema", default={})

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
    "%": operator.mod,  # polars mod is floored, like pandas (NOTE: no negative-operand % conformance case yet)
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
    if prop in columns and alias in columns:
        return prop
    return None


def _lower_function(node: FunctionCall, columns: Sequence[str]) -> Optional[pl.Expr]:
    """Lower a whitelisted scalar cypher function to polars, or None to defer. Only
    parity-verified mappings admitted; anything else returns None (caller NIEs, never guesses)."""
    import polars as pl  # function-local: polars is an optional dependency
    name = node.name.lower()
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


def _nan_guard(result: pl.Expr, op: str, left: pl.Expr, right: pl.Expr, ldt: Optional[pl.DataType], rdt: Optional[pl.DataType]) -> pl.Expr:
    """Mask a comparison so NaN compares IEEE/pandas/Cypher-style (false; ``!=`` true), not
    polars-style (NaN = largest). ``is_nan()`` applied only to float-OUTPUT operands; no-op
    for int/string/bool comparisons."""
    nan_terms = []
    if _dtype_is_float(ldt):
        nan_terms.append(left.is_nan())
    if _dtype_is_float(rdt):
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
        return pl.col(node.name) if node.name in columns else None
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
        # decline (NIE): int-literal division — Cypher folds 5/2 to 2 (truncating; x/0 errors) vs
        # polars true division 2.5, silently wrong inside a non-monotonic op (e.g. ORDER BY
        # n.val % (10/4) sorts differently); pandas folds it. Column / int is Float on both, so kept.
        if node.op == "/" and _is_int_literal(node.left) and _is_int_literal(node.right):
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
        result = _apply_binop(node.op, left, right)
        if result is not None and node.op in _NAN_GUARD_OPS:
            result = _nan_guard(result, node.op, left, right, ldt, rdt)
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


def lower_select_items(items: Sequence[Any], columns: Sequence[str]) -> Optional[List[Any]]:
    """Lower projection items [(alias, expr) | 'col'] to polars exprs, or None."""
    out: List[Any] = []
    for item in items:
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


def lower_order_by_keys(keys: Sequence[Any], columns: Sequence[str]) -> Optional[Tuple[List[Any], List[bool]]]:
    """Lower order_by [(expr, direction)] to (polars exprs, descending flags)."""
    exprs: List[Any] = []
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


def _lower_with_schema(table: Any, fn):
    """Run a lowering callable with the table schema published to ``_SCHEMA`` (float-operand
    inference for the NaN guard)."""
    token = _SCHEMA.set(dict(table.schema))
    try:
        return fn()
    finally:
        _SCHEMA.reset(token)


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


def _project_polars(g: Plottable, items: Sequence[Any], extend: bool) -> Optional[Plottable]:
    """Shared body of ``select_polars`` / ``with_columns_polars``; None if any item isn't
    lowerable (honest NIE, no pandas bridge)."""
    table = _active_table(g)
    exprs = _lower_with_schema(table, lambda: lower_select_items(items, list(table.columns)))
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


def select_polars(g: Plottable, items: Sequence[Any]) -> Optional[Plottable]:
    """Native polars projection (replaces the row table)."""
    return _project_polars(g, items, extend=False)


def with_columns_polars(g: Plottable, items: Sequence[Any]) -> Optional[Plottable]:
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
    filter_dict entries are scalar-equality conjuncts.
    """
    import polars as pl
    table = _active_table(g)
    columns = list(table.columns)
    preds: List[Any] = []
    if filter_dict:
        for col, val in filter_dict.items():
            if col not in columns or isinstance(val, dict):
                return None  # missing column / nested-struct value -> defer (NIE)
            if isinstance(val, (list, tuple, set)):
                # IN: `is_in` on a null cell -> null -> filter drops it, i.e. openCypher 3VL
                # (`null IN [...]` = null -> excluded), matching the filter_by_dict membership
                # fix. (Equality below also drops nulls: `null == v` -> null -> dropped.)
                preds.append(pl.col(col).is_in(list(val)))
            else:
                preds.append(pl.col(col) == val)
    if expr is not None:
        if not isinstance(expr, str):
            return None
        lowered = _lower_with_schema(table, lambda: lower_expr_str(expr, columns))
        if lowered is None:
            return None
        preds.append(lowered)
    if not preds:
        return g  # empty WHERE -> identity
    combined = preds[0]
    for pred in preds[1:]:
        combined = combined & pred
    return _rewrap(g, table.filter(combined))


def order_by_polars(g: Plottable, keys: Sequence[Any]) -> Optional[Plottable]:
    """Native polars sort; None if any key isn't lowerable."""
    table = _active_table(g)
    lowered = _lower_with_schema(table, lambda: lower_order_by_keys(keys, list(table.columns)))
    if lowered is None:
        return None
    exprs, descending = lowered
    # cypher ORDER BY puts NULLs last; polars defaults to nulls_last=False (pandas sort_values
    # default puts NaN last only for asc), so set nulls_last=True to match pandas na_position='last'.
    return _rewrap(g, table.sort(exprs, descending=descending, nulls_last=True))


# Native aggs: count/sum/avg/min/max/count_distinct/collect/collect_distinct; stdev/percentile
# etc. return None → caller declines (NIE).
def _agg_expr(func: str, expr: Optional[str], columns: Sequence[str], alias: str, schema: Optional[dict] = None) -> Optional[pl.Expr]:
    import polars as pl
    func = func.lower()
    if func == "count" and (expr is None or expr == "*"):
        return pl.len().alias(alias)
    if not isinstance(expr, str) or expr not in columns:
        return None
    col = pl.col(expr)
    # pandas aggs skip NaN (skipna); polars skips only NULL and treats NaN as a value (NaN == NaN
    # is True, so self-inequality can't detect it). For FLOAT columns convert in-query NaN -> null
    # first so every agg matches the oracle (pandas sum([nan, 1]) == 1 vs raw polars == nan).
    # fill_nan is float-only, hence the dtype gate. Stored NaN is nulled at ingestion; this covers
    # NaN created mid-query (e.g. 0.0/0.0).
    if schema is not None and _dtype_is_float(schema.get(expr)):
        col = col.fill_nan(None)
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


def group_by_polars(g: Plottable, keys: Sequence[Any], aggregations: Sequence[Any]) -> Optional[Plottable]:
    """Native polars group-by; None if a key/agg isn't lowerable. Matches pandas dropna=False
    (null keys kept) + non-null agg semantics; output order is first-occurrence (maintain_order),
    though the parity gate compares order-insensitively."""
    table = _active_table(g)
    cols = list(table.columns)
    if not keys or not all(isinstance(k, str) and k in cols for k in keys):
        return None
    aggs: List[Any] = []
    for agg in aggregations:
        if not isinstance(agg, (list, tuple)) or len(agg) not in (2, 3):
            return None
        alias = str(agg[0])
        func = str(agg[1])
        expr = agg[2] if len(agg) == 3 else None
        lowered = _agg_expr(func, expr, cols, alias, table.schema)
        if lowered is None:
            return None
        aggs.append(lowered)
    out = table.group_by(list(keys), maintain_order=True).agg(aggs)
    return _rewrap(g, out)


def unwind_polars(g: Plottable, expr: str, as_: str = "value") -> Optional[Plottable]:
    """Native UNWIND for a literal list: cross-join each row with the values (cypher per-row
    expansion; empty list → 0 rows); None → caller NIEs. List-column / expression unwinds
    (null/empty-element semantics) decline (NIE) for now."""
    import polars as pl
    from graphistry.compute.gfql.expr_parser import ListLiteral, Literal

    if not isinstance(expr, str):
        return None
    parse = _parser()
    if parse is None:
        return None
    try:
        node = parse(expr)
    except Exception:
        return None
    if not isinstance(node, ListLiteral) or not all(isinstance(it, Literal) for it in node.items):
        return None
    table = _active_table(g)
    if as_ in table.columns:
        return None
    values = [it.value for it in node.items if isinstance(it, Literal)]
    rhs = pl.DataFrame({as_: values})
    return _rewrap(g, table.join(rhs, how="cross"))


def select_extend_polars(g: Plottable, items: Sequence[Any]) -> Optional[Plottable]:
    """Native polars ``with_(items, extend=True)``: add/overwrite projected columns
    while keeping the existing row table (pandas ``assign`` semantics). Emitted by
    the bindings-path aggregate lowering (pre-aggregation group keys / agg args),
    so it is required for binding-row queries (#1709). None → NIE."""
    table = _active_table(g)
    exprs = _lower_with_schema(table, lambda: lower_select_items(items, list(table.columns)))
    if exprs is None:
        return None
    out = table.with_columns(exprs)
    if _select_emits_temporal_constructor_text(out):
        return None
    return _rewrap(g, out)


def binding_rows_polars(
    g: Plottable,
    binding_ops: Sequence[Any],
    attach_prop_aliases: Optional[List[str]] = None,
) -> Optional[Plottable]:
    """Native polars bindings-row table for FIXED-LENGTH connected patterns (#1709).

    Materializes one row per matched path for an alternating ``n/e/n/...`` pattern
    (the ``rows(binding_ops=...)`` op emitted by Cypher multi-alias lowering), with
    the same meaningful schema as the pandas engine: bare ``alias`` id columns,
    ``edge_alias.col`` edge-payload columns, and ``alias.{col}`` node-property
    columns per node alias. (The pandas frame additionally carries join-residue
    columns — raw ``node_id``, ``a__a_join__``, leaked ``__gfql_edge_index__`` —
    that no lowered query references; those are intentionally not replicated.)

    Returns None to DECLINE (caller raises the honest NIE) for anything outside
    the supported subset: variable-length/multi-hop edges, shortestPath scalar
    bindings, node ``query=`` / edge query or endpoint-match params, hop labels,
    HAS_-label destination disambiguation, seeded re-entry contexts, cartesian
    (node-only) mode, and the legacy ``alias_endpoints`` variant. NO-CHEATING:
    never bridges to pandas. Parity gate: differential tests vs the pandas oracle.
    """
    import polars as pl
    from graphistry.compute.ast import ASTEdge, ASTNode, from_json as ast_from_json
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin
    from graphistry.compute.gfql.same_path.edge_semantics import EdgeSemantics
    from .predicates import filter_by_dict_polars

    def _names(lf: Any) -> List[str]:
        # LazyFrame column names WITHOUT collecting data (schema-only resolve).
        return lf.collect_schema().names()

    nodes = g._nodes
    edges = g._edges
    node_id = g._node
    src = g._source
    dst = g._destination
    if nodes is None or edges is None or node_id is None or src is None or dst is None:
        return None
    if getattr(g, "_gfql_start_nodes", None) is not None:
        # Bounded re-entry seeds the first alias from carried rows — pandas-only.
        return None

    ops = [ast_from_json(op_json, validate=False) for op_json in binding_ops]
    # Shared validation (engine-agnostic): raises the canonical GFQLValidationError
    # for malformed op sequences / duplicate aliases — same error as pandas.
    RowPipelineMixin._gfql_validate_binding_ops(ops)
    if RowPipelineMixin._gfql_binding_ops_mode(ops) == "node_cartesian":
        return None  # MATCH (a), (b) cross joins: deferred (rare; own schema study)
    if RowPipelineMixin._gfql_is_shortest_path_scalar_binding_ops(ops):
        return None  # shortestPath scalar contract: BFS/native backends, pandas-only

    for idx, op in enumerate(ops):
        if idx % 2 == 0:
            if not isinstance(op, ASTNode) or getattr(op, "query", None) is not None:
                return None
        else:
            if not isinstance(op, ASTEdge):
                return None
            sem = EdgeSemantics.from_edge(op)
            if sem.is_multihop:
                # Bounded directed var-length (`-[*1..k]->`, graph-bench q3) is
                # supported via iterative pair joins; everything else declines:
                # unbounded (`[*]`, needs fixed-point + termination error),
                # undirected multihop (immediate-backtrack avoidance not ported),
                # and aliased var-length edges (pandas rejects those outright).
                if (
                    op.direction == "undirected"
                    or bool(op.to_fixed_point)
                    or (op.max_hops is None and op.hops is None)
                    or isinstance(getattr(op, "_name", None), str)
                ):
                    return None
            if op.direction not in ("forward", "reverse", "undirected"):
                return None
            if any(
                getattr(op, attr, None) is not None
                for attr in (
                    "edge_query", "source_node_match", "destination_node_match",
                    "source_node_query", "destination_node_query",
                    "label_node_hops", "label_edge_hops",
                    "output_min_hops", "output_max_hops",
                )
            ):
                return None
            if bool(getattr(op, "label_seeds", False)) or bool(getattr(op, "include_zero_hop_seed", False)):
                return None
            # Duplicate-id + HAS_<Label> endpoint disambiguation: pandas has a
            # bespoke candidate-domain rule here; decline rather than diverge.
            if RowPipelineMixin._gfql_has_edge_destination_label_col(op, nodes.columns) is not None:
                return None

    node_id = str(node_id)
    src = str(src)
    dst = str(dst)

    try:
        # Build the WHOLE binding table as ONE deferred pl.LazyFrame and collect
        # ONCE on the active target (#1709 laziness): under engine='polars-gpu' the
        # entire join chain + property attach runs on cudf_polars in a single GPU
        # collect (~4-5× vs CPU on the join phase — de-risk probe 2026-07-06);
        # under 'polars' it collects on CPU (parity-identical). NO-CHEATING: a
        # GPU-incapable plan node makes `collect` raise NotImplementedError (honest
        # NIE → use engine='pandas'/'polars'), never a silent CPU fallback.
        nodes_lf = nodes.lazy()
        edges_lf = edges.lazy()
        seed_nodes = filter_by_dict_polars(nodes_lf, getattr(ops[0], "filter_dict", None))
        state = seed_nodes.select(pl.col(node_id).alias("__current__"))
        alias_frames: dict = {}
        node_aliases: List[str] = []
        first_alias = getattr(ops[0], "_name", None)
        if isinstance(first_alias, str):
            state = state.with_columns(pl.col("__current__").alias(first_alias))
            alias_frames[first_alias] = seed_nodes
            node_aliases.append(first_alias)

        for edge_idx in range(1, len(ops), 2):
            edge_op = ops[edge_idx]
            sem = EdgeSemantics.from_edge(edge_op)
            edges_f = filter_by_dict_polars(edges_lf, getattr(edge_op, "edge_match", None))
            edge_alias = getattr(edge_op, "_name", None)
            if isinstance(edge_alias, str):
                payload_renames = {
                    col: f"{edge_alias}.{col}"
                    for col in _names(edges_f)
                    if col not in (src, dst)
                }
            else:
                # Unaliased edge payload is unaddressable downstream; carrying it
                # unprefixed (as pandas does) only risks column collisions.
                edges_f = edges_f.select([src, dst])
                payload_renames = {}
            if sem.is_undirected:
                fwd = edges_f.rename({src: "__from__", dst: "__to__"})
                rev = edges_f.rename({dst: "__from__", src: "__to__"})
                oriented = pl.concat([fwd, rev.select(_names(fwd))], how="vertical")
            else:
                join_col, result_col = (dst, src) if edge_op.direction == "reverse" else (src, dst)
                oriented = edges_f.rename({join_col: "__from__", result_col: "__to__"})
            if payload_renames:
                oriented = oriented.rename(payload_renames)
            # Column collision between edge payload and accumulated state → decline
            # (pandas resolves via merge suffixes; unreferenced-by-queries either way).
            overlap = (set(_names(oriented)) - {"__from__"}) & set(_names(state))
            if overlap:
                return None
            if sem.is_multihop:
                # Bounded directed var-length: iterative pair joins, one row per
                # distinct edge sequence (Cypher path multiplicity — pairs NOT
                # deduped, so parallel edges multiply per hop, matching pandas
                # `_gfql_multihop_binding_rows`). Zero-hop rows (min 0) keep the
                # seed row (endpoint == start), also matching pandas.
                # Same defaults as the pandas builder: bare hops=k means exactly-k.
                min_hops = edge_op.min_hops if edge_op.min_hops is not None else (
                    edge_op.hops if edge_op.hops is not None else 1
                )
                max_hops = edge_op.max_hops if edge_op.max_hops is not None else edge_op.hops
                pairs = oriented.select(["__from__", "__to__"])
                state_cols = _names(state)
                reachable = [state] if min_hops == 0 else []
                current = state
                # Lazy: build all max_hops iterations (no eager .height early-break —
                # empty intermediates lazily join to empty, so the result is
                # identical; the pandas break is an optimization, not semantics).
                for _hop in range(1, int(max_hops) + 1):
                    current = (
                        current.join(pairs, left_on="__current__", right_on="__from__", how="inner")
                        .drop("__current__")
                        .rename({"__to__": "__current__"})
                        .select(state_cols)
                    )
                    if _hop >= min_hops:
                        reachable.append(current)
                state = pl.concat(reachable, how="vertical") if reachable else state.limit(0)
            else:
                state = (
                    state.join(oriented, left_on="__current__", right_on="__from__", how="inner")
                    .drop("__current__")
                    .rename({"__to__": "__current__"})
                )

            next_op = ops[edge_idx + 1]
            next_nodes = filter_by_dict_polars(nodes_lf, getattr(next_op, "filter_dict", None))
            state = state.join(
                next_nodes.select(node_id).unique(),
                left_on="__current__",
                right_on=node_id,
                how="semi",
            )
            next_alias = getattr(next_op, "_name", None)
            if isinstance(next_alias, str):
                state = state.with_columns(pl.col("__current__").alias(next_alias))
                alias_frames[next_alias] = next_nodes
                node_aliases.append(next_alias)

        # #1711 projection-pushdown: attach_prop_aliases (from the cypher lowering)
        # names node aliases whose PROPERTIES are referenced downstream; others skip
        # the property join (their bare id column suffices). None = attach all.
        attach_set = None if attach_prop_aliases is None else set(attach_prop_aliases)
        for alias in node_aliases:
            if attach_set is not None and alias not in attach_set:
                continue  # properties unreferenced — keep only the bare id column
            lookup_src = alias_frames[alias]
            lookup = lookup_src.select(
                [pl.col(node_id), pl.col(node_id).alias(f"{alias}.{node_id}")]
                + [
                    pl.col(col).alias(f"{alias}.{col}")
                    for col in _names(lookup_src)
                    if col != node_id
                ]
            )
            if (set(_names(lookup)) - {node_id}) & set(_names(state)):
                return None
            state = state.join(lookup, left_on=alias, right_on=node_id, how="left")
        state = state.drop("__current__")
        # Single collect on the active target (CPU / GPU). Deferred SchemaError
        # (int/float join-key dtype divergence pandas unifies implicitly) surfaces
        # here → decline honestly; a GPU-incapable node raises NotImplementedError
        # (from the lazy `collect` NO-CHEATING contract), which we let propagate.
        out_df = _lazy_collect(state)
    except pl.exceptions.SchemaError:
        return None

    out = _rewrap(g, out_df)
    edge_aliases = {
        alias
        for op in ops[1::2]
        for alias in [getattr(op, "_name", None)]
        if isinstance(alias, str)
    }
    setattr(out, "_gfql_rows_edge_aliases", edge_aliases)
    return out


def can_select_native(items: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_select_items(items, columns) is not None


def can_order_by_native(keys: Sequence[Any], columns: Sequence[str]) -> bool:
    return lower_order_by_keys(keys, columns) is not None
