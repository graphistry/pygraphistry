"""Cypher ``CALL graphistry.sem_decide`` / ``CALL graphistry.sem_map``: option parsing, YIELD columns and execution.

The call reads the rows of the clause before it, runs the same kernel as ``g.sem_decide`` / ``g.sem_map``, and
keeps the upstream columns beside the yielded ones. ``$param`` values are already resolved when options are parsed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

from graphistry.Engine import Engine, EngineAbstract, is_polars_df, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.call.validation import validate_call_params
from graphistry.compute.gfql.cypher.ast import CallClause, CypherYieldItem
from graphistry.compute.typing import DataFrameT

if TYPE_CHECKING:
    from graphistry.compute.execution_context import ExecutionContext
    from graphistry.compute.gfql.cypher.call_procedures import CompiledCypherProcedureCall
    from graphistry.compute.gfql.cypher.lowering import CompiledCypherQuery
    from graphistry.compute.gfql.policy import PolicyDict

OptionValue = Union[None, bool, int, float, str, List["OptionValue"], Dict[str, "OptionValue"]]

DECIDE = "graphistry.sem_decide"
MAP = "graphistry.sem_map"

DEFAULT_PROVIDER = "default"
DEFAULT_TASK_IDS = {DECIDE: "cypher_sem_decide", MAP: "cypher_sem_map"}

# Keys that pass straight through to the kernel.
_COMMON_KEYS = frozenset({
    "provider", "task_id", "task_version", "id_col", "batch_size", "max_concurrency", "deadline_s", "budget",
    "max_rows", "column_max_chars", "overwrite", "dedupe", "template",
})
_DECIDE_KEYS = _COMMON_KEYS | frozenset({
    "question", "questions", "kind", "options", "levels", "evidence", "out_col", "score_col", "status_col",
    "min_p", "require_probabilities", "max_usd", "max_requests",
})
_MAP_KEYS = _COMMON_KEYS | frozenset({
    "question", "instruction", "out_col", "evidence", "max_output_tokens", "max_requests",
})
_DECIDE_SUFFIXES = (("", "value"), ("__p", "score"), ("__status", "status"), ("__dist", "distribution"),
                    ("__probability_kind", "probability_kind"), ("__error_code", "error_code"), ("__meta", "meta"))
_MAP_SUFFIXES = (("", "value"), ("__status", "status"), ("__error_code", "error_code"), ("__meta", "meta"))


def _fail(call: CallClause, message: str, *, value: OptionValue = None, key: Optional[str] = None,
          suggestion: str = "") -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field="call.args" if key is None else f"call.args.{key}",
        value=value if value is not None else call.procedure,
        suggestion=suggestion or "See the semantic CALL reference: CALL graphistry.sem_decide({question: ..., evidence: [...]}) YIELD verdict, score, status.",
        line=call.span.line,
        column=call.span.column,
        language="cypher",
    )


def _options(call: CallClause, params: Optional[Mapping[str, OptionValue]]) -> Dict[str, OptionValue]:
    if len(call.args) != 1:
        raise _fail(call, f"CALL {call.procedure} takes one options map, for example {{question: $q, evidence: ['text']}} or $opts")
    from graphistry.compute.gfql.cypher.call_procedures import _eval_constant_call_arg
    from graphistry.compute.gfql.cypher.lowering import _parse_row_expr

    arg = call.args[0]
    node = _parse_row_expr(arg.text, params=params, field="call.args", line=arg.span.line, column=arg.span.column)
    value = _eval_constant_call_arg(node, call=call, raw_text=arg.text)
    if not isinstance(value, dict):
        raise _fail(call, f"CALL {call.procedure} options must be a map", value=arg.text)
    return dict(value)


def _only_keys(call: CallClause, options: Mapping[str, OptionValue], allowed: FrozenSet[str]) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise _fail(
            call, f"Unknown option {unknown[0]!r} for CALL {call.procedure}", key=unknown[0],
            suggestion=f"Use only: {', '.join(sorted(allowed))}.",
        )


def _text(call: CallClause, options: Mapping[str, OptionValue], key: str, default: Optional[str] = None) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or not value:
        raise _fail(call, f"CALL {call.procedure} option {key!r} must be a non-empty string", value=value, key=key)
    return value


def _evidence(call: CallClause, options: Mapping[str, OptionValue], carried: Sequence[str]) -> Dict[str, OptionValue]:
    has_evidence, has_template = "evidence" in options, "template" in options
    if has_evidence == has_template:
        raise _fail(call, f"CALL {call.procedure} needs exactly one of 'evidence' (column list) or 'template'",
                    key="evidence")
    if has_template:
        return {"template": options["template"]}
    cols = options["evidence"]
    if not isinstance(cols, list) or not cols or not all(isinstance(col, str) for col in cols):
        raise _fail(call, "CALL option 'evidence' must be a non-empty list of column names", value=str(cols),
                    key="evidence")
    if carried:
        missing = [col for col in cols if col not in carried]
        if missing:
            raise _fail(
                call, f"CALL evidence column {missing[0]!r} is not a column of the clause before the CALL",
                value=missing[0], key="evidence",
                suggestion=f"Project it first, for example WITH ... AS {missing[0]}. Available: {', '.join(carried)}.",
            )
    return {"evidence_cols": list(cols)}


def _budget(call: CallClause, options: Mapping[str, OptionValue]) -> Dict[str, OptionValue]:
    budget: Dict[str, OptionValue] = {}
    raw = options.get("budget")
    if raw is not None:
        if not isinstance(raw, dict):
            raise _fail(call, "CALL option 'budget' must be a map", key="budget")
        budget.update(raw)
    if "max_usd" in options:
        if "max_cost_usd" in budget:
            raise _fail(call, "CALL options 'max_usd' and budget.max_cost_usd are the same limit; give one", key="max_usd")
        budget["max_cost_usd"] = options["max_usd"]
    if "max_requests" in options:
        if "max_requests" in budget:
            raise _fail(call, "CALL options 'max_requests' and budget.max_requests are the same limit; give one",
                        key="max_requests")
        budget["max_requests"] = options["max_requests"]
    return {"budget": budget} if budget else {}


def _decide_questions(call: CallClause, options: Mapping[str, OptionValue]) -> Tuple[List[OptionValue], str]:
    """Return the kernel questions and the id of the sugar question (empty for a raw ``questions`` list)."""
    if ("question" in options) == ("questions" in options):
        raise _fail(call, "CALL graphistry.sem_decide needs exactly one of 'question' (text) or 'questions' (list of maps)",
                    key="question")
    if "questions" in options:
        questions = options["questions"]
        if not isinstance(questions, list) or not questions:
            raise _fail(call, "CALL option 'questions' must be a non-empty list of {id, kind, text} maps", key="questions")
        for banned in ("kind", "options", "levels", "out_col", "score_col", "status_col"):
            if banned in options:
                raise _fail(call, f"CALL option {banned!r} applies to 'question', not to a 'questions' list", key=banned)
        return list(questions), ""
    qid = _text(call, options, "out_col", "verdict")
    question: Dict[str, OptionValue] = {"id": qid, "text": _text(call, options, "question")}
    if "levels" in options:
        question["kind"], question["levels"] = options.get("kind", "score"), options["levels"]
    elif "options" in options:
        question["kind"], question["options"] = options.get("kind", "choice"), options["options"]
    else:
        question["kind"] = options.get("kind", "predicate")
    return [question], qid


def _question_ids(call: CallClause, questions: Sequence[OptionValue]) -> List[str]:
    ids: List[str] = []
    for question in questions:
        qid = question.get("id") if isinstance(question, dict) else None
        if not isinstance(qid, str) or not qid:
            raise _fail(call, "Each CALL question needs a non-empty string 'id'", key="questions")
        ids.append(qid)
    return ids


def _available(call: CallClause, options: Mapping[str, OptionValue], ids: Sequence[str], sugar_id: str,
               ) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """Map each YIELD name to the kernel column it reads; also return the default YIELD names."""
    if call.procedure == MAP:
        out = _text(call, options, "out_col", "result")
        names = {out if kind == "value" else kind: out + suffix for suffix, kind in _MAP_SUFFIXES}
        return names, (out, "status")
    if sugar_id:
        score = _text(call, options, "score_col", "score")
        status = _text(call, options, "status_col", "status")
        rename = {"value": sugar_id, "score": score, "status": status}
        names = {rename.get(kind, kind): sugar_id + suffix for suffix, kind in _DECIDE_SUFFIXES}
        return names, (sugar_id, score, status)
    names = {qid + suffix: qid + suffix for qid in ids for suffix, _ in _DECIDE_SUFFIXES}
    return names, tuple(name for qid in ids for name in (qid, qid + "__p", qid + "__status"))


def _yield_columns(
    call: CallClause, available: Mapping[str, str], defaults: Sequence[str], carried: Sequence[str],
) -> Tuple[Tuple[str, str], ...]:
    items: Sequence[CypherYieldItem] = call.yield_items
    chosen: List[Tuple[str, str]] = []
    if not items:
        chosen = [(name, name) for name in defaults]
    for item in items:
        if item.name not in available:
            raise GFQLValidationError(
                ErrorCode.E108, "Cypher YIELD references an unknown output of this semantic CALL",
                field="yield", value=item.name, suggestion=f"Use one of: {', '.join(sorted(available))}.",
                line=item.span.line, column=item.span.column, language="cypher",
            )
        chosen.append((item.name, item.alias or item.name))
    seen: Dict[str, str] = {}
    for source, output in chosen:
        if output in seen or output in carried:
            reason = "is used twice" if output in seen else "is already a column of the clause before the CALL"
            raise _fail(call, f"YIELD name {output!r} {reason}", value=output, key="yield",
                        suggestion="Rename it with AS.")
        seen[output] = source
    return tuple((available[source], output) for source, output in chosen)


def compile_semantic_call(
    call: CallClause,
    *,
    carried: Sequence[str],
    params: Optional[Mapping[str, OptionValue]],
) -> "CompiledCypherProcedureCall":
    from graphistry.compute.gfql.cypher.call_procedures import CompiledCypherProcedureCall, ProcedureOutputColumn

    options = _options(call, params)
    decide = call.procedure == DECIDE
    _only_keys(call, options, _DECIDE_KEYS if decide else _MAP_KEYS)
    kernel: Dict[str, OptionValue] = {
        "provider": options.get("provider", DEFAULT_PROVIDER),
        "task_id": options.get("task_id", DEFAULT_TASK_IDS[call.procedure]),
    }
    for key in _COMMON_KEYS - {"provider", "task_id", "budget", "template"}:
        if key in options:
            kernel[key] = options[key]
    kernel.update(_evidence(call, options, carried))
    kernel.update(_budget(call, options))
    ids: List[str] = []
    sugar_id = ""
    if decide:
        questions, sugar_id = _decide_questions(call, options)
        ids = _question_ids(call, questions)
        kernel["questions"] = questions
        for key in ("min_p", "require_probabilities"):
            if key in options:
                kernel[key] = options[key]
    else:
        if ("question" in options) == ("instruction" in options):
            raise _fail(call, "CALL graphistry.sem_map needs exactly one of 'instruction' or 'question' (the instruction text)",
                        key="instruction")
        kernel["instruction"] = _text(call, options, "instruction" if "instruction" in options else "question")
        kernel["out_col"] = _text(call, options, "out_col", "result")
        if "max_output_tokens" in options:
            kernel["max_output_tokens"] = options["max_output_tokens"]
    function = "sem_decide" if decide else "sem_map"
    validate_call_params(function, kernel)  # type: ignore[arg-type]
    available, defaults = _available(call, options, ids, sugar_id)
    outputs = tuple(
        ProcedureOutputColumn(source_name=source, output_name=output)
        for source, output in _yield_columns(call, available, defaults, carried)
    )
    return CompiledCypherProcedureCall(
        procedure=call.procedure,
        backend="semantic",
        algorithm=function,
        output_columns=outputs,
        result_kind="rows",
        row_kind="node",
        call_function=function,
        call_params=kernel,
        carried_columns=tuple(carried),
        line=call.span.line,
        column=call.span.column,
    )


def with_prefix(
    compiled_call: "CompiledCypherProcedureCall", prefix: Optional["CompiledCypherQuery"],
) -> "CompiledCypherProcedureCall":
    return replace(compiled_call, prefix=prefix)


def _select_and_rename(frame: DataFrameT, columns: List[str], renames: Dict[str, str]) -> DataFrameT:
    if is_polars_df(frame):
        return frame.select(columns).rename(renames)
    return frame[columns].rename(columns=renames)


def execute_semantic_call(
    rows: Plottable,
    compiled_call: "CompiledCypherProcedureCall",
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional["PolicyDict"],
    context: Optional["ExecutionContext"],
) -> Plottable:
    """Run the kernel over ``rows`` (the table before the CALL); keep carried columns, add the yielded ones."""
    from graphistry.compute.gfql.call.executor import execute_call

    from graphistry.compute.semantic.explain import _host_frame

    assert compiled_call.call_function is not None
    resolved = resolve_engine(EngineAbstract(engine) if isinstance(engine, str) else engine, rows)
    if rows._nodes is not None and is_polars_df(rows._nodes):
        # The kernels read pandas or cuDF, as g.sem_decide reads a Polars graph; the caller converts the result back.
        host = rows.bind()
        host._nodes = _host_frame(rows._nodes)
        host._edges = None if rows._edges is None else _host_frame(rows._edges)
        rows, resolved = host, Engine.PANDAS
    result = execute_call(rows, compiled_call.call_function, dict(compiled_call.call_params), resolved,
                          policy=policy, context=context)
    frame = result._nodes
    if frame is None:
        raise GFQLValidationError(
            ErrorCode.E108, "The semantic CALL needs a row table before it", field="call",
            value=compiled_call.procedure, suggestion="Project columns with WITH before the CALL.", language="cypher",
        )
    assert rows._nodes is not None
    carried = [str(col) for col in rows._nodes.columns]
    clash = [out.output_name for out in compiled_call.output_columns if out.output_name in carried]
    if clash:
        raise GFQLValidationError(
            ErrorCode.E108, f"YIELD name {clash[0]!r} is already a column of the clause before the CALL",
            field="yield", value=clash[0], suggestion="Rename it with AS.", language="cypher",
        )
    sources = [out.source_name for out in compiled_call.output_columns]
    renames = {out.source_name: out.output_name for out in compiled_call.output_columns
               if out.source_name != out.output_name}
    projected = _select_and_rename(frame, carried + sources, renames)
    out = result.bind()
    out._nodes = projected
    out._edges = result._edges[:0] if result._edges is not None else projected[:0]
    out._node = str(projected.columns[0]) if len(projected.columns) > 0 else None
    return out
