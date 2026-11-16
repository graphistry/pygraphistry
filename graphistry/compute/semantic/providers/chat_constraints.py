"""Compile a decision's answer codes to chat API constraints."""
from typing import Dict, Tuple

from graphistry.compute.semantic.providers.http import JSONObject


GRAMMAR_FIELDS: Dict[str, Tuple[str, str]] = {
    "vllm_choice": ("structured_outputs", "choice"),
    "vllm_regex": ("structured_outputs", "regex"),
    "vllm_grammar": ("structured_outputs", "grammar"),
    "guided_choice": ("guided_choice", "choice"),
    "guided_regex": ("guided_regex", "regex"),
    "guided_grammar": ("guided_grammar", "grammar"),
    "sglang_regex": ("regex", "regex"),
    "sglang_ebnf": ("ebnf", "grammar"),
}
CONSTRAINT_KEYS = frozenset({
    "response_format", "tools", "tool_choice", "functions", "function_call", "structured_outputs",
    "guided_choice", "guided_regex", "guided_grammar", "guided_json", "regex", "ebnf", "json_schema",
    "logprobs", "top_logprobs",
})
JSON_INSTRUCTION = ' Reply with a JSON object containing only "answer", whose value is one answer code.'


def answer_schema(codes: str) -> JSONObject:
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": list(codes)}},
            "required": ["answer"], "additionalProperties": False}


def constraint_fields(strategy: str, codes: str) -> JSONObject:
    schema = answer_schema(codes)
    function: JSONObject = {"name": "answer", "parameters": schema, "description": "The code of the answer."}
    if strategy == "json_schema":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
    if strategy == "json_mode":
        return {"response_format": {"type": "json_object"}}
    if strategy == "tool":
        return {"tools": [{"type": "function", "function": function}],
                "tool_choice": {"type": "function", "function": {"name": "answer"}}}
    if strategy == "function":
        return {"functions": [function], "function_call": {"name": "answer"}}
    if strategy in GRAMMAR_FIELDS:
        field, kind = GRAMMAR_FIELDS[strategy]
        constraint = (list(codes) if kind == "choice" else "(" + "|".join(codes) + ")" if kind == "regex"
                      else "root ::= " + " | ".join('"' + code + '"' for code in codes))
        return {field: {kind: constraint}} if field == "structured_outputs" else {field: constraint}
    return {}
