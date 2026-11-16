"""Caller requirements for measured decision probabilities; no provider ABI changes."""
from dataclasses import replace

from graphistry.compute.semantic.contract import DecisionProvider, DecisionQuestion, DecisionResult
from graphistry.compute.semantic.providers.chat import AnthropicMessagesProvider, OpenAIChatProvider
from graphistry.compute.semantic.providers.langchain import LangChainDecisionProvider
from graphistry.compute.semantic.rules import ProbabilityRequirement


def known_probability_gap(provider: DecisionProvider, requirement: ProbabilityRequirement) -> bool:
    """Only exact shipped chat types have known semantics; custom providers/subclasses are checked per answer."""
    if requirement is False:
        return False
    if type(provider) in (AnthropicMessagesProvider, LangChainDecisionProvider):
        return True
    if isinstance(provider, OpenAIChatProvider) and type(provider) is OpenAIChatProvider:
        if requirement == "native":
            return True
        return provider.strategy != "logprobs" and (
            provider.strategy != "auto" or "logprobs" not in provider.probe_strategies)
    return False


def enforce_probability(question: DecisionQuestion, result: DecisionResult,
                        requirement: ProbabilityRequirement) -> DecisionResult:
    """Apply after content/alignment validation and before min_p; preserve answerless uncertainty/errors/skips."""
    if requirement is False or result.status in ("error", "skipped") or result.value is None:
        return result
    kinds = ("native_decision",) if requirement == "native" else ("native_decision", "token_logprob")
    key = ("true" if result.value else "false") if question.kind == "predicate" else str(result.value)
    if result.probability_kind in kinds and result.distribution is not None and key in result.distribution:
        return result
    return replace(result, value=None, distribution=None, confidence=None, probability_kind="none", status="error",
                   error_code="probabilities_unavailable", status_reason=None)
